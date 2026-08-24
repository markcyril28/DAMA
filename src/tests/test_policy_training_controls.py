import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import dama.ai.ml.trainer as trainer_module
from dama.ai.ml import checkpoint_acceptance
from dama.ai.ml.corpus import SnapshotDecision, canonical_state_key
from dama.ai.ml.model_vs_algo import opening_suite_identity
from dama.ai.ml.trainer import (
    Trainer,
    TrainingConfig,
    TrainingStats,
    activate_enhanced_stage,
    config_from_yaml,
    load_config_from_yaml,
    validate_recovery_experiment_config,
)


def test_selfplay_executor_shutdown_captures_workers_before_nonblocking_shutdown() -> None:
    class _Process:
        def __init__(self, alive: bool) -> None:
            self.alive = alive
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_calls = []

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def join(self, timeout=None) -> None:
            self.join_calls.append(timeout)

    class _Executor:
        def __init__(self, process) -> None:
            self._processes = {1: process}
            self.shutdown_calls = []

        def shutdown(self, wait=True, cancel_futures=False) -> None:
            self.shutdown_calls.append((wait, cancel_futures))
            self._processes = None

    process = _Process(alive=False)
    executor = _Executor(process)

    trainer_module._shutdown_selfplay_executor(executor, timeout=0)

    assert executor.shutdown_calls == [(False, True)]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.join_calls


def test_selfplay_executor_shutdown_escalates_only_lingering_workers() -> None:
    class _Process:
        def __init__(self, alive: bool) -> None:
            self.alive = alive
            self.terminate_calls = 0
            self.kill_calls = 0

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def join(self, timeout=None) -> None:
            return None

    class _Executor:
        def __init__(self, processes) -> None:
            self._processes = {index: process for index, process in enumerate(processes)}

        def shutdown(self, wait=True, cancel_futures=False) -> None:
            self._processes = None

    graceful = _Process(alive=False)
    stuck = _Process(alive=True)
    executor = _Executor([graceful, stuck])

    trainer_module._shutdown_selfplay_executor(executor, timeout=0)

    assert graceful.terminate_calls == 0
    assert graceful.kill_calls == 0
    assert stuck.terminate_calls == 1
    assert stuck.kill_calls == 1


def test_training_stats_separate_current_dataset_and_historical_loss() -> None:
    stats = TrainingStats.from_dict({
        "best_loss": 0.0345,
        "loss_history": [{"step": 10, "loss": 0.9}],
    })
    stats.current_train_loss = 0.9
    stats.current_dataset_best_train_loss = 0.8
    payload = stats.to_dict()

    assert payload["current_train_loss"] == 0.9
    assert payload["current_dataset_best_train_loss"] == 0.8
    assert payload["historical_best_train_loss"] == 0.0345
    assert payload["best_loss"] == 0.0345


def test_dataset_fingerprint_resets_only_current_dataset_baseline() -> None:
    holder = object.__new__(Trainer)
    holder.stats = TrainingStats(
        current_dataset_best_train_loss=0.4,
        historical_best_train_loss=0.03,
        best_loss=0.03,
        dataset_fingerprint="old",
    )
    holder._active_snapshot_manifest = {}

    Trainer._activate_dataset_manifest(holder, {
        "fingerprint": "new",
        "version": 2,
        "files": [],
        "metrics": {},
        "teacher_settings": {},
        "noise_settings": {},
        "generation_settings": {},
    })
    assert holder.stats.current_dataset_best_train_loss == float("inf")
    assert holder.stats.historical_best_train_loss == 0.03

    holder.stats.current_dataset_best_train_loss = 0.5
    Trainer._activate_dataset_manifest(holder, holder._active_snapshot_manifest)
    assert holder.stats.current_dataset_best_train_loss == 0.5


def test_checkpoint_load_restores_loss_baselines_monotonically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint baselines survive a missing/stale stats sidecar."""
    import torch

    checkpoint = tmp_path / "model_step_12.pt"
    torch.save({
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "step": 12,
        "epoch": 3,
        "loss": 0.4,
        "current_dataset_best_train_loss": 0.4,
        "historical_best_train_loss": 0.03,
        "dataset_fingerprint": "checkpoint-corpus",
    }, checkpoint)

    class _Model:
        def load_state_dict(self, *_args, **_kwargs):
            return [], []

    class _Optimizer:
        param_groups = [{"lr": 1e-3}]

        def load_state_dict(self, _state):
            return None

    holder = object.__new__(Trainer)
    holder.device = torch.device("cpu")
    holder.config = TrainingConfig(learning_rate=2e-3)
    holder.model = _Model()
    holder.optimizer = _Optimizer()
    holder.scheduler = None
    holder.scaler = None
    holder.stats = TrainingStats(
        current_dataset_best_train_loss=0.5,
        historical_best_train_loss=0.02,
    )
    holder.step = 0
    holder.epoch = 0
    holder.best_loss = float("inf")
    holder._has_non_finite_tensors = lambda: False

    Trainer._load_checkpoint(holder, str(checkpoint))

    # Never worsen a baseline already recorded in the sidecar, while a fresh
    # sidecar (the common recovery case) can be populated from the checkpoint.
    assert holder.stats.current_dataset_best_train_loss == pytest.approx(0.4)
    assert holder.stats.historical_best_train_loss == pytest.approx(0.02)
    assert holder.stats.best_loss == pytest.approx(0.02)

    # A checkpoint from another corpus must not contaminate the active
    # dataset's current-loss baseline.
    holder.stats.dataset_fingerprint = "different-corpus"
    holder.stats.current_dataset_best_train_loss = 0.5
    Trainer._load_checkpoint(holder, str(checkpoint))
    assert holder.stats.current_dataset_best_train_loss == pytest.approx(0.5)


def test_checkpoint_load_resets_optimizer_param_groups_to_config(
    tmp_path: Path,
) -> None:
    import torch

    source_parameters = [
        torch.nn.Parameter(torch.tensor([1.0])),
        torch.nn.Parameter(torch.tensor([2.0])),
    ]
    source_optimizer = torch.optim.AdamW([
        {
            "params": [source_parameters[0]],
            "lr": 7e-4,
            "weight_decay": 1e-5,
        },
        {
            "params": [source_parameters[1]],
            "lr": 8e-4,
            "weight_decay": 2e-5,
        },
    ])
    checkpoint = tmp_path / "model_step_12.pt"
    torch.save({
        "model_state_dict": {},
        "optimizer_state_dict": source_optimizer.state_dict(),
        "step": 12,
    }, checkpoint)

    class _Model:
        def load_state_dict(self, *_args, **_kwargs):
            return [], []

    target_parameters = [
        torch.nn.Parameter(torch.tensor([3.0])),
        torch.nn.Parameter(torch.tensor([4.0])),
    ]
    holder = object.__new__(Trainer)
    holder.device = torch.device("cpu")
    holder.config = TrainingConfig(
        learning_rate=2e-4,
        weight_decay=1e-4,
    )
    holder.model = _Model()
    holder.optimizer = torch.optim.AdamW([
        {"params": [target_parameters[0]]},
        {"params": [target_parameters[1]]},
    ])
    holder.scheduler = None
    holder.scaler = None
    holder.stats = TrainingStats()
    holder.step = 0
    holder.epoch = 0
    holder.best_loss = float("inf")
    holder._has_non_finite_tensors = lambda: False

    Trainer._load_checkpoint(holder, str(checkpoint))

    for group in holder.optimizer.param_groups:
        assert group["lr"] == pytest.approx(holder.config.learning_rate)
        assert group["weight_decay"] == pytest.approx(
            holder.config.weight_decay)


def test_promotion_metadata_records_live_optimizer_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "total_states": 4,
        "correct_states": 2,
        "top1_teacher_agreement": 0.5,
        "decision_states": 3,
        "decision_correct_states": 1,
        "decision_top1_teacher_agreement": 1 / 3,
        "forced_move_fraction": 0.25,
    }
    monkeypatch.setattr(
        trainer_module,
        "evaluate_teacher_agreement",
        lambda *_args, **_kwargs: result,
    )
    captured = {}

    class _Registry:
        def consider(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                promoted=True,
                reason="best_held_out_teacher_agreement",
                record={
                    **kwargs,
                    "promoted": True,
                    "reason": "best_held_out_teacher_agreement",
                },
            )

    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        learning_rate=2e-4,
        weight_decay=1e-4,
    )
    holder.optimizer = SimpleNamespace(param_groups=[{
        "lr": 3e-4,
        "weight_decay": 1e-5,
    }])
    holder.model = object()
    holder.stats = TrainingStats(dataset_fingerprint="dataset")
    holder.step = 12
    holder.epoch = 3
    holder._frozen_suite_entries = [object()]
    holder._frozen_suite_manifest = {"suite_sha256": "suite"}
    holder._promotion_registry = _Registry()

    selection = Trainer._evaluate_teacher_promotion(
        holder, tmp_path / "model_step_12.pt")

    context = captured["comparison_context"]
    assert context["learning_rate"] == pytest.approx(
        holder.optimizer.param_groups[0]["lr"])
    assert context["weight_decay"] == pytest.approx(
        holder.optimizer.param_groups[0]["weight_decay"])
    assert selection["promotion"]["promoted"] is True


def test_saved_checkpoint_carries_declared_dataset_provenance(
    tmp_path: Path,
) -> None:
    """P0 requires every checkpoint to record what corpus produced it.

    The audit verified these keys by deserializing live artifacts only; nothing
    asserted them, so a refactor could drop one silently and the loss-semantics
    and promotion tests would all still pass.
    """
    import threading
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    manifest = {
        "manifest_path": str(tmp_path / "snapshot_v000001" / "manifest.json"),
        "version": 1,
        "files": [{"name": "replay_a.jsonl", "sha256": "aa", "size_bytes": 10}],
        "metrics": {
            "post_dedup_unique_state_count": 4321,
            "forced_move_rate": 0.191,
            "forced_move_count": 825,
        },
        "teacher_settings": {"difficulty": "hard"},
        "noise_settings": {"played_action_probability": 0.10,
                           "label_is_teacher": True},
        "generation_settings": {"algorithm_fraction": 0.70,
                                "model_fraction": 0.30},
    }

    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        latest_path=str(tmp_path / "latest.pt"),
        learning_rate=2e-4,
        weight_decay=1e-4,
    )
    holder.model = SimpleNamespace(state_dict=lambda: {}, arch_params={})
    holder.optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {}, "param_groups": []})
    holder.stats = TrainingStats(
        dataset_fingerprint="corpus-fingerprint",
        dataset_metadata={"fingerprint": "corpus-fingerprint", "version": 1},
    )
    holder.step = 136000
    holder.epoch = 7
    holder.scheduler = None
    holder.scaler = None
    holder.stats_collector = None
    holder.log_file = str(tmp_path / "train.jsonl")
    holder.device = torch.device("cpu")
    holder._checkpoint_thread = None
    holder._active_snapshot_manifest = manifest
    holder._evaluate_validation_loss = lambda: 0.42
    holder._evaluate_teacher_promotion = lambda _path: None
    holder._live_optimizer_context = lambda: {
        "learning_rate": 2e-4, "weight_decay": 1e-4}
    holder._snapshot_stats = lambda: {}
    holder._save_stats = lambda **_kwargs: None
    holder._put_status = lambda _message: None

    path = Trainer._save_checkpoint(holder, loss=0.9)
    thread = holder._checkpoint_thread
    assert isinstance(thread, threading.Thread)
    thread.join(timeout=30)
    assert not thread.is_alive()

    saved = torch.load(path, map_location="cpu", weights_only=False)

    assert saved["dataset_fingerprint"] == "corpus-fingerprint"
    assert saved["dataset_metadata"]["version"] == 1
    assert saved["snapshot_manifest_path"] == manifest["manifest_path"]
    assert saved["snapshot_file_list"] == manifest["files"]
    assert saved["snapshot_unique_state_count"] == 4321
    assert saved["snapshot_forced_move_rate"] == pytest.approx(0.191)
    assert saved["snapshot_forced_move_count"] == 825
    assert saved["teacher_settings"] == manifest["teacher_settings"]
    assert saved["noise_settings"] == manifest["noise_settings"]
    assert saved["generation_settings"] == manifest["generation_settings"]
    # Never call a checkpoint "best" from training loss alone.
    assert saved["selection_basis"] == "held_out_teacher_agreement"


def test_checkpoint_collision_still_measures_without_writing(
    tmp_path: Path,
) -> None:
    """A step collision must not silently skip validation and agreement.

    Resuming from step 134,000 into a directory already holding checkpoints
    136,000-196,000 made the collision path return before
    _evaluate_validation_loss() and _evaluate_teacher_promotion(), so ~62,000
    steps produced no measurement at all. The checkpoint, the latest alias and
    the append-only promotion registry must still be left untouched.
    """
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    existing = checkpoint_dir / "model_step_136000.pt"
    torch.save({"marker": "original"}, existing)
    original_bytes = existing.read_bytes()

    calls = []

    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        latest_path=str(tmp_path / "latest.pt"),
    )
    holder.step = 136000
    holder.stats = TrainingStats()
    holder._checkpoint_thread = None
    holder._verify_existing_checkpoint_collision = (
        lambda _p: calls.append("verified"))
    holder._evaluate_validation_loss = (
        lambda: (calls.append("val_loss"), 0.4242)[1])
    holder._record_validation_stats = (
        lambda v: calls.append(("recorded", v)))
    holder._evaluate_teacher_promotion = (
        lambda _p: (calls.append("agreement"), {"promotion": {}})[1])
    holder._save_stats = lambda **_k: calls.append("stats_saved")

    returned = Trainer._save_checkpoint(holder, loss=0.9)

    assert returned == str(existing)
    # Measured...
    assert "val_loss" in calls, "collision path skipped validation loss"
    assert ("recorded", 0.4242) in calls, "validation loss was not recorded"
    assert "agreement" in calls, "collision path skipped teacher agreement"
    assert "stats_saved" in calls, "measurements were not made durable"
    # ...but wrote nothing.
    assert existing.read_bytes() == original_bytes
    assert not (tmp_path / "latest.pt").exists()
    assert holder._checkpoint_thread is None, "collision must not start a write"


def test_checkpoint_load_rewinds_newer_sidecar_histories(
    tmp_path: Path,
) -> None:
    import torch

    checkpoint = tmp_path / "model_step_12.pt"
    torch.save({
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "step": 12,
        "epoch": 3,
        "loss": 0.4,
        "current_train_loss": 0.4,
        "current_dataset_best_train_loss": 0.35,
        "historical_best_train_loss": 0.03,
        "validation_loss": 0.45,
        "dataset_fingerprint": "checkpoint-corpus",
        "dataset_metadata": {"version": 2},
        "generation_cycles_completed": 2,
    }, checkpoint)

    class _Model:
        def load_state_dict(self, *_args, **_kwargs):
            return [], []

    class _Optimizer:
        param_groups = [{"lr": 1e-3}]

        def load_state_dict(self, _state):
            return None

    holder = object.__new__(Trainer)
    holder.device = torch.device("cpu")
    holder.config = TrainingConfig(learning_rate=2e-3)
    holder.model = _Model()
    holder.optimizer = _Optimizer()
    holder.scheduler = None
    holder.scaler = None
    holder.stats = TrainingStats(
        total_steps=15,
        epochs_completed=5,
        current_train_loss=0.1,
        current_dataset_best_train_loss=0.1,
        historical_best_train_loss=0.02,
        dataset_fingerprint="future-corpus",
        dataset_metadata={"version": 3},
        generation_cycles_completed=9,
        loss_history=[
            {"step": 12, "loss": 0.4},
            {"step": 15, "loss": 0.1},
        ],
        val_loss_history=[
            {"step": 12, "val_loss": 0.5},
            {"step": 15, "val_loss": 0.2},
        ],
        lr_history=[{"step": 12}, {"step": 15}],
        gpu_mem_history=[{"step": 12}, {"step": 15}],
        step_times=[{"step": 12}, {"step": 15}],
        test_history=[{"step": 12}, {"step": 15}],
        teacher_agreement_history=[
            {"step": 12, "top1_teacher_agreement": 0.3},
            {"step": 15, "top1_teacher_agreement": 0.6},
        ],
        promotion_history=[{"step": 12}, {"step": 15}],
        acceptance_history=[{"step": 12}, {"step": 15}],
    )
    holder.step = 0
    holder.epoch = 0
    holder.best_loss = float("inf")
    holder._has_non_finite_tensors = lambda: False

    Trainer._load_checkpoint(holder, str(checkpoint))

    assert holder.step == 12
    assert holder.epoch == 3
    assert holder.stats.total_steps == 12
    assert holder.stats.epochs_completed == 3
    assert holder.stats.current_train_loss == pytest.approx(0.4)
    assert holder.stats.dataset_fingerprint == "checkpoint-corpus"
    assert holder.stats.dataset_metadata == {"version": 2}
    assert holder.stats.current_dataset_best_train_loss == pytest.approx(0.35)
    assert holder.stats.historical_best_train_loss == pytest.approx(0.02)
    assert holder.stats.best_val_loss == pytest.approx(0.45)
    assert holder.stats.best_teacher_agreement == pytest.approx(0.3)
    assert holder.stats.generation_cycles_completed == 2
    for name in (
        "loss_history",
        "val_loss_history",
        "lr_history",
        "gpu_mem_history",
        "step_times",
        "test_history",
        "teacher_agreement_history",
        "promotion_history",
        "acceptance_history",
    ):
        assert all(entry.get("step", 0) <= 12 for entry in getattr(holder.stats, name))


def test_checkpoint_collision_fails_closed_without_overwriting(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    baseline = checkpoint_dir / "model_step_134000.pt"
    baseline.write_bytes(b"preserved baseline")
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(checkpoint_dir=str(checkpoint_dir))
    holder.step = 134000
    holder._checkpoint_thread = None

    with pytest.raises(RuntimeError, match="cannot be verified"):
        Trainer._save_checkpoint(holder, 1.0)

    assert baseline.read_bytes() == b"preserved baseline"


def test_verified_checkpoint_collision_preserves_existing_step(
    tmp_path: Path,
) -> None:
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model_step_134000.pt"
    torch.save({
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "step": 134000,
    }, checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(checkpoint_dir=str(checkpoint_dir))
    holder.step = 134000
    holder._checkpoint_thread = None

    saved = Trainer._save_checkpoint(holder, 1.0)

    assert Path(saved) == checkpoint
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


def test_recovery_checkpoint_collision_requires_registry_hash(
    tmp_path: Path,
) -> None:
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model_step_136000.pt"
    torch.save({
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict": {
            "param_groups": [{"lr": 2e-4, "weight_decay": 1e-4}],
        },
        "step": 136000,
        "recovery_experiment": {
            "enabled": True,
            "baseline_sha256": "ABC123",
            "training_stage": "policy_only",
        },
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    records = ({
        "step": 136000,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "comparison_context": {
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
        },
    },)
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        recovery_enforced=True,
        recovery_baseline_sha256="ABC123",
        policy_stage="policy_only",
    )
    holder.step = 136000
    holder._checkpoint_thread = None
    holder._promotion_registry = SimpleNamespace(records=lambda: records)

    assert Trainer._save_checkpoint(holder, 1.0) == str(checkpoint)

    holder._promotion_registry = SimpleNamespace(records=lambda: ({
        **records[0],
        "checkpoint_sha256": "0" * 64,
    },))
    with pytest.raises(RuntimeError, match="hash does not match"):
        Trainer._save_checkpoint(holder, 1.0)


def test_recovery_checkpoint_collision_rejects_false_optimizer_metadata(
    tmp_path: Path,
) -> None:
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model_step_136000.pt"
    torch.save({
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict": {
            "param_groups": [{"lr": 2e-4, "weight_decay": 1e-5}],
        },
        "step": 136000,
        "recovery_experiment": {
            "enabled": True,
            "baseline_sha256": "ABC123",
            "training_stage": "policy_only",
        },
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    records = ({
        "step": 136000,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "comparison_context": {
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
        },
    },)
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        recovery_enforced=True,
        recovery_baseline_sha256="ABC123",
        policy_stage="policy_only",
    )
    holder.step = 136000
    holder._checkpoint_thread = None
    holder._promotion_registry = SimpleNamespace(records=lambda: records)

    with pytest.raises(RuntimeError, match="optimizer state does not match"):
        Trainer._save_checkpoint(holder, 1.0)


def _cycle_allocator_holder(tmp_path: Path, generation_cycles: int) -> Trainer:
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(replay_dir=str(tmp_path / "replay"))
    holder.stats = TrainingStats(
        generation_cycles_completed=generation_cycles,
    )
    return holder


def test_generation_cycle_allocator_rewinds_stale_missing_stats_from_replay(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "replay_existing.jsonl").write_text(
        json.dumps({"game_id": "cycle-000027-model-000001"}) + "\n",
        encoding="utf-8",
    )
    holder = _cycle_allocator_holder(tmp_path, 0)

    assert holder._allocate_generation_cycle_id() == 28
    assert holder.stats.generation_cycles_completed == 0


def test_generation_cycle_allocator_never_collides_with_stats_or_replay(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "replay_existing.jsonl").write_text(
        json.dumps({"generation_cycle_id": 27}) + "\n",
        encoding="utf-8",
    )
    holder = _cycle_allocator_holder(tmp_path, 32)

    assert holder._allocate_generation_cycle_id() == 32


def test_generation_cycle_allocator_starts_at_zero_for_empty_replay(
    tmp_path: Path,
) -> None:
    (tmp_path / "replay").mkdir()
    holder = _cycle_allocator_holder(tmp_path, 0)

    assert holder._allocate_generation_cycle_id() == 0


def test_generation_cycle_allocator_caches_unchanged_replay_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    first = replay_dir / "replay_first.jsonl"
    second = replay_dir / "replay_second.jsonl"
    first.write_text(json.dumps({"game_id": "cycle-000003-model-1"}) + "\n")
    second.write_text(json.dumps({"game_id": "cycle-000007-model-1"}) + "\n")
    holder = _cycle_allocator_holder(tmp_path, 0)

    real_open = Path.open
    opened = []

    def counting_open(path, *args, **kwargs):
        if path.name.startswith("replay_"):
            opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)

    assert holder._durable_generation_cycle_max() == 7
    assert len(opened) == 2
    opened.clear()
    assert holder._durable_generation_cycle_max() == 7
    assert opened == []

    first.write_text(json.dumps({"game_id": "cycle-000011-model-1"}) + "\n")
    third = replay_dir / "replay_third.jsonl"
    third.write_text(json.dumps({"generation_cycle_id": 13}) + "\n")
    opened.clear()
    assert holder._durable_generation_cycle_max() == 13
    assert {path.name for path in opened} == {first.name, third.name}

    second.unlink()
    opened.clear()
    assert holder._durable_generation_cycle_max() == 13
    assert opened == []
    assert second.resolve() not in holder._generation_cycle_file_cache


def test_selfplay_entries_record_cycle_and_behavior_provenance() -> None:
    entries = [{"game_id": "cycle-000028-model-000001"}, "not-a-dict"]

    Trainer._annotate_selfplay_entries(
        entries,
        cycle_id=28,
        behavior_step=136000,
        behavior_id="trainer-step-136000",
        behavior_checkpoint_sha256="AB" * 32,
    )

    assert entries[0]["generation_cycle_id"] == 28
    assert entries[0]["model_behavior_step"] == 136000
    assert entries[0]["model_behavior_id"] == "trainer-step-136000"
    assert entries[0]["model_behavior_checkpoint_sha256"] == "AB" * 32


def test_changed_stage_contract_waits_for_matching_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_manifest = tmp_path / "snapshot_v000001" / "manifest.json"
    new_manifest = tmp_path / "snapshot_v000002" / "manifest.json"

    class FakeSnapshotManager:
        def __init__(self) -> None:
            self.calls = 0
            self.progress = False
            self.loaded = None

        def consider_snapshot(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return SnapshotDecision(
                    False,
                    "fresh rate below threshold",
                    old_manifest,
                    {"fresh_unique_state_rate": 0.0},
                )
            return SnapshotDecision(
                True,
                "admitted",
                new_manifest,
                {"fresh_unique_state_rate": 0.50},
            )

        def snapshot_matches_settings(self, path, **_kwargs):
            return Path(path) == new_manifest

        def eligible_replay_files(self):
            count = 2 if self.progress else 1
            return [tmp_path / f"eligible_{index}.jsonl" for index in range(count)], {}

        def load_split(self, path, max_train_entries=0):
            self.loaded = (Path(path), max_train_entries)
            return ["train"], ["validation"], {
                "fingerprint": "enhanced",
                "version": 2,
            }

    manager = FakeSnapshotManager()
    holder = object.__new__(Trainer)
    holder._snapshot_manager = manager
    holder.config = SimpleNamespace(selfplay_games=72, replay_max_entries=500)
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._stopped = False
    holder.step = 7
    holder._corpus_settings = lambda *_, **__: (
        {"stage": "enhanced"},
        {"played_action_probability": 0.10},
        {"current_model_inference_depth": 2},
    )
    selfplay_calls = []
    def _run_selfplay(games: int, return_behavior_step: bool = False) -> int | tuple[int, int]:
        selfplay_calls.append(games)
        manager.progress = True
        if return_behavior_step:
            return 0, holder.step
        return 0
    holder.run_selfplay = _run_selfplay
    holder._service_control_queue = lambda: None
    activated = []
    holder._activate_dataset_manifest = lambda manifest: activated.append(manifest)
    monkeypatch.setattr(
        trainer_module,
        "analyze_replay_files",
        lambda files: (
            {
                "records": len(files),
                "state_set_sha256": f"states-{len(files)}",
            },
            set(),
        ),
    )

    train, validation = Trainer._prepare_training_split(holder)

    assert train == ["train"]
    assert validation == ["validation"]
    assert selfplay_calls == [72]
    assert manager.loaded == (new_manifest, 500)
    assert activated == [{"fingerprint": "enhanced", "version": 2}]


def test_prepare_training_split_captures_behavior_step_before_selfplay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_manifest = Path("snapshot_v000001")
    new_manifest = Path("snapshot_v000002")
    observed_steps = []

    class FakeSnapshotManager:
        def __init__(self) -> None:
            self.calls = 0
            self.loaded = None
            self.progress = False

        def consider_snapshot(self, **kwargs):
            self.calls += 1
            observed_steps.append(
                int(kwargs["generation_settings"]["model_behavior_step"])
            )
            if self.calls == 1:
                return SnapshotDecision(
                    False,
                    "warming",
                    old_manifest,
                    {"fresh_unique_state_rate": 0.0},
                )
            return SnapshotDecision(
                True,
                "admitted",
                new_manifest,
                {"fresh_unique_state_rate": 0.50},
            )

        def snapshot_matches_settings(self, *_args, **_kwargs):
            return self.calls == 2

        def eligible_replay_files(self):
            count = 2 if self.progress else 1
            return [Path(f"replay_{index}.jsonl") for index in range(count)], {}

        def load_split(self, path, max_train_entries=0):
            self.loaded = (Path(path), max_train_entries)
            return ["train"], ["validation"], {
                "fingerprint": "enhanced",
                "version": 2,
            }

    manager = FakeSnapshotManager()
    holder = object.__new__(Trainer)
    holder._snapshot_manager = manager
    holder.config = SimpleNamespace(
        selfplay_games=72,
        replay_max_entries=500,
        teacher_difficulty="hard",
        teacher_target_type="hard",
        teacher_soft_temperature=1.0,
        teacher_score_depth=3,
        teacher_value_scale=1000.0,
        teacher_hard_label_blend=0.25,
        selfplay_noise_prob=0.10,
        selfplay_opening_plies=(0,),
        selfplay_opening_seed=20260819,
        selfplay_max_moves=200,
        selfplay_difficulties=None,
        algo_vs_algo_enabled=False,
        algo_vs_algo_games=0,
        trajectory_algorithm_fraction=0.70,
        trajectory_model_fraction=0.30,
        selfplay_opponent_focus="algorithm",
        selfplay_focus_side="both",
        symmetry_augmentation=False,
        inference_depth=1,
    )
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._stopped = False
    def _corpus_settings(
        *args, model_behavior_step=None, **_kwargs
    ) -> tuple[dict, dict, dict]:
        step = int(model_behavior_step) if model_behavior_step is not None else holder.step
        return (
            {"stage": "enhanced"},
            {"played_action_probability": 0.10},
            {"current_model_inference_depth": 2, "model_behavior_step": step},
        )
    holder._corpus_settings = _corpus_settings
    holder._service_control_queue = lambda: None
    holder._activate_dataset_manifest = lambda manifest: None
    holder.step = 11
    def _run_selfplay(
        games: int, return_behavior_step: bool = False
    ) -> int | tuple[int, int]:
        holder.step = 99
        manager.progress = True
        if return_behavior_step:
            return 0, 99
        return 0
    holder.run_selfplay = _run_selfplay

    monkeypatch.setattr(
        trainer_module,
        "analyze_replay_files",
        lambda files: (
            {"records": len(files)},
            set(),
        ),
    )
    train, validation = Trainer._prepare_training_split(holder)

    assert train == ["train"]
    assert validation == ["validation"]
    assert holder.step == 99
    assert observed_steps == [11, 99]
    assert manager.calls == 2


def test_background_selfplay_uses_snapshot_step_from_selfplay_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    new_manifest = Path("snapshot_v000002")
    observed_steps = []

    class _FakeThread:
        def __init__(self, target, daemon=False):
            self._target = target
            self._running = False

        def start(self):
            self._running = True
            try:
                self._target()
            finally:
                self._running = False

        def is_alive(self):
            return self._running

        def join(self, _timeout=None):
            return None

    class FakeSnapshotManager:
        def consider_snapshot(self, **kwargs):
            observed_steps.append(
                int(kwargs["generation_settings"]["model_behavior_step"])
            )
            return SnapshotDecision(
                True,
                "admitted",
                new_manifest,
                {"fresh_unique_state_rate": 0.50},
            )

        def snapshot_matches_settings(self, *_args, **_kwargs):
            return True

        def eligible_replay_files(self):
            return [Path("replay.json")], {}

        def load_split(self, path, max_train_entries=0):
            return ["train"], ["validation"], {
                "fingerprint": "enhanced",
                "version": 2,
            }

    holder = object.__new__(Trainer)
    holder._snapshot_manager = FakeSnapshotManager()
    holder.config = SimpleNamespace(replay_max_files=60, replay_max_entries=500)
    holder.config.max_moves_per_sample = 32
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._bg_selfplay_thread = None
    holder._bg_selfplay_dataset = None
    holder._bg_selfplay_entries = None
    holder._bg_selfplay_incremental = None
    holder._bg_snapshot_manifest = None
    holder._bg_validation_entries = None
    holder._bg_selfplay_lock = trainer_module.threading.Lock()
    holder._stopped = False
    holder._paused = False
    holder._data_ready_event = SimpleNamespace(set=lambda: None)
    holder.step = 23
    holder._corpus_settings = lambda model_behavior_step=None, **_: (
        {"difficulty": "hard", "target_type": "hard"},
        {"played_action_probability": 0.10},
        {"model_behavior_step": model_behavior_step},
    )
    def _run_selfplay(
        num_games: int, **_kwargs: object
    ) -> tuple[int, int]:
        holder.step = 99
        holder._stopped = True
        return 0, 23
    holder.run_selfplay = _run_selfplay

    monkeypatch.setattr(trainer_module.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        trainer_module.CachedTensorDataset,
        "from_entries",
        staticmethod(lambda *_args, **_kwargs: "dataset"),
    )

    Trainer._start_background_selfplay(holder, 72)

    assert observed_steps == [23]
    assert holder._bg_snapshot_manifest["fingerprint"] == "enhanced"


def test_runtime_model_root_and_cleanup_removes_temporary_files(
    tmp_path: Path,
) -> None:
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(runtime_model_root=str(tmp_path / "logs" / "runtime_models"))
    holder._runtime_model_dir = None

    temp_path = holder._runtime_model_path("temp_selfplay_model.pt")
    temp_path2 = holder._runtime_model_path("temp_async_test.pt")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text("temporary", encoding="utf-8")
    temp_path2.write_text("shared", encoding="utf-8")

    # Verify overlap-safe cleanup unlinks only the target file.
    holder._cleanup_runtime_model_file(temp_path)
    assert not temp_path.exists()
    assert temp_path2.exists()
    assert temp_path.parent.exists()

    # Remove the remaining file and confirm the directory is cleaned up.
    holder._cleanup_runtime_model_file(temp_path2)
    holder._cleanup_runtime_models_dir()
    assert not temp_path.parent.exists()


def test_existing_frozen_suite_overlap_is_filtered_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "frozen_suite.jsonl"
    suite_path.write_text("existing\n", encoding="utf-8")
    state = {
        "p1_men": [[0, 1]],
        "p1_kings": [],
        "p2_men": [[7, 6]],
        "p2_kings": [],
        "turn": 1,
        "move_count": 4,
    }
    overlap_key = canonical_state_key(state)
    captured = {}

    class FakeSnapshotManager:
        def __init__(self) -> None:
            self.external_keys = set()

        def eligible_replay_files(self):
            return [tmp_path / "replay_repaired.jsonl"], {}

        def set_external_validation_state_keys(self, keys):
            self.external_keys = set(keys)

    manager = FakeSnapshotManager()
    holder = object.__new__(Trainer)
    holder._snapshot_manager = manager
    holder.config = SimpleNamespace(
        validation_enabled=True,
        frozen_suite_path=str(suite_path),
        frozen_suite_auto_create=True,
        frozen_suite_size=1,
        frozen_suite_seed=99,
        teacher_difficulty="hard",
        selfplay_opening_plies=(2, 4),
        selfplay_noise_prob=0.10,
        selfplay_max_moves=200,
    )
    monkeypatch.setattr(
        trainer_module,
        "analyze_replay_files",
        lambda _files: ({"records": 1}, {overlap_key}),
    )
    monkeypatch.setattr(
        trainer_module,
        "create_frozen_teacher_suite",
        lambda *_args, **kwargs: captured.update(kwargs) or {},
    )
    monkeypatch.setattr(
        trainer_module,
        "load_frozen_teacher_suite",
        lambda *_args, **_kwargs: (
            [SimpleNamespace(state=state)],
            {"suite_sha256": "a" * 64},
        ),
    )

    Trainer._ensure_frozen_teacher_suite(holder)

    assert captured["exclude_state_keys"] is None
    assert manager.external_keys == {overlap_key}


def _valid_recovery_config(checkpoint: Path) -> TrainingConfig:
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    root = checkpoint.parent
    policy_namespace = "policy_arm"
    enhanced_namespace = "enhanced_arm"
    return TrainingConfig(
        resume=str(checkpoint),
        recovery_enforced=True,
        recovery_baseline_path=str(checkpoint),
        recovery_baseline_sha256=digest,
        validation_enabled=True,
        validation_fraction=0.15,
        frozen_suite_size=5000,
        teacher_agreement_threshold=0.50,
        snapshot_enabled=True,
        snapshot_min_fresh_fraction=0.50,
        teacher_difficulty="hard",
        selfplay_noise_prob=0.10,
        trajectory_algorithm_fraction=0.70,
        trajectory_model_fraction=0.30,
        selfplay_games=72,
        algo_vs_algo_enabled=True,
        algo_vs_algo_games=168,
        selfplay_opponent_focus="algorithm",
        algo_vs_algo_difficulties=["easy", "medium", "hard"],
        selfplay_opening_plies=(2, 4, 6),
        test_games=100,
        test_vs_algo=True,
        test_promoted_only=True,
        test_opening_plies=(2, 4, 6, 8),
        test_opponents=("random", "easy"),
        test_confidence_method="wilson_score",
        policy_stage="policy_only",
        value_head_enabled=False,
        inference_depth=1,
        teacher_target_type="hard",
        checkpoint_dir=str(root / f"checkpoints_{policy_namespace}"),
        runtime_model_root=str(root / f"runtime_{policy_namespace}"),
        latest_path=str(root / f"latest_{policy_namespace}.pt"),
        promoted_path=str(root / f"promoted_{policy_namespace}.pt"),
        accepted_path=str(root / f"accepted_{policy_namespace}.pt"),
        replay_dir=str(root / f"replay_{policy_namespace}"),
        log_dir=str(root / f"logs_{policy_namespace}"),
        stats_file=str(root / f"stats_{policy_namespace}.json"),
        promotion_registry=str(
            root / f"registry_{policy_namespace}" / "promotions.jsonl"),
        acceptance_dir=str(root / f"acceptance_{policy_namespace}"),
        stats_output_dir=str(root / f"stats_output_{policy_namespace}"),
        snapshot_root=str(root / f"snapshots_{policy_namespace}"),
        ram_cache_file=str(root / f"cache_{policy_namespace}.pt"),
        policy_output_namespace=policy_namespace,
        enhanced_output_namespace=enhanced_namespace,
    )


def test_recovery_gate_accepts_only_exact_baseline_and_controls(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_step_134000.pt"
    checkpoint.write_bytes(b"immutable baseline")
    config = _valid_recovery_config(checkpoint)

    validate_recovery_experiment_config(config)
    with pytest.raises(ValueError, match="resume-latest"):
        validate_recovery_experiment_config(config, resume_latest_requested=True)

    other = tmp_path / "model_step_136000.pt"
    other.write_bytes(b"other")
    config.resume = str(other)
    with pytest.raises(ValueError, match="resume only"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_accepts_an_approved_numbered_anchor_other_than_134000(
    tmp_path: Path,
) -> None:
    """The 2026-08-24 continuation resumes step 174000, not step 134000.

    The gate used to pin the anchor by filename to model_step_134000.pt, which
    would reject the approved continuation outright.  Identity is now carried
    by the SHA-256 pin; the filename rule only keeps the anchor an immutable
    numbered checkpoint rather than a moving alias.
    """
    checkpoint = tmp_path / "model_step_174000.pt"
    checkpoint.write_bytes(b"approved continuation anchor")
    config = _valid_recovery_config(checkpoint)

    validate_recovery_experiment_config(config)


@pytest.mark.parametrize(
    "anchor_name",
    ["latest.pt", "promoted_policy.pt", "model_step_.pt", "model_step_174000.pth"],
)
def test_recovery_gate_refuses_an_anchor_that_is_not_a_numbered_checkpoint(
    tmp_path: Path,
    anchor_name: str,
) -> None:
    """Aliases are republished in place, so pinning one is unverifiable."""
    checkpoint = tmp_path / anchor_name
    checkpoint.write_bytes(b"moving alias")
    config = _valid_recovery_config(checkpoint)

    with pytest.raises(ValueError, match="numbered checkpoint"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_refuses_an_anchor_whose_digest_does_not_match(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_step_174000.pt"
    checkpoint.write_bytes(b"approved continuation anchor")
    config = _valid_recovery_config(checkpoint)
    checkpoint.write_bytes(b"a different checkpoint entirely")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_accepts_another_path_to_the_same_baseline_file(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_step_134000.pt"
    checkpoint.write_bytes(b"immutable baseline")
    same_file_alias = tmp_path / "baseline-path-alias.pt"
    os.link(checkpoint, same_file_alias)
    config = _valid_recovery_config(checkpoint)
    config.resume = str(same_file_alias)

    validate_recovery_experiment_config(config)


def test_recovery_gate_rejects_unready_data_controls(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_step_134000.pt"
    checkpoint.write_bytes(b"immutable baseline")
    config = _valid_recovery_config(checkpoint)
    config.snapshot_min_fresh_fraction = 0.49
    config.trajectory_model_fraction = 0.0

    with pytest.raises(ValueError, match="50% fresh"):
        validate_recovery_experiment_config(config)


@pytest.mark.parametrize("alias_name", ["latest_path", "promoted_path", "accepted_path"])
def test_recovery_gate_rejects_alias_targeting_baseline(
    tmp_path: Path, alias_name: str,
) -> None:
    checkpoint = tmp_path / "model_step_134000.pt"
    checkpoint.write_bytes(b"immutable baseline")
    config = _valid_recovery_config(checkpoint)
    setattr(config, alias_name, str(checkpoint))

    with pytest.raises(ValueError, match="recovery baseline"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_requires_checkpoint_output_directory_isolation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_step_134000.pt"
    checkpoint.write_bytes(b"immutable baseline")
    config = _valid_recovery_config(checkpoint)
    config.checkpoint_dir = str(checkpoint.parent)

    with pytest.raises(ValueError, match="contains the recovery baseline"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_rejects_alias_inside_checkpoint_dir_or_existing_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model_step_134000.pt"
    checkpoint.write_bytes(b"immutable baseline")
    numbered = checkpoint_dir / "model_step_135000.pt"
    numbered.write_bytes(b"existing checkpoint")
    config = _valid_recovery_config(checkpoint)
    config.checkpoint_dir = str(checkpoint_dir)
    config.promoted_path = str(numbered)

    with pytest.raises(ValueError, match="outside checkpoint directory"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_rejects_colliding_aliases(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_step_134000.pt"
    checkpoint.write_bytes(b"immutable baseline")
    config = _valid_recovery_config(checkpoint)
    config.promoted_path = config.latest_path

    with pytest.raises(ValueError, match="aliases must be distinct"):
        validate_recovery_experiment_config(config)


def test_alias_publication_preserves_source_and_existing_checkpoint_is_unchanged(
    tmp_path: Path,
) -> None:
    import torch

    source = tmp_path / "model_step_134000.pt"
    torch.save({
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "step": 134000,
    }, source)
    destination = tmp_path / "latest.pt"
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    Trainer._publish_checkpoint_alias(source, destination)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert destination.read_bytes() == source.read_bytes()

    holder = object.__new__(Trainer)
    holder._checkpoint_thread = None
    holder.config = SimpleNamespace(
        checkpoint_dir=str(tmp_path),
        recovery_enforced=False,
    )
    holder.step = 134000
    assert Trainer._save_checkpoint(holder, 1.0) == str(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_rollback_accepts_the_anchor_whose_embedded_lineage_predates_it(
    tmp_path: Path,
) -> None:
    """The continuation anchor embeds the *previous* run's baseline hash.

    Step 174000 was written by the wd1e4 run, so its stored
    ``recovery_experiment.baseline_sha256`` is the step-134000 digest, while
    the continuation config pins the 174000 *file* digest. Selecting a rollback
    target must therefore identify the anchor by its file hash -- comparing the
    embedded lineage instead would reject the approved anchor and leave the run
    with no admissible checkpoint at all.
    """
    import torch

    anchor = tmp_path / "model_step_174000.pt"
    anchor.write_bytes(b"the approved continuation anchor")
    anchor_digest = hashlib.sha256(anchor.read_bytes()).hexdigest().upper()

    checkpoint_dir = tmp_path / "checkpoints_continuation"
    checkpoint_dir.mkdir()

    def _write(step: int, baseline: str) -> Path:
        path = checkpoint_dir / f"model_step_{step}.pt"
        torch.save({
            "model_state_dict": {"w": torch.zeros(2)},
            "step": step,
            "recovery_experiment": {
                "enabled": True,
                "baseline_sha256": baseline,
                "training_stage": "policy_only",
            },
        }, path)
        return path

    descendant = _write(176000, anchor_digest)
    _write(178000, "7238CD80" + "0" * 56)  # a foreign lineage

    holder = SimpleNamespace(
        config=SimpleNamespace(
            resume=str(anchor),
            recovery_baseline_sha256=anchor_digest,
            policy_stage="policy_only",
            checkpoint_dir=str(checkpoint_dir),
            teacher_agreement_threshold=0.50,
            policy_gate_promotion_registry=None,
        ),
        step=174000,
        _checkpoint_file_sha256=Trainer._checkpoint_file_sha256,
    )
    assert Trainer._verified_recovery_rollback_checkpoint(holder) == anchor.resolve()

    # A descendant that embeds the new baseline is preferred once passed.
    holder.step = 176000
    assert (Trainer._verified_recovery_rollback_checkpoint(holder)
            == descendant.resolve())

    # A descendant of a foreign lineage never qualifies, even when it is newest.
    holder.step = 178000
    assert (Trainer._verified_recovery_rollback_checkpoint(holder)
            == descendant.resolve())

    # If the anchor file itself changes, it stops being the approved anchor.
    anchor.write_bytes(b"tampered")
    holder.step = 174000
    with pytest.raises(RuntimeError, match="No verified checkpoint remains"):
        Trainer._verified_recovery_rollback_checkpoint(holder)


def _enhanced_stage_acceptance_fixture(tmp_path: Path) -> dict:
    """Build a complete, genuinely passing enhanced-stage unlock on disk.

    Returns the durable inputs so a caller can corrupt exactly one of them and
    assert the gate fails closed. Audit Suggestion 4 named the previous fixture
    directly: it used opening_suite_id "sha256:test-suite" and 55/100 teacher
    counts, so it demonstrated a gate that could not tell a real acceptance
    report from an invented one.
    """
    baseline = tmp_path / "model_step_134000.pt"
    baseline.write_bytes(b"immutable baseline")
    config = _valid_recovery_config(baseline)
    policy_checkpoint_dir = Path(config.checkpoint_dir)
    policy_checkpoint_dir.mkdir(parents=True)
    promoted = policy_checkpoint_dir / "model_step_140000.pt"
    promoted.write_bytes(b"promoted policy")
    promoted_sha256 = hashlib.sha256(promoted.read_bytes()).hexdigest()
    suite = tmp_path / "frozen.jsonl"
    suite.write_bytes(b"frozen suite")
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    suite.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps({"suite_sha256": suite_sha256}), encoding="utf-8")

    registry = Path(config.promotion_registry)
    registry.parent.mkdir(parents=True)
    registry_record = {
        "promoted": True,
        "step": 140000,
        "training_stage": "policy_only",
        "teacher_agreement": 0.55,
        "teacher_agreement_threshold": 0.50,
        "teacher_correct_states": 2750,
        "teacher_total_states": 5000,
        "checkpoint_path": str(promoted),
        "checkpoint_sha256": promoted_sha256,
        "suite_fingerprint": suite_sha256,
    }

    def _write_registry() -> None:
        registry.write_text(
            json.dumps(registry_record) + "\n", encoding="utf-8")

    _write_registry()

    config.resume = str(promoted)
    config.frozen_suite_path = str(suite)
    activate_enhanced_stage(config, inference_depth=3)

    accepted_alias = Path(config.policy_output_paths["accepted_path"])
    accepted_alias.write_bytes(promoted.read_bytes())
    acceptance_dir = Path(config.policy_output_paths["acceptance_dir"])
    acceptance_dir.mkdir(parents=True)
    suite_id = opening_suite_identity(
        config.test_opening_seed, config.test_opening_plies, 50)
    common = {
        "model_path": str(promoted),
        "algo_difficulty": "easy",
        "opening_seed": config.test_opening_seed,
        "opening_plies": list(config.test_opening_plies),
        "opening_suite_id": suite_id,
        "opening_suite_size": 50,
        "ml_inference_depth": 1,
    }
    random_result = {
        "ml_wins": 90, "draws": 0, "algo_wins": 10,
        "ml_as_p1_wins": 45, "ml_as_p1_draws": 0, "ml_as_p1_losses": 5,
        "ml_as_p2_wins": 45, "ml_as_p2_draws": 0, "ml_as_p2_losses": 5,
        "opponent_type": "random", **common,
    }
    easy_result = {
        "ml_wins": 65, "draws": 10, "algo_wins": 25,
        "ml_as_p1_wins": 32, "ml_as_p1_draws": 5, "ml_as_p1_losses": 13,
        "ml_as_p2_wins": 33, "ml_as_p2_draws": 5, "ml_as_p2_losses": 12,
        "opponent_type": "algorithm", **common,
    }
    decision = checkpoint_acceptance.evaluate_acceptance_gates(
        0.55, random_result, easy_result)
    report = {
        "passed": True,
        "step": 140000,
        "training_stage": "policy_only",
        "checkpoint_path": str(promoted),
        "checkpoint_sha256": promoted_sha256,
        "frozen_suite_fingerprint": suite_sha256,
        "task_id": checkpoint_acceptance.acceptance_task_id(
            str(promoted), 140000),
        "teacher_agreement_counts": {
            "correct_states": 2750, "total_states": 5000},
        "selection_sequence": [
            "held_out_teacher_agreement",
            "random_game_strength",
            "easy_game_strength",
        ],
        "opening_seed": config.test_opening_seed,
        "opening_plies": list(config.test_opening_plies),
        "opening_suite_id": suite_id,
        "inference_depth": 1,
        "max_moves": config.selfplay_max_moves,
        "num_workers": 4,
        "ci_method": decision.to_dict()["ci_method"],
        "checks": decision.checks,
        "thresholds": decision.thresholds,
        "metrics": decision.metrics,
        "random": random_result,
        "easy": easy_result,
    }
    report_path = acceptance_dir / "acceptance_step_140000.json"

    def _write_report() -> None:
        report_path.write_text(json.dumps(report), encoding="utf-8")

    _write_report()
    validate_recovery_experiment_config(config)
    return {
        "config": config,
        "registry_record": registry_record,
        "write_registry": _write_registry,
        "report": report,
        "write_report": _write_report,
        "suite_id": suite_id,
    }


def test_opening_suite_identity_reproduces_what_a_real_run_stamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run both paths: the gate's recomputation against the tester's own stamp.

    The gate refuses any acceptance report whose ``opening_suite_id`` differs
    from ``opening_suite_identity()``.  That protects nothing unless the value
    a genuine ``ModelVsAlgoTester`` run writes is the same function of the
    declared seed, plies, and suite size -- and no test exercised that: every
    fixture compared the public helper against itself, so the equivalence was
    asserted only by inspection.  Here ``run_tests`` builds and stamps its own
    id through its own code path, with only game execution stubbed out.
    """
    from dama.ai.ml import model_vs_algo

    def _fake_batch(batch):
        (_model, _difficulty, opponent, ml_player_values,
         _max_moves, openings, depth) = batch
        return [
            {
                "result": "draw",
                "ml_player": player_value,
                "winner": None,
                "num_moves": 12,
                "ml_moves": 6,
                "algo_moves": 6,
                "game_time_ms": 1.0,
                "opponent_type": opponent,
                "opening_plies": opening[0],
                "opening_seed": opening[1],
                "ml_inference_depth": depth,
            }
            for player_value, opening in zip(ml_player_values, openings)
        ]

    class _UnavailablePool:
        """Force the in-process sequential path so the stub is the one used."""

        def __init__(self, *args, **kwargs):
            raise OSError("no process pool in test")

    monkeypatch.setattr(model_vs_algo, "_play_test_games_batch", _fake_batch)
    monkeypatch.setattr(model_vs_algo, "ProcessPoolExecutor", _UnavailablePool)

    stats_dir = tmp_path / "stats"
    tester = model_vs_algo.ModelVsAlgoTester(
        model_path=str(tmp_path / "model.pt"),
        algo_difficulty="easy",
        num_workers=1,
        stats_dir=str(stats_dir),
        opening_plies=(4, 6, 8),
        opening_seed=20260824,
    )
    stats = tester.run_tests(num_games=4)

    assert stats.opening_suite_size == 2
    assert stats.opening_suite_id == opening_suite_identity(
        20260824, (4, 6, 8), 2)
    assert stats.opening_suite_id != opening_suite_identity(
        20260824, (4, 6, 8), 50)

    persisted = json.loads(
        (stats_dir / "latest_test.json").read_text(encoding="utf-8"))
    assert persisted["opening_suite_id"] == stats.opening_suite_id


def test_acceptance_gate_recomputes_the_opening_suite_identity(
    tmp_path: Path,
) -> None:
    """A matching-but-invented suite id no longer proves the declared protocol."""
    fixture = _enhanced_stage_acceptance_fixture(tmp_path)
    report = fixture["report"]

    # The exact value the superseded fixture used, applied consistently to
    # every field the gate previously compared against.
    report["opening_suite_id"] = "sha256:test-suite"
    report["random"]["opening_suite_id"] = "sha256:test-suite"
    report["easy"]["opening_suite_id"] = "sha256:test-suite"
    fixture["write_report"]()
    with pytest.raises(ValueError, match="declared opening suite"):
        validate_recovery_experiment_config(fixture["config"])


def test_acceptance_gate_rejects_a_suite_id_only_one_record_carries(
    tmp_path: Path,
) -> None:
    fixture = _enhanced_stage_acceptance_fixture(tmp_path)
    report = fixture["report"]
    report["easy"]["opening_suite_id"] = "sha256:" + "0" * 64
    fixture["write_report"]()
    with pytest.raises(ValueError, match="declared opening suite"):
        validate_recovery_experiment_config(fixture["config"])


def test_acceptance_gate_requires_exactly_the_frozen_suite_size(
    tmp_path: Path,
) -> None:
    """55/100 satisfied every provenance comparison the gate used to make."""
    fixture = _enhanced_stage_acceptance_fixture(tmp_path)
    fixture["registry_record"]["teacher_correct_states"] = 55
    fixture["registry_record"]["teacher_total_states"] = 100
    fixture["write_registry"]()
    fixture["report"]["teacher_agreement_counts"] = {
        "correct_states": 55, "total_states": 100}
    fixture["write_report"]()
    with pytest.raises(ValueError, match="not the required 5000"):
        validate_recovery_experiment_config(fixture["config"])


def test_acceptance_gate_requires_counts_to_reproduce_the_agreement(
    tmp_path: Path,
) -> None:
    fixture = _enhanced_stage_acceptance_fixture(tmp_path)
    # Correct suite size, but the counts do not divide to the promoted 0.55.
    fixture["registry_record"]["teacher_correct_states"] = 2751
    fixture["write_registry"]()
    fixture["report"]["teacher_agreement_counts"] = {
        "correct_states": 2751, "total_states": 5000}
    fixture["write_report"]()
    with pytest.raises(ValueError, match="quotient of its recorded counts"):
        validate_recovery_experiment_config(fixture["config"])


def test_acceptance_gate_rejects_impossible_teacher_counts(
    tmp_path: Path,
) -> None:
    fixture = _enhanced_stage_acceptance_fixture(tmp_path)
    fixture["registry_record"]["teacher_correct_states"] = 5001
    fixture["write_registry"]()
    fixture["report"]["teacher_agreement_counts"] = {
        "correct_states": 5001, "total_states": 5000}
    fixture["write_report"]()
    with pytest.raises(ValueError, match="teacher_correct_states exceeds"):
        validate_recovery_experiment_config(fixture["config"])


def test_acceptance_gate_rejects_missing_teacher_counts(
    tmp_path: Path,
) -> None:
    fixture = _enhanced_stage_acceptance_fixture(tmp_path)
    fixture["registry_record"].pop("teacher_correct_states")
    fixture["registry_record"].pop("teacher_total_states")
    fixture["write_registry"]()
    fixture["report"]["teacher_agreement_counts"] = {
        "correct_states": None, "total_states": None}
    fixture["write_report"]()
    with pytest.raises(ValueError, match="no held-out teacher-agreement counts"):
        validate_recovery_experiment_config(fixture["config"])


def test_enhanced_stage_resumes_only_from_recorded_policy_promotion(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "model_step_134000.pt"
    baseline.write_bytes(b"immutable baseline")
    config = _valid_recovery_config(baseline)
    policy_checkpoint_dir = Path(config.checkpoint_dir)
    policy_checkpoint_dir.mkdir(parents=True)
    promoted = policy_checkpoint_dir / "model_step_140000.pt"
    promoted.write_bytes(b"promoted policy")
    promoted_sha256 = hashlib.sha256(promoted.read_bytes()).hexdigest()
    suite = tmp_path / "frozen.jsonl"
    suite.write_bytes(b"frozen suite")
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    suite.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
        "suite_sha256": suite_sha256,
    }), encoding="utf-8")
    registry = Path(config.promotion_registry)
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({
        "promoted": True,
        "step": 140000,
        "training_stage": "policy_only",
        "teacher_agreement": 0.55,
        "teacher_agreement_threshold": 0.50,
        # A real frozen-suite measurement: 2750/5000 = 0.55. The former
        # 55/100 fixture asserted a gate that could pass on a suite fifty
        # times smaller than the approved one.
        "teacher_correct_states": 2750,
        "teacher_total_states": 5000,
        "checkpoint_path": str(promoted),
        "checkpoint_sha256": promoted_sha256,
        "suite_fingerprint": suite_sha256,
    }) + "\n", encoding="utf-8")

    config.resume = str(promoted)
    config.frozen_suite_path = str(suite)
    activate_enhanced_stage(config, inference_depth=3)

    assert config.teacher_target_type == "distribution"
    assert config.value_head_enabled is True
    assert config.pipeline_mode == "alternate"
    assert config.inference_depth == 3
    assert config.policy_gate_promotion_registry == str(registry)
    assert config.promotion_registry != str(registry)
    for field_name in trainer_module._STAGE_MUTABLE_PATH_FIELDS:
        assert getattr(config, field_name) != config.policy_output_paths[field_name]
        assert config.enhanced_output_namespace in getattr(config, field_name)

    with pytest.raises(ValueError, match="terminal passing policy acceptance"):
        validate_recovery_experiment_config(config)

    accepted_alias = Path(config.policy_output_paths["accepted_path"])
    accepted_alias.write_bytes(promoted.read_bytes())
    acceptance_dir = Path(config.policy_output_paths["acceptance_dir"])
    acceptance_dir.mkdir(parents=True)
    # Derived exactly as ModelVsAlgoTester derives it, so the gate's own
    # recomputation is checked against the real identity rather than a token.
    expected_suite_id = opening_suite_identity(
        config.test_opening_seed, config.test_opening_plies, 50)
    random_result = {
        "ml_wins": 90,
        "draws": 0,
        "algo_wins": 10,
        "ml_as_p1_wins": 45,
        "ml_as_p1_draws": 0,
        "ml_as_p1_losses": 5,
        "ml_as_p2_wins": 45,
        "ml_as_p2_draws": 0,
        "ml_as_p2_losses": 5,
        "model_path": str(promoted),
        "opponent_type": "random",
        "algo_difficulty": "easy",
        "opening_seed": config.test_opening_seed,
        "opening_plies": list(config.test_opening_plies),
        "opening_suite_id": expected_suite_id,
        "opening_suite_size": 50,
        "ml_inference_depth": 1,
    }
    easy_result = {
        "ml_wins": 65,
        "draws": 10,
        "algo_wins": 25,
        "ml_as_p1_wins": 32,
        "ml_as_p1_draws": 5,
        "ml_as_p1_losses": 13,
        "ml_as_p2_wins": 33,
        "ml_as_p2_draws": 5,
        "ml_as_p2_losses": 12,
        "model_path": str(promoted),
        "opponent_type": "algorithm",
        "algo_difficulty": "easy",
        "opening_seed": config.test_opening_seed,
        "opening_plies": list(config.test_opening_plies),
        "opening_suite_id": expected_suite_id,
        "opening_suite_size": 50,
        "ml_inference_depth": 1,
    }
    decision = checkpoint_acceptance.evaluate_acceptance_gates(
        0.55, random_result, easy_result)
    expected_task_id = checkpoint_acceptance.acceptance_task_id(
        str(promoted), 140000)
    acceptance_report_path = acceptance_dir / "acceptance_step_140000.json"
    acceptance_report = {
        "passed": True,
        "step": 140000,
        "training_stage": "policy_only",
        "checkpoint_path": str(promoted),
        "checkpoint_sha256": promoted_sha256,
        "frozen_suite_fingerprint": suite_sha256,
        "task_id": expected_task_id,
        "teacher_agreement_counts": {
            "correct_states": 2750,
            "total_states": 5000,
        },
        "selection_sequence": [
            "held_out_teacher_agreement",
            "random_game_strength",
            "easy_game_strength",
        ],
        "opening_seed": config.test_opening_seed,
        "opening_plies": list(config.test_opening_plies),
        "opening_suite_id": expected_suite_id,
        "inference_depth": 1,
        "max_moves": config.selfplay_max_moves,
        "num_workers": 4,
        "ci_method": decision.to_dict()["ci_method"],
        "checks": decision.checks,
        "thresholds": decision.thresholds,
        "metrics": decision.metrics,
        "random": random_result,
        "easy": easy_result,
    }
    acceptance_report_path.write_text(
        json.dumps(acceptance_report), encoding="utf-8")

    validate_recovery_experiment_config(config)

    # Every durable input that can unlock P5 is independently fail-closed.
    registry_record = json.loads(registry.read_text(encoding="utf-8"))
    registry_record["teacher_agreement_threshold"] = 0.51
    registry.write_text(json.dumps(registry_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recorded promoted policy-only"):
        validate_recovery_experiment_config(config)
    registry_record["teacher_agreement_threshold"] = 0.50
    registry.write_text(json.dumps(registry_record) + "\n", encoding="utf-8")

    acceptance_report["task_id"] = "wrong-task"
    acceptance_report_path.write_text(
        json.dumps(acceptance_report), encoding="utf-8")
    with pytest.raises(ValueError, match="unmatched provenance: task_id"):
        validate_recovery_experiment_config(config)
    acceptance_report["task_id"] = expected_task_id
    acceptance_report_path.write_text(
        json.dumps(acceptance_report), encoding="utf-8")

    accepted_alias.write_bytes(b"wrong accepted checkpoint")
    with pytest.raises(ValueError, match="accepted policy alias"):
        validate_recovery_experiment_config(config)
    accepted_alias.write_bytes(promoted.read_bytes())

    pending_task = checkpoint_acceptance.make_pending_acceptance_task(
        str(promoted),
        step=140000,
        teacher_agreement=0.55,
        opening_plies=config.test_opening_plies,
        opening_seed=config.test_opening_seed,
        inference_depth=1,
        max_moves=config.selfplay_max_moves,
        num_workers=min(config.cpu_workers, 4),
        training_stage="policy_only",
        checkpoint_sha256=promoted_sha256,
        suite_fingerprint=suite_sha256,
        teacher_correct_states=2750,
        teacher_total_states=5000,
    )
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        acceptance_dir, pending_task)
    with pytest.raises(ValueError, match="still pending"):
        validate_recovery_experiment_config(config)
    pending_path.unlink()

    policy_accepted = Path(config.policy_output_paths["accepted_path"])
    config.policy_output_paths["accepted_path"] = str(promoted)
    config.accepted_path = str(
        promoted).replace(
            config.policy_output_namespace,
            config.enhanced_output_namespace,
            1,
        )
    with pytest.raises(ValueError, match="alias must remain outside"):
        validate_recovery_experiment_config(config)
    config.policy_output_paths["accepted_path"] = str(policy_accepted)
    config.accepted_path = str(policy_accepted).replace(
        config.policy_output_namespace,
        config.enhanced_output_namespace,
        1,
    )

    policy_replay = config.policy_output_paths["replay_dir"]
    config.replay_dir = policy_replay
    with pytest.raises(ValueError, match="output isolation"):
        validate_recovery_experiment_config(config)
    config.replay_dir = policy_replay.replace(
        config.policy_output_namespace, config.enhanced_output_namespace)

    unpromoted = policy_checkpoint_dir / "model_step_142000.pt"
    unpromoted.write_bytes(b"unpromoted")
    config.resume = str(unpromoted)
    with pytest.raises(ValueError, match="recorded promoted policy-only"):
        validate_recovery_experiment_config(config)


def test_enhanced_stage_rejects_incomplete_namespace_without_mutating_config(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "model_step_134000.pt"
    baseline.write_bytes(b"immutable baseline")
    config = _valid_recovery_config(baseline)
    original_paths = {
        field_name: getattr(config, field_name)
        for field_name in trainer_module._STAGE_MUTABLE_PATH_FIELDS
    }
    config.ram_cache_file = str(tmp_path / "cache-without-policy-token.pt")

    with pytest.raises(ValueError, match="ram_cache_file"):
        activate_enhanced_stage(config)

    assert config.policy_stage == "policy_only"
    assert config.policy_output_paths == {}
    assert config.policy_gate_promotion_registry is None
    for field_name, original in original_paths.items():
        if field_name == "ram_cache_file":
            continue
        assert getattr(config, field_name) == original


@pytest.mark.parametrize(
    ("profile", "expected_model_games", "expected_algorithm_games"),
    [(None, 72, 168), ("server", 360, 840)],
)
def test_policy_yaml_resolves_all_recovery_controls(
    profile: str | None,
    expected_model_games: int,
    expected_algorithm_games: int,
) -> None:
    root = Path(__file__).resolve().parents[2]
    # Retired to config/superseded/ on 2026-08-24; still the audited
    # record of what the wd1e4 run resolved.
    path = (root / "config" / "superseded"
            / "training_config_policy_distillation.yaml")
    config = config_from_yaml(load_config_from_yaml(str(path), profile))

    validate_recovery_experiment_config(config)
    assert config.selfplay_games == expected_model_games
    assert config.algo_vs_algo_games == expected_algorithm_games
    assert config.selfplay_noise_prob == pytest.approx(0.10)
    assert config.teacher_difficulty == "hard"
    assert config.validation_fraction == pytest.approx(0.15)
    assert config.snapshot_min_fresh_fraction == pytest.approx(0.50)
    assert config.test_promoted_only is True
    assert config.test_games == 100
    assert config.policy_output_namespace == "policy_distillation_recovery_wd1e4"
    assert config.enhanced_output_namespace == "policy_distillation_enhanced_p5_wd1e4"


def _rng_state_holder():
    """A minimal Trainer stand-in for the RNG-restore paths."""
    holder = object.__new__(Trainer)
    holder.device = None
    return holder


def test_cuda_rng_states_reach_the_setters_as_cpu_byte_tensors(monkeypatch) -> None:
    """The regression that aborted every CUDA resume during ``__init__``.

    ``_load_checkpoint`` loads with ``map_location=self.device``, so a saved
    RNG state -- which torch always writes on the CPU -- comes back on the GPU.
    ``set_rng_state_all`` rejects that with "RNG state must be a
    torch.ByteTensor", and the branch that called it was the one branch that
    did not move its state back. Assert the contract at the setter boundary so
    the check holds on a CPU-only runner too.
    """
    import torch

    received: dict = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda states: received.__setitem__("all", list(states)))
    monkeypatch.setattr(torch, "set_rng_state", lambda state: received.__setitem__("torch", state))

    # float64 stands in for "not what the setter accepts" without needing a GPU.
    saved = torch.arange(16, dtype=torch.float64)
    _rng_state_holder()._restore_rng_state({
        "torch": torch.arange(8, dtype=torch.float64),
        "cuda_all": [saved],
    })

    assert received["torch"].dtype is torch.uint8
    assert received["torch"].device.type == "cpu"
    (cuda_state,) = received["all"]
    assert cuda_state.dtype is torch.uint8
    assert cuda_state.device.type == "cpu"
    # The state itself must survive the conversion unchanged.
    assert torch.equal(cuda_state, saved.to(torch.uint8))


def test_as_cpu_byte_state_moves_off_the_device_it_was_mapped_onto() -> None:
    """Pin the device move itself, which a CPU-only runner cannot observe."""
    import torch

    calls: list = []

    class _Recorder:
        def detach(self):
            return self

        def to(self, **kwargs):
            calls.append(kwargs)
            return torch.zeros(4, dtype=torch.uint8)

    Trainer._as_cpu_byte_state(_Recorder())
    assert calls == [{"device": "cpu", "dtype": torch.uint8}]


def test_rng_restore_seeds_the_overlap_when_the_gpu_count_differs(monkeypatch) -> None:
    """``set_rng_state_all`` indexes devices positionally.

    A checkpoint carrying more states than this host has devices would target a
    device that does not exist; fewer would silently leave the trailing ones
    unseeded. Seed the overlap per device instead.
    """
    import torch

    seeded: list = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: seeded.append("all"))
    monkeypatch.setattr(
        torch.cuda, "set_rng_state",
        lambda state, index=None: seeded.append((index, state.dtype, state.device.type)))

    _rng_state_holder()._restore_rng_state({
        "cuda_all": [
            torch.zeros(16, dtype=torch.uint8),
            torch.ones(16, dtype=torch.uint8),
        ],
    })

    assert seeded == [(0, torch.uint8, "cpu")]


def test_a_bad_rng_state_warns_instead_of_losing_the_resume(
    tmp_path: Path, capsys
) -> None:
    """Reproducibility is never worth aborting a resume for."""
    import torch

    checkpoint = tmp_path / "model_step_5.pt"
    torch.save({
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "step": 5,
        "epoch": 1,
        "rng_state": {"python": "not-a-valid-python-rng-state"},
    }, checkpoint)

    class _Model:
        def load_state_dict(self, *_args, **_kwargs):
            return [], []

    class _Optimizer:
        param_groups = [{"lr": 1e-3}]

        def load_state_dict(self, _state):
            return None

    holder = object.__new__(Trainer)
    holder.device = torch.device("cpu")
    holder.config = TrainingConfig(learning_rate=1e-3)
    holder.model = _Model()
    holder.optimizer = _Optimizer()
    holder.scheduler = None
    holder.scaler = None
    holder.stats = TrainingStats()
    holder.step = 0
    holder.epoch = 0
    holder.best_loss = float("inf")
    holder._has_non_finite_tensors = lambda: False

    Trainer._load_checkpoint(holder, str(checkpoint))

    assert holder.step == 5
    assert "Could not restore RNG state" in capsys.readouterr().out


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_cuda_resume_restores_rng_from_a_gpu_mapped_checkpoint(tmp_path: Path) -> None:
    """End-to-end on real hardware: save on CUDA, reload mapped to CUDA."""
    import torch

    checkpoint = tmp_path / "model_step_7.pt"
    torch.save({
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "cuda_all": torch.cuda.get_rng_state_all(),
        },
    }, checkpoint)

    saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
    assert saved["rng_state"]["cuda_all"][0].device.type == "cuda"

    _rng_state_holder()._restore_rng_state(saved["rng_state"])

    assert torch.equal(
        torch.cuda.get_rng_state(), saved["rng_state"]["cuda_all"][0].cpu())
