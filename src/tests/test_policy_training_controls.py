import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import dama.ai.ml.trainer as trainer_module
from dama.ai.ml import checkpoint_acceptance
from dama.ai.ml import device as ml_device
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


def test_algorithm_opening_schedule_rotates_full_strata_across_cycles() -> None:
    choices = (2, 4, 6, 8)
    algorithm_indices = range(72, 72 + 168)

    per_cycle = []
    for cycle_id in range(577, 581):
        seed_base = 20260819 + cycle_id * 1_000_003
        assigned = [
            trainer_module._training_opening_assignment(
                choices,
                seed_base,
                game_index,
                cycle_rotation=cycle_id,
            )
            for game_index in algorithm_indices
        ]
        per_cycle.append(assigned)
        assert {
            depth: sum(value[0] == depth for value in assigned)
            for depth in choices
        } == {depth: 42 for depth in choices}
        assert [value[1] for value in assigned] == [
            seed_base + game_index for game_index in algorithm_indices
        ]

    for offset in range(168):
        assert {
            per_cycle[cycle][offset][0] for cycle in range(4)
        } == set(choices)

    holder = object.__new__(Trainer)
    holder.config = TrainingConfig()
    holder.step = 0
    generation = Trainer._corpus_settings(holder)[2]
    assert generation["algorithm_opening_schedule"] == "cycle_rotated_v1"


def test_selfplay_task_stamping_computes_each_opening_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def assign(choices, seed_base, game_index, *, cycle_rotation=0):
        calls.append((choices, seed_base, game_index, cycle_rotation))
        return choices[(game_index + cycle_rotation) % len(choices)], seed_base + game_index

    monkeypatch.setattr(trainer_module, "_training_opening_assignment", assign)
    tasks = [("easy",), ("hard",)]
    stamped = trainer_module._stamp_selfplay_tasks(
        tasks,
        opening_choices=(2, 4, 6, 8),
        opening_seed_base=100,
        start_index=72,
        cycle_rotation=3,
        trajectory_source="algorithm",
        game_id_kind="algorithm",
        cycle_id=7,
        teacher_difficulty="hard",
    )
    model_stamped = trainer_module._stamp_selfplay_tasks(
        [("medium", "ml")],
        opening_choices=(2, 4),
        opening_seed_base=200,
        start_index=0,
        cycle_rotation=0,
        trajectory_source="current_model",
        game_id_kind="model",
        cycle_id=8,
        teacher_difficulty="hard",
        inference_depth=1,
    )

    assert stamped == [
        ("easy", 8, 172, "algorithm", "cycle-000007-algorithm-000000", "hard"),
        ("hard", 2, 173, "algorithm", "cycle-000007-algorithm-000001", "hard"),
    ]
    assert model_stamped == [
        ("medium", "ml", 2, 200, "current_model", "cycle-000008-model-000000", "hard", 1),
    ]
    assert calls == [
        ((2, 4, 6, 8), 100, 72, 3),
        ((2, 4, 6, 8), 100, 73, 3),
        ((2, 4), 200, 0, 0),
    ]


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


def test_selfplay_batch_consumer_releases_future_delivery_list() -> None:
    first = {"state": 1}
    second = {"state": 2}
    delivered = [first, second]
    retained = []

    def _consume(entries, game_count):
        assert game_count == 2
        retained.extend(entries)

    trainer_module._consume_and_release_selfplay_batch(
        delivered, 2, _consume)

    assert delivered == []
    assert retained == [first, second]

    failed = [{"state": 3}]

    def _fail(_entries, _game_count):
        raise RuntimeError("synthetic consume failure")

    with pytest.raises(RuntimeError, match="synthetic consume failure"):
        trainer_module._consume_and_release_selfplay_batch(
            failed, 1, _fail)
    assert failed == []

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
        param_groups = [{"lr": 1e-3, "params": []}]

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


def _cross_backend_checkpoint_holder(
    tmp_path: Path,
    source_format,
    target_format,
    device: str = "cpu",
    fused: bool = False,
):
    import torch

    # A conv weight trained under one memory format (CUDA: channels_last;
    # MPS/CPU: contiguous) and resumed under the other.
    source_parameter = torch.nn.Parameter(
        torch.randn(4, 3, 3, 3).contiguous(memory_format=source_format))
    source_parameter.grad = torch.randn_like(source_parameter)
    source_optimizer = torch.optim.AdamW([source_parameter], lr=1e-3)
    source_optimizer.step()
    optimizer_state = source_optimizer.state_dict()
    # load_state_dict() adopts the saved groups' flags, so the checkpoint
    # must record the fused run that wrote it for the target to stay fused.
    optimizer_state["param_groups"][0]["fused"] = fused
    checkpoint = tmp_path / "model_step_12.pt"
    torch.save({
        "model_state_dict": {},
        "optimizer_state_dict": optimizer_state,
        "step": 12,
    }, checkpoint)

    class _Model:
        def load_state_dict(self, *_args, **_kwargs):
            return [], []

    target_parameter = torch.nn.Parameter(
        torch.randn(4, 3, 3, 3, device=device).contiguous(
            memory_format=target_format))
    holder = object.__new__(Trainer)
    holder.device = torch.device(device)
    holder.config = TrainingConfig(learning_rate=2e-4, weight_decay=1e-4)
    holder.model = _Model()
    holder.optimizer = torch.optim.AdamW([target_parameter], fused=fused)
    holder.scheduler = None
    holder.scaler = None
    holder.stats = TrainingStats()
    holder.step = 0
    holder.epoch = 0
    holder.best_loss = float("inf")
    holder._has_non_finite_tensors = lambda: False
    return holder, source_optimizer, target_parameter, checkpoint


@pytest.mark.parametrize("direction", ["cuda_to_mac", "mac_to_cuda"])
def test_checkpoint_load_lays_optimizer_moments_out_like_parameters(
    tmp_path: Path,
    direction: str,
) -> None:
    import torch

    formats = (torch.channels_last, torch.contiguous_format)
    if direction == "mac_to_cuda":
        formats = formats[::-1]
    holder, source_optimizer, target_parameter, checkpoint = (
        _cross_backend_checkpoint_holder(tmp_path, *formats))

    Trainer._load_checkpoint(holder, str(checkpoint))

    source_state = next(iter(source_optimizer.state.values()))
    loaded_state = holder.optimizer.state[target_parameter]
    for key in ("exp_avg", "exp_avg_sq"):
        assert loaded_state[key].stride() == target_parameter.stride()
        assert torch.equal(loaded_state[key], source_state[key])


@pytest.mark.skipif(
    not ml_device.mps_available(), reason="needs an Apple GPU (MPS)")
def test_cuda_checkpoint_resumes_fused_adamw_on_mps(tmp_path: Path) -> None:
    import torch

    holder, _source_optimizer, target_parameter, checkpoint = (
        _cross_backend_checkpoint_holder(
            tmp_path, torch.channels_last, torch.contiguous_format,
            device="mps", fused=True))

    Trainer._load_checkpoint(holder, str(checkpoint))
    target_parameter.grad = torch.randn_like(target_parameter)
    holder.optimizer.step()
    torch.mps.synchronize()

    assert torch.isfinite(target_parameter.detach().cpu()).all()


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
    Trainer._wait_for_checkpoint_writer(holder, timeout=30)
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


def test_latest_checkpoint_copy_fallback_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A hardlink failure must not expose a partial latest checkpoint."""
    import shutil
    import threading

    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    latest_path = tmp_path / "latest.pt"
    previous_latest = b"verified previous latest checkpoint"
    latest_path.write_bytes(previous_latest)

    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        latest_path=str(latest_path),
    )
    holder.model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.arange(4096)},
        arch_params={},
    )
    holder.optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {}, "param_groups": []})
    holder.stats = TrainingStats()
    holder.step = 2000
    holder.epoch = 1
    holder.scheduler = None
    holder.scaler = None
    holder.stats_collector = None
    holder.log_file = str(tmp_path / "train.jsonl")
    holder.device = torch.device("cpu")
    holder._checkpoint_thread = None
    holder._active_snapshot_manifest = {}
    holder._evaluate_validation_loss = lambda: None
    holder._evaluate_teacher_promotion = lambda _path: None
    holder._live_optimizer_context = lambda: {}
    holder._snapshot_stats = lambda: {}
    holder._save_stats = lambda **_kwargs: None
    holder._put_status = lambda _message: None
    holder._prune_old_checkpoints = lambda _path: []

    copy_opened = threading.Event()
    allow_copy_to_finish = threading.Event()
    copy_destinations = []
    real_copyfileobj = shutil.copyfileobj

    def _fail_hardlink(_source, _destination):
        raise OSError("forced hardlink fallback")

    def _paused_copyfileobj(source, destination, *args, **kwargs):
        copy_destinations.append(Path(destination.name))
        destination.write(b"partial checkpoint")
        destination.flush()
        copy_opened.set()
        assert allow_copy_to_finish.wait(timeout=5)
        destination.seek(0)
        destination.truncate()
        return real_copyfileobj(source, destination, *args, **kwargs)

    monkeypatch.setattr(trainer_module.os, "link", _fail_hardlink)
    monkeypatch.setattr(shutil, "copyfileobj", _paused_copyfileobj)

    Trainer._save_checkpoint(holder, loss=0.5)
    assert copy_opened.wait(timeout=5)
    try:
        assert latest_path.read_bytes() == previous_latest
        assert len(copy_destinations) == 1
        assert copy_destinations[0].parent == tmp_path
        assert copy_destinations[0].name.startswith("latest.pt.")
        assert copy_destinations[0].suffix == ".tmp"
    finally:
        allow_copy_to_finish.set()

    thread = holder._checkpoint_thread
    assert isinstance(thread, threading.Thread)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert torch.load(
        latest_path, map_location="cpu", weights_only=False,
    )["step"] == 2000


def test_latest_checkpoint_copy_failure_removes_partial_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed alias copy must preserve the old alias and release its temp."""
    import shutil

    source = tmp_path / "source.pt"
    latest_path = tmp_path / "latest.pt"
    temporary = tmp_path / "latest.pt.tmp"
    source.write_bytes(b"complete checkpoint")
    latest_path.write_bytes(b"verified previous latest checkpoint")

    def _fail_hardlink(_source, _destination):
        raise OSError("forced hardlink fallback")

    def _fail_partial_copy(_source, destination, *_args, **_kwargs):
        destination.write(b"partial checkpoint")
        raise OSError(28, "simulated alias disk full")

    monkeypatch.setattr(trainer_module.os, "link", _fail_hardlink)
    monkeypatch.setattr(shutil, "copyfileobj", _fail_partial_copy)

    with pytest.raises(OSError, match="simulated alias disk full"):
        Trainer._publish_checkpoint_alias(source, latest_path)

    assert latest_path.read_bytes() == b"verified previous latest checkpoint"
    assert not temporary.exists()
    assert list(tmp_path.glob("latest.pt.*.tmp")) == []


@pytest.mark.parametrize("copy_fallback", [False, True])
def test_checkpoint_alias_commits_bytes_and_directory_before_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    copy_fallback: bool,
) -> None:
    """A successful alias has durable bytes and a durable public pathname."""
    import shutil

    source = tmp_path / "source.pt"
    destination = tmp_path / "latest.pt"
    source.write_bytes(b"complete checkpoint")
    events = []
    real_fsync = os.fsync
    real_link = os.link
    real_replace = os.replace

    def tracking_fsync(fd):
        mode = os.fstat(fd).st_mode
        events.append("directory_fsync" if stat.S_ISDIR(mode) else "file_fsync")
        return real_fsync(fd)

    def tracking_replace(source_path, destination_path):
        events.append("replace")
        return real_replace(source_path, destination_path)

    if copy_fallback:
        def fail_hardlink(_source, _destination):
            raise OSError("forced hardlink fallback")

        monkeypatch.setattr(trainer_module.os, "link", fail_hardlink)
    else:
        monkeypatch.setattr(trainer_module.os, "link", real_link)
    monkeypatch.setattr(shutil, "copy2", shutil.copy2)
    monkeypatch.setattr(trainer_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(trainer_module.os, "replace", tracking_replace)

    Trainer._publish_checkpoint_alias(source, destination)

    expected = (
        ["file_fsync", "replace", "directory_fsync"]
        if copy_fallback else ["replace", "directory_fsync"]
    )
    assert events == expected
    assert destination.read_bytes() == source.read_bytes()


def test_checkpoint_alias_copy_fsync_failure_preserves_previous_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Copied bytes must reach storage before replacing a usable alias."""
    import shutil

    source = tmp_path / "source.pt"
    destination = tmp_path / "latest.pt"
    temporary = tmp_path / "latest.pt.tmp"
    source.write_bytes(b"complete checkpoint")
    destination.write_bytes(b"previous checkpoint")

    def fail_hardlink(_source, _destination):
        raise OSError("forced hardlink fallback")

    def fail_file_fsync(_fd):
        raise OSError(5, "simulated alias file fsync failure")

    monkeypatch.setattr(trainer_module.os, "link", fail_hardlink)
    monkeypatch.setattr(shutil, "copy2", shutil.copy2)
    monkeypatch.setattr(trainer_module.os, "fsync", fail_file_fsync)

    with pytest.raises(OSError, match="alias file fsync failure"):
        Trainer._publish_checkpoint_alias(source, destination)

    assert destination.read_bytes() == b"previous checkpoint"
    assert not temporary.exists()
    assert list(tmp_path.glob("latest.pt.*.tmp")) == []


def test_checkpoint_alias_reports_directory_fsync_failure_without_residue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A post-replace commit failure remains visible to checkpoint recovery."""
    source = tmp_path / "source.pt"
    destination = tmp_path / "latest.pt"
    temporary = tmp_path / "latest.pt.tmp"
    source.write_bytes(b"complete checkpoint")
    destination.write_bytes(b"previous checkpoint")

    def fail_directory_fsync(_path):
        raise OSError(5, "simulated alias directory fsync failure")

    monkeypatch.setattr(trainer_module, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(OSError, match="alias directory fsync failure"):
        Trainer._publish_checkpoint_alias(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert not temporary.exists()


def test_checkpoint_writer_failure_propagates_after_join(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A daemon-writer error must fail the training thread after its join."""
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        latest_path=str(tmp_path / "latest.pt"),
    )
    holder.model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.tensor([1.0])},
        arch_params={},
    )
    holder.optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {}, "param_groups": []})
    holder.stats = TrainingStats()
    holder.step = 2000
    holder.epoch = 1
    holder.scheduler = None
    holder.scaler = None
    holder.stats_collector = None
    holder.log_file = str(tmp_path / "train.jsonl")
    holder.device = torch.device("cpu")
    holder._checkpoint_thread = None
    holder._checkpoint_write_error = None
    holder._active_snapshot_manifest = {}
    holder._evaluate_validation_loss = lambda: None
    holder._evaluate_teacher_promotion = lambda _path: None
    holder._live_optimizer_context = lambda: {}
    holder._snapshot_stats = lambda: {}
    holder._save_stats = lambda **_kwargs: None
    holder._put_status = lambda _message: None

    def _fail_alias(_self: Trainer, _source: Path, _destination: Path) -> None:
        raise OSError("forced final alias publication failure")

    monkeypatch.setattr(Trainer, "_publish_checkpoint_alias", _fail_alias)

    Trainer._save_checkpoint(holder, loss=0.5)
    with pytest.raises(
        RuntimeError,
        match=(
            "Checkpoint save error during latest alias publication at step 2000: "
            "OSError: forced final alias publication failure"
        ),
    ):
        Trainer._wait_for_checkpoint_writer(holder)

    assert (checkpoint_dir / "model_step_002000.pt").is_file()
    assert not Path(holder.config.latest_path).exists()


@pytest.mark.parametrize(
    "failure_stage", ["serialization", "file_fsync", "directory_fsync"],
)
def test_checkpoint_serialization_failure_removes_partial_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    """A failed numbered write must not publish or consume disk indefinitely."""
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        latest_path=str(tmp_path / "latest.pt"),
    )
    holder.model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.tensor([1.0])},
        arch_params={},
    )
    holder.optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {}, "param_groups": []})
    holder.stats = TrainingStats()
    holder.step = 2000
    holder.epoch = 1
    holder.scheduler = None
    holder.scaler = None
    holder.stats_collector = None
    holder.log_file = str(tmp_path / "train.jsonl")
    holder.device = torch.device("cpu")
    holder._checkpoint_thread = None
    holder._checkpoint_write_error = None
    holder._active_snapshot_manifest = {}
    holder._evaluate_validation_loss = lambda: None
    holder._evaluate_teacher_promotion = lambda _path: None
    holder._live_optimizer_context = lambda: {}
    holder._snapshot_stats = lambda: {}
    holder._save_stats = lambda **_kwargs: None
    holder._put_status = lambda _message: None

    def _fail_partial_save(_payload, destination):
        # The writer serializes into memory before its temporary exists.
        if hasattr(destination, "write"):
            destination.write(b"partial checkpoint")
        else:
            Path(destination).write_bytes(b"partial checkpoint")
        raise OSError(28, "simulated checkpoint disk full")

    if failure_stage == "serialization":
        monkeypatch.setattr(trainer_module.torch, "save", _fail_partial_save)
        error_text = "simulated checkpoint disk full"
    elif failure_stage == "file_fsync":
        def _fail_fsync(_fd):
            raise OSError(5, "simulated checkpoint file fsync failure")

        monkeypatch.setattr(trainer_module.os, "fsync", _fail_fsync)
        error_text = "simulated checkpoint file fsync failure"
    else:
        real_fsync = os.fsync
        fsync_calls = 0

        def _fail_directory_fsync(fd):
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError(5, "simulated checkpoint directory fsync failure")
            return real_fsync(fd)

        monkeypatch.setattr(
            trainer_module.os, "fsync", _fail_directory_fsync,
        )
        error_text = "simulated checkpoint directory fsync failure"

    Trainer._save_checkpoint(holder, loss=0.5)
    with pytest.raises(
        RuntimeError, match=error_text,
    ):
        Trainer._wait_for_checkpoint_writer(holder)

    expected_paths = (
        [checkpoint_dir / "model_step_002000.pt"]
        if failure_stage == "directory_fsync" else []
    )
    assert list(checkpoint_dir.iterdir()) == expected_paths
    assert not Path(holder.config.latest_path).exists()


def test_numbered_checkpoint_and_directory_are_fsynced_before_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Recovery bytes and their directory entry precede alias publication."""
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model_step_002000.pt"
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        latest_path=str(tmp_path / "latest.pt"),
    )
    holder.model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.tensor([1.0])},
        arch_params={},
    )
    holder.optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {}, "param_groups": []})
    holder.stats = TrainingStats()
    holder.step = 2000
    holder.epoch = 1
    holder.scheduler = None
    holder.scaler = None
    holder.stats_collector = None
    holder.log_file = str(tmp_path / "train.jsonl")
    holder.device = torch.device("cpu")
    holder._checkpoint_thread = None
    holder._active_snapshot_manifest = {}
    holder._evaluate_validation_loss = lambda: None
    holder._evaluate_teacher_promotion = lambda _path: None
    holder._live_optimizer_context = lambda: {}
    holder._snapshot_stats = lambda: {}
    holder._save_stats = lambda **_kwargs: None
    holder._put_status = lambda _message: None
    holder._prune_old_checkpoints = lambda _path: []

    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd):
        mode = os.fstat(fd).st_mode
        events.append("directory_fsync" if stat.S_ISDIR(mode) else "file_fsync")
        return real_fsync(fd)

    def tracking_replace(source, destination):
        if Path(destination) == checkpoint_path:
            events.append("numbered_replace")
        return real_replace(source, destination)

    def tracking_alias(_self, source, destination):
        events.append("latest_alias")
        os.link(source, destination)

    monkeypatch.setattr(trainer_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(trainer_module.os, "replace", tracking_replace)
    monkeypatch.setattr(Trainer, "_publish_checkpoint_alias", tracking_alias)

    Trainer._save_checkpoint(holder, loss=0.5)
    Trainer._wait_for_checkpoint_writer(holder, timeout=30)

    assert events == [
        "file_fsync", "numbered_replace", "directory_fsync", "latest_alias",
    ]
    assert torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )["step"] == 2000


def test_checkpoint_directory_fsync_keeps_native_windows_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native Windows keeps atomic rename without unsupported directory fds."""

    def fail_open(*_args, **_kwargs):
        raise AssertionError("native Windows must not open a directory fd")

    monkeypatch.setattr(trainer_module.os, "name", "nt")
    monkeypatch.setattr(trainer_module.os, "open", fail_open)

    trainer_module._fsync_directory(tmp_path)


def test_checkpoint_namespace_commits_parent_on_every_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A numbered checkpoint directory is durable before writers can use it."""
    checkpoint_dir = tmp_path / "models" / "checkpoints"
    checkpoint_dir.parent.mkdir()
    synced = []

    def tracking_sync(path: Path) -> None:
        assert checkpoint_dir.is_dir()
        assert not list(checkpoint_dir.iterdir())
        synced.append(Path(path))

    monkeypatch.setattr(trainer_module, "_fsync_directory", tracking_sync)

    assert trainer_module._prepare_checkpoint_directory(
        checkpoint_dir) == checkpoint_dir
    assert trainer_module._prepare_checkpoint_directory(
        checkpoint_dir) == checkpoint_dir
    assert synced == [checkpoint_dir.parent, checkpoint_dir.parent]


def test_checkpoint_namespace_commit_failure_precedes_model_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing namespace boundary must stop construction before the model."""
    checkpoint_dir = tmp_path / "models" / "checkpoints"
    checkpoint_dir.parent.mkdir()
    log_dir = tmp_path / "logs"

    def fail_parent_sync(path: Path) -> None:
        assert Path(path) == checkpoint_dir.parent
        assert checkpoint_dir.is_dir()
        raise OSError(5, "simulated checkpoint namespace sync failure")

    def unexpected_model(*_args, **_kwargs):
        raise AssertionError("model construction must follow namespace commit")

    monkeypatch.setattr(trainer_module, "_fsync_directory", fail_parent_sync)
    monkeypatch.setattr(trainer_module, "create_model", unexpected_model)

    config = TrainingConfig(
        device="cpu",
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
    )
    with pytest.raises(OSError, match="checkpoint namespace sync failure"):
        Trainer(config)

    assert checkpoint_dir.is_dir()
    assert not list(checkpoint_dir.iterdir())
    assert not log_dir.exists()


@pytest.mark.parametrize("drop_point", ["numbered", "latest"])
def test_checkpoint_writer_repairs_numbered_file_lost_during_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
    drop_point: str,
) -> None:
    """A transient DrvFS disappearance must not leave only the latest alias."""
    import threading
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model_step_002000.pt"
    latest_path = tmp_path / "latest.pt"

    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        latest_path=str(latest_path),
    )
    holder.model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.tensor([1.0])},
        arch_params={},
    )
    holder.optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {}, "param_groups": []})
    holder.stats = TrainingStats()
    holder.step = 2000
    holder.epoch = 1
    holder.scheduler = None
    holder.scaler = None
    holder.stats_collector = None
    holder.log_file = str(tmp_path / "train.jsonl")
    holder.device = torch.device("cpu")
    holder._checkpoint_thread = None
    holder._active_snapshot_manifest = {}
    holder._evaluate_validation_loss = lambda: None
    holder._evaluate_teacher_promotion = lambda _path: None
    holder._live_optimizer_context = lambda: {}
    holder._snapshot_stats = lambda: {}
    holder._save_stats = lambda **_kwargs: None
    holder._put_status = lambda _message: None

    real_replace = trainer_module.os.replace
    removed_once = False

    def _replace_then_drop_numbered(source, destination):
        nonlocal removed_once
        real_replace(source, destination)
        destination = Path(destination)
        should_drop = (
            (drop_point == "numbered" and destination == checkpoint_path)
            or (drop_point == "latest" and destination == latest_path)
        )
        if should_drop and not removed_once:
            checkpoint_path.unlink()
            removed_once = True

    monkeypatch.setattr(
        trainer_module.os, "replace", _replace_then_drop_numbered)

    returned = Trainer._save_checkpoint(holder, loss=0.5)
    thread = holder._checkpoint_thread
    assert isinstance(thread, threading.Thread)
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert returned == str(checkpoint_path)
    assert removed_once
    assert checkpoint_path.is_file()
    assert latest_path.is_file()
    assert torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )["step"] == 2000
    assert "Numbered checkpoint disappeared" in capsys.readouterr().out


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
        param_groups = [{"lr": 1e-3, "params": []}]

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


def test_delayed_stats_snapshot_cannot_replace_newer_progress(
    tmp_path: Path,
) -> None:
    """Background checkpoint I/O must not move the stats sidecar backward."""
    import threading

    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        stats_file=str(tmp_path / "training_stats.json"),
        log_dir=str(tmp_path),
        recovery_enforced=False,
    )
    holder.stats = TrainingStats()
    holder._stats_write_lock = threading.RLock()
    holder._stats_snapshot_generation = 0
    holder._stats_persisted_generation = -1
    holder._update_training_progress_report = lambda _path: None

    holder.step = 100
    holder.stats.loss_history.append({"step": 100, "loss": 1.0})
    delayed_checkpoint_snapshot = Trainer._snapshot_stats(holder)

    holder.step = 200
    holder.stats.loss_history.append({"step": 200, "loss": 0.8})
    current_progress_snapshot = Trainer._snapshot_stats(holder)

    # Reproduce the physical completion order of a slow checkpoint writer:
    # current progress lands first, then the older captured snapshot arrives.
    Trainer._save_stats(holder, _snapshot=current_progress_snapshot)
    Trainer._save_stats(holder, _snapshot=delayed_checkpoint_snapshot)

    stored = json.loads(Path(holder.config.stats_file).read_text())
    assert stored["total_steps"] == 200
    assert [entry["step"] for entry in stored["loss_history"]] == [100, 200]
    assert trainer_module._STATS_WRITE_GENERATION_KEY not in stored


def test_stats_writer_commits_bytes_and_public_name_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Strict stats success means both data and its directory entry landed."""
    import threading

    holder = object.__new__(Trainer)
    stats_path = tmp_path / "training_stats.json"
    holder.config = SimpleNamespace(stats_file=str(stats_path))
    holder.stats = TrainingStats()
    holder.step = 140000
    holder._stats_write_lock = threading.RLock()
    holder._stats_snapshot_generation = 0
    holder._stats_persisted_generation = -1
    holder._update_training_progress_report = lambda _path: None

    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd: int) -> None:
        events.append("file_fsync")
        real_fsync(fd)

    def tracking_replace(source, destination) -> None:
        events.append("replace")
        real_replace(source, destination)

    def tracking_directory_fsync(path: Path) -> None:
        events.append("directory_fsync")
        assert Path(path) == stats_path.parent

    monkeypatch.setattr(trainer_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(trainer_module.os, "replace", tracking_replace)
    monkeypatch.setattr(
        trainer_module, "_fsync_directory", tracking_directory_fsync)

    assert holder._save_stats(_raise_on_error=True) is True
    assert events == ["file_fsync", "replace", "directory_fsync"]
    assert json.loads(stats_path.read_text(encoding="utf-8"))["total_steps"] == 140000
    assert not list(tmp_path.glob("*.tmp"))


def test_stats_writer_does_not_acknowledge_directory_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A visible replacement is retriable until its directory is committed."""
    import threading

    holder = object.__new__(Trainer)
    stats_path = tmp_path / "training_stats.json"
    holder.config = SimpleNamespace(stats_file=str(stats_path))
    holder.stats = TrainingStats()
    holder.step = 140000
    holder._stats_write_lock = threading.RLock()
    holder._stats_snapshot_generation = 0
    holder._stats_persisted_generation = -1
    holder._update_training_progress_report = lambda _path: None

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError(5, "simulated stats directory commit failure")

    monkeypatch.setattr(
        trainer_module, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(OSError, match="stats directory commit failure"):
        holder._save_stats(_raise_on_error=True)

    # The complete replacement may already be visible, but the failed call
    # must not advance the acknowledged generation or strand its temporary.
    assert json.loads(stats_path.read_text(encoding="utf-8"))["total_steps"] == 140000
    assert holder._stats_persisted_generation == -1
    assert not list(tmp_path.glob("*.tmp"))

    monkeypatch.setattr(trainer_module, "_fsync_directory", lambda _path: None)
    assert holder._save_stats(_raise_on_error=True) is True
    assert holder._stats_persisted_generation == 2


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


def _text_replay_opens(monkeypatch: pytest.MonkeyPatch) -> list:
    """Count whole-file ('r') replay reads, i.e. exact-fallback scans."""
    real_open = Path.open
    scanned = []

    def counting_open(path, mode="r", *args, **kwargs):
        if Path(path).name.startswith("replay_") and "r" in mode and "b" not in mode:
            scanned.append(Path(path))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    return scanned


def test_generation_cycle_scan_reads_only_shard_tails(tmp_path, monkeypatch):
    """Cold start answers from tails/sidecar, never whole-corpus scans."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    lines = "\n".join(
        json.dumps({"generation_cycle_id": cycle}) + "\n" for cycle in range(20)
    )
    (replay_dir / "replay_a.jsonl").write_text(lines, encoding="utf-8")
    (replay_dir / "replay_b.jsonl").write_text(
        json.dumps({"game_id": "cycle-000031-model-000001"}) + "\n",
        encoding="utf-8",
    )
    scanned = _text_replay_opens(monkeypatch)

    first = _cycle_allocator_holder(tmp_path, 0)
    assert first._durable_generation_cycle_max() == 31
    assert scanned == [], "cold start must not run the exact whole-file scan"
    sidecar = json.loads(
        (replay_dir / "generation_cycle_cache.json").read_text(encoding="utf-8"))
    assert sidecar["schema"] == 1
    assert sidecar["entries"]["replay_b.jsonl"][2] == 31

    # A brand-new process (empty in-memory cache) is answered entirely from
    # the persisted sidecar identities.
    second = _cycle_allocator_holder(tmp_path, 0)
    scanned.clear()
    assert second._durable_generation_cycle_max() == 31
    assert scanned == []


def test_generation_cycle_sidecar_rescans_only_grown_shards(tmp_path, monkeypatch):
    """Appending new records invalidates just the grown shard's entry."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "replay_a.jsonl").write_text(
        json.dumps({"generation_cycle_id": 7}) + "\n", encoding="utf-8")
    holder = _cycle_allocator_holder(tmp_path, 0)
    assert holder._durable_generation_cycle_max() == 7

    # Simulate the next self-play cycle appending to the newest shard.
    with (replay_dir / "replay_a.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"generation_cycle_id": 12}) + "\n")

    next_holder = _cycle_allocator_holder(tmp_path, 0)
    scanned = _text_replay_opens(monkeypatch)
    assert next_holder._durable_generation_cycle_max() == 12
    assert scanned == []
    sidecar = json.loads(
        (replay_dir / "generation_cycle_cache.json").read_text(encoding="utf-8"))
    assert sidecar["entries"]["replay_a.jsonl"][2] == 12


def test_generation_cycle_corrupt_sidecar_self_heals(tmp_path, monkeypatch):
    """A corrupt sidecar is ignored and rebuilt from re-derived answers."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "replay_a.jsonl").write_text(
        json.dumps({"generation_cycle_id": 9}) + "\n", encoding="utf-8")
    (replay_dir / "generation_cycle_cache.json").write_text(
        "{not json", encoding="utf-8")

    scanned = _text_replay_opens(monkeypatch)
    holder = _cycle_allocator_holder(tmp_path, 0)
    assert holder._durable_generation_cycle_max() == 9
    assert scanned == [], "tail proof makes the corrupt sidecar irrelevant"
    sidecar = json.loads(
        (replay_dir / "generation_cycle_cache.json").read_text(encoding="utf-8"))
    assert sidecar["entries"]["replay_a.jsonl"][2] == 9

    scanned.clear()
    healed = _cycle_allocator_holder(tmp_path, 0)
    assert healed._durable_generation_cycle_max() == 9
    assert scanned == [], "healed sidecar answers future launches outright"


def test_generation_cycle_sidecar_caches_no_cycle_legacy_shard(tmp_path, monkeypatch):
    """A known no-cycle shard must not force a full scan on every restart."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "replay_legacy.jsonl").write_text(
        json.dumps({"game_id": "legacy-game-without-cycle"}) + "\n",
        encoding="utf-8",
    )

    scanned = _text_replay_opens(monkeypatch)
    first = _cycle_allocator_holder(tmp_path, 0)
    assert first._durable_generation_cycle_max() == -1
    assert len(scanned) == 1
    sidecar = json.loads(
        (replay_dir / "generation_cycle_cache.json").read_text(encoding="utf-8"))
    assert sidecar["entries"]["replay_legacy.jsonl"][2] == -1

    scanned.clear()
    second = _cycle_allocator_holder(tmp_path, 0)
    assert second._durable_generation_cycle_max() == -1
    assert scanned == []


def test_generation_cycle_sidecar_prunes_all_removed_shards(tmp_path):
    """Rotation removes sidecar state even when it removes every shard."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    shard = replay_dir / "replay_a.jsonl"
    shard.write_text(json.dumps({"generation_cycle_id": 17}) + "\n")
    assert _cycle_allocator_holder(tmp_path, 0)._durable_generation_cycle_max() == 17

    shard.unlink()
    assert _cycle_allocator_holder(tmp_path, 0)._durable_generation_cycle_max() == -1
    sidecar = json.loads(
        (replay_dir / "generation_cycle_cache.json").read_text(encoding="utf-8"))
    assert sidecar["entries"] == {}


def test_generation_cycle_sidecar_commits_its_public_name(
    tmp_path, monkeypatch,
):
    """A successful restart-cache publication commits bytes then its name."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    holder = object.__new__(Trainer)
    entries = {"replay_a.jsonl": (123, 456, 17)}
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd):
        events.append("file_fsync")
        real_fsync(fd)

    def tracking_replace(source, destination):
        events.append("replace")
        real_replace(source, destination)

    def tracking_directory_fsync(path):
        events.append("directory_fsync")
        assert Path(path) == replay_dir

    monkeypatch.setattr(trainer_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(trainer_module.os, "replace", tracking_replace)
    monkeypatch.setattr(
        trainer_module, "_fsync_directory", tracking_directory_fsync)

    assert holder._save_generation_cycle_sidecar(replay_dir, entries) is True
    assert events == ["file_fsync", "replace", "directory_fsync"]
    assert holder._load_generation_cycle_sidecar(replay_dir) == entries
    assert not list(replay_dir.glob(".generation_cycle_cache.*.tmp"))


def test_generation_cycle_sidecar_reports_directory_commit_failure(
    tmp_path, monkeypatch,
):
    """The optional cache stays fail-open when its directory cannot sync."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    holder = object.__new__(Trainer)
    entries = {"replay_a.jsonl": (123, 456, 17)}

    def fail_directory_fsync(_path):
        raise OSError(5, "simulated generation-cache directory commit failure")

    monkeypatch.setattr(
        trainer_module, "_fsync_directory", fail_directory_fsync)
    assert holder._save_generation_cycle_sidecar(replay_dir, entries) is False
    assert holder._load_generation_cycle_sidecar(replay_dir) == entries
    assert not list(replay_dir.glob(".generation_cycle_cache.*.tmp"))

    monkeypatch.setattr(trainer_module, "_fsync_directory", lambda _path: None)
    assert holder._save_generation_cycle_sidecar(replay_dir, entries) is True


def test_generation_cycle_scan_skips_shard_deleted_before_exact_fallback(
    tmp_path, monkeypatch,
):
    """A rotation race cannot crash allocation after the identity check."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    shard = replay_dir / "replay_a.jsonl"
    shard.write_text(json.dumps({"generation_cycle_id": 17}) + "\n")

    def delete_before_fallback(_holder, path):
        path.unlink()
        return None

    monkeypatch.setattr(
        Trainer, "_generation_cycle_tail_max", delete_before_fallback,
    )
    holder = _cycle_allocator_holder(tmp_path, 0)
    assert holder._durable_generation_cycle_max() == -1
    assert shard.resolve() not in holder._generation_cycle_file_cache


def test_generation_cycle_truncated_tail_uses_exact_scan(tmp_path, monkeypatch):
    """A torn final write cannot be proven from the tail; scan decides."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    torn = json.dumps({"generation_cycle_id": 15}) + "\n"
    torn += '{"generation_cycle_id": 16, "state": {"unfinis'
    (replay_dir / "replay_a.jsonl").write_text(torn, encoding="utf-8")

    assert Trainer._generation_cycle_tail_max(
        replay_dir / "replay_a.jsonl") is None
    scanned = _text_replay_opens(monkeypatch)
    holder = _cycle_allocator_holder(tmp_path, 0)
    assert holder._durable_generation_cycle_max() == 15
    assert len(scanned) == 1
    # The exact answer is then persisted for future launches.
    sidecar = json.loads(
        (replay_dir / "generation_cycle_cache.json").read_text(encoding="utf-8"))
    assert sidecar["entries"]["replay_a.jsonl"][2] == 15


@pytest.mark.parametrize("checkpoint_sha256", ["AB" * 32, None])
def test_selfplay_entries_record_cycle_and_behavior_provenance(
    checkpoint_sha256,
) -> None:
    entries = [{"game_id": "cycle-000028-model-000001"}, "not-a-dict"]

    Trainer._annotate_selfplay_entries(
        entries,
        cycle_id=28,
        behavior_step=136000,
        behavior_id="trainer-step-136000",
        behavior_checkpoint_sha256=checkpoint_sha256,
    )

    assert entries[0]["generation_cycle_id"] == 28
    assert entries[0]["model_behavior_step"] == 136000
    assert entries[0]["model_behavior_id"] == "trainer-step-136000"
    if checkpoint_sha256 is None:
        assert "model_behavior_checkpoint_sha256" not in entries[0]
    else:
        assert entries[0]["model_behavior_checkpoint_sha256"] == checkpoint_sha256
    assert list(entries[0]) == [
        "game_id",
        "generation_cycle_id",
        "model_behavior_step",
        "model_behavior_id",
        *(["model_behavior_checkpoint_sha256"] if checkpoint_sha256 else []),
    ]


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
    observed_handoffs = []
    replay_stats_handoff = object()

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
            observed_handoffs.append(kwargs["replay_file_stats_handoff"])
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
    holder.replay_buffer = SimpleNamespace(
        cleanup_old_files=lambda: 0,
        take_replay_file_stats_handoff=lambda: replay_stats_handoff,
    )
    holder._bg_selfplay_thread = None
    holder._bg_selfplay_dataset = None
    holder._bg_selfplay_entries = None
    holder._bg_selfplay_incremental = None
    holder._bg_snapshot_manifest = None
    holder._bg_validation_entries = None
    holder._bg_selfplay_lock = trainer_module.threading.Lock()
    holder._bg_selfplay_stop_event = trainer_module.threading.Event()
    holder._stopped = False
    holder._paused = False
    holder._data_ready_event = SimpleNamespace(
        set=holder._bg_selfplay_stop_event.set
    )
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
    assert observed_handoffs == [replay_stats_handoff]
    assert holder._bg_snapshot_manifest["fingerprint"] == "enhanced"


@pytest.mark.parametrize("failure_stage", ["load", "tensorize"])
@pytest.mark.parametrize(
    "followup", ["rejected", "admitted", "still_fails", "stop", "never_admitted"])
def test_background_snapshot_retries_admitted_preparation(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str, followup: str,
) -> None:
    """A preparation failure must not lose an already admitted snapshot."""
    calls = {"cycles": 0, "loads": [], "backoffs": [], "published": []}
    admitted_path = Path("snapshot_v000002/manifest.json")
    newer_path = Path("snapshot_v000003/manifest.json")

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target

        def start(self):
            self.target()

    def fail_preparation():
        return len(calls["loads"]) == 1 or followup == "still_fails"

    class Manager:
        def consider_snapshot(self, **_kwargs):
            if calls["cycles"] == 1 and followup != "never_admitted":
                return SnapshotDecision(True, "admitted", admitted_path, {})
            if calls["cycles"] == 2 and followup == "admitted":
                return SnapshotDecision(True, "admitted", newer_path, {})
            return SnapshotDecision(
                False, "below freshness floor", admitted_path, {})

        def load_split(self, path, max_train_entries=0):
            assert max_train_entries == 500
            calls["loads"].append(path)
            if failure_stage == "load" and fail_preparation():
                raise OSError("synthetic snapshot read failure")
            version = 3 if path == newer_path else 2
            return [f"train-{version}"], [f"validation-{version}"], {
                "version": version,
                "metrics": {"fresh_unique_state_rate": 0.5},
            }

    def tensorize(entries, **_kwargs):
        if failure_stage == "tensorize" and fail_preparation():
            raise RuntimeError("synthetic tensor preparation failure")
        return list(entries)

    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        replay_max_files=60, replay_max_entries=500, max_moves_per_sample=32,
    )
    holder._snapshot_manager = Manager()
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._bg_selfplay_thread = None
    holder._bg_selfplay_stop_event = trainer_module.threading.Event()
    holder._bg_selfplay_lock = trainer_module.threading.Lock()
    holder._bg_selfplay_dataset = None
    holder._bg_selfplay_incremental = None
    holder._bg_selfplay_entries = None
    holder._bg_snapshot_manifest = None
    holder._bg_validation_entries = None
    holder._stopped = holder._paused = False
    holder._wait_for_selfplay_disk_headroom = lambda: True
    holder._corpus_settings = lambda **_kwargs: ({}, {}, {})
    activations = []
    holder._activate_dataset_manifest = lambda manifest: activations.append(
        manifest["version"])
    validations = []
    holder._set_validation_entries = validations.append

    def collect():
        # Consume immediately, so later rejected cycles cannot re-publish the
        # same successful snapshot just because its handoff slot is empty.
        dataset, incremental = holder._collect_background_selfplay()
        calls["published"].append(dataset)
        assert incremental is None

    holder._data_ready_event = SimpleNamespace(set=collect)

    def run_selfplay(*_args, **_kwargs):
        calls["cycles"] += 1
        if calls["cycles"] == 4:
            holder._bg_selfplay_stop_event.set()
        return 1, 23

    def backoff(timeout=None):
        calls["backoffs"].append(timeout)
        if followup == "stop":
            holder._bg_selfplay_stop_event.set()
        return holder._bg_selfplay_stop_event.is_set()

    holder.run_selfplay = run_selfplay
    monkeypatch.setattr(holder._bg_selfplay_stop_event, "wait", backoff)
    monkeypatch.setattr(trainer_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        trainer_module.CachedTensorDataset, "from_entries", staticmethod(tensorize))

    Trainer._start_background_selfplay(holder, 72)

    if followup == "stop":
        assert calls == {
            "cycles": 1, "loads": [admitted_path],
            "backoffs": [2.0], "published": [],
        }
    elif followup == "never_admitted":
        assert calls == {
            "cycles": 4, "loads": [], "backoffs": [], "published": [],
        }
    elif followup == "still_fails":
        assert calls == {
            "cycles": 4, "loads": [admitted_path] * 3,
            "backoffs": [2.0] * 3, "published": [],
        }
    else:
        version = 3 if followup == "admitted" else 2
        assert calls == {
            "cycles": 4,
            "loads": [admitted_path, newer_path if version == 3 else admitted_path],
            "backoffs": [2.0], "published": [[f"train-{version}"]],
        }
        assert activations == [version]
        assert validations == [[f"validation-{version}"]]
    if followup in {"stop", "still_fails", "never_admitted"}:
        assert activations == validations == []


@pytest.mark.parametrize("tensorize_fails", [False, True])
def test_background_snapshot_releases_cycle_locals(
    monkeypatch: pytest.MonkeyPatch, tensorize_fails: bool,
) -> None:
    """Only the pending handoff may own payloads during the next generation."""
    import weakref

    class Payload(list):
        pass

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target

        def start(self):
            self.target()

    refs = {}
    observations = {}

    class Manager:
        def consider_snapshot(self, **_kwargs):
            return SimpleNamespace(admitted=True, manifest_path="snapshot")

        def load_split(self, *_args, **_kwargs):
            train = Payload(["train"])
            validation = Payload(["validation"])
            refs["train"] = weakref.ref(train)
            refs["validation"] = weakref.ref(validation)
            return train, validation, {
                "version": 2, "metrics": {"fresh_unique_state_rate": 0.5},
            }

    def tensorize(entries, **_kwargs):
        assert entries == ["train"]
        if tensorize_fails:
            raise RuntimeError("synthetic tensorization failure")
        dataset = Payload(["copied tensors"])
        refs["dataset"] = weakref.ref(dataset)
        return dataset

    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        replay_max_files=60, replay_max_entries=500, max_moves_per_sample=32,
    )
    holder._snapshot_manager = Manager()
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._bg_selfplay_thread = None
    holder._bg_selfplay_stop_event = trainer_module.threading.Event()
    holder._bg_selfplay_lock = trainer_module.threading.Lock()
    holder._data_ready_event = trainer_module.threading.Event()
    holder._bg_selfplay_dataset = None
    holder._bg_selfplay_incremental = None
    holder._bg_selfplay_entries = None
    holder._bg_snapshot_manifest = None
    holder._bg_validation_entries = None
    holder._stopped = holder._paused = False
    holder._wait_for_selfplay_disk_headroom = lambda: True
    holder._corpus_settings = lambda **_kwargs: ({}, {}, {})
    holder._activate_dataset_manifest = lambda manifest: observations.update(
        version=manifest["version"])
    holder._set_validation_entries = lambda entries: observations.update(
        validation=list(entries))

    def run_selfplay(*_args, **_kwargs):
        if refs:
            holder._bg_selfplay_stop_event.set()
            observations["train_released"] = refs["train"]() is None
            if not tensorize_fails:
                observations["pending_owned"] = (
                    refs["dataset"]() is holder._bg_selfplay_dataset
                    and refs["validation"]() is holder._bg_validation_entries
                )
                dataset, incremental = holder._collect_background_selfplay()
                observations["dataset"] = list(dataset)
                observations["incremental"] = incremental
                del dataset
                observations["dataset_released"] = refs["dataset"]() is None
            observations["validation_released"] = refs["validation"]() is None
        return 1, 23

    holder.run_selfplay = run_selfplay
    monkeypatch.setattr(trainer_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        trainer_module.CachedTensorDataset, "from_entries", staticmethod(tensorize))
    Trainer._start_background_selfplay(holder, 72)

    assert observations["train_released"]
    assert observations["validation_released"]
    if tensorize_fails:
        assert holder._bg_selfplay_dataset is None
        assert not holder._data_ready_event.is_set()
    else:
        assert observations["pending_owned"]
        assert observations["dataset"] == ["copied tensors"]
        assert observations["incremental"] is None
        assert observations["validation"] == ["validation"]
        assert observations["version"] == 2
        assert observations["dataset_released"]


def test_background_selfplay_stop_event_prevents_another_snapshot_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

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

    class _SnapshotManager:
        def consider_snapshot(self, **_kwargs):
            raise AssertionError(
                "shutdown must skip corpus work after the in-flight cycle"
            )

    holder = object.__new__(Trainer)
    holder._snapshot_manager = _SnapshotManager()
    holder.config = SimpleNamespace(replay_max_files=60)
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._bg_selfplay_thread = None
    holder._bg_selfplay_stop_event = trainer_module.threading.Event()
    holder._stopped = False
    holder._paused = False

    def _run_selfplay(_num_games, **_kwargs):
        calls.append("cycle")
        holder._bg_selfplay_stop_event.set()
        return 1, 23

    holder.run_selfplay = _run_selfplay
    monkeypatch.setattr(trainer_module.threading, "Thread", _FakeThread)

    Trainer._start_background_selfplay(holder, 72)

    assert calls == ["cycle"]
    assert holder._stopped is False


def test_background_selfplay_waits_for_storage_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

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

    class _StopOnWaitEvent:
        def __init__(self):
            self._set = False

        def is_set(self):
            return self._set

        def clear(self):
            self._set = False

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            self._set = True
            return True

    holder = object.__new__(Trainer)
    holder._snapshot_manager = object()
    holder.config = SimpleNamespace(
        replay_dir="/replay",
        replay_max_files=60,
        selfplay_min_free_disk_gb=10.0,
    )
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._bg_selfplay_thread = None
    holder._bg_selfplay_stop_event = _StopOnWaitEvent()
    holder._data_ready_event = trainer_module.threading.Event()
    holder._stopped = False
    holder._paused = False
    holder.run_selfplay = lambda *_args, **_kwargs: calls.append("generate")

    monkeypatch.setattr(trainer_module.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        trainer_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=9 * 1024 ** 3),
    )

    Trainer._start_background_selfplay(holder, 72)

    assert calls == [("wait", 30.0)]
    assert holder._stopped is False


def test_background_selfplay_rechecks_storage_before_snapshot_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

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

    class _StopOnWaitEvent:
        def __init__(self):
            self._set = False

        def is_set(self):
            return self._set

        def clear(self):
            self._set = False

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            self._set = True
            return True

    class _SnapshotManager:
        def consider_snapshot(self, **_kwargs):
            calls.append("consider")
            raise AssertionError("low storage must block snapshot admission")

    free_values = iter((11 * 1024 ** 3, 9 * 1024 ** 3))
    holder = object.__new__(Trainer)
    holder._snapshot_manager = _SnapshotManager()
    holder.config = SimpleNamespace(
        replay_dir="/replay",
        replay_max_files=60,
        selfplay_min_free_disk_gb=10.0,
    )
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._bg_selfplay_thread = None
    holder._bg_selfplay_stop_event = _StopOnWaitEvent()
    holder._data_ready_event = trainer_module.threading.Event()
    holder._stopped = False
    holder._paused = False
    holder.run_selfplay = lambda *_args, **_kwargs: (
        calls.append("generate") or (1, 23)
    )

    monkeypatch.setattr(trainer_module.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        trainer_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=next(free_values)),
    )

    Trainer._start_background_selfplay(holder, 72)

    assert calls == ["generate", ("wait", 30.0)]
    assert holder._stopped is False


def test_background_selfplay_stop_during_admission_skips_snapshot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

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

    class _SnapshotManager:
        def consider_snapshot(self, **_kwargs):
            calls.append("consider")
            holder._bg_selfplay_stop_event.set()
            return SimpleNamespace(
                admitted=True,
                manifest_path="snapshot/manifest.json",
            )

        def load_split(self, *_args, **_kwargs):
            raise AssertionError(
                "shutdown must leave the admitted snapshot for next launch"
            )

    holder = object.__new__(Trainer)
    holder._snapshot_manager = _SnapshotManager()
    holder.config = SimpleNamespace(
        replay_max_files=60,
        replay_max_entries=1_000_000,
    )
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._bg_selfplay_thread = None
    holder._bg_selfplay_stop_event = trainer_module.threading.Event()
    holder._data_ready_event = trainer_module.threading.Event()
    holder._stopped = False
    holder._paused = False
    holder.run_selfplay = lambda *_args, **_kwargs: (1, 23)
    holder._corpus_settings = lambda **_kwargs: ({}, {}, {})

    monkeypatch.setattr(trainer_module.threading, "Thread", _FakeThread)

    Trainer._start_background_selfplay(holder, 72)

    assert calls == ["consider"]
    assert holder._stopped is False


def test_background_selfplay_stop_during_snapshot_load_skips_tensorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

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

    class _SnapshotManager:
        def consider_snapshot(self, **_kwargs):
            calls.append("consider")
            return SimpleNamespace(
                admitted=True,
                manifest_path="snapshot/manifest.json",
            )

        def load_split(self, *_args, **_kwargs):
            calls.append("load")
            holder._bg_selfplay_stop_event.set()
            return [object()], [], {"version": 1, "metrics": {}}

    holder = object.__new__(Trainer)
    holder._snapshot_manager = _SnapshotManager()
    holder.config = SimpleNamespace(
        replay_max_files=60,
        replay_max_entries=1_000_000,
        max_moves_per_sample=32,
    )
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
    holder._bg_selfplay_thread = None
    holder._bg_selfplay_stop_event = trainer_module.threading.Event()
    holder._data_ready_event = trainer_module.threading.Event()
    holder._stopped = False
    holder._paused = False
    holder.run_selfplay = lambda *_args, **_kwargs: (1, 23)
    holder._corpus_settings = lambda **_kwargs: ({}, {}, {})

    def _fail_tensorize(*_args, **_kwargs):
        raise AssertionError("shutdown must skip snapshot tensorization")

    monkeypatch.setattr(trainer_module.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        trainer_module.CachedTensorDataset,
        "from_entries",
        staticmethod(_fail_tensorize),
    )

    Trainer._start_background_selfplay(holder, 72)

    assert calls == ["consider", "load"]
    assert holder._stopped is False


def test_background_selfplay_shutdown_signals_before_join() -> None:
    holder = object.__new__(Trainer)
    holder._bg_selfplay_stop_event = trainer_module.threading.Event()
    holder._data_ready_event = trainer_module.threading.Event()

    class _FakeThread:
        def __init__(self):
            self.alive = True
            self.joined = False

        def is_alive(self):
            return self.alive

        def join(self):
            assert holder._bg_selfplay_stop_event.is_set()
            self.joined = True
            self.alive = False

    thread = _FakeThread()
    holder._bg_selfplay_thread = thread

    Trainer._stop_background_selfplay(holder)

    assert thread.joined
    assert holder._data_ready_event.is_set()


def test_runtime_model_root_and_cleanup_removes_temporary_files(
    tmp_path: Path,
) -> None:
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(runtime_model_root=str(tmp_path / "logs" / "runtime_models"))
    holder._runtime_model_dir = None

    temp_path = holder._runtime_model_path("temp_selfplay_model.pt")
    temp_path2 = holder._runtime_model_path("temp_async_test.pt")
    assert not temp_path.parent.exists()
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


def test_runtime_model_checkpoint_staging_skips_only_optional_disk_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fork staging streams exact loadable bytes without a model-sized copy."""
    import torch

    checkpoint = {
        "model_state_dict": {"weight": torch.arange(6).reshape(2, 3)},
        "arch_params": {"channels": 2},
        "encoding_version": 2,
        "step": 17,
    }
    path = tmp_path / "runtime_models" / "process-test" / "temp_selfplay_model.pt"
    created_sinks = []
    real_sink = trainer_module._RuntimeCheckpointHashSink

    class TrackingSink(real_sink):
        def __init__(self) -> None:
            super().__init__()
            created_sinks.append(self)

    monkeypatch.setattr(
        trainer_module, "_RuntimeCheckpointHashSink", TrackingSink)

    memory_digest = trainer_module._stage_runtime_model_checkpoint(
        checkpoint, path, persist_to_disk=False)

    assert not path.exists()
    assert not path.parent.exists()
    assert len(memory_digest) == 64
    assert memory_digest == memory_digest.upper()
    assert len(created_sinks) == 1

    disk_digest = trainer_module._stage_runtime_model_checkpoint(
        checkpoint, path, persist_to_disk=True)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    assert disk_digest == hashlib.sha256(path.read_bytes()).hexdigest().upper()
    assert disk_digest == memory_digest
    assert created_sinks[0].tell() == path.stat().st_size
    assert loaded["step"] == checkpoint["step"]
    assert loaded["arch_params"] == checkpoint["arch_params"]
    assert torch.equal(
        loaded["model_state_dict"]["weight"],
        checkpoint["model_state_dict"]["weight"],
    )


def test_runtime_model_checkpoint_staging_preserves_prior_file_on_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial spawn fallback must never replace a usable runtime model."""
    import torch

    checkpoint = {
        "model_state_dict": {"weight": torch.arange(6).reshape(2, 3)},
        "arch_params": {"channels": 2},
        "encoding_version": 2,
        "step": 17,
    }
    path = tmp_path / "temp_selfplay_model.pt"
    prior = b"previous-valid-runtime-checkpoint"
    path.write_bytes(prior)
    temporary = path.with_name(path.name + ".tmp")
    real_open = Path.open

    class PartialWriter:
        def __enter__(self):
            self.handle = real_open(temporary, "wb")
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.handle.close()
            return False

        def write(self, payload):
            self.handle.write(memoryview(payload)[:37])
            self.handle.flush()
            raise OSError(28, "simulated runtime checkpoint disk full")

    def failing_open(target, mode="r", *args, **kwargs):
        if target == temporary and mode == "wb":
            return PartialWriter()
        return real_open(target, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match="simulated runtime checkpoint disk full"):
        trainer_module._stage_runtime_model_checkpoint(
            checkpoint, path, persist_to_disk=True)

    assert path.read_bytes() == prior
    assert not temporary.exists()


def test_fork_behavior_model_loads_live_state_without_detached_cpu_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sole CPU model receives the live state mapping directly."""
    source_state = {"weight": object()}

    class LiveModel:
        def state_dict(self):
            return source_state

    class ForkModel:
        def __init__(self) -> None:
            self.loaded_state = None
            self.eval_called = False
            self.cpu_called = False

        def load_state_dict(self, state) -> None:
            self.loaded_state = state

        def eval(self):
            self.eval_called = True
            return self

        def cpu(self):
            self.cpu_called = True
            return self

    fork_model = ForkModel()
    created_arch = []

    def fake_create_model(**arch):
        created_arch.append(arch)
        return fork_model

    monkeypatch.setattr(trainer_module, "create_model", fake_create_model)
    result = trainer_module._build_fork_behavior_model(
        LiveModel(), {"channels": 7})

    assert result is fork_model
    assert created_arch == [{"channels": 7, "initialize_weights": False}]
    assert fork_model.loaded_state is source_state
    assert fork_model.eval_called
    assert fork_model.cpu_called


def test_behavior_capture_and_optimizer_update_cannot_mix_revisions() -> None:
    """The behavior snapshot owns one complete model revision and step."""
    import threading

    import torch

    state_lock = threading.Lock()
    source = {
        "first": torch.tensor([0.0]),
        "second": torch.tensor([0.0]),
    }
    first_copied = threading.Event()
    continue_copy = threading.Event()
    update_finished = threading.Event()
    captured = []
    step = [7]

    def capture_state():
        result = {"first": source["first"].clone()}
        first_copied.set()
        assert continue_copy.wait(2.0)
        result["second"] = source["second"].clone()
        return result

    def capture_revision():
        captured.append(trainer_module._capture_model_revision(
            state_lock, lambda: step[0], capture_state))

    def optimizer_update():
        def mutate():
            source["first"].fill_(1.0)
            source["second"].fill_(1.0)
            step[0] += 1

        trainer_module._call_under_model_state_lock(state_lock, mutate)
        update_finished.set()

    capture_thread = threading.Thread(target=capture_revision)
    capture_thread.start()
    assert first_copied.wait(2.0)
    update_thread = threading.Thread(target=optimizer_update)
    update_thread.start()
    assert not update_finished.wait(0.05)
    continue_copy.set()
    capture_thread.join(2.0)
    update_thread.join(2.0)

    captured_step, captured_state = captured[0]
    assert captured_step == 7
    assert captured_state["first"].item() == 0.0
    assert captured_state["second"].item() == 0.0
    assert source["first"].item() == 1.0
    assert source["second"].item() == 1.0
    assert step[0] == 8


def test_cpu_behavior_fallback_owns_tensor_storage() -> None:
    """CPU fallback serialization must not alias later trainer updates."""
    import torch

    model = torch.nn.Linear(2, 1)
    copied = trainer_module._copy_behavior_state_to_cpu(
        model, torch.device("cpu"))
    original = {key: value.clone() for key, value in copied.items()}

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    for key, value in copied.items():
        assert torch.equal(value, original[key])
        assert value.data_ptr() != model.state_dict()[key].data_ptr()


@pytest.mark.parametrize(
    "accum_steps,batch_sizes,record_every",
    [
        (1, (2,), 1),
        (1, (2, 1), 1),
        (2, (2, 2), 1),
        (2, (2, 1), 1),
        (3, (2, 2, 1), 1),
        (2, (2, 1, 1, 1), 1),
        (2, (2, 2, 2, 1), 2),
    ],
)
def test_recorded_step_time_includes_forward_and_backward_work(
    accum_steps, batch_sizes, record_every,
) -> None:
    """Throughput counts and timing cover every accumulated microbatch."""
    import threading
    import time

    import torch

    state_lock = threading.Lock()

    class SlowModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward_padded(self, boards, move_features, move_counts):
            del boards, move_counts
            assert state_lock.locked()
            time.sleep(0.04)
            return self.scale * move_features.squeeze(-1)

    class CapturingStats:
        def __init__(self) -> None:
            self.steps = []

        def record_training_step(self, **record) -> None:
            self.steps.append(record)

        def record_epoch(self, **record) -> None:
            pass

    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        gradient_accumulation_steps=accum_steps,
        stats_record_every=record_every,
        stats_score_dist_every=100,
        stats_system_every=100,
        stats_model_health_every=100,
        checkpoint_every=100,
        train_steps=len(batch_sizes) // accum_steps,
        grad_clip_norm=None,
        amp=False,
        value_head_enabled=False,
        value_weight=0.15,
        policy_stage="policy_only",
        batch_size=2,
        learning_rate=0.1,
        thermal_enabled=False,
    )
    holder.device = torch.device("cpu")
    holder.model = SlowModel()
    holder.optimizer = torch.optim.SGD(holder.model.parameters(), lr=0.1)
    original_step = holder.optimizer.step

    def checked_step():
        assert state_lock.locked()
        return original_step()

    holder.optimizer.step = checked_step
    holder.scheduler = None
    holder.scaler = None
    holder.amp_dtype = torch.bfloat16
    holder.stats_collector = CapturingStats()
    holder._use_padded = True
    holder._compiled_fwd_loss = None
    holder._control_queue = None
    holder._stopped = False
    holder._paused = False
    holder._model_state_lock = state_lock
    holder.step = 0
    holder.epoch = 0
    holder._data_refreshed_pending = False
    holder._record_step_stats = lambda *args, **kwargs: None
    holder._update_process_title = lambda *args, **kwargs: None
    holder._compute_loss_padded = (
        lambda scores, move_counts, targets, reward_weights:
        torch.nn.functional.cross_entropy(scores, targets)
    )

    batch = (
        torch.zeros(2, 1),
        torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]]),
        torch.tensor([2, 2]),
        torch.tensor([0, 1]),
        torch.ones(2),
        torch.zeros(2),
    )
    holder.train_epoch([tuple(field[:size] for field in batch) for size in batch_sizes])

    expected_sizes = [
        sum(batch_sizes[start:start + accum_steps])
        for start in range(0, len(batch_sizes), accum_steps)
        if (start // accum_steps + 1) % record_every == 0
    ]
    recorded = holder.stats_collector.steps
    assert [record["batch_size"] for record in recorded] == expected_sizes
    assert all(record["step_time"] >= 0.04 * accum_steps for record in recorded)


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

    # Audit Suggestion 7 widened the gate to admit this namespace's own
    # lineage-verified checkpoints; a bare file outside it is still refused.
    other = tmp_path / "model_step_136000.pt"
    other.write_bytes(b"other")
    config.resume = str(other)
    with pytest.raises(ValueError, match="Recovery must resume from"):
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


def _acceptance_game_evidence(
    config: TrainingConfig,
    opponent_type: str,
    *,
    p1_wins: int,
    p1_draws: int,
    p1_losses: int,
    p2_wins: int,
    p2_draws: int,
    p2_losses: int,
) -> list[dict]:
    """Build the exact paired opening evidence behind acceptance aggregates."""
    games = []
    side_counts = {
        1: (p1_wins, p1_draws, p1_losses),
        2: (p2_wins, p2_draws, p2_losses),
    }
    for player, (wins, draws, losses) in side_counts.items():
        assert wins + draws + losses == 50
        results = ["ml_win"] * wins + ["draw"] * draws + ["algo_win"] * losses
        for index, result in enumerate(results):
            games.append({
                "result": result,
                "ml_player": player,
                "winner": (
                    player if result == "ml_win"
                    else None if result == "draw"
                    else 3 - player
                ),
                "opponent_type": opponent_type,
                "opening_plies": config.test_opening_plies[
                    index % len(config.test_opening_plies)],
                "opening_seed": config.test_opening_seed + index,
                "ml_inference_depth": 1,
            })
    return games


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
    promoted_sha256 = hashlib.sha256(promoted.read_bytes()).hexdigest().upper()
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
        "total_games": 100,
        "ml_wins": 90, "draws": 0, "algo_wins": 10,
        "ml_as_p1_wins": 45, "ml_as_p1_draws": 0, "ml_as_p1_losses": 5,
        "ml_as_p2_wins": 45, "ml_as_p2_draws": 0, "ml_as_p2_losses": 5,
        "games": _acceptance_game_evidence(
            config, "random",
            p1_wins=45, p1_draws=0, p1_losses=5,
            p2_wins=45, p2_draws=0, p2_losses=5,
        ),
        "opponent_type": "random", **common,
    }
    easy_result = {
        "total_games": 100,
        "ml_wins": 65, "draws": 10, "algo_wins": 25,
        "ml_as_p1_wins": 32, "ml_as_p1_draws": 5, "ml_as_p1_losses": 13,
        "ml_as_p2_wins": 33, "ml_as_p2_draws": 5, "ml_as_p2_losses": 12,
        "games": _acceptance_game_evidence(
            config, "algorithm",
            p1_wins=32, p1_draws=5, p1_losses=13,
            p2_wins=33, p2_draws=5, p2_losses=12,
        ),
        "opponent_type": "algorithm", **common,
    }
    decision = checkpoint_acceptance.evaluate_acceptance_gates(
        0.55, random_result, easy_result)
    report = {
        "schema_version": 1,
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


def test_enhanced_stage_rejects_incomplete_paired_game_evidence(
    tmp_path: Path,
) -> None:
    fixture = _enhanced_stage_acceptance_fixture(tmp_path)
    fixture["report"]["random"].pop("games")
    fixture["write_report"]()

    with pytest.raises(ValueError, match="complete paired-game protocol"):
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


def test_enhanced_stage_rejects_coerced_promotion_agreement(
    tmp_path: Path,
) -> None:
    fixture = _enhanced_stage_acceptance_fixture(tmp_path)
    fixture["registry_record"]["teacher_agreement"] = "0.55"
    fixture["write_registry"]()

    with pytest.raises(ValueError, match="recorded promoted policy-only"):
        validate_recovery_experiment_config(fixture["config"])


def test_enhanced_stage_rejects_coerced_report_agreement(
    tmp_path: Path,
) -> None:
    fixture = _enhanced_stage_acceptance_fixture(tmp_path)
    fixture["report"]["metrics"]["teacher_agreement"] = "0.55"
    fixture["write_report"]()

    with pytest.raises(ValueError, match="agreement provenance"):
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
    promoted_sha256 = hashlib.sha256(promoted.read_bytes()).hexdigest().upper()
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
        "total_games": 100,
        "ml_wins": 90,
        "draws": 0,
        "algo_wins": 10,
        "ml_as_p1_wins": 45,
        "ml_as_p1_draws": 0,
        "ml_as_p1_losses": 5,
        "ml_as_p2_wins": 45,
        "ml_as_p2_draws": 0,
        "ml_as_p2_losses": 5,
        "games": _acceptance_game_evidence(
            config, "random",
            p1_wins=45, p1_draws=0, p1_losses=5,
            p2_wins=45, p2_draws=0, p2_losses=5,
        ),
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
        "total_games": 100,
        "ml_wins": 65,
        "draws": 10,
        "algo_wins": 25,
        "ml_as_p1_wins": 32,
        "ml_as_p1_draws": 5,
        "ml_as_p1_losses": 13,
        "ml_as_p2_wins": 33,
        "ml_as_p2_draws": 5,
        "ml_as_p2_losses": 12,
        "games": _acceptance_game_evidence(
            config, "algorithm",
            p1_wins=32, p1_draws=5, p1_losses=13,
            p2_wins=33, p2_draws=5, p2_losses=12,
        ),
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
        "schema_version": 1,
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


@pytest.mark.parametrize(
    (
        "profile",
        "expected_compression",
        "expected_validation_cache",
        "expected_validation_compression",
    ),
    [
        (None, True, "dama_policy_distillation_recovery_c174k_validation_cache.pt", False),
        ("server", False, "policy_distillation_recovery_c174k_server_validation_cache.pt", False),
    ],
)
def test_active_policy_yaml_scopes_cache_compression_to_local(
    profile: str | None,
    expected_compression: bool,
    expected_validation_cache: str,
    expected_validation_compression: bool,
) -> None:
    """The disk-constrained local cache must not change server warm-start I/O."""
    root = Path(__file__).resolve().parents[2]
    path = root / "config" / "training_config_policy_distillation_c174k.yaml"
    config = config_from_yaml(load_config_from_yaml(str(path), profile))

    assert config.ram_cache_compress is expected_compression
    assert config.validation_tensor_cache_file.endswith(expected_validation_cache)
    assert config.validation_tensor_cache_compress is expected_validation_compression
    assert config.selfplay_min_free_disk_gb == pytest.approx(10.0)


@pytest.mark.parametrize("value", [True, -1, float("nan"), "ten"])
def test_selfplay_storage_floor_rejects_invalid_yaml_values(value) -> None:
    with pytest.raises(ValueError, match="minimum_free_disk_gb"):
        config_from_yaml({"selfplay": {"minimum_free_disk_gb": value}})


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
        param_groups = [{"lr": 1e-3, "params": []}]

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


# ---------------------------------------------------------------------------
# Audit Suggestion 7: lineage-verified continuation resume
# ---------------------------------------------------------------------------

def _stamped_checkpoint(
    path: Path,
    *,
    baseline_sha256: str,
    training_stage: str = "policy_only",
    enabled: bool = True,
    finite: bool = True,
) -> Path:
    """Write a checkpoint carrying the recovery lineage stamp this arm writes."""
    import torch

    weight = torch.ones(2, 2)
    if not finite:
        weight = weight * float("nan")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {"w": weight},
            "step": int(path.stem.rsplit("_", 1)[-1]),
            "recovery_experiment": {
                "enabled": enabled,
                "baseline_sha256": baseline_sha256,
                "training_stage": training_stage,
                "resume_anchor": "anchor",
            },
        },
        path,
    )
    return path


def _recovery_config_with_namespace(tmp_path: Path):
    """A valid recovery config whose checkpoint directory exists."""
    baseline = tmp_path / "model_step_174000.pt"
    baseline.write_bytes(b"approved continuation anchor")
    config = _valid_recovery_config(baseline)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return config, baseline


def test_recovery_gate_accepts_a_lineage_verified_continuation(tmp_path: Path) -> None:
    """A relaunch may resume its own newest verified checkpoint.

    Without this the experiment's ceiling was one uninterrupted session: every
    relaunch re-walked from the anchor, discarding the steps already trained
    and restarting the frozen-suite agreement series.
    """
    config, _baseline = _recovery_config_with_namespace(tmp_path)
    continuation = _stamped_checkpoint(
        Path(config.checkpoint_dir) / "model_step_202000.pt",
        baseline_sha256=config.recovery_baseline_sha256,
    )
    config.resume = str(continuation)

    validate_recovery_experiment_config(config)


def test_recovery_gate_refuses_a_stamped_checkpoint_from_another_namespace(
    tmp_path: Path,
) -> None:
    """Output isolation is per-directory, so the stamp alone is not enough."""
    config, _baseline = _recovery_config_with_namespace(tmp_path)
    foreign = _stamped_checkpoint(
        tmp_path / "foreign_namespace" / "model_step_202000.pt",
        baseline_sha256=config.recovery_baseline_sha256,
    )
    config.resume = str(foreign)

    with pytest.raises(ValueError, match="Recovery must resume from"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_refuses_an_unstamped_checkpoint_in_the_namespace(
    tmp_path: Path,
) -> None:
    """A file merely sitting in the directory proves no ancestry."""
    import torch

    config, _baseline = _recovery_config_with_namespace(tmp_path)
    unstamped = Path(config.checkpoint_dir) / "model_step_202000.pt"
    torch.save({"model_state_dict": {"w": torch.ones(2, 2)}}, unstamped)
    config.resume = str(unstamped)

    with pytest.raises(ValueError, match="Recovery must resume from"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_refuses_a_continuation_from_a_different_baseline(
    tmp_path: Path,
) -> None:
    """A checkpoint from a different recovery arm is not this arm's descendant."""
    config, _baseline = _recovery_config_with_namespace(tmp_path)
    wrong = _stamped_checkpoint(
        Path(config.checkpoint_dir) / "model_step_202000.pt",
        baseline_sha256="0" * 64,
    )
    config.resume = str(wrong)

    with pytest.raises(ValueError, match="Recovery must resume from"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_refuses_a_continuation_from_another_training_stage(
    tmp_path: Path,
) -> None:
    config, _baseline = _recovery_config_with_namespace(tmp_path)
    wrong_stage = _stamped_checkpoint(
        Path(config.checkpoint_dir) / "model_step_202000.pt",
        baseline_sha256=config.recovery_baseline_sha256,
        training_stage="enhanced",
    )
    config.resume = str(wrong_stage)

    with pytest.raises(ValueError, match="Recovery must resume from"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_refuses_a_continuation_holding_non_finite_weights(
    tmp_path: Path,
) -> None:
    """Impeccable lineage does not make NaN weights a valid resume point."""
    config, _baseline = _recovery_config_with_namespace(tmp_path)
    broken = _stamped_checkpoint(
        Path(config.checkpoint_dir) / "model_step_202000.pt",
        baseline_sha256=config.recovery_baseline_sha256,
        finite=False,
    )
    config.resume = str(broken)

    with pytest.raises(ValueError, match="Recovery must resume from"):
        validate_recovery_experiment_config(config)


def test_recovery_gate_refuses_a_stamped_alias_in_the_namespace(
    tmp_path: Path,
) -> None:
    """Aliases are republished in place, so resuming one is unverifiable."""
    config, _baseline = _recovery_config_with_namespace(tmp_path)
    alias = _stamped_checkpoint(
        Path(config.checkpoint_dir) / "model_step_202000.pt",
        baseline_sha256=config.recovery_baseline_sha256,
    )
    moving = Path(config.checkpoint_dir) / "latest_alias.pt"
    moving.write_bytes(alias.read_bytes())
    config.resume = str(moving)

    with pytest.raises(ValueError, match="Recovery must resume from"):
        validate_recovery_experiment_config(config)


def test_continuation_resolver_picks_the_newest_verified_checkpoint(
    tmp_path: Path,
) -> None:
    config, _baseline = _recovery_config_with_namespace(tmp_path)
    directory = Path(config.checkpoint_dir)
    for step in (176000, 202000, 178000):
        _stamped_checkpoint(
            directory / f"model_step_{step:06d}.pt",
            baseline_sha256=config.recovery_baseline_sha256,
        )
    # A later but unverifiable file must not win the selection.
    _stamped_checkpoint(
        directory / "model_step_204000.pt",
        baseline_sha256="0" * 64,
    )

    resolved = trainer_module.resolve_recovery_continuation_resume(config)

    assert resolved is not None
    assert resolved.name == "model_step_202000.pt"


def test_continuation_resolver_returns_none_on_an_empty_namespace(
    tmp_path: Path,
) -> None:
    """First launch of a namespace: the caller keeps the pinned anchor."""
    config, _baseline = _recovery_config_with_namespace(tmp_path)

    assert trainer_module.resolve_recovery_continuation_resume(config) is None


def test_continuation_resolver_requires_a_recovery_config(tmp_path: Path) -> None:
    config, _baseline = _recovery_config_with_namespace(tmp_path)
    config.recovery_enforced = False

    with pytest.raises(ValueError, match="recovery_experiment config"):
        trainer_module.resolve_recovery_continuation_resume(config)


def test_continuation_resolver_rejects_a_tensor_free_state_dict(
    tmp_path: Path,
) -> None:
    """A lineage stamp without any weight tensor proves nothing about finiteness."""
    import torch

    config, _baseline = _recovery_config_with_namespace(tmp_path)
    path = Path(config.checkpoint_dir) / "model_step_176000.pt"
    torch.save(
        {
            "model_state_dict": {"step": 176000},
            "recovery_experiment": {
                "enabled": True,
                "baseline_sha256": config.recovery_baseline_sha256,
                "training_stage": "policy_only",
                "resume_anchor": "anchor",
            },
        },
        path,
    )

    assert trainer_module.resolve_recovery_continuation_resume(config) is None


# ---------------------------------------------------------------------------
# Audit Suggestion 8: run provenance
# ---------------------------------------------------------------------------

def test_run_identity_names_the_session_and_its_resume_point(tmp_path: Path) -> None:
    """Two runs sharing a corpus were previously indistinguishable.

    ``dataset_fingerprint`` discriminates corpora, not sessions, so a numbered
    range spanning two runs read as one continuous trajectory.
    """
    from dama.ai.ml import run_status

    log_dir = tmp_path / "logs"
    run_status.begin_run(log_dir, pid=4321)
    holder = SimpleNamespace(
        config=SimpleNamespace(resume=str(tmp_path / "model_step_174000.pt")),
        step=174000,
    )

    Trainer._capture_run_identity(holder, str(log_dir))

    assert holder._run_identity["pid"] == 4321
    assert holder._run_identity["resume_step"] == 174000
    assert holder._run_identity["started_at"]
    assert holder._run_identity["run_id"].startswith("4321@")


def test_startup_reports_checkpoints_above_the_resume_point(
    tmp_path: Path, capsys
) -> None:
    """The mixed-lineage range is announced while the operator can still act."""
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    for step in (176000, 178000):
        (directory / f"model_step_{step:06d}.pt").write_bytes(b"x")
    holder = SimpleNamespace(
        config=SimpleNamespace(checkpoint_dir=str(directory)),
        step=174000,
    )

    Trainer._warn_about_checkpoints_above_resume_point(holder)

    captured = capsys.readouterr().out
    assert "above this run's resume point (step 174000)" in captured
    assert "176000 .. 178000" in captured
    assert "more than one lineage" in captured


def test_startup_is_quiet_when_no_checkpoint_is_ahead(tmp_path: Path, capsys) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "model_step_174000.pt").write_bytes(b"x")
    holder = SimpleNamespace(
        config=SimpleNamespace(checkpoint_dir=str(directory)),
        step=174000,
    )

    Trainer._warn_about_checkpoints_above_resume_point(holder)

    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Audit Suggestion 10: checkpoint retention
# ---------------------------------------------------------------------------

def _retention_holder(tmp_path: Path, keep: int, records=()) -> SimpleNamespace:
    from dama.ai.ml.teacher_validation import PromotionRegistry

    directory = tmp_path / "checkpoints"
    directory.mkdir(exist_ok=True)
    registry_path = tmp_path / "promotions.jsonl"
    if records:
        registry_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    holder = SimpleNamespace(
        config=SimpleNamespace(
            checkpoint_dir=str(directory),
            max_retained_checkpoints=keep,
            resume=None,
            recovery_baseline_path=None,
        ),
        _promotion_registry=PromotionRegistry(str(registry_path)),
    )
    holder._protected_checkpoint_paths = (
        Trainer._protected_checkpoint_paths.__get__(holder))
    return holder


def _write_steps(directory: Path, steps) -> None:
    for step in steps:
        (directory / f"model_step_{step:06d}.pt").write_bytes(b"checkpoint")


def test_checkpoint_retention_is_off_by_default(tmp_path: Path) -> None:
    """0 must preserve the historical keep-everything behaviour exactly."""
    holder = _retention_holder(tmp_path, keep=0)
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000])

    removed = Trainer._prune_old_checkpoints(holder, directory / "model_step_006000.pt")

    assert removed == []
    assert len(list(directory.glob("model_step_*.pt"))) == 3


def test_checkpoint_retention_keeps_only_the_newest_n(tmp_path: Path) -> None:
    holder = _retention_holder(tmp_path, keep=2)
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000, 8000])

    removed = Trainer._prune_old_checkpoints(holder, directory / "model_step_008000.pt")

    assert sorted(removed) == ["model_step_002000.pt", "model_step_004000.pt"]
    assert sorted(p.name for p in directory.glob("model_step_*.pt")) == [
        "model_step_006000.pt",
        "model_step_008000.pt",
    ]


def test_checkpoint_retention_commits_deleted_names_once(
    tmp_path: Path, monkeypatch,
) -> None:
    """One directory sync commits the complete retention deletion batch."""
    holder = _retention_holder(tmp_path, keep=2)
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000, 8000])
    syncs = []

    def track_sync(path: Path) -> None:
        assert not (directory / "model_step_002000.pt").exists()
        assert not (directory / "model_step_004000.pt").exists()
        syncs.append(Path(path))

    monkeypatch.setattr(trainer_module, "_fsync_directory", track_sync)

    removed = Trainer._prune_old_checkpoints(
        holder, directory / "model_step_008000.pt")

    assert sorted(removed) == [
        "model_step_002000.pt",
        "model_step_004000.pt",
    ]
    assert syncs == [directory]


def test_checkpoint_retention_directory_sync_failure_is_not_acknowledged(
    tmp_path: Path, monkeypatch,
) -> None:
    """A failed deletion commit must remain visible to the writer thread."""
    holder = _retention_holder(tmp_path, keep=2)
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000, 8000])

    def fail_sync(path: Path) -> None:
        assert Path(path) == directory
        raise OSError(5, "simulated checkpoint retention directory sync failure")

    monkeypatch.setattr(trainer_module, "_fsync_directory", fail_sync)

    with pytest.raises(
        OSError, match="simulated checkpoint retention directory sync failure",
    ):
        Trainer._prune_old_checkpoints(
            holder, directory / "model_step_008000.pt")


def test_checkpoint_retention_without_deletions_does_not_sync_directory(
    tmp_path: Path, monkeypatch,
) -> None:
    """A checkpoint set already within its bound pays no retention sync."""
    holder = _retention_holder(tmp_path, keep=2)
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000])

    def unexpected_sync(_path: Path) -> None:
        raise AssertionError("retention synced without deleting a checkpoint")

    monkeypatch.setattr(trainer_module, "_fsync_directory", unexpected_sync)

    assert Trainer._prune_old_checkpoints(
        holder, directory / "model_step_004000.pt") == []


def test_checkpoint_retention_never_deletes_a_promoted_checkpoint(
    tmp_path: Path,
) -> None:
    holder = _retention_holder(
        tmp_path,
        keep=1,
        records=[
            {
                "checkpoint_path": "checkpoints/model_step_002000.pt",
                "step": 2000,
                "teacher_agreement": 0.55,
                "promoted": True,
            },
        ],
    )
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000])

    removed = Trainer._prune_old_checkpoints(holder, directory / "model_step_006000.pt")

    assert removed == ["model_step_004000.pt"]
    assert (directory / "model_step_002000.pt").exists()


def test_checkpoint_retention_never_deletes_the_registry_best(tmp_path: Path) -> None:
    """Nothing has cleared the 0.50 gate on this arm.

    Protecting only promotions would therefore leave the entire agreement
    series prunable and discard the best result the run has produced.
    """
    holder = _retention_holder(
        tmp_path,
        keep=1,
        records=[
            {
                "checkpoint_path": "checkpoints/model_step_002000.pt",
                "step": 2000,
                "teacher_agreement": 0.4898,
                "promoted": False,
            },
            {
                "checkpoint_path": "checkpoints/model_step_004000.pt",
                "step": 4000,
                "teacher_agreement": 0.4744,
                "promoted": False,
            },
        ],
    )
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000])

    removed = Trainer._prune_old_checkpoints(holder, directory / "model_step_006000.pt")

    assert removed == ["model_step_004000.pt"]
    assert (directory / "model_step_002000.pt").exists()


def test_checkpoint_retention_never_deletes_the_resume_source(tmp_path: Path) -> None:
    holder = _retention_holder(tmp_path, keep=1)
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000])
    holder.config.resume = str(directory / "model_step_002000.pt")

    removed = Trainer._prune_old_checkpoints(holder, directory / "model_step_006000.pt")

    assert removed == ["model_step_004000.pt"]
    assert (directory / "model_step_002000.pt").exists()


def test_checkpoint_retention_fails_closed_on_an_unreadable_registry(
    tmp_path: Path,
) -> None:
    """An unreadable registry is not evidence that nothing is promoted."""
    holder = _retention_holder(tmp_path, keep=1)
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000])
    Path(holder._promotion_registry.path).write_text("{not json\n", encoding="utf-8")

    removed = Trainer._prune_old_checkpoints(holder, directory / "model_step_006000.pt")

    assert removed == []
    assert len(list(directory.glob("model_step_*.pt"))) == 3


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("teacher_agreement", True),
        ("teacher_agreement", "0.99"),
        ("teacher_agreement", float("nan")),
        ("promoted", 1),
        ("checkpoint_path", ["checkpoints/model_step_004000.pt"]),
    ),
)
def test_checkpoint_retention_fails_closed_on_malformed_registry_evidence(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    """Corrupt JSON values cannot redirect a destructive high-water mark."""
    records = [
        {
            "checkpoint_path": "checkpoints/model_step_002000.pt",
            "step": 2000,
            "teacher_agreement": 0.55,
            "promoted": False,
        },
        {
            "checkpoint_path": "checkpoints/model_step_004000.pt",
            "step": 4000,
            "teacher_agreement": 0.40,
            "promoted": False,
        },
    ]
    records[1][field] = value
    holder = _retention_holder(tmp_path, keep=1, records=records)
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000])

    removed = Trainer._prune_old_checkpoints(
        holder, directory / "model_step_006000.pt")

    assert removed == []
    assert sorted(path.name for path in directory.glob("model_step_*.pt")) == [
        "model_step_002000.pt",
        "model_step_004000.pt",
        "model_step_006000.pt",
    ]


# ---------------------------------------------------------------------------
# Proofread 2026-08-25 B1: protection must consume the retention budget
# ---------------------------------------------------------------------------

def test_checkpoint_retention_budget_is_met_despite_protected_files(
    tmp_path: Path,
) -> None:
    """Proofread 2026-08-25 B1.

    The old window slice (``candidates[:len(candidates)-keep_count]``) only
    ever examined the oldest few files, so protected entries inside that
    window were skipped without replacement and the live count crept above
    ``max_retained_checkpoints`` with every promotion.  With keep=2 and
    candidates [1000P, 2000, 3000, 4000] it deleted only 2000 and left three
    files on disk; the walk must instead delete both 2000 and 3000.
    """
    holder = _retention_holder(
        tmp_path,
        keep=2,
        records=[
            {
                "checkpoint_path": "checkpoints/model_step_001000.pt",
                "step": 1000,
                "teacher_agreement": 0.55,
                "promoted": True,
            },
        ],
    )
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [1000, 2000, 3000, 4000])

    removed = Trainer._prune_old_checkpoints(holder, directory / "model_step_004000.pt")

    assert sorted(removed) == [
        "model_step_002000.pt",
        "model_step_003000.pt",
    ]
    assert sorted(p.name for p in directory.glob("model_step_*.pt")) == [
        "model_step_001000.pt",
        "model_step_004000.pt",
    ]


def test_checkpoint_retention_walk_extends_past_protected_files(
    tmp_path: Path,
) -> None:
    """A protected entry in the middle of the walk extends the deletion window.

    With keep=1 and candidates [2000, 4000P, 6000, 8000], the old fixed
    window ``candidates[:3]`` deleted only 2000 (it stopped before 8000 and
    skipped 4000 without replacement), leaving three files on disk.  The
    oldest-first walk must continue past the protected 4000 and delete 6000
    too, meeting the budget among unprotected files.
    """
    holder = _retention_holder(
        tmp_path,
        keep=1,
        records=[
            {
                "checkpoint_path": "checkpoints/model_step_004000.pt",
                "step": 4000,
                "teacher_agreement": 0.55,
                "promoted": True,
            },
        ],
    )
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000, 8000])

    removed = Trainer._prune_old_checkpoints(holder, directory / "model_step_008000.pt")

    assert sorted(removed) == [
        "model_step_002000.pt",
        "model_step_006000.pt",
    ]
    assert sorted(p.name for p in directory.glob("model_step_*.pt")) == [
        "model_step_004000.pt",
        "model_step_008000.pt",
    ]


def test_checkpoint_retention_warns_when_protection_exceeds_the_budget(
    tmp_path: Path,
    capsys,
) -> None:
    """When protection alone pushes the live count over the budget, say so.

    Silence here is what let the unbounded growth go unnoticed: an operator
    watching the log must see that ``max_retained_checkpoints`` is not the
    number of checkpoints on disk.
    """
    records = [
        {
            "checkpoint_path": f"checkpoints/model_step_{step:06d}.pt",
            "step": step,
            "teacher_agreement": 0.55,
            "promoted": True,
        }
        for step in (2000, 4000)
    ]
    holder = _retention_holder(tmp_path, keep=1, records=records)
    directory = Path(holder.config.checkpoint_dir)
    _write_steps(directory, [2000, 4000, 6000])

    removed = Trainer._prune_old_checkpoints(holder, directory / "model_step_006000.pt")

    assert removed == []
    output = capsys.readouterr().out
    assert "[warn]" in output
    assert "max_retained_checkpoints" in output
    assert len(list(directory.glob("model_step_*.pt"))) == 3


# ---------------------------------------------------------------------------
# Proofread 2026-08-25 B2: prune-vs-rollback deletion race
# ---------------------------------------------------------------------------

def test_dead_epoch_rollback_tolerates_a_vanishing_pick(tmp_path: Path) -> None:
    """Proofread 2026-08-25 B2.

    Retention prunes inside the background checkpoint-writer thread while
    dead-epoch rollback globs ``model_step_*.pt`` on the main thread, so the
    newest glob hit can be unlinked milliseconds after selection.  Under
    recovery enforcement a vanished pick must fall back to the verified
    recovery rollback path instead of failing the run with "No verified
    checkpoint remains"; without enforcement it must fall back to the next
    remaining file.
    """
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    _write_steps(directory, [2000, 4000])
    holder = SimpleNamespace(
        config=SimpleNamespace(
            checkpoint_dir=str(directory),
            recovery_enforced=False,
        ),
        scaler=None,
    )
    loaded = []
    holder._load_checkpoint = lambda path: loaded.append(path)

    # Simulate the race: the pruner removes the picked file between the glob
    # and the load.  The rollback must retry with what is still on disk.
    def vanishing_pick(pattern):
        assert pattern == "model_step_*.pt"
        picks = sorted(directory.glob("model_step_*.pt"))
        target = picks[-1]
        target.unlink()  # background writer's prune wins the race
        return picks

    holder._rollback_checkpoint_candidates = vanishing_pick

    Trainer._rollback_after_dead_epoch(holder, reason="test")

    assert loaded == [str(directory / "model_step_002000.pt")]


def test_recovery_enforced_rollback_delegates_to_verified_lineage(
    tmp_path: Path,
) -> None:
    """Under recovery enforcement the fallback is the verified-lineage pick."""
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    _write_steps(directory, [2000, 4000])
    holder = SimpleNamespace(
        config=SimpleNamespace(
            checkpoint_dir=str(directory),
            recovery_enforced=True,
        ),
        scaler=None,
    )
    loaded = []
    holder._load_checkpoint = lambda path: loaded.append(path)
    verified = directory / "model_step_002000.pt"
    holder._verified_recovery_rollback_checkpoint = lambda: verified
    holder._rollback_checkpoint_candidates = (
        lambda pattern: sorted(directory.glob(pattern)))

    Trainer._rollback_after_dead_epoch(holder, reason="test")

    assert loaded == [str(verified)]


def test_recovery_enforced_race_fallback_skips_unstamped_files(
    tmp_path: Path,
) -> None:
    """A vanished verified pick must not let an unstamped file load.

    When the verified pick is pruned between selection and load (the
    prune-vs-rollback race), the fallback loop walks whatever files remain.
    Every other recovery path refuses checkpoints without the pinned lineage
    stamp; the race fallback must enforce the same invariant instead of
    silently loading a foreign-lineage state.
    """
    import torch

    directory = tmp_path / "checkpoints"
    directory.mkdir()
    stamped = _stamped_checkpoint(
        directory / "model_step_002000.pt",
        baseline_sha256="a" * 64,
    )
    # Newest file on disk carries no lineage stamp at all.
    (directory / "model_step_004000.pt").write_bytes(b"unstamped")
    holder = SimpleNamespace(
        config=SimpleNamespace(
            checkpoint_dir=str(directory),
            recovery_enforced=True,
            recovery_baseline_sha256="a" * 64,
            policy_stage="policy_only",
        ),
        scaler=None,
    )
    loaded = []
    holder._load_checkpoint = lambda path: loaded.append(path)

    holder.config.resume = str(tmp_path / "model_step_000000.pt")
    holder.step = 4000
    selector_calls = 0

    def vanishing_verified_pick():
        nonlocal selector_calls
        selector_calls += 1
        if selector_calls == 1:
            raise FileNotFoundError(str(directory / "model_step_999999.pt"))
        return Trainer._verified_recovery_rollback_checkpoint(holder)

    holder._verified_recovery_rollback_checkpoint = vanishing_verified_pick
    holder._rollback_checkpoint_candidates = (
        lambda pattern: sorted(directory.glob(pattern)))

    Trainer._rollback_after_dead_epoch(holder, reason="test")

    assert loaded == [str(stamped)]


def _tiny_replay_entries(count: int = 3):
    """Minimal real replay entries so tensorization runs on the CPU path."""
    from dama.game_state import GameState
    from dama.ai.ml.replay import ReplayEntry

    state = GameState.initial()
    moves = state.legal_moves()
    return [
        ReplayEntry(
            state=state.to_compact(),
            legal_moves=[m.to_dict() for m in moves[:5]],
            chosen_index=0,
            result=1,
            score=2.5,
        )
        for _ in range(count)
    ]


def test_prelaunch_free_ram_is_captured_onto_the_trainer():
    """Pass 117: the pre-load reading must land on the trainer instance.

    The RAM-cache gate measures free RAM only after the corpus inflates RSS,
    so it needs this earlier reading to judge host headroom honestly.
    """
    import psutil

    holder = object.__new__(Trainer)
    assert not hasattr(holder, "_prelaunch_free_ram_gb")

    Trainer._capture_prelaunch_free_ram(holder)

    expected = psutil.virtual_memory().available / (1024 ** 3)
    assert holder._prelaunch_free_ram_gb == pytest.approx(expected, abs=0.5)


def test_ram_cache_gate_falls_back_to_the_prelaunch_reading(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Pass 117: corpus growth must not silently demote launches.

    At the 60-file steady state the post-load reading sits below the 16 GB
    gate (~14.1 GB measured 2026-08-25) while the host still had ~20 GB free
    before the load.  With no fallback the gate declines the cache and every
    later launch trains on the slow standard path; with the pre-load reading
    the intended GPU-resident fast path engages again.
    """
    from dama.ai.ml import dataset as dataset_module
    from dama.ai.ml.dataset import FastBatchIterator, create_dataloader

    monkeypatch.setattr(dataset_module, "get_available_ram_gb", lambda: 14.0)
    monkeypatch.setattr(dataset_module, "get_total_ram_gb", lambda: 24.5)

    entries = _tiny_replay_entries()
    common = dict(
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        use_ram_cache=True,
        ram_threshold_gb=16.0,
        cache_file=str(tmp_path / "cache.pt"),
        device=None,
        capacity=0,
        max_moves_per_sample=8,
        amp_enabled=False,
    )

    declined = create_dataloader(entries, **common)
    assert not isinstance(declined, FastBatchIterator)

    accepted = create_dataloader(
        entries, prelaunch_free_ram_gb=19.6, **common)
    assert isinstance(accepted, FastBatchIterator)

    out = capsys.readouterr().out
    assert "gate satisfied by pre-load (19.6GB)" in out


def test_cached_tensor_dataset_save_compresses_and_preserves_prior_cache(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed compressed write must leave the last warm cache readable."""
    import torch

    from dama.ai.ml import dataset as dataset_module
    from dama.ai.ml.dataset import CachedTensorDataset

    dataset = CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    cache_file = tmp_path / "cache.pt"
    dataset.save(
        str(cache_file), metadata={"source": "test"}, compress=True)

    before = cache_file.read_bytes()
    assert before[:2] == b"\x1f\x8b"
    restored = CachedTensorDataset.load(str(cache_file), require_complete=True)
    assert restored.metadata["source"] == "test"
    for field in (
        "boards", "move_features", "move_counts", "targets",
        "reward_weights", "value_targets",
    ):
        assert torch.equal(getattr(restored, field), getattr(dataset, field))

    def fail_save(*_args, **_kwargs):
        raise OSError("simulated cache write failure")

    monkeypatch.setattr(dataset_module.torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated cache write failure"):
        dataset.save(
            str(cache_file), metadata={"source": "replacement"}, compress=True)

    assert cache_file.read_bytes() == before
    assert not list(tmp_path.glob(f".{cache_file.name}.*.tmp"))


def test_cached_tensor_dataset_save_commits_public_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The durable cache name is committed after its complete file bytes."""
    from dama.ai.ml import dataset as dataset_module
    from dama.ai.ml.dataset import CachedTensorDataset

    dataset = CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    cache_file = tmp_path / "cache.pt"
    events = []
    real_fsync = dataset_module.os.fsync
    real_replace = dataset_module.os.replace

    def tracking_fsync(fd):
        events.append("file_fsync")
        return real_fsync(fd)

    def tracking_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    def tracking_directory_fsync(path):
        if Path(path) == tmp_path.parent:
            events.append("namespace_fsync")
        else:
            events.append("directory_fsync")
            assert Path(path) == tmp_path

    monkeypatch.setattr(dataset_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(dataset_module.os, "replace", tracking_replace)
    monkeypatch.setattr(
        dataset_module, "_fsync_directory", tracking_directory_fsync)

    dataset.save(str(cache_file), metadata={"source": "test"}, compress=True)

    assert events == [
        "namespace_fsync", "file_fsync", "replace", "directory_fsync",
    ]
    assert cache_file.is_file()
    assert not list(tmp_path.glob(f".{cache_file.name}.*.tmp"))


def test_cached_tensor_dataset_reports_directory_commit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed directory commit is visible while complete bytes stay readable."""
    import torch

    from dama.ai.ml import dataset as dataset_module
    from dama.ai.ml.dataset import CachedTensorDataset

    dataset = CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    cache_file = tmp_path / "cache.pt"

    def fail_directory_fsync(path):
        if Path(path) == tmp_path:
            raise OSError("simulated cache directory fsync failure")

    monkeypatch.setattr(dataset_module, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(OSError, match="simulated cache directory fsync failure"):
        dataset.save(
            str(cache_file), metadata={"source": "replacement"}, compress=True)

    restored = CachedTensorDataset.load(str(cache_file), require_complete=True)
    assert restored.metadata["source"] == "replacement"
    for field in (
        "boards", "move_features", "move_counts", "targets",
        "reward_weights", "value_targets",
    ):
        assert torch.equal(getattr(restored, field), getattr(dataset, field))
    assert not list(tmp_path.glob(f".{cache_file.name}.*.tmp"))


def test_cached_tensor_dataset_commits_namespace_before_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache namespace must be durable before a file can be published."""
    from dama.ai.ml import dataset as dataset_module
    from dama.ai.ml.dataset import CachedTensorDataset

    dataset = CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    parent = tmp_path / "parent"
    parent.mkdir()
    cache_dir = parent / "cache_namespace"
    cache_file = cache_dir / "cache.pt"
    save_called = False
    real_directory_fsync = dataset_module._fsync_directory

    def fail_namespace_fsync(path):
        if Path(path) == parent:
            raise OSError("simulated cache namespace fsync failure")
        return real_directory_fsync(path)

    def unexpected_save(*_args, **_kwargs):
        nonlocal save_called
        save_called = True

    monkeypatch.setattr(
        dataset_module, "_fsync_directory", fail_namespace_fsync)
    monkeypatch.setattr(dataset_module.torch, "save", unexpected_save)

    with pytest.raises(OSError, match="simulated cache namespace fsync failure"):
        dataset.save(str(cache_file), metadata={"source": "test"})

    assert not save_called
    assert not cache_file.exists()
    assert not list(cache_dir.glob(f".{cache_file.name}.*.tmp"))

    monkeypatch.undo()
    dataset.save(str(cache_file), metadata={"source": "retry"})
    restored = CachedTensorDataset.load(str(cache_file), require_complete=True)
    assert restored.metadata["source"] == "retry"


def test_cached_tensor_dataset_loads_legacy_raw_cache(tmp_path: Path) -> None:
    """Compression keeps pre-existing direct torch.save cache files readable."""
    import torch

    from dama.ai.ml.dataset import CachedTensorDataset

    dataset = CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    cache_file = tmp_path / "legacy-cache.pt"
    torch.save({
        "boards": dataset.boards,
        "move_features": dataset.move_features,
        "move_counts": dataset.move_counts,
        "targets": dataset.targets,
        "reward_weights": dataset.reward_weights,
        "value_targets": dataset.value_targets,
        "metadata": {"legacy": True},
    }, cache_file)

    restored = CachedTensorDataset.load(str(cache_file), require_complete=True)
    assert restored.metadata == {"legacy": True}
    assert torch.equal(restored.boards, dataset.boards)
    assert torch.equal(restored.move_features, dataset.move_features)


def test_matching_raw_tensor_cache_migrates_when_compression_is_requested(
    tmp_path: Path,
) -> None:
    """A compression-enabled warm start atomically upgrades a valid raw cache."""
    import torch

    from dama.ai.ml.dataset import (
        CachedTensorDataset,
        load_matching_cached_tensor_dataset,
    )

    dataset = CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    cache_file = tmp_path / "legacy-cache.pt"
    source_key = {
        "cache_version": 3,
        "max_moves_per_sample": 8,
        "entry_count": len(dataset),
    }
    dataset.save(str(cache_file), metadata=source_key)
    assert cache_file.read_bytes()[:2] != b"\x1f\x8b"

    restored = load_matching_cached_tensor_dataset(
        str(cache_file),
        {"cache_version": 3, "max_moves_per_sample": 8},
        migrate_to_compressed=True,
    )

    assert restored is not None
    assert cache_file.read_bytes()[:2] == b"\x1f\x8b"
    assert restored.metadata["entry_count"] == len(dataset)
    assert torch.equal(restored.boards, dataset.boards)
    assert torch.equal(restored.move_features, dataset.move_features)


def test_manifest_keyed_tensor_cache_requires_the_complete_source_key(
    tmp_path: Path, monkeypatch
) -> None:
    """A warm snapshot cache cannot cross a source-fingerprint boundary."""
    from dama.ai.ml import dataset as dataset_module
    from dama.ai.ml.dataset import (
        FastBatchIterator,
        create_dataloader,
        load_matching_cached_tensor_dataset,
    )

    monkeypatch.setattr(dataset_module, "get_available_ram_gb", lambda: 20.0)
    monkeypatch.setattr(dataset_module, "get_total_ram_gb", lambda: 24.5)
    cache_file = tmp_path / "snapshot-cache.pt"
    source_key = {
        "cache_version": 3,
        "snapshot_train_cache_version": 2,
        "snapshot_fingerprint": "a" * 64,
        "external_validation_keys_sha256": "b" * 64,
        "validation_exclusion_keys_sha256": "c" * 64,
        "max_train_entries": 100,
        "max_moves_per_sample": 8,
        "encoding_version": trainer_module.ENCODING_VERSION,
        "policy_stage": "policy_only",
        "side_weight_balance_version": 1,
        "side_weight_balance": None,
    }
    loader = create_dataloader(
        _tiny_replay_entries(),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        use_ram_cache=True,
        ram_threshold_gb=16.0,
        cache_file=str(cache_file),
        device=None,
        capacity=0,
        max_moves_per_sample=8,
        amp_enabled=False,
        prelaunch_free_ram_gb=20.0,
        cache_metadata=source_key,
        load_existing_cache=False,
        compress_cache=True,
    )
    assert isinstance(loader, FastBatchIterator)
    assert cache_file.read_bytes()[:2] == b"\x1f\x8b"

    expected_key = dict(source_key)
    expected_key.pop("side_weight_balance")
    cached = load_matching_cached_tensor_dataset(str(cache_file), expected_key)
    assert cached is not None
    assert len(cached) == 3

    changed_key = dict(expected_key, snapshot_fingerprint="c" * 64)
    assert load_matching_cached_tensor_dataset(str(cache_file), changed_key) is None

    changed_holdout_key = dict(
        expected_key, validation_exclusion_keys_sha256="d" * 64)
    assert load_matching_cached_tensor_dataset(
        str(cache_file), changed_holdout_key) is None


def test_manifest_keyed_tensor_cache_rejects_malformed_cached_payload(
    tmp_path: Path, monkeypatch
) -> None:
    """A cache hit must fail closed on malformed tensors or balance metadata."""
    from dama.ai.ml import dataset as dataset_module
    from dama.ai.ml.dataset import load_matching_cached_tensor_dataset

    cache_file = tmp_path / "snapshot-cache.pt"
    cache_file.touch()
    expected_key = {
        "cache_version": 3,
        "snapshot_train_cache_version": 2,
        "snapshot_fingerprint": "a" * 64,
        "external_validation_keys_sha256": "b" * 64,
        "validation_exclusion_keys_sha256": "c" * 64,
        "max_train_entries": 100,
        "max_moves_per_sample": 8,
        "encoding_version": trainer_module.ENCODING_VERSION,
        "policy_stage": "policy_only",
        "side_weight_balance_version": 1,
    }
    cached_dataset = trainer_module.CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    cached_dataset.metadata = {
        **expected_key,
        "entry_count": len(cached_dataset),
        "side_weight_balance": None,
    }

    monkeypatch.setattr(
        dataset_module.CachedTensorDataset,
        "load",
        lambda _path, **_kwargs: cached_dataset,
    )
    assert load_matching_cached_tensor_dataset(str(cache_file), expected_key)
    valid_targets = cached_dataset.targets
    cached_dataset.targets = cached_dataset.targets[:-1]
    assert load_matching_cached_tensor_dataset(str(cache_file), expected_key) is None

    cached_dataset.targets = valid_targets
    valid_boards = cached_dataset.boards
    cached_dataset.boards = valid_boards[:, :-1]
    assert load_matching_cached_tensor_dataset(str(cache_file), expected_key) is None

    cached_dataset.boards = valid_boards
    cached_dataset.metadata["side_weight_balance"] = {"p1_count": 1}
    assert load_matching_cached_tensor_dataset(str(cache_file), expected_key) is None

    cached_dataset.metadata["side_weight_balance"] = {
        "p1_count": 1,
        "p2_count": 2,
        "p1_weight_before": "1.0",
        "p2_weight_before": 2.0,
        "p1_weight_after": 1.5,
        "p2_weight_after": 1.5,
    }
    assert load_matching_cached_tensor_dataset(str(cache_file), expected_key) is None

    empty_dataset = trainer_module.CachedTensorDataset.from_entries(
        [], max_moves_per_sample=8, show_progress=False)
    empty_dataset.metadata = {
        **expected_key,
        "entry_count": 0,
        "side_weight_balance": None,
    }
    monkeypatch.setattr(
        dataset_module.CachedTensorDataset,
        "load",
        lambda _path, **_kwargs: empty_dataset,
    )
    assert load_matching_cached_tensor_dataset(str(cache_file), expected_key) is None


def test_manifest_keyed_tensor_cache_rejects_missing_loss_tensors(
    tmp_path: Path,
) -> None:
    """A v3 cache cannot silently replace missing loss tensors with defaults."""
    import torch

    from dama.ai.ml.dataset import load_matching_cached_tensor_dataset

    cache_file = tmp_path / "snapshot-cache.pt"
    expected_key = {
        "cache_version": 3,
        "snapshot_train_cache_version": 2,
        "snapshot_fingerprint": "a" * 64,
        "external_validation_keys_sha256": "b" * 64,
        "validation_exclusion_keys_sha256": "c" * 64,
        "max_train_entries": 100,
        "max_moves_per_sample": 8,
        "encoding_version": trainer_module.ENCODING_VERSION,
        "policy_stage": "policy_only",
        "side_weight_balance_version": 1,
    }
    dataset = trainer_module.CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    torch.save({
        "boards": dataset.boards,
        "move_features": dataset.move_features,
        "move_counts": dataset.move_counts,
        "targets": dataset.targets,
        # Omit reward_weights and value_targets deliberately.
        "metadata": {
            **expected_key,
            "entry_count": len(dataset),
            "side_weight_balance": None,
        },
    }, cache_file)

    assert load_matching_cached_tensor_dataset(str(cache_file), expected_key) is None


def test_prepare_training_split_uses_verified_manifest_cache_before_train_parse(
    tmp_path: Path, monkeypatch
) -> None:
    """A matching cache bypasses only train-entry materialization."""
    cached_dataset = trainer_module.CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    cached_dataset.metadata = {
        "side_weight_balance": None,
    }
    manifest_path = tmp_path / "snapshot_v000001" / "manifest.json"
    context = SimpleNamespace(
        manifest={"fingerprint": "d" * 64},
        validation_keys={"held-out-state"},
    )

    class FakeSnapshotManager:
        external_validation_state_keys = {"e" * 64}

        def __init__(self) -> None:
            self.train_loaded = False

        def consider_snapshot(self, **_kwargs):
            return SnapshotDecision(True, "admitted", manifest_path, {})

        def snapshot_matches_settings(self, *_args, **_kwargs):
            return True

        def prepare_split(self, path, max_train_entries=0):
            assert path == manifest_path
            assert max_train_entries == 100
            return context

        def load_validation_entries(self, received):
            assert received is context
            return ["validation"]

        def load_train_entries(self, _received):
            self.train_loaded = True
            raise AssertionError("matching cache must skip train replay parsing")

    manager = FakeSnapshotManager()
    holder = object.__new__(Trainer)
    holder._snapshot_manager = manager
    holder.config = SimpleNamespace(
        selfplay_games=72,
        replay_max_entries=100,
        policy_stage="policy_only",
        ram_cache_enabled=True,
        ram_cache_file=str(tmp_path / "cache.pt"),
        ram_cache_threshold_gb=16.0,
        ram_cache_compress=True,
        max_moves_per_sample=8,
    )
    holder._stopped = False
    holder.step = 7
    holder._prelaunch_free_ram_gb = 20.0
    holder._corpus_settings = lambda *_, **__: ({}, {}, {})
    holder._activate_dataset_manifest = lambda _manifest: None
    holder._service_control_queue = lambda: None
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)

    observed = {}
    def load_cache(path, metadata, *, migrate_to_compressed=False):
        observed["path"] = path
        observed["metadata"] = metadata
        observed["migrate_to_compressed"] = migrate_to_compressed
        return cached_dataset

    monkeypatch.setattr(
        trainer_module, "load_matching_cached_tensor_dataset", load_cache)

    train, validation = Trainer._prepare_training_split(holder)

    assert train == []
    assert validation == ["validation"]
    assert manager.train_loaded is False
    assert holder._preloaded_snapshot_dataset is cached_dataset
    assert observed["path"] == str(tmp_path / "cache.pt")
    assert observed["metadata"]["snapshot_fingerprint"] == "d" * 64
    assert observed["migrate_to_compressed"] is True


def test_validation_tensor_cache_key_tracks_verified_sources() -> None:
    """A held-out cache must miss when any leakage-relevant source changes."""
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        policy_stage="policy_only",
        validation_tensor_cache_file="validation-cache.pt",
        max_moves_per_sample=8,
    )
    holder._snapshot_manager = SimpleNamespace(
        trained_ledger_source_sha256=lambda: "a" * 64)
    context = SimpleNamespace(
        validation_manifest={"files": [{"sha256": "source-a"}]},
        validation_keys={"held-out-state"},
    )

    baseline = Trainer._validation_tensor_cache_metadata(holder, context)
    assert baseline is not None

    context.validation_keys.add("frozen-suite-state")
    changed_exclusions = Trainer._validation_tensor_cache_metadata(holder, context)
    assert changed_exclusions is not None
    assert (
        changed_exclusions["validation_exclusion_keys_sha256"]
        != baseline["validation_exclusion_keys_sha256"]
    )

    context.validation_manifest["files"][0]["sha256"] = "source-b"
    changed_manifest = Trainer._validation_tensor_cache_metadata(holder, context)
    assert changed_manifest is not None
    assert (
        changed_manifest["validation_manifest_sha256"]
        != changed_exclusions["validation_manifest_sha256"]
    )

    holder._snapshot_manager = SimpleNamespace(
        trained_ledger_source_sha256=lambda: "b" * 64)
    changed_ledger = Trainer._validation_tensor_cache_metadata(holder, context)
    assert changed_ledger is not None
    assert (
        changed_ledger["trained_ledger_source_sha256"]
        != changed_manifest["trained_ledger_source_sha256"]
    )


def test_verified_validation_tensor_cache_bypasses_replay_materialization(
    tmp_path: Path, monkeypatch
) -> None:
    """A matching cache may skip only the already-verified held-out parse."""
    cached_dataset = trainer_module.CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    manifest_path = tmp_path / "snapshot_v000001" / "manifest.json"
    context = SimpleNamespace(
        manifest={"fingerprint": "d" * 64},
        validation_manifest={"files": [{"sha256": "validation-source"}]},
        validation_keys={"held-out-state"},
    )

    class FakeSnapshotManager:
        external_validation_state_keys = set()

        def consider_snapshot(self, **_kwargs):
            return SnapshotDecision(True, "admitted", manifest_path, {})

        def snapshot_matches_settings(self, *_args, **_kwargs):
            return True

        def prepare_split(self, path, max_train_entries=0):
            assert path == manifest_path
            assert max_train_entries == 100
            return context

        def trained_ledger_source_sha256(self):
            return "e" * 64

        def load_validation_entries(self, _received):
            raise AssertionError("matching cache must skip validation replay parsing")

        def load_train_entries(self, _received):
            return ["train"]

    holder = object.__new__(Trainer)
    holder._snapshot_manager = FakeSnapshotManager()
    holder.config = SimpleNamespace(
        selfplay_games=72,
        replay_max_entries=100,
        policy_stage="policy_only",
        ram_cache_enabled=False,
        ram_cache_file=None,
        ram_cache_threshold_gb=16.0,
        ram_cache_compress=True,
        validation_tensor_cache_file=str(tmp_path / "validation-cache.pt"),
        max_moves_per_sample=8,
    )
    holder._stopped = False
    holder.step = 7
    holder._corpus_settings = lambda *_, **__: ({}, {}, {})
    holder._activate_dataset_manifest = lambda _manifest: None
    holder._service_control_queue = lambda: None
    holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)

    expected = Trainer._validation_tensor_cache_metadata(holder, context)
    assert expected is not None
    cached_dataset.metadata = {
        **expected,
        "entry_count": len(cached_dataset),
        "validation_leakage": {
            "ledger_enabled": True,
            "all_time_trained_state_count": 9,
            "removed_validation_entry_count": 2,
            "removed_validation_state_count": 1,
            "retained_validation_entry_count": len(cached_dataset),
        },
    }
    observed = {}

    def load_cache(path, metadata, *, migrate_to_compressed=False):
        observed["path"] = path
        observed["metadata"] = metadata
        observed["migrate_to_compressed"] = migrate_to_compressed
        return cached_dataset

    monkeypatch.setattr(
        trainer_module, "load_matching_cached_tensor_dataset", load_cache)

    train, validation = Trainer._prepare_training_split(holder)

    assert train == ["train"]
    assert validation == []
    assert holder._preloaded_validation_dataset is cached_dataset
    assert holder._preloaded_validation_cache_metadata is None
    assert context.manifest["validation_leakage"]["retained_validation_entry_count"] == len(cached_dataset)
    assert observed["path"] == str(tmp_path / "validation-cache.pt")
    assert observed["metadata"] == expected
    assert observed["migrate_to_compressed"] is False


def test_validation_tensor_cache_rejects_malformed_leakage_accounting(
    tmp_path: Path, monkeypatch
) -> None:
    """A cache cannot publish invented held-out leakage statistics."""
    cached_dataset = trainer_module.CachedTensorDataset.from_entries(
        _tiny_replay_entries(), max_moves_per_sample=8, show_progress=False)
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        validation_tensor_cache_file=str(tmp_path / "validation-cache.pt"),
        validation_tensor_cache_compress=False,
    )
    expected = {
        "cache_version": 3,
        "validation_tensor_cache_version": 1,
        "validation_manifest_sha256": "a" * 64,
        "validation_exclusion_keys_sha256": "b" * 64,
        "trained_ledger_source_sha256": "c" * 64,
        "max_moves_per_sample": 8,
        "encoding_version": trainer_module.ENCODING_VERSION,
        "policy_stage": "policy_only",
    }
    cached_dataset.metadata = {
        **expected,
        "entry_count": len(cached_dataset),
        "validation_leakage": {
            "ledger_enabled": True,
            "all_time_trained_state_count": 9,
            "removed_validation_entry_count": 1,
            "removed_validation_state_count": 2,
            "retained_validation_entry_count": len(cached_dataset),
        },
    }
    monkeypatch.setattr(
        trainer_module,
        "load_matching_cached_tensor_dataset",
        lambda *_args, **_kwargs: cached_dataset,
    )

    assert Trainer._load_matching_validation_tensor_cache(holder, expected) is None


def test_validation_tensor_cache_persists_verified_leakage_metadata(
    tmp_path: Path,
) -> None:
    """The saved held-out cache carries its source key and leakage accounting."""
    from dama.ai.ml.dataset import load_matching_cached_tensor_dataset

    holder = object.__new__(Trainer)
    cache_path = tmp_path / "validation-cache.pt"
    expected = {
        "cache_version": 3,
        "validation_tensor_cache_version": 1,
        "validation_manifest_sha256": "a" * 64,
        "validation_exclusion_keys_sha256": "b" * 64,
        "trained_ledger_source_sha256": "c" * 64,
        "max_moves_per_sample": 8,
        "encoding_version": trainer_module.ENCODING_VERSION,
        "policy_stage": "policy_only",
    }
    holder.config = SimpleNamespace(
        policy_stage="policy_only",
        max_moves_per_sample=8,
        validation_tensor_cache_file=str(cache_path),
        ram_cache_compress=True,
    )
    holder._active_snapshot_manifest = {
        "validation_leakage": {
            "ledger_enabled": True,
            "all_time_trained_state_count": 9,
            "removed_validation_entry_count": 2,
            "removed_validation_state_count": 1,
            "retained_validation_entry_count": len(_tiny_replay_entries()),
        }
    }

    Trainer._set_validation_entries(
        holder, _tiny_replay_entries(), cache_metadata=expected)

    cached = load_matching_cached_tensor_dataset(
        str(cache_path), expected, migrate_to_compressed=False)
    assert cached is not None
    assert cached.metadata["validation_leakage"] == holder._active_snapshot_manifest[
        "validation_leakage"]


@pytest.mark.parametrize("stage", ["policy_only", "enhanced"])
@pytest.mark.parametrize("background_refresh", [False, True])
def test_validation_tensorization_releases_parsed_rows(
    stage: str, background_refresh: bool,
) -> None:
    """Validation needs tensors after publication, including snapshot refreshes."""
    import weakref

    import torch

    from dama.ai.ml.replay import ReplayEntry
    from dama.game_state import GameState

    state = GameState.initial()
    entries = [
        ReplayEntry(
            state=state.to_compact(),
            legal_moves=[move.to_dict() for move in state.legal_moves()],
            chosen_index=index, result=1, score=2.5,
        )
        for index in range(3)
    ]
    refs = [weakref.ref(entry) for entry in entries]
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        policy_stage=stage, max_moves_per_sample=32, batch_size=2,
        teacher_score_depth=1, teacher_soft_temperature=1.0,
        teacher_value_scale=1000.0, teacher_hard_label_blend=0.25,
    )
    if stage == "enhanced":
        expected = trainer_module.create_enhanced_dataloader(
            entries, batch_size=2, max_moves_per_sample=32,
            teacher_depth=1, temperature=1.0, value_scale=1000.0,
            hard_label_blend=0.25, shuffle=False, show_progress=False,
        ).dataset
    else:
        expected = trainer_module.CachedTensorDataset.from_entries(
            entries, max_moves_per_sample=32, show_progress=False)

    if background_refresh:
        training_dataset = object()
        manifest = {"version": 2}
        activated = []
        holder._activate_dataset_manifest = activated.append
        holder._bg_selfplay_lock = trainer_module.threading.Lock()
        holder._bg_selfplay_dataset = training_dataset
        holder._bg_selfplay_incremental = None
        holder._bg_snapshot_manifest = manifest
        holder._bg_validation_entries = entries
        assert holder._collect_background_selfplay() == (training_dataset, None)
        assert activated == [manifest]
        assert holder._bg_validation_entries is None
    else:
        holder._set_validation_entries(entries)

    # Do not clear the caller's rows in place; it still owns them until release.
    assert len(entries) == 3
    del entries
    assert all(ref() is None for ref in refs)
    assert holder._validation_entries == []
    assert len(holder._validation_dataloader) == 3
    fields = [
        "boards", "move_features", "move_counts", "targets",
        "reward_weights", "value_targets",
    ]
    if stage == "enhanced":
        fields.append("teacher_probabilities")
    for field in fields:
        assert torch.equal(
            getattr(holder._validation_dataloader, field), getattr(expected, field))


def test_snapshot_tensor_cache_requires_a_prelaunch_ram_measurement() -> None:
    """A failed RAM probe must decline the warm cache just like the cold path."""
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        policy_stage="policy_only",
        ram_cache_enabled=True,
        ram_cache_file="cache.pt",
        ram_cache_threshold_gb=16.0,
        replay_max_entries=100,
        max_moves_per_sample=8,
    )
    holder._snapshot_manager = SimpleNamespace(
        external_validation_state_keys=set())

    assert Trainer._snapshot_train_cache_metadata(
        holder, {"fingerprint": "a" * 64}, {"held-out-state"}) is None

    holder._prelaunch_free_ram_gb = 16.1
    first_key = Trainer._snapshot_train_cache_metadata(
        holder, {"fingerprint": "a" * 64}, {"held-out-state"})
    changed_holdout_key = Trainer._snapshot_train_cache_metadata(
        holder, {"fingerprint": "a" * 64},
        {"held-out-state", "newly-held-out-state"})
    assert first_key is not None
    assert changed_holdout_key is not None
    assert (
        first_key["validation_exclusion_keys_sha256"]
        != changed_holdout_key["validation_exclusion_keys_sha256"]
    )


def test_free_entry_lists_after_tensorize_clears_validation_mirror() -> None:
    """Pass 117: the release helper drops lists and the validation mirror."""
    holder = object.__new__(Trainer)
    holder._validation_entries = [object(), object()]

    first, second = Trainer._free_entry_lists_after_tensorize(
        holder, ["a"], ["b"])

    assert first is None and second is None
    assert holder._validation_entries == []
