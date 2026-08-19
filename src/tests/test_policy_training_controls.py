import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import dama.ai.ml.trainer as trainer_module
from dama.ai.ml.corpus import SnapshotDecision, canonical_state_key
from dama.ai.ml.trainer import (
    Trainer,
    TrainingConfig,
    TrainingStats,
    activate_enhanced_stage,
    config_from_yaml,
    load_config_from_yaml,
    validate_recovery_experiment_config,
)


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


def test_checkpoint_save_never_overwrites_existing_step(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    baseline = checkpoint_dir / "model_step_134000.pt"
    baseline.write_bytes(b"preserved baseline")
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(checkpoint_dir=str(checkpoint_dir))
    holder.step = 134000
    holder._checkpoint_thread = None

    saved = Trainer._save_checkpoint(holder, 1.0)

    assert Path(saved) == baseline
    assert baseline.read_bytes() == b"preserved baseline"


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
    holder._corpus_settings = lambda: (
        {"stage": "enhanced"},
        {"played_action_probability": 0.10},
        {"current_model_inference_depth": 2},
    )
    selfplay_calls = []
    holder.run_selfplay = lambda games: (
        selfplay_calls.append(games), setattr(manager, "progress", True)
    )[0]
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


def test_recovery_gate_rejects_unready_data_controls(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_step_134000.pt"
    checkpoint.write_bytes(b"immutable baseline")
    config = _valid_recovery_config(checkpoint)
    config.snapshot_min_fresh_fraction = 0.49
    config.trajectory_model_fraction = 0.0

    with pytest.raises(ValueError, match="50% fresh"):
        validate_recovery_experiment_config(config)


def test_enhanced_stage_resumes_only_from_recorded_policy_promotion(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "model_step_134000.pt"
    baseline.write_bytes(b"immutable baseline")
    promoted = tmp_path / "model_step_140000.pt"
    promoted.write_bytes(b"promoted policy")
    promoted_sha256 = hashlib.sha256(promoted.read_bytes()).hexdigest()
    suite = tmp_path / "frozen.jsonl"
    suite.write_bytes(b"frozen suite")
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    suite.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
        "suite_sha256": suite_sha256,
    }), encoding="utf-8")
    registry = tmp_path / "promotions.jsonl"
    registry.write_text(json.dumps({
        "promoted": True,
        "training_stage": "policy_only",
        "teacher_agreement": 0.55,
        "checkpoint_path": str(promoted),
        "checkpoint_sha256": promoted_sha256,
        "suite_fingerprint": suite_sha256,
    }) + "\n", encoding="utf-8")

    config = _valid_recovery_config(baseline)
    config.resume = str(promoted)
    config.promotion_registry = str(registry)
    config.frozen_suite_path = str(suite)
    activate_enhanced_stage(config, inference_depth=3)

    assert config.teacher_target_type == "distribution"
    assert config.value_head_enabled is True
    assert config.pipeline_mode == "alternate"
    assert config.inference_depth == 3

    validate_recovery_experiment_config(config)

    unpromoted = tmp_path / "model_step_142000.pt"
    unpromoted.write_bytes(b"unpromoted")
    config.resume = str(unpromoted)
    with pytest.raises(ValueError, match="recorded promoted policy-only"):
        validate_recovery_experiment_config(config)


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
    path = root / "config" / "training_config_policy_distillation.yaml"
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
