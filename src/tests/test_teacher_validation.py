import hashlib
import json
import os
from pathlib import Path
import stat

import pytest
import torch

import dama.ai.ml.teacher_validation as teacher_validation_module
from dama.ai.ml.replay import ReplayEntry
from dama.ai.ml.teacher_validation import (
    PromotionRegistry,
    evaluate_teacher_agreement,
    load_frozen_teacher_suite,
)


def _entry(chosen_index: int = 0, forced: bool = False) -> ReplayEntry:
    moves = [{"path": [[2, 1], [3, 0]], "captures": [], "promotion": False}]
    if not forced:
        moves.append({"path": [[2, 1], [3, 2]], "captures": [], "promotion": False})
    return ReplayEntry(
        state={
            "p1_men": [[2, 1]],
            "p1_kings": [],
            "p2_men": [[5, 0]],
            "p2_kings": [],
            "turn": 1,
            "move_count": 0,
        },
        legal_moves=moves,
        chosen_index=chosen_index,
        result=0,
    )


class _FirstMoveModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward_padded(self, boards, move_features, move_counts):
        scores = torch.zeros(
            boards.shape[0], move_features.shape[1], device=boards.device
        )
        scores[:, 0] = 1.0 + self.anchor
        invalid = torch.arange(move_features.shape[1], device=boards.device).unsqueeze(0)
        scores = scores.masked_fill(invalid >= move_counts.unsqueeze(1), float("-inf"))
        return scores


def test_teacher_agreement_reports_forced_fraction() -> None:
    entries = [_entry(0), _entry(1), _entry(0, forced=True)]
    result = evaluate_teacher_agreement(
        _FirstMoveModel(), entries, max_moves_per_sample=4, batch_size=2
    )

    assert result["correct_states"] == 2
    assert result["top1_teacher_agreement"] == 2 / 3
    assert result["decision_top1_teacher_agreement"] == 0.5
    assert result["forced_move_fraction"] == pytest.approx(1 / 3)


def test_frozen_suite_loader_checks_hash_count_and_uniqueness(tmp_path: Path) -> None:
    suite = tmp_path / "suite.jsonl"
    entry = _entry().to_dict()
    second = _entry().to_dict()
    second["state"] = dict(second["state"], p1_kings=[[4, 3]], p1_men=[])
    payload = json.dumps(entry) + "\n" + json.dumps(second) + "\n"
    suite.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(suite.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "state_count": 2,
        "suite_sha256": digest,
        "seed": 1,
        "teacher_difficulty": "hard",
        "opening_plies": [0, 2],
    }
    suite.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    entries, loaded = load_frozen_teacher_suite(str(suite), expected_count=2)
    assert len(entries) == 2
    assert loaded["suite_sha256"] == digest


def test_promotion_uses_agreement_not_training_loss(tmp_path: Path) -> None:
    registry = PromotionRegistry(str(tmp_path / "promotions.jsonl"), 0.50)
    below = registry.consider("step_1.pt", 1, 0.49, "suite", "data")
    first = registry.consider("step_2.pt", 2, 0.51, "suite", "data")
    worse = registry.consider("step_3.pt", 3, 0.505, "suite", "data")
    better = registry.consider("step_4.pt", 4, 0.55, "suite", "data")

    assert not below.promoted
    assert first.promoted
    assert not worse.promoted
    assert better.promoted
    assert "loss" not in better.record


def test_promotion_can_be_persisted_only_after_checkpoint_write(tmp_path: Path) -> None:
    path = tmp_path / "promotions.jsonl"
    registry = PromotionRegistry(str(path), 0.50)
    decision = registry.consider(
        "step_2.pt", 2, 0.51, "suite", "data", persist=False
    )

    assert decision.promoted
    assert not path.exists()

    registry.persist(decision)
    saved = json.loads(path.read_text(encoding="utf-8").strip())
    assert saved["checkpoint_path"] == "step_2.pt"
    assert saved["promoted"] is True


def test_promotion_registry_fsyncs_file_and_directory_around_replace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "promotions.jsonl"
    registry = PromotionRegistry(str(path), 0.50)
    decision = registry.consider(
        "step_2.pt", 2, 0.51, "suite", "data", persist=False
    )
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd):
        mode = os.fstat(fd).st_mode
        events.append(
            "directory_fsync" if stat.S_ISDIR(mode) else "file_fsync"
        )
        return real_fsync(fd)

    def tracking_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(teacher_validation_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(teacher_validation_module.os, "replace", tracking_replace)

    registry.persist(decision)

    assert events == ["file_fsync", "replace", "directory_fsync"]
    assert json.loads(path.read_text(encoding="utf-8"))["step"] == 2
    assert list(tmp_path.glob("promotions.jsonl.*.tmp")) == []


def test_promotion_registry_reports_directory_fsync_failure_without_residue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "promotions.jsonl"
    registry = PromotionRegistry(str(path), 0.50)
    decision = registry.consider(
        "step_2.pt", 2, 0.51, "suite", "data", persist=False
    )
    real_fsync = os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(5, "simulated promotion directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(
        teacher_validation_module.os, "fsync", fail_directory_fsync
    )

    with pytest.raises(OSError, match="promotion directory fsync failure"):
        registry.persist(decision)

    assert json.loads(path.read_text(encoding="utf-8"))["step"] == 2
    assert list(tmp_path.glob("promotions.jsonl.*.tmp")) == []


def test_frozen_suite_commits_bytes_before_atomic_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = tmp_path / "frozen.jsonl"
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd):
        mode = os.fstat(fd).st_mode
        events.append(
            "directory_fsync" if stat.S_ISDIR(mode) else "file_fsync"
        )
        return real_fsync(fd)

    def tracking_replace(source, destination):
        events.append(f"replace:{Path(destination).name}")
        return real_replace(source, destination)

    monkeypatch.setattr(teacher_validation_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(teacher_validation_module.os, "replace", tracking_replace)
    monkeypatch.setattr(
        teacher_validation_module,
        "get_best_move",
        lambda state, *_args, **_kwargs: state.legal_moves()[0],
    )

    manifest = teacher_validation_module.create_frozen_teacher_suite(
        str(suite), target_states=1, opening_plies=(0,), max_games=1
    )

    manifest_path = suite.with_suffix(suite.suffix + ".manifest.json")
    assert events == [
        "file_fsync",
        "replace:frozen.jsonl",
        "directory_fsync",
        "file_fsync",
        "replace:frozen.jsonl.manifest.json",
        "directory_fsync",
    ]
    assert json.loads(suite.read_text(encoding="utf-8"))["chosen_index"] == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert hashlib.sha256(suite.read_bytes()).hexdigest() == manifest["suite_sha256"]
    assert not list(tmp_path.glob("*.tmp"))


def test_frozen_suite_directory_commit_failure_stops_before_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = tmp_path / "frozen.jsonl"
    monkeypatch.setattr(
        teacher_validation_module,
        "get_best_move",
        lambda state, *_args, **_kwargs: state.legal_moves()[0],
    )
    monkeypatch.setattr(
        teacher_validation_module,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(
            OSError(5, "simulated suite directory fsync failure")
        ),
    )

    with pytest.raises(OSError, match="suite directory fsync failure"):
        teacher_validation_module.create_frozen_teacher_suite(
            str(suite), target_states=1, opening_plies=(0,), max_games=1
        )

    assert len(suite.read_text(encoding="utf-8").splitlines()) == 1
    assert not suite.with_suffix(suite.suffix + ".manifest.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_frozen_suite_manifest_failure_never_exposes_partial_public_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dama.ai.ml import run_status

    suite = tmp_path / "frozen.jsonl"
    monkeypatch.setattr(
        teacher_validation_module,
        "get_best_move",
        lambda state, *_args, **_kwargs: state.legal_moves()[0],
    )

    def fail_after_partial_manifest(_payload, handle, **_kwargs):
        handle.write('{"partial":')
        handle.flush()
        os.fsync(handle.fileno())
        raise OSError(28, "simulated frozen manifest disk full")

    monkeypatch.setattr(run_status.json, "dump", fail_after_partial_manifest)

    with pytest.raises(OSError, match="frozen manifest disk full"):
        teacher_validation_module.create_frozen_teacher_suite(
            str(suite), target_states=1, opening_plies=(0,), max_games=1
        )

    assert len(suite.read_text(encoding="utf-8").splitlines()) == 1
    assert not suite.with_suffix(suite.suffix + ".manifest.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_promotion_registry_write_failure_preserves_history_and_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "promotions.jsonl"
    registry = PromotionRegistry(str(path), 0.50)
    registry.consider("step_1.pt", 1, 0.51, "suite", "data")
    prior = path.read_bytes()
    failed = registry.consider(
        "step_2.pt", 2, 0.52, "suite", "data", persist=False)
    retry = registry.consider(
        "step_3.pt", 3, 0.53, "suite", "data", persist=False)

    original_fdopen = teacher_validation_module.os.fdopen

    class _FailingWriter:
        def __init__(self, handle) -> None:
            self._handle = handle
            self._writes = 0

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            return self._handle.__exit__(exc_type, exc, traceback)

        def write(self, payload):
            self._writes += 1
            if self._writes == 2:
                self._handle.write(payload[:19])
                self._handle.flush()
                os.fsync(self._handle.fileno())
                raise OSError(28, "simulated promotion registry disk full")
            return self._handle.write(payload)

        def flush(self):
            return self._handle.flush()

        def fileno(self):
            return self._handle.fileno()

    monkeypatch.setattr(
        teacher_validation_module.os,
        "fdopen",
        lambda fd, *args, **kwargs: _FailingWriter(
            original_fdopen(fd, *args, **kwargs)),
    )
    with pytest.raises(OSError, match="simulated promotion registry disk full"):
        registry.persist(failed)

    assert path.read_bytes() == prior
    assert list(tmp_path.glob("promotions.jsonl.*.tmp")) == []

    monkeypatch.setattr(teacher_validation_module.os, "fdopen", original_fdopen)
    registry.persist(retry)

    records = registry.records()
    assert [record["step"] for record in records] == [1, 3]


@pytest.mark.parametrize(
    "teacher_agreement",
    (True, "0.99", None, [], {}, float("nan"), float("inf"), -0.01, 1.01),
)
def test_malformed_current_agreement_cannot_create_promotion(
    tmp_path: Path,
    teacher_agreement,
) -> None:
    path = tmp_path / "promotions.jsonl"
    registry = PromotionRegistry(str(path), 0.50)

    with pytest.raises(ValueError, match="Promotion teacher agreement"):
        registry.consider(
            "invalid.pt", 1, teacher_agreement, "suite", "data")

    assert not path.exists()


@pytest.mark.parametrize(
    ("checkpoint_path", "step"),
    (
        (None, 2),
        ("", 2),
        (7, 2),
        ("valid.pt", True),
        ("valid.pt", "2"),
        ("valid.pt", -1),
    ),
)
def test_malformed_current_checkpoint_identity_cannot_create_promotion(
    tmp_path: Path,
    checkpoint_path,
    step,
) -> None:
    path = tmp_path / "promotions.jsonl"
    registry = PromotionRegistry(str(path), 0.50)

    with pytest.raises(ValueError, match="Promotion (checkpoint_path|step)"):
        registry.consider(
            checkpoint_path, step, 0.55, "suite", "data")

    assert not path.exists()


@pytest.mark.parametrize(
    "training_stage",
    (None, True, 7, [], {}, "invalid"),
)
def test_malformed_current_training_stage_cannot_create_promotion(
    tmp_path: Path,
    training_stage,
) -> None:
    path = tmp_path / "promotions.jsonl"
    registry = PromotionRegistry(str(path), 0.50)

    with pytest.raises(ValueError, match="Promotion training_stage"):
        registry.consider(
            "invalid.pt",
            1,
            0.55,
            "suite",
            "data",
            training_stage=training_stage,
        )

    assert not path.exists()


@pytest.mark.parametrize(
    "agreement_threshold",
    (True, "0.50", None, [], {}, float("nan"), float("inf"), -0.01, 1.01),
)
def test_malformed_current_threshold_cannot_create_registry(
    tmp_path: Path,
    agreement_threshold,
) -> None:
    path = tmp_path / "promotions.jsonl"

    with pytest.raises(ValueError, match="Promotion agreement threshold"):
        PromotionRegistry(str(path), agreement_threshold)

    assert not path.exists()


@pytest.mark.parametrize(
    "threshold_update",
    (
        {"teacher_agreement_threshold": 0.40},
        {"teacher_agreement_threshold": True},
        {"teacher_agreement_threshold": "0.50"},
        {"teacher_agreement_threshold": None},
        {},
    ),
)
def test_incomparable_registry_threshold_cannot_block_valid_promotion(
    tmp_path: Path,
    threshold_update,
) -> None:
    path = tmp_path / "promotions.jsonl"
    record = {
        "checkpoint_path": "malformed.pt",
        "step": 1,
        "teacher_agreement": 0.99,
        "suite_fingerprint": "suite",
        "dataset_fingerprint": "data",
        "training_stage": "policy_only",
        "comparison_context": {},
        "promoted": True,
    }
    record.update(threshold_update)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    decision = PromotionRegistry(str(path), 0.50).consider(
        "valid.pt", 2, 0.55, "suite", "data", persist=False)

    assert decision.promoted is True


@pytest.mark.parametrize(
    "training_stage",
    ("enhanced", None, True, 7, [], {}, "invalid"),
)
def test_incomparable_registry_training_stage_cannot_block_valid_promotion(
    tmp_path: Path,
    training_stage,
) -> None:
    path = tmp_path / "promotions.jsonl"
    path.write_text(json.dumps({
        "checkpoint_path": "old.pt",
        "step": 1,
        "teacher_agreement": 0.99,
        "teacher_agreement_threshold": 0.50,
        "suite_fingerprint": "suite",
        "dataset_fingerprint": "data",
        "training_stage": training_stage,
        "comparison_context": {},
        "promoted": True,
    }) + "\n", encoding="utf-8")

    decision = PromotionRegistry(str(path), 0.50).consider(
        "valid.pt",
        2,
        0.55,
        "suite",
        "data",
        training_stage="policy_only",
        persist=False,
    )

    assert decision.promoted is True


@pytest.mark.parametrize("promoted", (1, "false"))
def test_non_boolean_registry_decision_cannot_block_valid_promotion(
    tmp_path: Path,
    promoted,
) -> None:
    path = tmp_path / "promotions.jsonl"
    path.write_text(json.dumps({
        "checkpoint_path": "malformed.pt",
        "step": 1,
        "teacher_agreement": 0.99,
        "suite_fingerprint": "suite",
        "dataset_fingerprint": "data",
        "training_stage": "policy_only",
        "comparison_context": {},
        "promoted": promoted,
    }) + "\n", encoding="utf-8")

    decision = PromotionRegistry(str(path), 0.50).consider(
        "valid.pt", 2, 0.55, "suite", "data", persist=False)

    assert decision.promoted is True


@pytest.mark.parametrize(
    "teacher_agreement",
    (True, "0.99", None, [], {}, float("nan"), float("inf"), -0.01, 1.01),
)
def test_malformed_registry_agreement_cannot_block_valid_promotion(
    tmp_path: Path,
    teacher_agreement,
) -> None:
    path = tmp_path / "promotions.jsonl"
    path.write_text(json.dumps({
        "checkpoint_path": "malformed.pt",
        "step": 1,
        "teacher_agreement": teacher_agreement,
        "suite_fingerprint": "suite",
        "dataset_fingerprint": "data",
        "training_stage": "policy_only",
        "comparison_context": {},
        "promoted": True,
    }) + "\n", encoding="utf-8")

    decision = PromotionRegistry(str(path), 0.50).consider(
        "valid.pt", 2, 0.55, "suite", "data", persist=False)

    assert decision.promoted is True


@pytest.mark.parametrize(
    "identity_update",
    (
        {"checkpoint_path": None},
        {"checkpoint_path": ""},
        {"checkpoint_path": 7},
        {"step": True},
        {"step": "1"},
        {"step": -1},
    ),
)
def test_malformed_registry_checkpoint_identity_cannot_block_valid_promotion(
    tmp_path: Path,
    identity_update,
) -> None:
    path = tmp_path / "promotions.jsonl"
    record = {
        "checkpoint_path": "old.pt",
        "step": 1,
        "teacher_agreement": 0.99,
        "teacher_agreement_threshold": 0.50,
        "suite_fingerprint": "suite",
        "dataset_fingerprint": "data",
        "training_stage": "policy_only",
        "comparison_context": {},
        "promoted": True,
    }
    record.update(identity_update)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    decision = PromotionRegistry(str(path), 0.50).consider(
        "valid.pt", 2, 0.55, "suite", "data", persist=False)

    assert decision.promoted is True


@pytest.mark.parametrize(
    "record",
    ([], True, {"comparison_context": "invalid", "promoted": True}),
)
def test_malformed_registry_record_cannot_block_valid_promotion(
    tmp_path: Path,
    record,
) -> None:
    path = tmp_path / "promotions.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    decision = PromotionRegistry(str(path), 0.50).consider(
        "valid.pt", 2, 0.55, "suite", "data", persist=False)

    assert decision.promoted is True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG = "config/training_config_policy_distillation_c174k.yaml"


def _active_selfplay_and_validation() -> tuple[dict, dict]:
    import yaml

    raw = yaml.safe_load((PROJECT_ROOT / ACTIVE_CONFIG).read_text(encoding="utf-8"))
    return raw["selfplay"], raw["validation"]


def test_active_config_matches_the_frozen_suite_manifest():
    """Catch suite-contract drift at test time instead of first launch.

    ``Trainer._ensure_frozen_teacher_suite`` runs before any training and
    raises via ``create_frozen_teacher_suite`` when the active config's
    self-play settings disagree with the immutable suite manifest -- correct
    behaviour, but it costs a whole failed launch cycle to discover. This
    asserts the same equality here, where it fails in seconds. Mirrors the
    exact key set that creator validates (seed, teacher_difficulty,
    opening_plies, played_action_noise, max_moves_per_game, state_count).
    """

    suite_rel = "data/validation_policy_distillation/frozen_hard_5000.jsonl"
    suite_path = PROJECT_ROOT / suite_rel
    manifest_path = suite_path.with_suffix(suite_path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        pytest.skip(f"frozen teacher suite not present on this machine: {suite_rel}")

    selfplay_cfg, validation_cfg = _active_selfplay_and_validation()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected = {
        "seed": int(validation_cfg["frozen_suite_seed"]),
        "teacher_difficulty": str(selfplay_cfg["teacher_difficulty"]),
        "opening_plies": list(selfplay_cfg["opening_plies"]),
        "played_action_noise": float(selfplay_cfg["noise_prob"]),
        "max_moves_per_game": int(selfplay_cfg["max_moves_per_game"]),
        "state_count": int(validation_cfg["frozen_suite_size"]),
    }
    for key, value in expected.items():
        assert manifest.get(key) == value, (
            f"{ACTIVE_CONFIG} {key}={value!r} disagrees with the frozen suite "
            f"manifest {key}={manifest.get(key)!r}; the trainer would refuse "
            "to start (create_frozen_teacher_suite fails closed)"
        )
    # The configured path must be the suite the manifest belongs to, and the
    # suite bytes must still hash to the pinned fingerprint (immutability).
    assert str(validation_cfg["frozen_suite_path"]).replace("\\", "/") == suite_rel
    digest = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    assert manifest.get("suite_sha256") == digest


def test_current_snapshot_generation_matches_active_config():
    """The admitted corpus must share the active generation contract.

    A drift here does not wedge admissions (snapshot_matches_settings
    compares against each freshly written manifest), but it silently splits
    the lineage: new cycles admit under different opening/noise settings than
    every retained shard. Reading the CURRENT pointer also guards the
    missing-pointer corruption that consider_snapshot refuses to admit
    through (corpus.py fails closed when snapshots exist without one).
    """

    root = PROJECT_ROOT / "data/corpus_snapshots/policy_distillation_recovery_c174k"
    if not (root / "CURRENT").is_file():
        pytest.skip("c174k corpus snapshots not present on this machine")

    relative = (root / "CURRENT").read_text(encoding="utf-8").strip()
    assert relative, "CURRENT pointer exists but is empty"
    current_manifest = root / relative
    assert current_manifest.is_file(), (
        f"CURRENT points at {relative}, which does not resolve"
    )

    selfplay_cfg, _ = _active_selfplay_and_validation()
    manifest = json.loads(current_manifest.read_text(encoding="utf-8"))
    generation = manifest.get("generation_settings", {})
    noise = manifest.get("noise_settings", {})
    assert list(generation.get("opening_plies", [])) == list(
        selfplay_cfg["opening_plies"])
    assert float(noise.get("played_action_probability", -1)) == pytest.approx(
        float(selfplay_cfg["noise_prob"]))
