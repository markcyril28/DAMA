"""Mid-run reuse of unchanged held-out validation tensors.

The background producer re-parsed all held-out shards and the collector
re-tensorized the identical entry list on every admission, even though the
hold-out is immutable and its ledger-filtered entry set almost never changes
between admissions (measured 80.6 s parse + 1.1 s tensorize per admission on
the idle local host against snapshot v75).  Reuse is allowed only when the
verified validation manifest and the exact ledger-fingerprint subset that
filters its entries both match the identity recorded when the current tensors
were published; every missing proof takes the exact load path.
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import dama.ai.ml.corpus as corpus
from dama.ai.ml.corpus import (
    CorpusSnapshotManager,
    _merge_state_keys_file,
    _state_key_fingerprint,
    canonical_state_key,
)
import dama.ai.ml.trainer as trainer_module
from dama.ai.ml.trainer import Trainer, _VALIDATION_TENSORS_CURRENT


TEACHER = {"difficulty": "hard"}
NOISE = {"played_action_probability": 0.10, "label_is_teacher": True}
GENERATION = {"algorithm_fraction": 0.70, "model_fraction": 0.30}


def _state(index: int, turn: int = 1) -> dict:
    row = (index // 4) % 8
    col = (index * 2 + 1 - (row % 2)) % 8
    return {
        "p1_men": [[row, col]],
        "p1_kings": [],
        "p2_men": [[7 - row, 7 - col]],
        "p2_kings": [],
        "turn": turn,
        "move_count": index,
    }


def _entry(index: int) -> dict:
    return {
        "state": _state(index),
        "legal_moves": [
            {"path": [[0, 1], [1, 0]], "captures": [], "promotion": False},
            {"path": [[0, 1], [1, 2]], "captures": [], "promotion": False},
        ],
        "chosen_index": 0,
        "result": 0,
        "trajectory_source": "algorithm",
    }


def _write_replay(path: Path, indices) -> None:
    path.write_text(
        "".join(json.dumps(_entry(index), sort_keys=True) + "\n" for index in indices),
        encoding="utf-8",
    )


def _manager(replay_dir: Path, root: Path, **kwargs) -> CorpusSnapshotManager:
    options = {
        "validation_fraction": 0.25,
        "split_seed": 5,
        "min_fresh_fraction": 0.50,
        "grow_holdout": False,
        "trained_ledger_enabled": True,
    }
    options.update(kwargs)
    return CorpusSnapshotManager(str(replay_dir), str(root), **options)


def _admitted_manager(tmp_path: Path) -> CorpusSnapshotManager:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [index])
    manager = _manager(replay_dir, tmp_path / "root")
    assert manager.consider_snapshot(TEACHER, NOISE, GENERATION).admitted
    return manager


def _held_out_keys(manager: CorpusSnapshotManager) -> set:
    held = set()
    validation_manifest = json.loads(
        manager.validation_manifest_path.read_text(encoding="utf-8"))
    for record in validation_manifest["files"]:
        path = manager.validation_manifest_path.parent / record["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                held.add(canonical_state_key(json.loads(line)["state"]))
    assert held
    return held


def _holder(manager) -> Trainer:
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        policy_stage="policy_only", max_moves_per_sample=32)
    holder._snapshot_manager = manager
    holder._bg_selfplay_lock = threading.Lock()
    holder._validation_tensor_identity = None
    return holder


# ---------------------------------------------------------------------------
# Corpus: the parse-free leak subset must equal the parse-time filter
# ---------------------------------------------------------------------------

def test_leak_fingerprints_match_the_entries_load_validation_would_drop(
    tmp_path: Path,
) -> None:
    manager = _admitted_manager(tmp_path)
    context = manager.prepare_split(None, max_train_entries=0)
    assert manager.validation_leak_fingerprints(context) == frozenset()
    baseline = manager.load_validation_entries(context)
    assert baseline

    held = _held_out_keys(manager)
    _merge_state_keys_file(
        tmp_path / "root" / "ledger" / "trained_state_keys.txt.gz", held)
    reopened = _manager(tmp_path / "replay", tmp_path / "root")
    context2 = reopened.prepare_split(None, max_train_entries=0)
    leak = reopened.validation_leak_fingerprints(context2)
    assert leak == frozenset(_state_key_fingerprint(key) for key in held)
    assert reopened.load_validation_entries(context2) == []
    counts = context2.manifest["validation_leakage"]
    assert counts["removed_validation_state_count"] == len(held)
    assert counts["retained_validation_entry_count"] == 0


def test_context_records_stored_keys_without_external_exclusions(
    tmp_path: Path,
) -> None:
    manager = _admitted_manager(tmp_path)
    manager.set_external_validation_state_keys({"external-suite-key"})
    context = manager.prepare_split(None, max_train_entries=0)
    assert "external-suite-key" in context.validation_keys
    assert "external-suite-key" not in context.stored_validation_keys
    assert context.stored_validation_keys
    assert context.stored_validation_keys <= frozenset(context.validation_keys)


# ---------------------------------------------------------------------------
# Trainer: identity construction and the skip decision
# ---------------------------------------------------------------------------

def test_reuse_identity_changes_when_the_ledger_gains_a_held_state(
    tmp_path: Path,
) -> None:
    manager = _admitted_manager(tmp_path)
    holder = _holder(manager)
    context = manager.prepare_split(None, max_train_entries=0)
    first = holder._validation_reuse_identity(context)
    assert first is not None
    again = holder._validation_reuse_identity(
        manager.prepare_split(None, max_train_entries=0))
    assert again == first

    _merge_state_keys_file(
        tmp_path / "root" / "ledger" / "trained_state_keys.txt.gz",
        _held_out_keys(manager))
    reopened = _manager(tmp_path / "replay", tmp_path / "root")
    holder2 = _holder(reopened)
    changed = holder2._validation_reuse_identity(
        reopened.prepare_split(None, max_train_entries=0))
    assert changed is not None
    assert changed != first


def test_reuse_identity_requires_policy_stage_and_context_support(
    tmp_path: Path,
) -> None:
    manager = _admitted_manager(tmp_path)
    holder = _holder(manager)
    context = manager.prepare_split(None, max_train_entries=0)
    holder.config.policy_stage = "enhanced"
    assert holder._validation_reuse_identity(context) is None
    holder.config.policy_stage = "policy_only"
    legacy = SimpleNamespace(
        validation_manifest=context.validation_manifest,
        stored_validation_keys=frozenset(),
    )
    assert holder._validation_reuse_identity(legacy) is None


def test_load_or_reuse_skips_the_parse_only_with_a_proven_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _admitted_manager(tmp_path)
    holder = _holder(manager)
    parse_calls = []
    real_load = CorpusSnapshotManager.load_validation_entries

    def counting_load(self, context):
        parse_calls.append(context.manifest["version"])
        return real_load(self, context)

    monkeypatch.setattr(
        CorpusSnapshotManager, "load_validation_entries", counting_load)

    context = manager.prepare_split(None, max_train_entries=0)
    entries, identity = holder._load_or_reuse_validation_entries(context)
    assert entries and identity is not None
    assert parse_calls == [1]
    ledger_size = len(context.historically_trained)

    # Simulate the collector publishing tensors built from that hand-off.
    holder._validation_tensor_identity = {
        "identity": identity,
        "leakage": {
            key: context.manifest["validation_leakage"][key]
            for key in (
                "removed_validation_entry_count",
                "removed_validation_state_count",
                "retained_validation_entry_count",
            )
        },
    }
    context2 = manager.prepare_split(None, max_train_entries=0)
    reused, identity2 = holder._load_or_reuse_validation_entries(context2)
    assert reused is _VALIDATION_TENSORS_CURRENT
    assert identity2 == identity
    assert parse_calls == [1]
    rebuilt = context2.manifest["validation_leakage"]
    assert rebuilt["ledger_enabled"] is True
    assert rebuilt["all_time_trained_state_count"] == ledger_size
    assert rebuilt["retained_validation_entry_count"] == len(entries)

    # A ledger that gains a held state must invalidate the proof.
    _merge_state_keys_file(
        tmp_path / "root" / "ledger" / "trained_state_keys.txt.gz",
        _held_out_keys(manager))
    reopened = _manager(tmp_path / "replay", tmp_path / "root")
    holder._snapshot_manager = reopened
    context3 = reopened.prepare_split(None, max_train_entries=0)
    entries3, identity3 = holder._load_or_reuse_validation_entries(context3)
    assert entries3 == [] and entries3 is not _VALIDATION_TENSORS_CURRENT
    assert identity3 != identity
    assert parse_calls == [1, 1]


# ---------------------------------------------------------------------------
# Trainer: identity commit gating and the collector hand-off
# ---------------------------------------------------------------------------

def _commit_holder() -> Trainer:
    holder = object.__new__(Trainer)
    holder._bg_selfplay_lock = threading.Lock()
    holder._validation_tensor_identity = None
    return holder


def _leakage(retained: int) -> dict:
    return {
        "ledger_enabled": True,
        "all_time_trained_state_count": 10,
        "removed_validation_entry_count": 3,
        "removed_validation_state_count": 2,
        "retained_validation_entry_count": retained,
    }


def test_commit_requires_nonempty_tensors_and_validated_leakage() -> None:
    holder = _commit_holder()
    identity = {"validation_manifest_sha256": "a" * 64}

    holder._validation_dataloader = ["t"] * 4
    holder._active_snapshot_manifest = {"validation_leakage": _leakage(4)}
    holder._commit_validation_tensor_identity(identity)
    record = holder._validation_tensor_identity
    assert record is not None and record["identity"] is identity
    assert record["leakage"]["retained_validation_entry_count"] == 4

    # A count mismatch, an empty dataset, or a missing identity all clear it.
    holder._active_snapshot_manifest = {"validation_leakage": _leakage(5)}
    holder._commit_validation_tensor_identity(identity)
    assert holder._validation_tensor_identity is None

    holder._active_snapshot_manifest = {"validation_leakage": _leakage(4)}
    holder._validation_dataloader = []
    holder._commit_validation_tensor_identity(identity)
    assert holder._validation_tensor_identity is None

    holder._validation_dataloader = ["t"] * 4
    holder._commit_validation_tensor_identity(None)
    assert holder._validation_tensor_identity is None


def _collector_holder() -> tuple[Trainer, list, list]:
    holder = object.__new__(Trainer)
    holder._bg_selfplay_lock = threading.Lock()
    holder._bg_selfplay_dataset = ["dataset"]
    holder._bg_selfplay_incremental = None
    holder._bg_selfplay_entries = None
    holder._bg_snapshot_manifest = {"version": 7}
    activations, publishes = [], []
    holder._activate_dataset_manifest = activations.append
    holder._set_validation_entries = publishes.append
    return holder, activations, publishes


def test_collector_keeps_current_tensors_on_the_sentinel() -> None:
    holder, activations, publishes = _collector_holder()
    holder._bg_validation_entries = _VALIDATION_TENSORS_CURRENT
    holder._bg_validation_identity = {"identity": "unchanged"}
    holder._validation_tensor_identity = {"identity": "published"}
    holder._validation_dataloader = ["existing-tensors"]

    dataset, incremental = holder._collect_background_selfplay()
    assert dataset == ["dataset"] and incremental is None
    assert activations == [{"version": 7}]
    assert publishes == []
    assert holder._validation_dataloader == ["existing-tensors"]
    assert holder._validation_tensor_identity == {"identity": "published"}
    assert holder._bg_validation_identity is None


def test_collector_commits_the_handed_identity_after_a_rebuild() -> None:
    holder, activations, publishes = _collector_holder()
    entries = ["entry-a", "entry-b"]
    identity = {"validation_manifest_sha256": "b" * 64}
    holder._bg_validation_entries = entries
    holder._bg_validation_identity = identity
    committed = []
    holder._commit_validation_tensor_identity = committed.append

    dataset, incremental = holder._collect_background_selfplay()
    assert dataset == ["dataset"] and incremental is None
    assert activations == [{"version": 7}]
    assert publishes == [entries]
    assert committed == [identity]
    assert holder._bg_validation_identity is None
