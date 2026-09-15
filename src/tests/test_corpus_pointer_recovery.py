"""Damaged active pointers must not restart a configured corpus lineage."""

import json
import os
from pathlib import Path
import signal

import pytest

from dama.ai.ml.corpus import CorpusSnapshotManager


SETTINGS = (
    {"difficulty": "hard"},
    {"played_action_probability": 0.10, "label_is_teacher": True},
    {"algorithm_fraction": 0.70, "model_fraction": 0.30},
)


def _write_replay(path: Path, index: int) -> None:
    row = (index // 4) % 8
    col = (index * 2 + 1 - row % 2) % 8
    entry = {
        "state": {
            "p1_men": [[row, col]],
            "p1_kings": [],
            "p2_men": [[7 - row, 7 - col]],
            "p2_kings": [],
            "turn": 1,
            "move_count": index,
        },
        "legal_moves": [
            {"path": [[0, 1], [1, 0]], "captures": [], "promotion": False},
            {"path": [[0, 1], [1, 2]], "captures": [], "promotion": False},
        ],
        "chosen_index": 0,
        "result": 0,
        "trajectory_source": "algorithm",
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def _manager(tmp_path: Path, name: str, first_index: int, **kwargs):
    replay = tmp_path / f"{name}_replay"
    replay.mkdir()
    for index in range(4):
        _write_replay(replay / f"replay_{index}.jsonl", first_index + index)
    return CorpusSnapshotManager(
        str(replay), str(tmp_path / f"{name}_root"),
        validation_fraction=0.25, split_seed=5,
        min_fresh_fraction=0.50, grow_holdout=False, **kwargs,
    )


def _rebuilt_manager(tmp_path: Path, first_index: int = 20):
    base_manager = _manager(tmp_path, "base", 0)
    base = base_manager.consider_snapshot(*SETTINGS)
    assert base.admitted and base.manifest_path is not None
    base_manifest = json.loads(base.manifest_path.read_text(encoding="utf-8"))
    manager = _manager(
        tmp_path, "rebuilt", first_index,
        lineage_base_manifest=str(base.manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
    )
    return manager, base_manifest


@pytest.fixture
def active_lineage(tmp_path):
    manager, base_manifest = _rebuilt_manager(tmp_path)
    first = manager.consider_snapshot(*SETTINGS)
    assert first.admitted and first.manifest_path is not None
    unchanged = manager.consider_snapshot(*SETTINGS)
    assert not unchanged.admitted
    assert unchanged.reason == "unchanged"
    assert unchanged.metrics["fresh_unique_state_rate"] == 0.0
    return manager, first.manifest_path, base_manifest


def _damage_current(manager, damage: str) -> None:
    if damage == "missing":
        manager.current_pointer.unlink()
    elif damage in ("empty", "whitespace"):
        manager.current_pointer.write_text(
            "" if damage == "empty" else " \n\t", encoding="utf-8")
    elif damage == "missing_target":
        manager.current_pointer.write_text(
            "snapshot_v999999/manifest.json\n", encoding="utf-8")
    elif damage == "directory_target":
        # Preserve the authoritative manifest and make CURRENT name its parent.
        relative = manager.current_pointer.read_text(encoding="utf-8").strip()
        manager.current_pointer.write_text(
            Path(relative).parent.as_posix() + "\n", encoding="utf-8")
    else:
        raise AssertionError(f"Unknown pointer damage: {damage}")


@pytest.mark.parametrize("damage", [
    "missing", "empty", "whitespace", "missing_target", "directory_target",
])
def test_recovery_uses_current_json_without_readmitting_unchanged_replay(
    active_lineage, damage,
):
    manager, current_path, _base = active_lineage
    before = (manager.snapshot_root / "current.json").read_bytes()
    _damage_current(manager, damage)

    assert manager.current_manifest_path() == current_path
    path, manifest = manager._load_current_manifest()
    assert path == current_path
    assert manifest["fingerprint"] == json.loads(before)["fingerprint"]
    decision = manager.consider_snapshot(*SETTINGS)
    assert not decision.admitted
    assert decision.reason == "unchanged"
    assert decision.metrics["fresh_unique_state_rate"] == 0.0
    assert len(list(manager.snapshot_root.glob("snapshot_v*"))) == 1
    assert (manager.snapshot_root / "current.json").read_bytes() == before


@pytest.mark.parametrize("damage", [
    "missing", "empty", "missing_target", "directory_target",
])
def test_unrecoverable_pointer_cannot_fall_back_to_external_lineage_base(
    active_lineage, damage, monkeypatch,
):
    manager, _current_path, _base = active_lineage
    _damage_current(manager, damage)
    (manager.snapshot_root / "current.json").unlink()

    def unexpected_work(*args, **kwargs):
        pytest.fail("Unrecoverable pointer reached validation or lineage base")

    monkeypatch.setattr(manager, "_ensure_validation", unexpected_work)
    monkeypatch.setattr(manager, "_lineage_base", unexpected_work)
    with pytest.raises(RuntimeError, match="pointer is missing"):
        manager.consider_snapshot(*SETTINGS)
    assert len(list(manager.snapshot_root.glob("snapshot_v*"))) == 1


@pytest.mark.parametrize("damage", ["empty", "missing_target"])
@pytest.mark.parametrize("invalid", ["malformed_json", "invalid_record", "fingerprint"])
def test_recovery_rejects_invalid_redundant_evidence(active_lineage, damage, invalid):
    manager, current_path, _base = active_lineage
    _damage_current(manager, damage)
    pointer = manager.snapshot_root / "current.json"
    if invalid == "malformed_json":
        pointer.write_text("{", encoding="utf-8")
    else:
        record = json.loads(pointer.read_text(encoding="utf-8"))
        record["fingerprint"] = "not-a-sha256" if invalid == "invalid_record" else "0" * 64
        pointer.write_text(json.dumps(record), encoding="utf-8")

    for lookup in (manager.current_manifest_path, manager._load_current_manifest):
        with pytest.raises(RuntimeError, match="pointer|fingerprint"):
            lookup()
    with pytest.raises(RuntimeError, match="pointer|fingerprint"):
        manager.consider_snapshot(*SETTINGS)
    assert current_path.is_file()
    assert len(list(manager.snapshot_root.glob("snapshot_v*"))) == 1


def test_first_external_base_admission_still_records_the_verified_base(tmp_path):
    manager, base = _rebuilt_manager(tmp_path)
    first = manager.consider_snapshot(*SETTINGS)
    assert first.admitted
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["previous_fingerprint"] == base["fingerprint"]
    assert manifest["admission"]["previous_corpus_source"] == "lineage_base"
    assert first.metrics["fresh_unique_state_rate"] == 1.0


def test_first_external_base_admission_still_rejects_stale_replay(tmp_path):
    manager, _base = _rebuilt_manager(tmp_path, first_index=0)
    first = manager.consider_snapshot(*SETTINGS)
    assert not first.admitted
    assert first.reason == "unchanged"
    assert first.metrics["fresh_unique_state_rate"] == 0.0
    assert not list(manager.snapshot_root.glob("snapshot_v*"))


@pytest.mark.parametrize("target_kind", ["missing", "directory"])
def test_unavailable_redundant_target_cannot_reset_the_lineage(
    active_lineage, target_kind,
):
    manager, current_path, _base = active_lineage
    manager.current_pointer.unlink()
    pointer = manager.snapshot_root / "current.json"
    record = json.loads(pointer.read_text(encoding="utf-8"))
    record["manifest"] = (
        "snapshot_v999999/manifest.json" if target_kind == "missing"
        else current_path.parent.name)
    pointer.write_text(json.dumps(record), encoding="utf-8")

    assert manager.current_manifest_path() is None
    assert manager._load_current_manifest() == (None, None)
    with pytest.raises(RuntimeError, match="pointer is missing"):
        manager.consider_snapshot(*SETTINGS)
    assert len(list(manager.snapshot_root.glob("snapshot_v*"))) == 1


def test_healthy_current_remains_authoritative_over_redundant_pointer(active_lineage):
    manager, current_path, _base = active_lineage
    (manager.snapshot_root / "current.json").write_text("{", encoding="utf-8")

    assert manager.current_manifest_path() == current_path
    assert manager._load_current_manifest()[0] == current_path
    decision = manager.consider_snapshot(*SETTINGS)
    assert not decision.admitted
    assert decision.reason == "unchanged"


def test_readable_corrupt_current_manifest_is_not_hidden_by_recovery(active_lineage):
    manager, current_path, _base = active_lineage
    current_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError):
        manager._load_current_manifest()
    with pytest.raises(ValueError):
        manager.consider_snapshot(*SETTINGS)


def test_windows_directory_open_error_uses_redundant_pointer(
    active_lineage, monkeypatch,
):
    manager, current_path, _base = active_lineage
    _damage_current(manager, "directory_target")
    directory_target = current_path.parent
    real_open = os.open

    def windows_open(path, *args, **kwargs):
        if Path(path) == directory_target:
            # Windows reports EACCES, rather than EISDIR, when opening a directory.
            raise PermissionError("Access is denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", windows_open)
    assert manager._load_current_manifest()[0] == current_path
    decision = manager.consider_snapshot(*SETTINGS)
    assert not decision.admitted
    assert decision.reason == "unchanged"
    assert len(list(manager.snapshot_root.glob("snapshot_v*"))) == 1


def test_regular_manifest_permission_error_is_not_hidden_by_recovery(
    active_lineage, monkeypatch,
):
    manager, current_path, _base = active_lineage
    failure = PermissionError("Cannot read the active manifest")
    real_open = os.open

    def denied_open(path, *args, **kwargs):
        if Path(path) == current_path:
            raise failure
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", denied_open)
    with pytest.raises(PermissionError) as caught:
        manager._load_current_manifest()
    assert caught.value is failure


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(signal, "SIGALRM"),
    reason="FIFO recovery needs POSIX pipes and a bounded read deadline",
)
@pytest.mark.parametrize("pointer_kind", ["current", "redundant"])
def test_fifo_manifest_target_does_not_block_recovery(
    active_lineage, pointer_kind,
):
    manager, current_path, _base = active_lineage
    fifo = manager.snapshot_root / "damaged_manifest.json"
    os.mkfifo(fifo)
    if pointer_kind == "current":
        manager.current_pointer.write_text(fifo.name + "\n", encoding="utf-8")
    else:
        manager.current_pointer.unlink()
        redundant = manager.snapshot_root / "current.json"
        record = json.loads(redundant.read_text(encoding="utf-8"))
        record["manifest"] = fifo.name
        redundant.write_text(json.dumps(record), encoding="utf-8")

    def read_deadline(*_args):
        pytest.fail("FIFO target blocked before regular-file validation")

    previous_handler = signal.signal(signal.SIGALRM, read_deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        path, manifest = manager._load_current_manifest()
        if pointer_kind == "current":
            assert path == current_path
            assert manifest["fingerprint"] == json.loads(
                current_path.read_text(encoding="utf-8"))["fingerprint"]
        else:
            assert (path, manifest) == (None, None)
            with pytest.raises(RuntimeError, match="pointer is missing"):
                manager.consider_snapshot(*SETTINGS)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
    assert len(list(manager.snapshot_root.glob("snapshot_v*"))) == 1


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(signal, "SIGALRM"),
    reason="FIFO rejection needs POSIX pipes and a bounded read deadline",
)
@pytest.mark.parametrize("pointer_kind", ["current", "redundant"])
@pytest.mark.parametrize("lookup", ["path", "manifest", "admission"])
def test_fifo_pointer_fails_closed_without_waiting_for_a_writer(
    active_lineage, pointer_kind, lookup,
):
    manager, current_path, _base = active_lineage
    manifest_before = current_path.read_bytes()
    if pointer_kind == "current":
        pointer = manager.current_pointer
    else:
        manager.current_pointer.unlink()
        pointer = manager.snapshot_root / "current.json"
    pointer.unlink()
    os.mkfifo(pointer)

    def read_deadline(*_args):
        pytest.fail("FIFO control pointer blocked before regular-file validation")

    previous_handler = signal.signal(signal.SIGALRM, read_deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        with pytest.raises(RuntimeError, match="pointer"):
            if lookup == "path":
                manager.current_manifest_path()
            elif lookup == "manifest":
                manager._load_current_manifest()
            else:
                manager.consider_snapshot(*SETTINGS)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
    assert current_path.read_bytes() == manifest_before
    assert len(list(manager.snapshot_root.glob("snapshot_v*"))) == 1
