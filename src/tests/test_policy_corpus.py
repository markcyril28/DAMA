import gzip
import hashlib
import json
import os
from pathlib import Path
import threading

import pytest

import dama.ai.ml.corpus as corpus
from dama.ai.ml.corpus import (
    CorpusSnapshotManager,
    analyze_replay_files,
    canonical_state_key,
    split_replay_by_file,
)


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


def _entry(index: int, *, state: dict | None = None, forced: bool = False) -> dict:
    moves = [{"path": [[0, 1], [1, 0]], "captures": [], "promotion": False}]
    if not forced:
        moves.append({"path": [[0, 1], [1, 2]], "captures": [], "promotion": False})
    return {
        "state": state or _state(index),
        "legal_moves": moves,
        "chosen_index": 0,
        "result": 0,
        "trajectory_source": "algorithm",
    }


def _write_replay(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )


def test_canonical_state_ignores_move_count_and_normalizes_player_two() -> None:
    p1 = _state(0, turn=1)
    p1["move_count"] = 3
    p2 = {
        "p1_men": [[0, 0]],
        "p1_kings": [],
        "p2_men": [[7, 6]],
        "p2_kings": [],
        "turn": 2,
        "move_count": 99,
    }
    p1_equivalent = {
        "p1_men": [[0, 1]],
        "p1_kings": [],
        "p2_men": [[7, 7]],
        "p2_kings": [],
        "turn": 1,
        "move_count": 0,
    }

    assert canonical_state_key(p2) == canonical_state_key(p1_equivalent)
    changed_count = dict(p1_equivalent, move_count=200)
    assert canonical_state_key(changed_count) == canonical_state_key(p1_equivalent)


def test_whole_file_split_has_no_canonical_state_overlap(tmp_path: Path) -> None:
    files = []
    for index in range(8):
        path = tmp_path / f"replay_{index:02d}.jsonl"
        _write_replay(path, [_entry(index), _entry(index + 20)])
        files.append(path)

    train, validation = split_replay_by_file(files, validation_fraction=0.15, seed=77)
    train_keys = {canonical_state_key(entry.state) for entry in train}
    validation_keys = {canonical_state_key(entry.state) for entry in validation}

    assert validation
    assert train
    assert train_keys.isdisjoint(validation_keys)
    assert len(validation) % 2 == 0


def test_snapshot_gate_accepts_exact_half_fresh_and_preserves_prior(
    tmp_path: Path, monkeypatch,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_00_{index}.jsonl", [_entry(index, forced=index == 0)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
        # This case asserts the exact freshness arithmetic of the admission
        # gate, so the hold-out is pinned; growth has its own coverage.
        grow_holdout=False,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}

    first = manager.consider_snapshot(settings, noise, generation)
    assert first.admitted
    assert first.manifest_path is not None
    first_manifest_before = first.manifest_path.read_bytes()

    with first.manifest_path.open("r", encoding="utf-8") as handle:
        first_manifest = json.load(handle)
    previous_count = first_manifest["metrics"]["post_dedup_unique_state_count"]
    assert previous_count > 0

    for offset in range(previous_count):
        _write_replay(
            replay_dir / f"replay_01_{offset}.jsonl",
            [_entry(100 + offset)],
        )

    second = manager.consider_snapshot(settings, noise, generation)
    assert second.admitted
    assert second.metrics["fresh_unique_state_rate"] == pytest.approx(0.50)
    assert first.manifest_path.read_bytes() == first_manifest_before

    _write_replay(replay_dir / "replay_02_0.jsonl", [_entry(300)])
    from dama.ai.ml import corpus

    complement_calls = 0
    original_exclude = corpus._exclude_validation_state_keys

    def counted_exclude(*args, **kwargs):
        nonlocal complement_calls
        complement_calls += 1
        return original_exclude(*args, **kwargs)

    monkeypatch.setattr(
        corpus, "_exclude_validation_state_keys", counted_exclude)
    rejected = manager.consider_snapshot(settings, noise, generation)
    assert not rejected.admitted
    assert "below" in rejected.reason
    assert rejected.manifest_path == second.manifest_path
    assert complement_calls == 0


def _admit_series(
    manager: CorpusSnapshotManager,
    replay_dir: Path,
    cycles: int,
    start_cycle: int = 0,
) -> list:
    """Admit ``cycles`` snapshots, feeding all-fresh states each time."""
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}
    decisions = []
    for cycle in range(start_cycle, start_cycle + cycles):
        # The live pipeline rotates shards out via cleanup_old_files, so each
        # cycle presents an all-fresh corpus and clears the freshness gate.
        for stale in replay_dir.glob("replay_*.jsonl"):
            stale.unlink()
        for offset in range(4):
            _write_replay(
                replay_dir / f"replay_{cycle:02d}_{offset}.jsonl",
                [_entry(1000 * (cycle + 1) + offset)],
            )
        decision = manager.consider_snapshot(settings, noise, generation)
        assert decision.admitted, decision.reason
        decisions.append(decision)
    return decisions


def test_snapshot_retention_prunes_oldest_and_keeps_current(tmp_path: Path) -> None:
    """A positive retention cap must reclaim disk without breaking the active split.

    Each admission stores a full copy of the replay corpus, so an uncapped
    snapshot root grows without bound and fills the volume mid-run.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
        max_retained_snapshots=3,
    )
    decisions = _admit_series(manager, replay_dir, 5)

    kept = sorted(path.name for path in snapshot_root.glob("snapshot_v*"))
    assert kept == ["snapshot_v000003", "snapshot_v000004", "snapshot_v000005"]

    # The newest admission stays intact and remains loadable.
    current = decisions[-1]
    assert current.manifest_path is not None
    assert current.manifest_path.is_file()
    train_entries, _validation_entries, _manifest = manager.load_split()
    assert train_entries

    # Version numbering must not restart after pruning.
    assert manager._next_version() == 6
    later = _admit_series(manager, replay_dir, 1, start_cycle=5)[0]
    assert later.manifest_path is not None
    assert later.manifest_path.parent.name == "snapshot_v000006"

    # The frozen validation set is never a pruning candidate.
    assert (snapshot_root / "validation" / "manifest.json").is_file()


def test_snapshot_retention_commits_deletion_batch_before_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful retention call commits removed snapshot names once."""
    snapshot_root = tmp_path / "snapshots"
    for version in range(1, 4):
        (snapshot_root / f"snapshot_v{version:06d}").mkdir(parents=True)
    manager = CorpusSnapshotManager(
        str(tmp_path / "replay"),
        str(snapshot_root),
        max_retained_snapshots=2,
    )
    events = []
    real_rmtree = corpus.shutil.rmtree
    real_fsync_directory = corpus.run_status._fsync_directory

    def tracking_rmtree(path: Path) -> None:
        events.append(f"removed:{Path(path).name}")
        real_rmtree(path)

    def tracking_fsync_directory(path: Path) -> None:
        if Path(path) == snapshot_root:
            events.append("snapshot_root_committed")
        real_fsync_directory(path)

    monkeypatch.setattr(corpus.shutil, "rmtree", tracking_rmtree)
    monkeypatch.setattr(
        corpus.run_status, "_fsync_directory", tracking_fsync_directory
    )

    removed = manager._prune_old_snapshots(
        snapshot_root / "snapshot_v000003"
    )

    assert removed == ["snapshot_v000001"]
    assert events == ["removed:snapshot_v000001", "snapshot_root_committed"]


def test_snapshot_retention_commit_failure_is_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retention cannot report reclaimed storage before its directory commit."""
    snapshot_root = tmp_path / "snapshots"
    for version in range(1, 3):
        (snapshot_root / f"snapshot_v{version:06d}").mkdir(parents=True)
    manager = CorpusSnapshotManager(
        str(tmp_path / "replay"),
        str(snapshot_root),
        max_retained_snapshots=1,
    )

    def fail_snapshot_root_commit(path: Path) -> None:
        if Path(path) == snapshot_root:
            raise OSError(5, "simulated snapshot retention directory sync failure")

    monkeypatch.setattr(
        corpus.run_status, "_fsync_directory", fail_snapshot_root_commit
    )

    with pytest.raises(OSError, match="snapshot retention directory sync failure"):
        manager._prune_old_snapshots(snapshot_root / "snapshot_v000002")

    assert not (snapshot_root / "snapshot_v000001").exists()
    assert (snapshot_root / "snapshot_v000002").is_dir()


def test_snapshot_retention_without_deletion_does_not_sync_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An at-cap snapshot root pays no retention directory-sync cost."""
    snapshot_root = tmp_path / "snapshots"
    for version in range(1, 3):
        (snapshot_root / f"snapshot_v{version:06d}").mkdir(parents=True)
    manager = CorpusSnapshotManager(
        str(tmp_path / "replay"),
        str(snapshot_root),
        max_retained_snapshots=2,
    )

    def unexpected_sync(path: Path) -> None:
        raise AssertionError(f"unexpected directory sync: {path}")

    monkeypatch.setattr(
        corpus.run_status, "_fsync_directory", unexpected_sync
    )

    assert manager._prune_old_snapshots(
        snapshot_root / "snapshot_v000002"
    ) == []


def test_snapshot_retention_disabled_by_default_keeps_every_snapshot(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
    )
    assert manager.max_retained_snapshots == 0
    _admit_series(manager, replay_dir, 4)
    assert len(list(snapshot_root.glob("snapshot_v*"))) == 4


def test_snapshot_retention_rejects_negative_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CorpusSnapshotManager(
            str(tmp_path / "replay"),
            str(tmp_path / "snapshots"),
            max_retained_snapshots=-1,
        )


# ---------------------------------------------------------------------------
# Proofread 2026-08-25 C1: leftover version directory must not break admission
# ---------------------------------------------------------------------------

def test_admission_fails_closed_on_a_leftover_version_directory(
    tmp_path: Path,
) -> None:
    """Proofread 2026-08-25 C1.

    ``os.replace(staging, final_dir)`` has no guard for ``final_dir`` already
    existing: a half-written earlier ``snapshot_vNNNNNN`` makes admission fail
    with errno 39, and an *empty* leftover is silently adopted under a
    lineage that belongs to nothing.  A leftover target must fail closed.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_00_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}

    # Simulate a concurrent/littered admission: while this manager is
    # building its staging tree, a half-written earlier attempt appears at
    # the same version number it is about to claim.
    real_next_version = manager._next_version

    def colliding_next_version() -> int:
        version = real_next_version()
        leftover = snapshot_root / f"snapshot_v{version:06d}"
        leftover.mkdir(parents=True, exist_ok=True)
        (leftover / "manifest.json").write_text("{ truncated", encoding="utf-8")
        return version

    manager._next_version = colliding_next_version

    with pytest.raises(RuntimeError, match="already exists"):
        manager.consider_snapshot(settings, noise, generation)

    # The leftover itself is untouched evidence; nothing was admitted.
    assert (snapshot_root / "snapshot_v000001" / "manifest.json").read_text(
        encoding="utf-8"
    ) == "{ truncated"
    assert manager.current_manifest_path() is None
    # The staging tree of the refused attempt did not leak either.
    assert not list(snapshot_root.glob(".snapshot_v*"))


def test_empty_leftover_version_directory_also_fails_closed(
    tmp_path: Path,
) -> None:
    """An empty leftover must not be silently adopted as the new snapshot."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_00_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}

    real_next_version = manager._next_version

    def colliding_next_version() -> int:
        version = real_next_version()
        (snapshot_root / f"snapshot_v{version:06d}").mkdir(parents=True, exist_ok=True)
        return version

    manager._next_version = colliding_next_version

    with pytest.raises(RuntimeError, match="already exists"):
        manager.consider_snapshot(settings, noise, generation)

    assert manager.current_manifest_path() is None


def _validation_manifest(snapshot_root: Path) -> dict:
    return json.loads(
        (snapshot_root / "validation" / "manifest.json").read_text(encoding="utf-8")
    )


def test_initial_validation_commits_shard_names_before_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validation manifest cannot authorize uncommitted shard names."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(
            replay_dir / f"replay_{index:02d}.jsonl", [_entry(index)]
        )
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=5,
    )
    events = []
    real_fsync_directory = corpus.run_status._fsync_directory
    real_write_json_atomic = corpus._write_json_atomic

    def tracking_fsync_directory(path: Path) -> None:
        if Path(path).name == "files":
            events.append("validation_files_committed")
        real_fsync_directory(path)

    def tracking_write_json_atomic(path: Path, payload: dict) -> None:
        if Path(path).name == "manifest.json":
            events.append("validation_manifest_started")
        real_write_json_atomic(path, payload)

    monkeypatch.setattr(
        corpus.run_status, "_fsync_directory", tracking_fsync_directory
    )
    monkeypatch.setattr(corpus, "_write_json_atomic", tracking_write_json_atomic)

    manager._ensure_validation(sorted(replay_dir.glob("replay_*.jsonl")))

    assert events.index("validation_files_committed") < events.index(
        "validation_manifest_started"
    )


def test_initial_validation_shard_commit_failure_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed shard-name commit cannot publish or strand a validation set."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(
            replay_dir / f"replay_{index:02d}.jsonl", [_entry(index)]
        )
    snapshot_root = tmp_path / "snapshots"
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
    )
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    real_fsync_directory = corpus.run_status._fsync_directory
    failed = False

    def fail_first_validation_files_commit(path: Path) -> None:
        nonlocal failed
        if Path(path).name == "files" and not failed:
            failed = True
            raise OSError(5, "simulated validation shard directory sync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(
        corpus.run_status,
        "_fsync_directory",
        fail_first_validation_files_commit,
    )

    with pytest.raises(OSError, match="validation shard directory sync failure"):
        manager._ensure_validation(files)

    assert failed
    assert not manager.validation_manifest_path.exists()
    assert not manager.validation_manifest_path.parent.exists()
    assert not list(snapshot_root.glob(".validation*"))

    manifest, state_keys = manager._ensure_validation(files)
    assert manifest["files"]
    assert state_keys


def test_validation_growth_commits_new_shard_names_before_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Append-only growth commits copied names before changing authority."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(8):
        _write_replay(
            replay_dir / f"replay_a{index}.jsonl", [_entry(index)]
        )
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=5,
    )
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    initial, _keys = manager._ensure_validation(files)
    held = {str(record["name"]) for record in initial["files"]}
    for name in held:
        (replay_dir / name).unlink()
    for index in range(len(held)):
        _write_replay(
            replay_dir / f"replay_b{index}.jsonl", [_entry(100 + index)]
        )

    events = []
    validation_files = manager.validation_manifest_path.parent / "files"
    real_fsync_directory = corpus.run_status._fsync_directory
    real_write_json_atomic = corpus._write_json_atomic

    def tracking_fsync_directory(path: Path) -> None:
        if Path(path) == validation_files:
            events.append("validation_files_committed")
        real_fsync_directory(path)

    def tracking_write_json_atomic(path: Path, payload: dict) -> None:
        if Path(path) == manager.validation_manifest_path:
            events.append("validation_manifest_started")
        real_write_json_atomic(path, payload)

    monkeypatch.setattr(
        corpus.run_status, "_fsync_directory", tracking_fsync_directory
    )
    monkeypatch.setattr(corpus, "_write_json_atomic", tracking_write_json_atomic)

    grown, _keys = manager._ensure_validation(
        sorted(replay_dir.glob("replay_*.jsonl"))
    )

    assert grown.get("growth_history")
    assert events.index("validation_files_committed") < events.index(
        "validation_manifest_started"
    )


def test_validation_holdout_grows_append_only_toward_configured_share(
    tmp_path: Path,
) -> None:
    """Audit finding F3: a frozen hold-out decays far below its approved share.

    The split is created from whatever files exist at creation time and, before
    this behaviour, never grew -- so an approved 15% realized 1.7% against a
    rolling corpus while the manifest still declared 0.15.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(2):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)
    created = _validation_manifest(snapshot_root)
    assert len(created["files"]) == 1
    held_at_creation = {record["name"] for record in created["files"]}

    # Grow the corpus well beyond the creation-time size.
    for index in range(18):
        _write_replay(replay_dir / f"replay_grow_{index:02d}.jsonl", [_entry(500 + index)])
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)

    split = manifest["split"]
    assert split["fraction"] == 0.25
    assert split["validation_file_count"] == round(len(files) * 0.25)
    assert split["realized_file_fraction"] == pytest.approx(0.25, abs=0.03)

    # Append-only: every originally held file is still held.
    held_now = {record["name"] for record in manifest["files"]}
    assert held_at_creation <= held_now
    assert manifest["growth_history"][-1]["source_file_count"] == len(files)

    # The regenerated key set must satisfy the integrity contract.
    reloaded, _state_keys = manager._ensure_validation(files)
    assert len(reloaded["files"]) == len(manifest["files"])


def test_warm_validation_verification_skips_redundant_existence_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opening the manifest and key member already proves they exist."""

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(
            replay_dir / f"replay_{index:02d}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir), str(snapshot_root),
        validation_fraction=0.25, split_seed=5,
    )
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, state_keys = manager._ensure_validation(files)
    manifest_path = manager.validation_manifest_path
    key_path = manifest_path.parent / manifest["state_keys_file"]
    real_exists = Path.exists
    real_is_file = Path.is_file

    def guarded_exists(path):
        if path == manifest_path:
            raise AssertionError("validation manifest received a pre-open stat")
        return real_exists(path)

    def guarded_is_file(path):
        if path == key_path:
            raise AssertionError("state-key member received a pre-cache stat")
        return real_is_file(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    reloaded, reloaded_keys = manager._ensure_validation(files)
    assert reloaded == manifest
    assert reloaded_keys == state_keys


def test_validation_growth_never_releases_a_held_file(tmp_path: Path) -> None:
    """States may move train -> validation, never the reverse."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(2):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.30,
        split_seed=11,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)
    held = {record["name"] for record in _validation_manifest(snapshot_root)["files"]}

    for round_index in range(4):
        for index in range(5):
            _write_replay(
                replay_dir / f"replay_r{round_index}_{index}.jsonl",
                [_entry(2000 + round_index * 100 + index)],
            )
        files = sorted(replay_dir.glob("replay_*.jsonl"))
        manifest, _keys = manager._ensure_validation(files)
        now = {record["name"] for record in manifest["files"]}
        assert held <= now, "growth released a previously held validation file"
        held = now
        # A training file must always survive the split.
        assert len(now) < len(files)


def test_validation_holdout_growth_can_be_disabled(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(2):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
        grow_holdout=False,
    )
    _admit_series(manager, replay_dir, 1)
    for index in range(18):
        _write_replay(replay_dir / f"replay_grow_{index:02d}.jsonl", [_entry(500 + index)])
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)

    assert len(manifest["files"]) == 1
    assert "growth_history" not in manifest
    # The realized share is still reported honestly.
    assert manifest["split"]["realized_file_fraction"] == pytest.approx(1 / len(files))


def test_windows_written_relative_paths_still_resolve(tmp_path: Path) -> None:
    """Finding F5: a backslash-separated pointer must not reset the lineage.

    The native-Windows launcher wrote `..\\validation\\manifest.json` into
    snapshot v11. On WSL that path does not resolve, current_manifest_path()
    returns None, and consider_snapshot takes the no-previous-corpus branch --
    skipping the >=50% freshness floor, which is how v12 was admitted.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir), str(snapshot_root),
        validation_fraction=0.25, split_seed=5,
        min_fresh_fraction=0.50, grow_holdout=False,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}
    first = manager.consider_snapshot(settings, noise, generation)
    assert first.admitted

    # Everything written must be POSIX-separated.
    pointer = (snapshot_root / "CURRENT").read_text(encoding="utf-8").strip()
    assert "\\" not in pointer
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert "\\" not in manifest["validation_manifest"]

    # A pointer left behind by Windows must still resolve on this host.
    (snapshot_root / "CURRENT").write_text(
        pointer.replace("/", "\\") + "\n", encoding="utf-8")
    assert manager.current_manifest_path() is not None, (
        "backslash pointer did not resolve -- the lineage would silently reset")

    # And the freshness gate must therefore still be enforced.
    _write_replay(replay_dir / "replay_new.jsonl", [_entry(950)])
    decision = manager.consider_snapshot(settings, noise, generation)
    assert not decision.admitted
    assert "below" in decision.reason


def test_current_pointer_lookup_opens_without_redundant_exists_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    target = snapshot_root / "snapshot_v000001" / "manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    pointer = snapshot_root / "CURRENT"
    pointer.write_text("snapshot_v000001/manifest.json\n", encoding="utf-8")
    manager = CorpusSnapshotManager(
        str(tmp_path / "replay"), str(snapshot_root))
    real_exists = Path.exists

    def guarded_exists(path):
        if path == pointer:
            raise AssertionError("CURRENT received a pre-read stat")
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    assert manager.current_manifest_path() == target


def test_snapshot_gate_loads_current_manifest_without_target_pre_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate opens CURRENT's target once and validates that descriptor."""

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(
            replay_dir / f"replay_{index:02d}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir), str(snapshot_root),
        validation_fraction=0.25, split_seed=5,
    )
    settings = ({"difficulty": "hard"}, {"noise": 0.1}, {"mix": "fixed"})
    first = manager.consider_snapshot(*settings)
    assert first.admitted
    current_target = manager.current_manifest_path()
    assert current_target is not None
    real_is_file = Path.is_file

    def guarded_is_file(path):
        if path == current_target:
            raise AssertionError("CURRENT target received a pre-open stat")
        return real_is_file(path)

    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    second = manager.consider_snapshot(*settings)
    assert not second.admitted
    assert second.reason == "unchanged"


def test_load_current_manifest_rejects_non_regular_target(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshots"
    target = snapshot_root / "snapshot_v000001" / "manifest.json"
    target.mkdir(parents=True)
    (snapshot_root / "CURRENT").write_text(
        "snapshot_v000001/manifest.json\n", encoding="utf-8")
    manager = CorpusSnapshotManager(
        str(tmp_path / "replay"), str(snapshot_root))

    assert manager._load_current_manifest() == (None, None)


def test_current_json_recovers_a_missing_current_pointer(tmp_path: Path) -> None:
    """The redundant committed pointer must recover a lost CURRENT name."""

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index:02d}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir), str(snapshot_root),
        validation_fraction=0.25, split_seed=5,
    )
    decision = manager.consider_snapshot(
        {"difficulty": "hard"}, {"noise": 0.1}, {"mix": "fixed"})
    assert decision.admitted

    manager.current_pointer.unlink()

    assert manager.current_manifest_path() == decision.manifest_path
    path, manifest = manager._load_current_manifest()
    assert path == decision.manifest_path
    assert manifest is not None
    assert manifest["fingerprint"] == json.loads(
        (snapshot_root / "current.json").read_text(encoding="utf-8")
    )["fingerprint"]


def test_current_json_fallback_rejects_a_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    """A recovery pointer is authoritative only for its exact manifest."""

    snapshot_root = tmp_path / "snapshots"
    manifest = snapshot_root / "snapshot_v000001" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"fingerprint": "a" * 64}), encoding="utf-8")
    (snapshot_root / "current.json").write_text(json.dumps({
        "manifest": "snapshot_v000001/manifest.json",
        "fingerprint": "b" * 64,
    }), encoding="utf-8")
    manager = CorpusSnapshotManager(
        str(tmp_path / "replay"), str(snapshot_root))

    with pytest.raises(RuntimeError, match="fingerprint does not match"):
        manager.current_manifest_path()
    with pytest.raises(RuntimeError, match="fingerprint does not match"):
        manager._load_current_manifest()


def test_atomic_current_pointer_syncs_file_and_directory_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd):
        events.append("file_fsync")
        return real_fsync(fd)

    def tracking_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    def tracking_directory_fsync(path):
        events.append("directory_fsync")
        assert path == tmp_path

    monkeypatch.setattr(corpus.os, "fsync", tracking_fsync)
    monkeypatch.setattr(corpus.os, "replace", tracking_replace)
    monkeypatch.setattr(
        corpus.run_status, "_fsync_directory", tracking_directory_fsync)
    target = tmp_path / "CURRENT"

    corpus._write_text_atomic(target, "snapshot_v000001/manifest.json\n")

    assert events == ["file_fsync", "replace", "directory_fsync"]
    assert target.read_text(encoding="utf-8") == (
        "snapshot_v000001/manifest.json\n")


def test_atomic_current_pointer_failure_preserves_old_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "CURRENT"
    target.write_text("snapshot_v000001/manifest.json\n", encoding="utf-8")

    def fail_fsync(_fd):
        raise OSError(5, "simulated pointer fsync failure")

    monkeypatch.setattr(corpus.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated pointer fsync failure"):
        corpus._write_text_atomic(
            target, "snapshot_v000002/manifest.json\n")

    assert target.read_text(encoding="utf-8") == (
        "snapshot_v000001/manifest.json\n")
    assert not list(tmp_path.glob("CURRENT.*.tmp"))


def test_atomic_current_pointer_reports_post_replace_directory_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible complete pointer is not reported durable if its name is not."""

    target = tmp_path / "CURRENT"
    target.write_text("snapshot_v000001/manifest.json\n", encoding="utf-8")

    def fail_directory_fsync(_path):
        raise OSError(5, "simulated pointer directory fsync failure")

    monkeypatch.setattr(
        corpus.run_status, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(
        OSError, match="simulated pointer directory fsync failure"
    ):
        corpus._write_text_atomic(
            target, "snapshot_v000002/manifest.json\n")

    assert target.read_text(encoding="utf-8") == (
        "snapshot_v000002/manifest.json\n")
    assert not list(tmp_path.glob("CURRENT.*.tmp"))


def test_lost_current_pointer_fails_closed_instead_of_skipping_freshness_gate(
    tmp_path: Path,
) -> None:
    """A vanished CURRENT pointer must not silently disable the freshness gate.

    The gate reads ``current_path is not None``, so before this a lost pointer
    made every candidate admissible and recorded a 100% fresh rate that is only
    an artifact of an empty previous-key set -- which is how snapshot_v000012
    entered the policy-distillation namespace ungated.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
        grow_holdout=False,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}

    first = manager.consider_snapshot(settings, noise, generation)
    assert first.admitted

    # Simulate the pointer loss observed in production.
    (snapshot_root / "CURRENT").unlink()
    (snapshot_root / "current.json").unlink()

    _write_replay(replay_dir / "replay_new.jsonl", [_entry(900)])
    with pytest.raises(RuntimeError, match="pointer is missing"):
        manager.consider_snapshot(settings, noise, generation)

    # Restoring the pointer restores normal gated behaviour.
    (snapshot_root / "CURRENT").write_text(
        "snapshot_v000001/manifest.json\n", encoding="utf-8")
    decision = manager.consider_snapshot(settings, noise, generation)
    assert not decision.admitted
    assert "below" in decision.reason


def test_first_ever_admission_still_allowed_without_a_pointer(
    tmp_path: Path,
) -> None:
    """The fail-closed check must not break a genuinely empty namespace."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
        grow_holdout=False,
    )
    assert manager.current_manifest_path() is None
    assert manager._snapshot_dirs() == []
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10, "label_is_teacher": True},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    assert decision.admitted


def test_growth_never_absorbs_a_shard_an_admitted_snapshot_trained_on(
    tmp_path: Path,
) -> None:
    """A hold-out built from already-trained shards measures memorisation.

    The 2026-08-22 growth absorbed 8 shards that were training files in the
    active snapshot, leaving 87.93% of the hold-out already fit by the model.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(6):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.50,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)
    trained = manager._trained_shard_names()
    assert trained, "the admitted snapshot must report its training shards"

    # Offer fresh, never-trained shards alongside the trained ones.
    for index in range(6):
        _write_replay(
            replay_dir / f"replay_fresh_{index}.jsonl", [_entry(400 + index)])
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)

    added = {
        name
        for event in manifest.get("growth_history", [])
        for name in event["added_files"]
    }
    assert added, "growth should still absorb the untrained shards"
    assert not (added & trained), (
        f"growth absorbed trained shards: {sorted(added & trained)}")


def test_holdout_reports_how_much_of_it_is_still_in_the_corpus(
    tmp_path: Path,
) -> None:
    """A hold-out whose shards rotated out must not advertise its target share."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)
    held = {record["name"] for record in manifest["files"]}
    assert held

    # Rotate every held shard out of the replay window.
    for path in list(replay_dir.glob("replay_*.jsonl")):
        if path.name in held:
            path.unlink()
    for index in range(8):
        _write_replay(replay_dir / f"replay_after_{index}.jsonl", [_entry(700 + index)])
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)
    split = manifest["split"]

    assert split["validation_files_in_corpus"] < split["validation_file_count"]
    assert split["realized_in_corpus_fraction"] < split["realized_file_fraction"]
    # Both ratios stay in range even when held shards outlive the window.
    assert 0.0 <= split["realized_file_fraction"] <= 1.0
    assert 0.0 <= split["realized_in_corpus_fraction"] <= 1.0


def test_growth_resumes_after_the_holdout_rotates_out_of_the_window(
    tmp_path: Path,
) -> None:
    """End-to-end wiring: _grow_validation must pass the *present* count.

    Sized so the two accountings disagree absolutely: all-time accounting sees
    the target already met and grows nothing, present accounting sees a
    hold-out that covers none of the live corpus and reopens every slot.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(8):
        _write_replay(replay_dir / f"replay_a{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)
    held = {record["name"] for record in manifest["files"]}
    assert len(held) == 2, "target for 8 files at 25% is 2"

    # Every held shard leaves the window; the corpus size is unchanged.
    for name in held:
        (replay_dir / name).unlink()
    for index in range(len(held)):
        _write_replay(replay_dir / f"replay_b{index}.jsonl", [_entry(900 + index)])

    files = sorted(replay_dir.glob("replay_*.jsonl"))
    assert len(files) == 8
    manifest, _keys = manager._ensure_validation(files)

    history = manifest.get("growth_history", [])
    assert history, (
        "hold-out covers none of the live corpus but growth did not reopen -- "
        "the quota is counting all-time held shards")
    assert manifest["split"]["validation_files_in_corpus"] >= 1


def test_growth_quota_measures_against_holdout_files_still_present(
    tmp_path: Path,
) -> None:
    """The quota must count present hold-out shards, not all-time ones.

    Counting all-time held shards lets a hold-out whose files have rotated out
    of the replay window report its target share while covering none of the
    live corpus -- F3 recurring at 15% magnitude instead of 1.7%.
    """
    manager = CorpusSnapshotManager(
        str(tmp_path / "replay"),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=5,
    )
    # 20 files -> target 5. Nine shards were absorbed all-time, but only one
    # still exists in the window, so eight slots must reopen.
    assert manager._validation_growth_quota(
        total_files=20, held=1, candidates=10, all_time_held=9) == 4
    # With every held shard still present the quota is satisfied and closed.
    assert manager._validation_growth_quota(
        total_files=20, held=5, candidates=10, all_time_held=5) == 0
    # The survivor guard still applies.
    assert manager._validation_growth_quota(
        total_files=20, held=0, candidates=1, all_time_held=0) == 0


def test_holdout_growth_is_bounded_by_the_file_ceiling(tmp_path: Path) -> None:
    """Present-held accounting must not grow the manifest once per rotation."""
    from dama.ai.ml.corpus import HOLDOUT_FILE_CEILING

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)

    # Repeatedly rotate the whole window so the quota keeps reopening.
    for cycle in range(12):
        for path in list(replay_dir.glob("replay_*.jsonl")):
            path.unlink()
        for index in range(4):
            _write_replay(
                replay_dir / f"replay_c{cycle}_{index}.jsonl",
                [_entry(5000 + cycle * 50 + index)],
            )
        files = sorted(replay_dir.glob("replay_*.jsonl"))
        manifest, _keys = manager._ensure_validation(files)

    target = max(1, round(4 * 0.25))
    assert len(manifest["files"]) <= HOLDOUT_FILE_CEILING * target


def test_ceiling_bound_holdout_skips_trained_shard_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound hold-out has no eligibility question left to answer."""
    from dama.ai.ml.corpus import HOLDOUT_FILE_CEILING

    manager = CorpusSnapshotManager(
        str(tmp_path / "replay"),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=5,
    )
    # Four live files target one held file and cap the append-only manifest at
    # three.  Rotate all three historical held shards out so only the ceiling,
    # rather than present-file coverage, closes the growth quota.
    manifest = {
        "files": [
            {"name": f"held_old_{index}.jsonl"}
            for index in range(HOLDOUT_FILE_CEILING)
        ]
    }
    files = [tmp_path / f"replay_live_{index}.jsonl" for index in range(4)]

    def fail_if_scanned() -> set[str]:
        raise AssertionError("ceiling-bound growth scanned trained shards")

    monkeypatch.setattr(manager, "_trained_shard_names", fail_if_scanned)
    assert manager._grow_validation(manifest, files) is None


def test_snapshot_load_removes_validation_overlap(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(7):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.15,
        split_seed=11,
        min_fresh_fraction=0.50,
    )
    external_key = canonical_state_key(_state(2))
    manager.set_external_validation_state_keys({external_key})
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    train, validation, manifest = manager.load_split(decision.manifest_path)

    train_keys = {canonical_state_key(entry.state) for entry in train}
    validation_keys = {canonical_state_key(entry.state) for entry in validation}
    assert train_keys.isdisjoint(validation_keys)
    assert external_key not in train_keys
    assert manifest["admission"]["passed"] is True
    assert manifest["metrics"]["forced_move_rate"] >= 0.0
    assert manifest["metrics"]["external_validation_state_count"] == 1
    assert manifest["metrics"]["validation_overlap_state_count_removed"] == (
        manifest["metrics"]["unique_state_count"]
        - manifest["metrics"]["post_dedup_unique_state_count"]
    )


def test_snapshot_settings_match_rejects_previous_stage_contract(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=17,
        min_fresh_fraction=0.50,
    )
    policy_teacher = {"stage": "policy_only", "target_type": "hard"}
    enhanced_teacher = {"stage": "enhanced", "target_type": "distribution"}
    noise = {"played_action_probability": 0.10}
    policy_generation = {"current_model_inference_depth": 1}
    enhanced_generation = {"current_model_inference_depth": 2}

    first = manager.consider_snapshot(
        policy_teacher, noise, policy_generation
    )
    rejected = manager.consider_snapshot(
        enhanced_teacher, noise, enhanced_generation
    )

    assert first.admitted
    assert not rejected.admitted
    assert rejected.manifest_path == first.manifest_path
    assert not manager.snapshot_matches_settings(
        rejected.manifest_path,
        enhanced_teacher,
        noise,
        enhanced_generation,
    )


def test_snapshot_settings_match_treats_behavior_identity_as_provenance(
    tmp_path: Path,
) -> None:
    """Behavior-step drift alone must not read as a contract change.

    Training always proceeds past the standing snapshot's admission step
    within a session, and dead-epoch rollback can rewind behind it, so
    neither direction of behavior-identity drift distinguishes a healthy
    relaunch from a healthy running session. Real contract drift (noise,
    generation schema) must still mismatch.
    """

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=17,
        min_fresh_fraction=0.50,
    )
    teacher = {"stage": "policy_only", "target_type": "hard"}
    noise = {"played_action_probability": 0.10}
    admitted_generation = {
        "current_model_inference_depth": 1,
        "model_behavior_id": "trainer-step-312036",
        "model_behavior_step": 312036,
    }

    first = manager.consider_snapshot(teacher, noise, admitted_generation)
    assert first.admitted

    resumed_generation = dict(
        admitted_generation,
        model_behavior_id="trainer-step-314000",
        model_behavior_step=314000,
    )
    assert manager.snapshot_matches_settings(
        first.manifest_path, teacher, noise, resumed_generation
    )

    rolled_back_generation = dict(
        admitted_generation,
        model_behavior_id="trainer-step-306000",
        model_behavior_step=306000,
    )
    assert manager.snapshot_matches_settings(
        first.manifest_path, teacher, noise, rolled_back_generation
    )

    assert not manager.snapshot_matches_settings(
        first.manifest_path,
        teacher,
        {"played_action_probability": 0.0},
        resumed_generation,
    )

    assert not manager.snapshot_matches_settings(
        first.manifest_path,
        teacher,
        noise,
        dict(resumed_generation, current_model_inference_depth=2),
    )


def test_snapshot_load_rejects_tampered_training_shard(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=19,
        min_fresh_fraction=0.50,
    )
    decision = manager.consider_snapshot({}, {}, {})
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    shard = decision.manifest_path.parent / manifest["files"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"tampered\n")

    with pytest.raises(RuntimeError, match="integrity verification"):
        manager.load_split(decision.manifest_path)


def test_snapshot_load_rejects_tampered_state_key_digest(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=23,
        min_fresh_fraction=0.50,
    )
    decision = manager.consider_snapshot({}, {}, {})
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    state_keys = decision.manifest_path.parent / manifest["state_keys_file"]
    with gzip.open(state_keys, "wt", encoding="ascii", newline="\n") as handle:
        handle.write("0" * 64 + "\n")

    with pytest.raises(RuntimeError, match="canonical-state fingerprint"):
        manager.load_split(decision.manifest_path)


def test_snapshot_shards_link_immutable_sources_and_report_cross_cycle_repeats(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    state = _state(4)
    first = replay_dir / "replay_cycle_a.jsonl"
    second = replay_dir / "replay_cycle_b.jsonl"
    repeated_a = _entry(4, state=state)
    repeated_a["game_id"] = "cycle-000001-algorithm-000001"
    repeated_b = _entry(4, state=state)
    repeated_b["game_id"] = "cycle-000002-algorithm-000001"
    unique_b = _entry(200)
    unique_b["game_id"] = "cycle-000002-algorithm-000002"
    _write_replay(first, [repeated_a])
    _write_replay(second, [repeated_b, unique_b])

    metrics, _ = analyze_replay_files([first, second])
    assert metrics["cross_file_repeated_state_count"] == 1
    assert metrics["cross_file_unique_state_count"] == 1
    assert metrics["cross_cycle_repeated_state_count"] == 1
    assert metrics["cross_cycle_unique_state_count"] == 1

    manager = CorpusSnapshotManager(
        str(replay_dir), str(tmp_path / "snapshots"),
        validation_fraction=0.1, split_seed=7, min_fresh_fraction=0.0,
    )
    decision = manager.consider_snapshot({}, {}, {})
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    shard = decision.manifest_path.parent / manifest["files"][0]["path"]
    source = replay_dir / manifest["files"][0]["name"]
    assert manifest["files"][0]["storage"] == "source_hardlink"
    assert shard.stat().st_ino == source.stat().st_ino

    source.unlink()
    assert shard.is_file()


def test_existing_validation_manifest_must_match_split_contract(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    snapshot_root = tmp_path / "snapshots"
    manager = CorpusSnapshotManager(
        str(replay_dir), str(snapshot_root), validation_fraction=0.25,
        split_seed=11, min_fresh_fraction=0.50,
    )
    manager.consider_snapshot({}, {}, {})
    manifest_path = snapshot_root / "validation" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"]["seed"] = 12
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="split fraction and seed"):
        manager.consider_snapshot({}, {}, {})


def _contract_entry(index: int, source: str, game_id: str) -> dict:
    entry = _entry(index)
    entry.update({
        "played_index": 0,
        "trajectory_source": source,
        "was_exploration": False,
        "teacher_difficulty": "hard",
        "opening_plies": 2,
        "game_id": game_id,
    })
    return entry


def test_recovery_snapshots_exclude_legacy_files_and_audit_exact_game_mix(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_replay(replay_dir / "replay_legacy.jsonl", [_entry(900)])
    for file_index in range(2):
        entries = []
        for game_index in range(10):
            source = "algorithm" if game_index < 7 else "current_model"
            entries.append(_contract_entry(
                file_index * 100 + game_index,
                source,
                f"{file_index}-{game_index}",
            ))
        _write_replay(replay_dir / f"replay_repaired_{file_index}.jsonl", entries)

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.50,
        split_seed=3,
        min_fresh_fraction=0.50,
        enforce_policy_contract=True,
        allowed_opening_plies=(2, 4, 6, 8),
    )
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    _, _, manifest = manager.load_split(decision.manifest_path)

    assert decision.admitted
    assert "replay_legacy.jsonl" in manifest["metrics"]["rejected_replay_files"]
    assert manifest["metrics"]["source_game_counts"] == {
        "algorithm": 7,
        "current_model": 3,
    }


def test_replay_audit_catches_late_contract_violation():
    from dama.ai.ml import corpus

    valid = _contract_entry(1, "algorithm", "late-check-1")
    invalid = _contract_entry(2, "algorithm", "late-check-2")
    invalid["chosen_index"] = 999

    # Use the real iterator only for the assertion setup below; replacing it
    # keeps this test independent of JSONL formatting and file I/O.
    original = corpus._iter_entry_dicts
    try:
        corpus._iter_entry_dicts = lambda _path: iter((valid, invalid))
        result = corpus.audit_policy_replay_file(
            Path("sentinel.jsonl"), (2, 4, 6, 8))
    finally:
        corpus._iter_entry_dicts = original
    assert result["valid"] is False
    assert result["records"] == 2
    assert result["errors"]["invalid_teacher_index"] == 1


def test_replay_analysis_cache_matches_cold_scan_and_skips_reparse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    shared = _state(0)
    first = _entry(0, state=shared)
    first["game_id"] = "cycle-000001-algorithm-000001"
    first["trajectory_source"] = "algorithm"
    second = _entry(1, forced=True)
    second["game_id"] = "cycle-000001-algorithm-000002"
    left = tmp_path / "replay_left.jsonl"
    _write_replay(left, [first, second])

    repeated = _entry(0, state=shared)
    repeated["game_id"] = "cycle-000002-current-model-000001"
    repeated["trajectory_source"] = "current_model"
    unique = _entry(2)
    unique["game_id"] = "cycle-000002-current-model-000002"
    right = tmp_path / "replay_right.jsonl"
    _write_replay(right, [repeated, unique])

    previous = {canonical_state_key(shared)}
    cold = corpus.analyze_replay_files([left, right], previous)

    original_iterator = corpus._iter_entry_dicts_with_digest

    def fail_if_reparsed(_path, _digest):
        raise AssertionError("unchanged replay file was reparsed")

    monkeypatch.setattr(corpus, "_iter_entry_dicts_with_digest", fail_if_reparsed)
    warm = corpus.analyze_replay_files([left, right], previous)

    assert warm == cold
    monkeypatch.setattr(corpus, "_iter_entry_dicts_with_digest", original_iterator)


def test_replay_analysis_cache_invalidates_on_file_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_mutating.jsonl"
    _write_replay(path, [_entry(0)])
    cold_before = corpus.analyze_replay_files([path])

    _write_replay(path, [_entry(0), _entry(20, forced=True)])
    original_iterator = corpus._iter_entry_dicts_with_digest
    parse_count = 0

    def counted_iterator(file_path, digest):
        nonlocal parse_count
        parse_count += 1
        yield from original_iterator(file_path, digest)

    monkeypatch.setattr(corpus, "_iter_entry_dicts_with_digest", counted_iterator)
    warm_after_mutation = corpus.analyze_replay_files([path])
    assert parse_count == 1

    monkeypatch.setattr(corpus, "_iter_entry_dicts_with_digest", original_iterator)
    corpus._clear_replay_file_cache()
    cold_after_mutation = corpus.analyze_replay_files([path])

    assert warm_after_mutation == cold_after_mutation
    assert warm_after_mutation != cold_before


def test_replay_hash_cache_reuses_and_invalidates_by_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_hash.jsonl"
    _write_replay(path, [_entry(0)])
    original_hash = corpus._sha256_file_uncached
    hash_calls = 0

    def counted_hash(file_path):
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(file_path)

    monkeypatch.setattr(corpus, "_sha256_file_uncached", counted_hash)
    first = corpus.replay_file_sha256(path)
    assert corpus.replay_file_sha256(path) == first
    assert hash_calls == 1

    _write_replay(path, [_entry(0), _entry(20)])
    changed = corpus.replay_file_sha256(path)
    assert changed != first
    assert hash_calls == 2


def test_large_manifest_hashes_in_parallel_and_still_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One directory scan feeds overlapping hashes without weakening checks."""
    from dama.ai.ml import corpus

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=5,
    )
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    assert decision.manifest_path is not None
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    expected = {
        decision.manifest_path.parent / record["path"]: record["sha256"]
        for record in manifest["files"]
    }
    assert len(expected) > 1

    monkeypatch.setattr(corpus, "_PARALLEL_MANIFEST_HASH_MIN_BYTES", 0)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    calls = 0
    active = 0
    peak_active = 0
    identities = {}
    scan_calls = []
    real_scandir = corpus.os.scandir

    def counted_scandir(path):
        scan_calls.append(Path(path))
        return real_scandir(path)

    def coordinated_hash(path: Path, identity) -> str:
        nonlocal calls, active, peak_active
        identities[path] = identity
        with lock:
            calls += 1
            ordinal = calls
            active += 1
            peak_active = max(peak_active, active)
        try:
            if ordinal <= 2:
                barrier.wait(timeout=2.0)
            return expected[path]
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(corpus.os, "scandir", counted_scandir)
    monkeypatch.setattr(
        corpus, "_replay_file_sha256_for_identity", coordinated_hash)
    verified = manager._verify_manifest_integrity(
        decision.manifest_path, manifest, "training_snapshot")
    assert verified
    assert calls == len(expected)
    assert peak_active >= 2
    assert scan_calls == [decision.manifest_path.parent / "files"] * 2
    assert set(identities) == set(expected)
    assert all(
        identity == corpus._replay_file_identity(path)
        for path, identity in identities.items()
    )

    bad_path = next(iter(expected))
    manager._manifest_integrity_identity_cache.clear()
    monkeypatch.setattr(
        corpus,
        "_replay_file_sha256_for_identity",
        lambda path, _identity: (
            "0" * 64 if path == bad_path else expected[path]),
    )
    with pytest.raises(RuntimeError, match="failed integrity verification"):
        manager._verify_manifest_integrity(
            decision.manifest_path, manifest, "training_snapshot")

    replaced = False

    def replace_during_cached_hash(path: Path, _identity) -> str:
        nonlocal replaced
        if path == bad_path and not replaced:
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            replaced = True
        return expected[path]

    monkeypatch.setattr(
        corpus,
        "_replay_file_sha256_for_identity",
        replace_during_cached_hash,
    )
    manager._manifest_integrity_identity_cache.clear()
    with pytest.raises(RuntimeError, match="failed integrity verification"):
        manager._verify_manifest_integrity(
            decision.manifest_path, manifest, "training_snapshot")
    assert replaced


def test_warm_manifest_integrity_uses_one_fail_closed_identity_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully verified immutable declaration needs one later recheck."""
    from dama.ai.ml import corpus

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=5,
    )
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    assert decision.manifest_path is not None
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    expected_keys = manager._verify_manifest_integrity(
        decision.manifest_path, manifest, "training_snapshot")

    files_dir = decision.manifest_path.parent / "files"
    scan_calls = []
    real_scandir = corpus.os.scandir

    def counted_scandir(path):
        scan_calls.append(Path(path))
        return real_scandir(path)

    def unexpected_digest(_path, _identity):
        raise AssertionError("unchanged verified shard was re-digested")

    monkeypatch.setattr(corpus.os, "scandir", counted_scandir)
    monkeypatch.setattr(
        corpus, "_replay_file_sha256_for_identity", unexpected_digest)
    assert manager._verify_manifest_integrity(
        decision.manifest_path, manifest, "training_snapshot") == expected_keys
    assert scan_calls == [files_dir]


def test_warm_manifest_integrity_reverifies_changed_identity(
    tmp_path: Path,
) -> None:
    """A changed identity falls back to full digests before it is accepted."""

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=5,
    )
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    assert decision.manifest_path is not None
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    expected_keys = manager._verify_manifest_integrity(
        decision.manifest_path, manifest, "training_snapshot")
    stored = decision.manifest_path.parent / manifest["files"][0]["path"]

    replacement = stored.with_suffix(".replacement")
    replacement.write_bytes(stored.read_bytes())
    replacement.replace(stored)
    assert manager._verify_manifest_integrity(
        decision.manifest_path, manifest, "training_snapshot") == expected_keys

    stored.write_bytes(stored.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="failed integrity verification"):
        manager._verify_manifest_integrity(
            decision.manifest_path, manifest, "training_snapshot")


def test_malformed_replay_json_is_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_malformed.jsonl"
    path.write_text(json.dumps(_entry(0)) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid replay JSON"):
        corpus.analyze_replay_files([path])

    original_iterator = corpus._iter_entry_dicts_with_digest
    parse_count = 0

    def counted_iterator(file_path, digest):
        nonlocal parse_count
        parse_count += 1
        yield from original_iterator(file_path, digest)

    monkeypatch.setattr(corpus, "_iter_entry_dicts_with_digest", counted_iterator)
    with pytest.raises(ValueError, match="Invalid replay JSON"):
        corpus.analyze_replay_files([path])
    assert parse_count == 1


def test_semantically_malformed_replay_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_semantically_malformed.jsonl"
    path.write_text(
        json.dumps(_entry(0)) + "\n" + json.dumps({"state": {}}) + "\n",
        encoding="utf-8",
    )
    first, _ = corpus.analyze_replay_files([path])
    assert first["malformed_records"] == 1

    original_iterator = corpus._iter_entry_dicts_with_digest
    parse_count = 0

    def counted_iterator(file_path, digest):
        nonlocal parse_count
        parse_count += 1
        yield from original_iterator(file_path, digest)

    monkeypatch.setattr(corpus, "_iter_entry_dicts_with_digest", counted_iterator)
    second, _ = corpus.analyze_replay_files([path])
    assert second == first
    assert parse_count == 1


def test_replay_audit_cache_reuses_unchanged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_audited.jsonl"
    _write_replay(path, [_contract_entry(1, "algorithm", "audit-1")])
    cold = corpus.audit_policy_replay_file(path, (2, 4, 6, 8))

    def fail_if_reparsed(_path, _digest):
        raise AssertionError("unchanged replay file was reparsed")

    monkeypatch.setattr(corpus, "_iter_entry_dicts_with_digest", fail_if_reparsed)
    warm = corpus.audit_policy_replay_file(path, (2, 4, 6, 8))
    assert warm == cold


def test_replay_audit_populates_analysis_cache_in_same_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold contract check must supply the later diversity analysis."""
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_fused.jsonl"
    entries = [
        _contract_entry(
            index,
            "algorithm" if index < 7 else "current_model",
            f"fused-{index}",
        )
        for index in range(10)
    ]
    _write_replay(path, entries)

    original_iterator = corpus._iter_entry_dicts_with_digest
    original_hash = corpus._sha256_file_uncached
    parse_calls = 0
    hash_calls = 0

    def counted_iterator(file_path, digest):
        nonlocal parse_calls
        parse_calls += 1
        yield from original_iterator(file_path, digest)

    def counted_hash(file_path):
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(file_path)

    monkeypatch.setattr(corpus, "_iter_entry_dicts_with_digest", counted_iterator)
    monkeypatch.setattr(corpus, "_sha256_file_uncached", counted_hash)

    audit = corpus.audit_policy_replay_file(path, (2, 4, 6, 8))
    metrics, keys = corpus.analyze_replay_files([path])
    digest = corpus.replay_file_sha256(path)

    assert audit["valid"]
    assert audit["records"] == 10
    assert audit["source_game_counts"] == {
        "algorithm": 7, "current_model": 3}
    assert metrics["records"] == 10
    assert metrics["malformed_records"] == 0
    assert metrics["source_game_counts"] == audit["source_game_counts"]
    assert metrics["state_set_sha256"] == corpus._state_set_digest(keys)
    assert len(digest) == 64
    assert parse_calls == 1, "analysis reparsed a contract-valid shard"
    assert hash_calls == 0, "analysis or digest lookup re-read the shard"


def test_fused_audit_digest_matches_exact_jsonl_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-pass digest covers original bytes, not normalized text."""
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_exact_bytes.jsonl"
    entries = [
        _contract_entry(
            index,
            "algorithm" if index < 7 else "current_model",
            f"exact-α-{index}",
        )
        for index in range(10)
    ]
    # Cross the reader's 1 MiB block boundary inside one JSON record.
    entries[0]["game_id"] = "exact-long-" + ("x" * (1024 * 1024))
    raw_lines = [
        json.dumps(entry, ensure_ascii=False, sort_keys=True).encode("utf-8")
        for entry in entries
    ]
    payload = b"\r\n" + b"\r\n".join(raw_lines[:-1]) + b"\r\n" + raw_lines[-1]
    path.write_bytes(payload)

    def forbid_second_read(_path):
        raise AssertionError("fused analysis performed a separate digest read")

    monkeypatch.setattr(corpus, "_sha256_file_uncached", forbid_second_read)
    audit = corpus.audit_policy_replay_file(path, (2, 4, 6, 8))

    assert audit["valid"]
    assert corpus.replay_file_sha256(path) == hashlib.sha256(payload).hexdigest()
    assert corpus.analyze_replay_files([path])[0]["records"] == 10


def test_replay_iterators_share_the_accelerated_json_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both corpus readers must route decoding through replay's codec."""
    from dama.ai.ml import corpus

    path = tmp_path / "replay_shared_codec.jsonl"
    entry = _contract_entry(1, "algorithm", "shared-codec")
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    argument_types = []

    def tracked_loads(payload):
        argument_types.append(type(payload))
        return json.loads(payload)

    monkeypatch.setattr(corpus, "_replay_json_loads", tracked_loads)
    assert list(corpus._iter_entry_dicts(path)) == [entry]
    digest = hashlib.sha256()
    assert list(corpus._iter_entry_dicts_with_digest(path, digest)) == [entry]
    assert argument_types == [str, bytes]
    assert digest.hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_fused_audit_rejects_without_publishing_partial_analysis(
    tmp_path: Path,
) -> None:
    """The fused reader keeps the original first-error fail-closed boundary."""
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_invalid_fused.jsonl"
    valid = _contract_entry(1, "algorithm", "fused-valid")
    invalid = _contract_entry(2, "algorithm", "fused-invalid")
    invalid["chosen_index"] = 999
    path.write_text(
        json.dumps(valid) + "\n" + json.dumps(invalid) + "\nnot-json\n",
        encoding="utf-8",
    )

    result = corpus.audit_policy_replay_file(path, (2, 4, 6, 8))
    identity = corpus._replay_file_identity(path).as_key()

    assert not result["valid"]
    assert result["records"] == 2
    assert result["errors"] == {"invalid_teacher_index": 1}
    assert identity not in corpus._REPLAY_ANALYSIS_CACHE
    assert identity not in corpus._REPLAY_HASH_CACHE


def test_legacy_replay_audit_fails_fast(monkeypatch):
    from dama.ai.ml import corpus

    legacy = {"state": {}, "legal_moves": []}
    consumed = []

    def sentinel_iterator(_path):
        consumed.append(1)
        yield legacy
        raise AssertionError("legacy audit must stop after first contract error")

    monkeypatch.setattr(corpus, "_iter_entry_dicts", sentinel_iterator)
    result = corpus.audit_policy_replay_file(Path("legacy.jsonl"), (2, 4, 6, 8))
    assert result["valid"] is False
    assert result["records"] == 1
    assert result["errors"] == {"missing_legal_moves": 1}
    assert consumed == [1]


def test_partial_cycle_replay_file_is_rejected_by_per_file_split(
    tmp_path: Path,
) -> None:
    """A killed cycle leaves an off-ratio file; it must not poison the corpus."""
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    def _cycle(name: str, algorithm: int, model: int, base: int) -> None:
        entries = []
        for index in range(algorithm + model):
            source = "algorithm" if index < algorithm else "current_model"
            entries.append(_contract_entry(base + index, source, f"{name}-{index}"))
        _write_replay(replay_dir / name, entries)

    _cycle("replay_complete_0.jsonl", 7, 3, 0)
    _cycle("replay_complete_1.jsonl", 7, 3, 100)
    # Interrupted cycle: the model trajectories were still running when the
    # process died, so the file holds 7/2 instead of 7/3.
    _cycle("replay_partial.jsonl", 7, 2, 200)

    partial = corpus.audit_policy_replay_file(
        replay_dir / "replay_partial.jsonl", (2, 4, 6, 8))
    assert partial["valid"] is False
    assert partial["errors"] == {"unbalanced_policy_trajectory_split": 1}

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.50,
        split_seed=3,
        min_fresh_fraction=0.50,
        enforce_policy_contract=True,
        allowed_opening_plies=(2, 4, 6, 8),
    )
    eligible, rejected = manager.eligible_replay_files()
    assert [path.name for path in eligible] == [
        "replay_complete_0.jsonl", "replay_complete_1.jsonl",
    ]
    assert set(rejected) == {"replay_partial.jsonl"}

    # The aggregate contract holds again once the partial file is excluded.
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    assert decision.admitted
    assert decision.metrics["source_game_counts"] == {
        "algorithm": 7, "current_model": 3,
    }


def test_snapshot_load_accepts_windows_written_manifest_paths(tmp_path: Path) -> None:
    """A manifest written by the native-Windows launcher must load on WSL.

    ``str(Path("files") / name)`` emits the *host* separator, so a snapshot or
    hold-out grown on native Windows records ``files\\replay_*.jsonl``.  Read
    back on Linux that is a single filename containing a backslash: every stored
    shard "fails integrity verification" while sitting untouched on disk.  Both
    the training snapshot and the frozen validation manifest are rewritten here
    because ``_grow_validation`` carries old records forward verbatim, so a
    single Windows run leaves the two manifests permanently mixed.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=29,
        min_fresh_fraction=0.50,
    )
    decision = manager.consider_snapshot({}, {}, {})

    for manifest_path in (
        decision.manifest_path,
        manager.validation_manifest_path,
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["files"], manifest_path
        for record in manifest["files"]:
            record["path"] = record["path"].replace("/", "\\")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    train_entries, validation_entries, _ = manager.load_split(
        decision.manifest_path)
    assert train_entries
    assert validation_entries


def test_prepared_snapshot_split_matches_the_legacy_eager_load(tmp_path: Path) -> None:
    """Deferring train materialization must preserve split membership exactly."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(
            replay_dir / f"replay_{index}.jsonl",
            [_entry(index), _entry(index + 20)],
        )
    manager = CorpusSnapshotManager(
        str(replay_dir), str(snapshot_root), validation_fraction=0.25,
        split_seed=29, min_fresh_fraction=0.50, grow_holdout=False,
    )
    decision = manager.consider_snapshot({}, {}, {})
    assert decision.manifest_path is not None

    context = manager.prepare_split(decision.manifest_path)
    deferred_validation = manager.load_validation_entries(context)
    deferred_train = manager.load_train_entries(context)
    eager_train, eager_validation, eager_manifest = manager.load_split(
        decision.manifest_path)

    assert [entry.to_dict() for entry in deferred_train] == [
        entry.to_dict() for entry in eager_train
    ]
    assert [entry.to_dict() for entry in deferred_validation] == [
        entry.to_dict() for entry in eager_validation
    ]
    assert context.manifest["validation_leakage"] == eager_manifest[
        "validation_leakage"
    ]


# ---------------------------------------------------------------------------
# Audit Suggestion 9: the hold-out's freshness tax must be attributable
# ---------------------------------------------------------------------------

def test_holdout_growth_records_the_freshness_it_cost(tmp_path: Path) -> None:
    """A growth event cancels a cycle's freshness gain; the manifest must say so.

    ``_grow_validation`` can only choose never-trained shards -- a trained one
    would measure memorisation, not generalisation -- so the shards eligible
    for the hold-out are exactly the fresh ones.  Each growth event therefore
    moves a whole fresh shard out of training and flattens that cycle's
    freshness reading, which from the logs is indistinguishable from the
    generator having saturated.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_00_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
        grow_holdout=True,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}

    first = manager.consider_snapshot(settings, noise, generation)
    assert first.admitted

    # Enough new shards that the 25% quota reopens and growth takes one.
    for offset in range(8):
        _write_replay(
            replay_dir / f"replay_01_{offset}.jsonl",
            [_entry(100 + offset)],
        )

    second = manager.consider_snapshot(settings, noise, generation)

    assert second.metrics["holdout_growth_files"], (
        "the quota should have reopened and moved at least one shard")
    assert second.metrics["fresh_states_transferred_to_holdout"] > 0
    assert (
        second.metrics["fresh_states_transferred_to_holdout"]
        <= second.metrics["states_transferred_to_holdout"]
    )
    # The counterfactual is what the cycle would have read without the
    # transfer, so it can never be the lower of the two.
    assert (
        second.metrics["fresh_unique_state_rate_without_holdout_growth"]
        >= second.metrics["fresh_unique_state_rate"]
    )


def test_a_cycle_without_holdout_growth_reports_no_freshness_cost(
    tmp_path: Path,
) -> None:
    """No growth event, no cost -- and never last cycle's cost repeated."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_00_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
        grow_holdout=False,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}

    decision = manager.consider_snapshot(settings, noise, generation)

    assert decision.admitted
    assert decision.metrics["holdout_growth_files"] == []
    assert decision.metrics["fresh_states_transferred_to_holdout"] == 0
    assert decision.metrics["states_transferred_to_holdout"] == 0
    assert (
        decision.metrics["fresh_unique_state_rate_without_holdout_growth"]
        == pytest.approx(decision.metrics["fresh_unique_state_rate"])
    )


def _shard_reuse_manager(tmp_path: Path, **kwargs) -> CorpusSnapshotManager:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir(exist_ok=True)
    snapshot_root = tmp_path / "snapshots"
    base = dict(
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
        grow_holdout=False,
    )
    base.update(kwargs)
    return CorpusSnapshotManager(str(replay_dir), str(snapshot_root), **base)


_SETTINGS = {"difficulty": "hard"}
_NOISE = {"played_action_probability": 0.10, "label_is_teacher": True}
_GENERATION = {"algorithm_fraction": 0.70, "model_fraction": 0.30}


def test_snapshot_reuses_immutable_shards_via_hardlink(tmp_path: Path) -> None:
    """Survivors link from the predecessor and rotated shards from replay.

    The steady-state corpus rotates one shard per cycle, so per-admission
    copying falls to zero. Reused entries are byte-identical by construction
    (same inode), which makes this a pure storage optimization rather than a
    semantic change.
    """
    manager = _shard_reuse_manager(tmp_path)
    replay_dir = tmp_path / "replay"
    for index in range(4):
        _write_replay(replay_dir / f"replay_00_{index}.jsonl", [_entry(index)])

    first = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)
    assert first.admitted
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert all(
        record["storage"] == "source_hardlink"
        for record in first_manifest["files"]
    )
    assert first_manifest["admission"]["reused_shard_count"] == 0
    assert (
        first_manifest["admission"]["source_linked_shard_count"]
        == len(first_manifest["files"])
    )
    assert first_manifest["admission"]["copied_shard_count"] == 0

    # Cycle 2 keeps three shards byte-identical and adds one new file.
    for offset in range(4):
        _write_replay(
            replay_dir / f"replay_01_{offset}.jsonl",
            [_entry(100 + offset)],
        )
    second = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)
    assert second.admitted
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    storage_by_name = {
        record["name"]: record["storage"] for record in second_manifest["files"]
    }
    # Shards present in both admissions' train splits survived unchanged and
    # must be linked, not copied.  (Some seed shards may live in the hold-out
    # instead of the train split, depending on the hash-ranked selection.)
    first_names = {
        record["name"] for record in first_manifest["files"]
    }
    second_names = set(storage_by_name)
    survivors = first_names & second_names
    assert survivors, "test setup expected at least one surviving train shard"
    for name in survivors:
        assert storage_by_name[name] == "hardlink", (
            f"{name} survived unchanged but was stored as {storage_by_name[name]}"
        )
    rotated = {n for n in second_names if n.startswith("replay_01")}
    assert rotated
    for name in rotated:
        assert storage_by_name[name] == "source_hardlink"
    admission = second_manifest["admission"]
    assert admission["reused_shard_count"] == len(survivors)
    assert admission["source_linked_shard_count"] == len(rotated)
    assert admission["copied_shard_count"] == 0
    assert admission["reused_shard_bytes"] > 0
    assert admission["source_linked_shard_bytes"] > 0

    # Inode identity: reuse shares data instead of duplicating it.
    prev_dir = first.manifest_path.parent / "files"
    curr_dir = second.manifest_path.parent / "files"
    for name in survivors:
        assert (
            prev_dir.joinpath(name).stat().st_ino
            == curr_dir.joinpath(name).stat().st_ino
        ), f"{name} was copied, not hardlinked"
    for name in rotated:
        assert (
            replay_dir.joinpath(name).stat().st_ino
            == curr_dir.joinpath(name).stat().st_ino
        ), f"{name} was copied instead of linked from replay"

    # Replay-window rotation unlinks only one name. The snapshot's immutable
    # inode remains complete and loadable through its own link.
    replay_dir.joinpath(next(iter(rotated))).unlink()

    # The reused snapshot still loads through full integrity verification.
    train_entries, _validation, _manifest = manager.load_split()
    assert train_entries


def test_snapshot_commits_shard_directory_before_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nested files/ transaction must precede snapshot publication."""
    manager = _shard_reuse_manager(tmp_path)
    replay_dir = tmp_path / "replay"
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(
            replay_dir / f"replay_00_{index}.jsonl", [_entry(index)]
        )

    shard_directory_committed = False
    activation_observed = False
    real_fsync_directory = corpus.run_status._fsync_directory
    real_replace = corpus.os.replace

    def tracking_fsync_directory(path: Path) -> None:
        nonlocal shard_directory_committed
        if Path(path).name == "files":
            shard_directory_committed = True
        real_fsync_directory(path)

    def tracking_replace(source, destination) -> None:
        nonlocal activation_observed
        destination_path = Path(destination)
        if (
            destination_path.parent == snapshot_root
            and destination_path.name.startswith("snapshot_v")
        ):
            assert shard_directory_committed
            activation_observed = True
        real_replace(source, destination)

    monkeypatch.setattr(
        corpus.run_status, "_fsync_directory", tracking_fsync_directory
    )
    monkeypatch.setattr(corpus.os, "replace", tracking_replace)

    decision = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)

    assert decision.admitted
    assert shard_directory_committed
    assert activation_observed
    assert manager.load_split(decision.manifest_path)[0]


def test_shard_directory_commit_failure_cannot_activate_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed files/ commit leaves no pointer and permits a clean retry."""
    manager = _shard_reuse_manager(tmp_path)
    replay_dir = tmp_path / "replay"
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(
            replay_dir / f"replay_00_{index}.jsonl", [_entry(index)]
        )

    real_fsync_directory = corpus.run_status._fsync_directory
    failed = False

    def fail_first_shard_directory_commit(path: Path) -> None:
        nonlocal failed
        if Path(path).name == "files" and not failed:
            failed = True
            raise OSError(5, "simulated shard directory sync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(
        corpus.run_status,
        "_fsync_directory",
        fail_first_shard_directory_commit,
    )

    with pytest.raises(OSError, match="shard directory sync failure"):
        manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)

    assert failed
    assert not manager.current_pointer.exists()
    assert not (snapshot_root / "current.json").exists()
    assert not list(snapshot_root.glob("snapshot_v*"))
    assert not list(snapshot_root.glob(".snapshot_v*"))

    retry = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)
    assert retry.admitted
    assert manager.load_split(retry.manifest_path)[0]


def test_snapshot_reuse_disabled_copies_everything(tmp_path: Path) -> None:
    """reuse_previous_shards=False restores the copy-everything behaviour."""
    manager = _shard_reuse_manager(tmp_path, reuse_previous_shards=False)
    replay_dir = tmp_path / "replay"
    for index in range(4):
        _write_replay(replay_dir / f"replay_00_{index}.jsonl", [_entry(index)])
    first = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)
    assert first.admitted

    for offset in range(4):
        _write_replay(
            replay_dir / f"replay_01_{offset}.jsonl",
            [_entry(100 + offset)],
        )
    second = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)
    assert second.admitted
    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert all(record["storage"] == "copy" for record in manifest["files"])
    assert manifest["admission"]["reused_shard_count"] == 0


def test_snapshot_reuse_falls_back_to_copy_when_link_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filesystem refusal (cross-device, permissions) degrades to copy."""
    manager = _shard_reuse_manager(tmp_path)
    replay_dir = tmp_path / "replay"
    for index in range(4):
        _write_replay(replay_dir / f"replay_00_{index}.jsonl", [_entry(index)])
    first = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)
    assert first.admitted

    def refuse_link(src, dst):
        raise OSError("simulated cross-device link")

    monkeypatch.setattr(
        "dama.ai.ml.corpus.os.link", lambda s, d: refuse_link(s, d)
    )

    for offset in range(4):
        _write_replay(
            replay_dir / f"replay_01_{offset}.jsonl",
            [_entry(100 + offset)],
        )
    second = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)
    assert second.admitted
    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert all(record["storage"] == "copy" for record in manifest["files"])
    assert manifest["admission"]["reused_shard_count"] == 0
    train_entries, _validation, _manifest = manager.load_split()
    assert train_entries


def test_snapshot_copy_fallback_fsyncs_complete_shard_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copied inode must be durable before a manifest can authorize it."""
    source = tmp_path / "replay_source.jsonl"
    destination = tmp_path / "stored" / "replay_source.jsonl"
    destination.parent.mkdir()
    payload = json.dumps(_entry(7), sort_keys=True).encode("utf-8") + b"\n"
    source.write_bytes(payload)
    fsync_sizes = []
    real_fsync = os.fsync

    def tracking_fsync(fd: int) -> None:
        fsync_sizes.append(os.fstat(fd).st_size)
        real_fsync(fd)

    monkeypatch.setattr(corpus.os, "fsync", tracking_fsync)

    assert corpus._store_shard(source, destination) == "copy"
    assert fsync_sizes == [len(payload)]
    assert destination.read_bytes() == payload


def test_snapshot_copy_fsync_failure_removes_uncommitted_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed copied-inode sync cannot leave an admissible destination."""
    source = tmp_path / "replay_source.jsonl"
    destination = tmp_path / "stored" / "replay_source.jsonl"
    destination.parent.mkdir()
    payload = json.dumps(_entry(8), sort_keys=True).encode("utf-8") + b"\n"
    source.write_bytes(payload)

    def fail_fsync(_fd: int) -> None:
        raise OSError(5, "simulated copied shard sync failure")

    monkeypatch.setattr(corpus.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="copied shard sync failure"):
        corpus._store_shard(source, destination)

    assert not destination.exists()


def test_snapshot_source_hardlink_fails_closed_on_identity_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source replacement around link publication cannot enter a snapshot."""
    from dama.ai.ml import corpus

    source = tmp_path / "replay_race.jsonl"
    destination = tmp_path / "stored.jsonl"
    payload = json.dumps(_entry(1)).encode("utf-8") + b"\n"
    source.write_bytes(payload)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    real_link = os.link

    def link_then_replace(src, dst):
        real_link(src, dst)
        replacement = source.with_suffix(".replacement")
        replacement.write_bytes(payload + b" ")
        os.replace(replacement, source)

    monkeypatch.setattr(corpus.os, "link", link_then_replace)
    with pytest.raises(RuntimeError, match="changed during snapshot storage"):
        corpus._store_shard(
            source,
            destination,
            hardlink_source=True,
            expected_size=len(payload),
            expected_sha256=expected_sha256,
        )
    assert not destination.exists()


def test_snapshot_reuse_avoids_corrupted_predecessor(
    tmp_path: Path,
) -> None:
    """A mutated predecessor is bypassed for the verified live source.

    This is the fail-closed half of the contract.  Deliberately corrupts the
    OLD snapshot's stored shard (same size) to prove the digest gate notices;
    the old snapshot's integrity is intentionally sacrificed by the scenario.
    """
    manager = _shard_reuse_manager(tmp_path)
    replay_dir = tmp_path / "replay"
    sources = {}
    for index in range(4):
        path = replay_dir / f"replay_00_{index}.jsonl"
        _write_replay(path, [_entry(index)])
        sources[path.name] = path.read_bytes()

    first = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)
    assert first.admitted
    prev_files = first.manifest_path.parent / "files"

    # Corrupt one predecessor shard in place at equal size.
    victim = prev_files / "replay_00_0.jsonl"
    original = bytearray(victim.read_bytes())
    original[0] = original[0] ^ 0xFF
    victim.write_bytes(bytes(original))

    # Live corpus keeps pristine bytes under replacement inodes. The first
    # snapshot remains intentionally corrupt instead of mutating through its
    # hardlink a second time.
    for name, payload in sources.items():
        replacement = replay_dir / f".{name}.replacement"
        replacement.write_bytes(payload)
        os.replace(replacement, replay_dir / name)

    for offset in range(4):
        _write_replay(
            replay_dir / f"replay_01_{offset}.jsonl",
            [_entry(100 + offset)],
        )
    second = manager.consider_snapshot(_SETTINGS, _NOISE, _GENERATION)
    assert second.admitted
    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    storage = {r["name"]: r["storage"] for r in manifest["files"]}
    first_names = {
        r["name"] for r in
        json.loads(first.manifest_path.read_text(encoding="utf-8"))["files"]
    }
    survivors = first_names & set(storage)
    assert "replay_00_0.jsonl" not in survivors or (
        storage.get("replay_00_0.jsonl") == "source_hardlink"
    )
    # The corrupted shard must never be linked; any surviving intact sibling
    # from the previous snapshot should be.
    for name in survivors:
        if name == "replay_00_0.jsonl":
            continue
        assert storage[name] == "hardlink", (
            f"{name} was {storage[name]}, expected hardlink for intact shard"
        )
