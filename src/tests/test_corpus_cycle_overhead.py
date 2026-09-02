"""[Pass 182] Per-cycle corpus admission overhead contracts.

The replay digest and audit caches must hold a whole admission working set.

One ``consider_snapshot()`` touches every replay shard in the rolling window
plus every held-out shard (61 + 27 = 88 identities on the c174k window), in
the same order every cycle.  Under the shared 64-entry LRU that cyclic access
pattern evicted every entry before its next use, so each self-play cycle
re-hashed all 88 shards: 1.33 GB through drvfs, 12.3 s of a 12.9 s admission
check.  Digests and audits are a few hundred bytes each; only the per-file
analyses (every canonical state key of a shard, ~6 MB each) stay tightly
bounded.  These tests pin that a cyclic sweep wider than the analysis bound
never re-hashes or re-audits an unchanged file, that the analysis cache still
honours its own bound, and that a replaced file still evicts its stale entry.
"""

import errno
import json
import os
from pathlib import Path
import threading

import pytest

from dama.ai.ml import corpus


def _state(index: int) -> dict:
    row = (index // 4) % 8
    col = (index * 2 + 1 - (row % 2)) % 8
    return {
        "p1_men": [[row, col]],
        "p1_kings": [],
        "p2_men": [[7 - row, 7 - col]],
        "p2_kings": [],
        "turn": 1,
        "move_count": index,
    }


def _write_shard(path: Path, index: int) -> None:
    """Ten complete, contract-valid trajectories per shard (exact 70/30 split)."""
    lines = []
    for game in range(10):
        source = "algorithm" if game < 7 else "current_model"
        entry = {
            "state": _state(index * 10 + game),
            "legal_moves": [
                {"path": [[0, 1], [1, 0]], "captures": [], "promotion": False},
                {"path": [[0, 1], [1, 2]], "captures": [], "promotion": False},
            ],
            "chosen_index": 0,
            "played_index": 0,
            "result": 0,
            "trajectory_source": source,
            "was_exploration": False,
            "teacher_difficulty": "hard",
            "opening_plies": 2,
            "game_id": f"cycle-{index}-game-{game}",
        }
        lines.append(json.dumps(entry, sort_keys=True) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


@pytest.fixture
def shards(tmp_path: Path):
    corpus._clear_replay_file_cache()
    count = corpus._REPLAY_FILE_CACHE_MAX + 24  # wider than the analysis bound
    paths = []
    for index in range(count):
        path = tmp_path / f"replay_{index:04d}.jsonl"
        _write_shard(path, index)
        paths.append(path)
    yield paths
    corpus._clear_replay_file_cache()


def _spy(monkeypatch, name):
    calls = []
    real = getattr(corpus, name)

    def wrapper(*args, **kwargs):
        calls.append(args[0] if args else None)
        return real(*args, **kwargs)

    monkeypatch.setattr(corpus, name, wrapper)
    return calls


def test_digest_bound_covers_window_holdout_and_snapshot_verification():
    # 60-file window + 27-file hold-out ceiling + a 60-file snapshot
    # verification at startup, with headroom for a larger server window.
    assert corpus._REPLAY_DIGEST_CACHE_MAX >= 4 * corpus._REPLAY_FILE_CACHE_MAX
    assert corpus._REPLAY_DIGEST_CACHE_MAX >= 60 + 27 + 60
    assert corpus._REPLAY_IDENTITY_MAP_MAX >= corpus._REPLAY_DIGEST_CACHE_MAX


def test_replay_scan_reuses_metadata_and_preserves_mtime_name_order(tmp_path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    newest = replay_dir / "replay_z.jsonl"
    tied_b = replay_dir / "replay_b.jsonl"
    tied_a = replay_dir / "replay_a.jsonl"
    for path in (newest, tied_b, tied_a):
        path.write_text("{}\n", encoding="utf-8")
    os.utime(tied_a, ns=(10, 10))
    os.utime(tied_b, ns=(10, 10))
    os.utime(newest, ns=(20, 20))
    (replay_dir / "replay_directory.jsonl").mkdir()
    (replay_dir / "other.jsonl").write_text("{}\n", encoding="utf-8")

    manager = corpus.CorpusSnapshotManager(
        str(replay_dir), str(tmp_path / "snapshots"))
    files, identities = manager._replay_files_with_identities()

    assert files == [tied_a, tied_b, newest]
    assert manager.replay_files() == files
    assert set(identities) == set(files)
    for path, identity in identities.items():
        assert identity == corpus._replay_file_identity(path)


def test_external_validation_exclusion_avoids_full_union() -> None:
    """The small frozen suite is removed without copying the large hold-out."""

    class UnionForbiddenSet(set):
        def union(self, *_others):
            raise AssertionError("validation exclusion materialized a full union")

    state_keys = {"train", "held", "frozen", "both"}
    validation_keys = UnionForbiddenSet({"held", "both"})
    external_keys = {"frozen", "both"}

    remaining = corpus._exclude_validation_state_keys(
        state_keys, validation_keys, external_keys)
    overlap_count = corpus._validation_overlap_state_count(
        state_keys, validation_keys, external_keys)

    assert remaining == {"train"}
    assert overlap_count == 3
    assert state_keys == {"train", "held", "frozen", "both"}
    assert validation_keys == {"held", "both"}
    assert external_keys == {"frozen", "both"}


def test_large_directory_metadata_reads_overlap_and_preserve_order(monkeypatch):
    """The drvfs identity batch is bounded, parallel, and deterministic."""
    monkeypatch.setattr(
        corpus, "_PARALLEL_METADATA_STAT_MIN_PROBE_NS", 0)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    peak_active = 0

    class Entry:
        def __init__(self, index):
            self.index = index

        def stat(self):
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                if self.index in (4, 5):
                    barrier.wait(timeout=2.0)
                return self.index
            finally:
                with lock:
                    active -= 1

    entries = [
        Entry(index)
        for index in range(corpus._PARALLEL_METADATA_STAT_MIN_FILES)
    ]
    results = corpus._directory_entry_stats(entries)

    assert peak_active >= 2
    assert [entry.index for entry, _result, _error in results] == list(
        range(len(entries)))
    assert [result for _entry, result, _error in results] == list(
        range(len(entries)))
    assert all(error is None for _entry, _result, error in results)


def test_fast_directory_metadata_batch_stays_synchronous(monkeypatch):
    """Fast server filesystems do not pay thread-pool setup per scan."""
    monkeypatch.setattr(
        corpus, "_PARALLEL_METADATA_STAT_MIN_PROBE_NS", 10**18)
    caller = threading.get_ident()

    class Entry:
        def stat(self):
            return threading.get_ident()

    entries = [
        Entry()
        for _index in range(corpus._PARALLEL_METADATA_STAT_MIN_FILES)
    ]
    results = corpus._directory_entry_stats(entries)

    assert [result for _entry, result, _error in results] == [
        caller
    ] * len(entries)


def test_directory_metadata_strategy_is_reused_per_parent(monkeypatch):
    """Only the first slow scan spends serial calls on latency detection."""
    corpus._clear_replay_file_cache()
    monkeypatch.setattr(
        corpus, "_PARALLEL_METADATA_STAT_MIN_PROBE_NS", 0)
    caller = threading.get_ident()

    class Entry:
        def __init__(self, index):
            self.path = f"/slow-metadata/replay_{index}.jsonl"

        def stat(self):
            return threading.get_ident()

    entries = [
        Entry(index)
        for index in range(corpus._PARALLEL_METADATA_STAT_MIN_FILES)
    ]
    first = corpus._directory_entry_stats(entries)
    second = corpus._directory_entry_stats(entries)

    assert [result for _entry, result, _error in first][
        :corpus._METADATA_STAT_PROBE_FILES
    ] == [caller] * corpus._METADATA_STAT_PROBE_FILES
    assert all(
        result != caller for _entry, result, _error in second)
    corpus._clear_replay_file_cache()


@pytest.mark.skipif(
    not corpus._HAS_FAST_STAT,
    reason="compiled POSIX metadata accelerator is not built",
)
def test_native_metadata_batch_matches_stat_and_preserves_errors(
    tmp_path, monkeypatch,
):
    """The compiled batch returns exact stat fields, order, and errno."""

    paths = []
    for index in range(12):
        path = tmp_path / f"replay_{index:02d}.jsonl"
        path.write_text(f"{index}\n", encoding="utf-8")
        paths.append(path)
    with os.scandir(tmp_path) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    missing_entry = entries[-1]
    paths[-1].unlink()

    real_fast_stat = corpus._fast_stat_paths
    calls = []

    def tracked_fast_stat(raw_paths, workers):
        calls.append((list(raw_paths), workers))
        return real_fast_stat(raw_paths, workers)

    monkeypatch.setattr(corpus, "_fast_stat_paths", tracked_fast_stat)
    monkeypatch.setattr(corpus, "_FAST_METADATA_STAT_ENABLED", True)
    results = corpus._parallel_directory_entry_stats(entries, workers=8)

    assert calls == [([entry.path for entry in entries], len(entries))]
    assert [entry for entry, _result, _error in results] == entries
    for entry, result, error in results[:-1]:
        expected = os.stat(entry.path)
        assert error is None
        assert result == corpus._FastStatResult(
            st_mode=expected.st_mode,
            st_dev=expected.st_dev,
            st_ino=expected.st_ino,
            st_size=expected.st_size,
            st_mtime_ns=expected.st_mtime_ns,
        )
    entry, result, error = results[-1]
    assert entry is missing_entry
    assert result is None
    assert isinstance(error, FileNotFoundError)
    assert error.errno == errno.ENOENT


@pytest.mark.skipif(
    not corpus._HAS_FAST_STAT,
    reason="compiled POSIX metadata accelerator is not built",
)
def test_native_metadata_helpers_are_joined_before_return(tmp_path):
    """No native helper thread may survive into the next self-play fork."""

    task_dir = Path("/proc/self/task")
    if not task_dir.is_dir():
        pytest.skip("Linux task accounting is unavailable")
    paths = []
    for index in range(40):
        path = tmp_path / f"replay_{index:02d}.jsonl"
        path.write_text(f"{index}\n", encoding="utf-8")
        paths.append(path)

    before = {path.name for path in task_dir.iterdir()}
    for _round in range(8):
        results = corpus._fast_stat_paths(paths, 32)
        assert len(results) == len(paths)
        assert all(fields is not None and error == 0
                   for fields, error in results)
    after = {path.name for path in task_dir.iterdir()}

    assert after == before


@pytest.mark.skipif(
    not corpus._HAS_FAST_STAT,
    reason="compiled POSIX metadata accelerator is not built",
)
def test_native_identity_recheck_stats_exact_paths_without_rescan(
    tmp_path, monkeypatch,
):
    """A proven-slow final transaction needs no second directory listing."""

    paths = []
    for index in range(12):
        path = tmp_path / f"replay_{index:02d}.jsonl"
        path.write_text(f"{index}\n", encoding="utf-8")
        paths.append(path)
    identities = {path: corpus._replay_file_identity(path) for path in paths}
    parent = os.path.abspath(str(tmp_path))
    monkeypatch.setitem(
        corpus._METADATA_STAT_PARALLEL_BY_PARENT, parent, True)
    real_fast_stat = corpus._fast_stat_paths
    calls = []

    def tracked_fast_stat(raw_paths, workers):
        calls.append((list(raw_paths), workers))
        return real_fast_stat(raw_paths, workers)

    monkeypatch.setattr(corpus, "_fast_stat_paths", tracked_fast_stat)
    monkeypatch.setattr(
        corpus.os,
        "scandir",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("native exact-path verification rescanned a directory")
        ),
    )

    corpus.CorpusSnapshotManager._verify_replay_file_identities(identities)

    assert calls == [([os.fspath(path) for path in paths], len(paths))]


@pytest.mark.skipif(
    not corpus._HAS_FAST_STAT,
    reason="compiled POSIX metadata accelerator is not built",
)
def test_native_identity_recheck_fails_closed_on_missing_or_changed_path(
    tmp_path, monkeypatch,
):
    """Direct native stats retain deletion and replacement rejection."""

    paths = []
    for index in range(12):
        path = tmp_path / f"replay_{index:02d}.jsonl"
        path.write_text(f"before-{index}\n", encoding="utf-8")
        paths.append(path)
    identities = {path: corpus._replay_file_identity(path) for path in paths}
    monkeypatch.setitem(
        corpus._METADATA_STAT_PARALLEL_BY_PARENT,
        os.path.abspath(str(tmp_path)),
        True,
    )

    paths[-1].unlink()
    with pytest.raises(RuntimeError, match="disappeared during corpus analysis"):
        corpus.CorpusSnapshotManager._verify_replay_file_identities(identities)

    paths[-1].write_text("replacement-is-longer\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed during corpus analysis"):
        corpus.CorpusSnapshotManager._verify_replay_file_identities(identities)


@pytest.mark.skipif(
    not corpus._HAS_FAST_STAT,
    reason="compiled POSIX metadata accelerator is not built",
)
def test_native_identity_recheck_falls_back_when_accelerator_fails(
    tmp_path, monkeypatch,
):
    """An optional native failure retains the exact directory verifier."""

    paths = []
    for index in range(12):
        path = tmp_path / f"replay_{index:02d}.jsonl"
        path.write_text(f"{index}\n", encoding="utf-8")
        paths.append(path)
    identities = {path: corpus._replay_file_identity(path) for path in paths}
    monkeypatch.setitem(
        corpus._METADATA_STAT_PARALLEL_BY_PARENT,
        os.path.abspath(str(tmp_path)),
        True,
    )
    monkeypatch.setattr(
        corpus,
        "_fast_stat_paths",
        lambda _paths, _workers: (_ for _ in ()).throw(
            RuntimeError("synthetic native failure")
        ),
    )

    corpus.CorpusSnapshotManager._verify_replay_file_identities(identities)


@pytest.mark.skipif(
    not corpus._HAS_FAST_STAT,
    reason="compiled POSIX metadata accelerator is not built",
)
def test_native_identity_recheck_stays_serial_on_fast_parent(
    tmp_path, monkeypatch,
):
    """A low-latency server filesystem does not pay native thread setup."""

    paths = []
    for index in range(12):
        path = tmp_path / f"replay_{index:02d}.jsonl"
        path.write_text(f"{index}\n", encoding="utf-8")
        paths.append(path)
    identities = {path: corpus._replay_file_identity(path) for path in paths}
    monkeypatch.setitem(
        corpus._METADATA_STAT_PARALLEL_BY_PARENT,
        os.path.abspath(str(tmp_path)),
        False,
    )
    calls = []
    real_fast_stat = corpus._fast_stat_paths

    def tracked_fast_stat(raw_paths, workers):
        calls.append((list(raw_paths), workers))
        return real_fast_stat(raw_paths, workers)

    monkeypatch.setattr(corpus, "_fast_stat_paths", tracked_fast_stat)

    corpus.CorpusSnapshotManager._verify_replay_file_identities(identities)

    assert calls == []


def test_admission_identity_snapshot_recheck_fails_closed_on_replacement(tmp_path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    path = replay_dir / "replay_a.jsonl"
    path.write_text("before\n", encoding="utf-8")
    manager = corpus.CorpusSnapshotManager(
        str(replay_dir), str(tmp_path / "snapshots"))
    files, identities = manager._replay_files_with_identities()
    manager._verify_replay_file_identities(identities)

    replacement = replay_dir / "replacement.tmp"
    replacement.write_text("after\n", encoding="utf-8")
    replacement.replace(path)

    with pytest.raises(RuntimeError, match="changed during corpus analysis"):
        manager._verify_replay_file_identities(identities)


def test_admission_identity_recheck_scans_each_parent_once(
    tmp_path, monkeypatch,
):
    directories = [tmp_path / "left", tmp_path / "right"]
    paths = []
    for directory in directories:
        directory.mkdir()
        for index in range(3):
            path = directory / f"replay_{index}.jsonl"
            path.write_text(f"{directory.name}-{index}\n", encoding="utf-8")
            paths.append(path)
    identities = {path: corpus._replay_file_identity(path) for path in paths}

    real_scandir = os.scandir
    scanned = []

    def tracked_scandir(parent):
        scanned.append(Path(parent))
        return real_scandir(parent)

    def forbid_individual_stat(_path):
        raise AssertionError("identity recheck issued a per-file Path.stat")

    monkeypatch.setattr(corpus.os, "scandir", tracked_scandir)
    monkeypatch.setattr(corpus, "_replay_file_identity", forbid_individual_stat)

    corpus.CorpusSnapshotManager._verify_replay_file_identities(identities)

    assert scanned == directories


def test_admission_identity_recheck_fails_closed_on_disappearance(tmp_path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    path = replay_dir / "replay_a.jsonl"
    path.write_text("before\n", encoding="utf-8")
    identity = corpus._replay_file_identity(path)
    path.unlink()

    with pytest.raises(RuntimeError, match="disappeared during corpus analysis"):
        corpus.CorpusSnapshotManager._verify_replay_file_identities(
            {path: identity})


def test_cyclic_digest_sweep_wider_than_analysis_bound_never_rehashes(
        shards, monkeypatch):
    hashes = _spy(monkeypatch, "_sha256_file_uncached")
    first = [corpus.replay_file_sha256(path) for path in shards]
    assert len(hashes) == len(shards)
    for _sweep in range(3):
        again = [corpus.replay_file_sha256(path) for path in shards]
        assert again == first
    assert len(hashes) == len(shards), "an unchanged shard was re-hashed"
    assert len(corpus._REPLAY_HASH_CACHE) == len(shards)


def test_cyclic_audit_sweep_wider_than_analysis_bound_never_reparses(
        shards, monkeypatch):
    scans = _spy(monkeypatch, "_scan_replay_file_analysis")
    first = [corpus.audit_policy_replay_file(path, (2, 4, 6, 8)) for path in shards]
    assert len(scans) == len(shards)
    for _sweep in range(3):
        again = [corpus.audit_policy_replay_file(path, (2, 4, 6, 8))
                 for path in shards]
        assert again == first
    assert len(scans) == len(shards), "an unchanged shard was re-audited"
    assert len(corpus._REPLAY_AUDIT_CACHE) == len(shards)
    # A valid cold audit also supplies analysis and digest facts.  Their
    # independent bounds remain unchanged.
    assert len(corpus._REPLAY_ANALYSIS_CACHE) == corpus._REPLAY_FILE_CACHE_MAX
    assert len(corpus._REPLAY_HASH_CACHE) == len(shards)


def test_analysis_cache_keeps_its_own_tight_bound(shards, monkeypatch):
    reads = _spy(monkeypatch, "_read_replay_file_analysis")
    for path in shards:
        corpus._cached_replay_file_analysis(path)
    assert len(reads) == len(shards)
    assert len(corpus._REPLAY_ANALYSIS_CACHE) == corpus._REPLAY_FILE_CACHE_MAX
    # The digests those analyses produced are all retained, even though the
    # analyses themselves were evicted.
    assert len(corpus._REPLAY_HASH_CACHE) == len(shards)
    hashes = _spy(monkeypatch, "_sha256_file_uncached")
    for path in shards:
        corpus.replay_file_sha256(path)
    assert hashes == []


def test_uniform_cycle_analysis_stores_one_shard_level_id(shards):
    """Normal ReplayWriter shards do not retain one cycle set per state."""
    analysis = corpus._cached_replay_file_analysis(shards[0])
    assert analysis.uniform_generation_cycle == "0"
    assert analysis.state_counts
    assert analysis.state_cycles == {}


def test_mixed_cycle_analysis_retains_exact_per_state_provenance(tmp_path):
    """A genuinely mixed or legacy shard still takes the exact general path."""
    path = tmp_path / "replay_mixed.jsonl"

    def entry(state, game_id):
        return {
            "state": state,
            "legal_moves": [
                {"path": [[0, 1], [1, 0]], "captures": [], "promotion": False}
            ],
            "game_id": game_id,
            "trajectory_source": "legacy",
        }

    path.write_text(
        "".join(
            json.dumps(value) + "\n"
            for value in (
                entry(_state(0), "cycle-a-game-0"),
                entry(_state(1), None),
                entry(_state(0), "cycle-b-game-1"),
            )
        ),
        encoding="utf-8",
    )
    analysis = corpus._read_replay_file_analysis(
        path, corpus._replay_file_identity(path).as_key())
    key_a = corpus.canonical_state_key(_state(0))
    key_without_provenance = corpus.canonical_state_key(_state(1))

    assert analysis.uniform_generation_cycle is None
    assert analysis.state_cycles[key_a] == frozenset({"a", "b"})
    assert key_without_provenance not in analysis.state_cycles


def test_single_cycle_with_missing_provenance_keeps_observed_keys(tmp_path):
    """A legacy state without an id prevents the uniform-shard proof only."""
    path = tmp_path / "replay_partial_cycle.jsonl"

    def entry(state, game_id):
        return {
            "state": state,
            "legal_moves": [
                {"path": [[0, 1], [1, 0]], "captures": [], "promotion": False}
            ],
            "game_id": game_id,
        }

    path.write_text(
        json.dumps(entry(_state(0), "cycle-a-game-0"))
        + "\n"
        + json.dumps(entry(_state(1), None))
        + "\n",
        encoding="utf-8",
    )
    analysis = corpus._read_replay_file_analysis(
        path, corpus._replay_file_identity(path).as_key())
    key_a = corpus.canonical_state_key(_state(0))
    key_without_provenance = corpus.canonical_state_key(_state(1))

    assert analysis.uniform_generation_cycle is None
    assert analysis.state_cycles == {key_a: frozenset({"a"})}
    assert key_without_provenance not in analysis.state_cycles


def test_replaced_file_still_evicts_its_stale_entries(shards, monkeypatch):
    target = shards[0]
    before = corpus.replay_file_sha256(target)
    audit_before = corpus.audit_policy_replay_file(target, (2, 4, 6, 8))
    assert audit_before["valid"]
    hashes = _spy(monkeypatch, "_sha256_file_uncached")
    # Atomic replacement, as a rewritten shard would be: new inode, new bytes.
    temp = target.with_suffix(".tmp")
    temp.write_text('{"state": {}}\n', encoding="utf-8")
    temp.replace(target)
    after = corpus.replay_file_sha256(target)
    assert after != before
    assert hashes == [target]
    assert not corpus.audit_policy_replay_file(target, (2, 4, 6, 8))["valid"]
    path_key = corpus._replay_file_identity(target).resolved_path
    identities = [key for key in corpus._REPLAY_HASH_CACHE if key[0] == path_key]
    assert len(identities) == 1, "the stale generation must not linger"


def test_analyze_replay_files_digest_opt_out_changes_only_the_digest(shards):
    files = shards[:6]
    with_digest, keys = corpus.analyze_replay_files(files, set())
    without, keys_again = corpus.analyze_replay_files(
        files, set(), include_state_digest=False)
    assert keys == keys_again
    assert with_digest["state_set_sha256"] == corpus._state_set_digest(keys)
    assert without["state_set_sha256"] is None
    without.pop("state_set_sha256")
    with_digest.pop("state_set_sha256")
    assert without == with_digest


def test_admitted_snapshot_fingerprints_the_post_dedup_training_keys(tmp_path):
    """consider_snapshot() must still record and verify the digest of the
    deduplicated training keys after opting out of the pre-dedup digest."""
    corpus._clear_replay_file_cache()
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(6):
        _write_shard(replay_dir / f"replay_{index:04d}.jsonl", index)
    manager = corpus.CorpusSnapshotManager(
        str(replay_dir), str(tmp_path / "snapshots"),
        validation_fraction=0.2, split_seed=3, min_fresh_fraction=0.5,
        enforce_policy_contract=True, allowed_opening_plies=(2, 4, 6, 8),
    )
    decision = manager.consider_snapshot(
        teacher_settings={"difficulty": "hard"},
        noise_settings={"played_action_probability": 0.1},
        generation_settings={"algorithm_fraction": 0.7, "model_fraction": 0.3},
    )
    assert decision.admitted, decision.reason
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    stored_keys = corpus._read_state_keys(
        decision.manifest_path.parent / manifest["state_keys_file"])
    assert manifest["metrics"]["state_set_sha256"] == corpus._state_set_digest(stored_keys)
    assert manifest["metrics"]["state_set_sha256"] is not None
    # The integrity verification that every load performs re-derives it.
    train, validation, loaded = manager.load_split(decision.manifest_path)
    assert train and validation
    corpus._clear_replay_file_cache()


def test_rejected_snapshot_defers_digest_but_unchanged_reuses_verified_digest(
    tmp_path, monkeypatch,
):
    """Only a changed candidate below the freshness gate may omit its digest."""
    corpus._clear_replay_file_cache()
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_shard(replay_dir / f"replay_{index:04d}.jsonl", index)
    manager = corpus.CorpusSnapshotManager(
        str(replay_dir), str(tmp_path / "snapshots"),
        validation_fraction=0.25, split_seed=3, min_fresh_fraction=0.5,
        enforce_policy_contract=True, allowed_opening_plies=(2, 4, 6, 8),
        grow_holdout=False,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.1}
    generation = {"algorithm_fraction": 0.7, "model_fraction": 0.3}

    admitted = manager.consider_snapshot(settings, noise, generation)
    assert admitted.admitted
    # Populate both immutable key-file cache entries before forbidding another
    # sorted digest.  The next unchanged decision must reuse the verified one.
    assert manager.consider_snapshot(settings, noise, generation).reason == "unchanged"
    expected_digest = admitted.metrics["state_set_sha256"]

    def unexpected_digest(_keys):
        raise AssertionError("rejected or unchanged candidate sorted its state keys")

    monkeypatch.setattr(corpus, "_state_set_digest", unexpected_digest)
    unchanged = manager.consider_snapshot(settings, noise, generation)
    assert unchanged.reason == "unchanged"
    assert unchanged.metrics["state_set_sha256"] == expected_digest

    changed = replay_dir / "replay_9999.jsonl"
    _write_shard(changed, 9999)
    rows = [json.loads(line) for line in changed.read_text().splitlines()]
    for row in rows:
        row["state"]["p1_kings"] = row["state"].pop("p1_men")
    changed.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    rejected = manager.consider_snapshot(settings, noise, generation)
    assert not rejected.admitted
    assert "below" in rejected.reason
    assert rejected.metrics["state_set_sha256"] is None
    corpus._clear_replay_file_cache()


def test_manifest_key_file_cache_reuses_and_invalidates_by_identity(
    tmp_path, monkeypatch,
):
    """Warm admissions keep two immutable key members, never stale bytes."""
    manager = corpus.CorpusSnapshotManager(
        str(tmp_path / "replay"), str(tmp_path / "snapshots"))
    key_file = tmp_path / "canonical_state_keys.txt.gz"
    corpus._write_state_keys(key_file, {"a", "b"})

    real_read = corpus._read_state_keys
    reads = []

    def tracked_read(path):
        reads.append(Path(path))
        return real_read(path)

    monkeypatch.setattr(corpus, "_read_state_keys", tracked_read)
    first_keys, first_digest = manager._cached_state_key_file(key_file)
    warm_keys, warm_digest = manager._cached_state_key_file(key_file)
    assert warm_keys is first_keys
    assert warm_digest == first_digest == corpus._state_set_digest({"a", "b"})
    assert reads == [key_file]

    # Atomic replacement changes the identity.  The old pathname generation
    # must be evicted immediately rather than consuming one of the two slots.
    corpus._write_state_keys(key_file, {"c", "d", "e", "f"})
    replaced_keys, replaced_digest = manager._cached_state_key_file(key_file)
    assert replaced_keys == {"c", "d", "e", "f"}
    assert replaced_digest == corpus._state_set_digest(replaced_keys)
    assert reads == [key_file, key_file]
    assert len(manager._state_key_file_cache) == 1


def test_manifest_key_file_cache_is_bounded_to_active_split(tmp_path):
    """Historical snapshots cannot accumulate decompressed key sets in RAM."""
    manager = corpus.CorpusSnapshotManager(
        str(tmp_path / "replay"), str(tmp_path / "snapshots"))
    paths = []
    for index in range(corpus._STATE_KEY_FILE_CACHE_MAX + 1):
        path = tmp_path / f"keys_{index}.txt.gz"
        corpus._write_state_keys(path, {f"key-{index}"})
        manager._cached_state_key_file(path)
        paths.append(path)

    assert len(manager._state_key_file_cache) == corpus._STATE_KEY_FILE_CACHE_MAX
    cached_paths = {identity[0] for identity in manager._state_key_file_cache}
    assert str(paths[0].absolute()) not in cached_paths
    assert {str(path.absolute()) for path in paths[1:]} == cached_paths


def _reference_analyze(files, previous_state_keys=None, analyses=None):
    """The pre-Pass-182 merge, verbatim, as the exactness oracle."""
    from collections import Counter, defaultdict
    previous = previous_state_keys or set()
    unique_keys = set()
    state_counts = Counter()
    state_files = defaultdict(set)
    state_cycles = defaultdict(set)
    source_counts = Counter()
    game_sources = {}
    forced = malformed = total = 0
    for path in files:
        path = Path(path)
        analysis = analyses[path]
        total += analysis.records
        malformed += analysis.malformed_records
        forced += analysis.forced_move_count
        source_counts.update(analysis.source_counts)
        game_sources.update(analysis.game_sources)
        for key, count in analysis.state_counts.items():
            state_counts[key] += count
            unique_keys.add(key)
            state_files[key].add(path.name)
        if analysis.uniform_generation_cycle is not None:
            for key in analysis.state_counts:
                state_cycles[key].add(analysis.uniform_generation_cycle)
        else:
            for key, cycles in analysis.state_cycles.items():
                state_cycles[key].update(cycles)
    new_unique = unique_keys.difference(previous)
    fresh_records = sum(c for k, c in state_counts.items() if k not in previous)
    cross_file_states = sum(1 for n in state_files.values() if len(n) > 1)
    cross_file_unique_states = sum(1 for n in state_files.values() if len(n) == 1)
    cross_file_duplicate_records = sum(
        max(0, state_counts[k] - 1) for k, n in state_files.items() if len(n) > 1)
    observed = {k: c for k, c in state_cycles.items() if c}
    cross_cycle_states = sum(1 for c in observed.values() if len(c) > 1)
    cross_cycle_unique_states = sum(1 for c in observed.values() if len(c) == 1)
    metrics = {
        "records": total,
        "malformed_records": malformed,
        "unique_state_count": len(unique_keys),
        "unique_state_rate": (len(unique_keys) / total) if total else 0.0,
        "forced_move_count": forced,
        "forced_move_rate": (forced / total) if total else 0.0,
        "cross_file_repeated_state_count": cross_file_states,
        "cross_file_unique_state_count": cross_file_unique_states,
        "cross_file_duplicate_record_count": cross_file_duplicate_records,
        "cross_file_unique_state_rate": (
            cross_file_unique_states / len(unique_keys) if unique_keys else 0.0),
        "cross_cycle_observed_state_count": len(observed),
        "cross_cycle_repeated_state_count": cross_cycle_states,
        "cross_cycle_unique_state_count": cross_cycle_unique_states,
        "cross_cycle_unique_state_rate": (
            cross_cycle_unique_states / len(observed) if observed else 0.0),
        "new_unique_state_count": len(new_unique),
        "fresh_unique_state_rate": (len(new_unique) / len(unique_keys)) if unique_keys else 0.0,
        "fresh_record_rate": (fresh_records / total) if total else 0.0,
        "source_counts": dict(sorted(source_counts.items())),
        "source_game_counts": dict(sorted(Counter(game_sources.values()).items())),
        "state_set_sha256": corpus._state_set_digest(unique_keys),
    }
    return metrics, unique_keys


def _analysis(records, state_counts, state_cycles, sources, games, forced=0, malformed=0):
    frozen_cycles = {k: frozenset(v) for k, v in state_cycles.items()}
    cycle_ids = set().union(*frozen_cycles.values()) if frozen_cycles else set()
    return corpus._ReplayFileAnalysis(
        identity=("x",), sha256="0" * 64, records=records,
        malformed_records=malformed, forced_move_count=forced,
        state_counts=dict(state_counts),
        state_cycles=frozen_cycles,
        uniform_generation_cycle=(
            next(iter(cycle_ids))
            if len(cycle_ids) == 1
            and set(frozen_cycles) == set(state_counts)
            and all(frozen_cycles.values())
            else None
        ),
        source_counts=dict(sources), game_sources=dict(games),
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("basename_collisions", [False, True])
def test_merge_restructuring_is_exact_against_reference(
    monkeypatch, tmp_path, seed, basename_collisions,
):
    """Exercise both the unique-basename fast path and its exact fallback.

    Random file sets contain cross-file repeats, cross-cycle repeats, empty
    cycle sets, and a previous-key set.  The collision arm also includes
    repeated file names in a second directory and a duplicated path.
    """
    import random
    rng = random.Random(seed)
    keys = [f"k{i:03d}" for i in range(60)]
    analyses = {}
    files = []
    for index in range(12):
        directory = tmp_path / ("a" if index < 9 else "b")
        directory.mkdir(exist_ok=True)
        # In the collision arm, index 9..11 reuse names from a/ (same name,
        # different directory).  Otherwise every logical file name is unique.
        name_index = index if not basename_collisions or index < 9 else index - 9
        name = f"replay_{name_index:02d}.jsonl"
        path = directory / name
        chosen = rng.sample(keys, rng.randint(1, 25))
        counts = {k: rng.randint(1, 3) for k in chosen}
        cycles = {}
        for k in chosen:
            if rng.random() < 0.7:
                cycles[k] = {f"c{rng.randint(0, 5)}" for _ in range(rng.randint(0, 2))}
        sources = {"algorithm": rng.randint(0, 9), "current_model": rng.randint(0, 4)}
        games = {f"g{rng.randint(0, 30)}": rng.choice(["algorithm", "current_model"])
                 for _ in range(rng.randint(0, 6))}
        analyses[path] = _analysis(
            sum(counts.values()), counts, cycles, sources, games,
            forced=rng.randint(0, 5), malformed=rng.randint(0, 2))
        files.append(path)
    if basename_collisions:
        files.append(files[rng.randrange(len(files))])  # a duplicated path
    previous = set(rng.sample(keys, 20))

    monkeypatch.setattr(corpus, "_cached_replay_file_analysis", lambda p: analyses[Path(p)])
    expected_metrics, expected_keys = _reference_analyze(files, previous, analyses)
    metrics, result_keys = corpus.analyze_replay_files(files, previous)
    assert result_keys == expected_keys
    assert metrics == expected_metrics
    without, _ = corpus.analyze_replay_files(files, previous, include_state_digest=False)
    assert without["state_set_sha256"] is None
    without.pop("state_set_sha256")
    expected_metrics.pop("state_set_sha256")
    assert without == expected_metrics


def test_merge_shares_immutable_cycle_sets_without_aliasing_bugs(monkeypatch, tmp_path):
    shared = frozenset({"c1"})
    a = tmp_path / "replay_a.jsonl"
    b = tmp_path / "replay_b.jsonl"
    analyses = {
        a: _analysis(2, {"k": 1, "j": 1}, {"k": shared, "j": shared}, {}, {}),
        b: _analysis(1, {"k": 1}, {"k": {"c2"}}, {}, {}),
    }
    monkeypatch.setattr(corpus, "_cached_replay_file_analysis", lambda p: analyses[Path(p)])
    metrics, keys = corpus.analyze_replay_files([a, b])
    assert keys == {"k", "j"}
    assert metrics["cross_cycle_repeated_state_count"] == 1   # k saw c1 and c2
    assert metrics["cross_cycle_unique_state_count"] == 1     # j saw only c1
    assert metrics["cross_file_repeated_state_count"] == 1
    assert metrics["cross_file_unique_state_count"] == 1
    assert metrics["cross_file_duplicate_record_count"] == 1
    assert shared == frozenset({"c1"})  # the input frozenset was never mutated


def test_uniform_cycle_shortcut_falls_back_for_ambiguous_shards(monkeypatch, tmp_path):
    """Same-cycle shards and states without provenance keep exact semantics."""
    a = tmp_path / "replay_a.jsonl"
    b = tmp_path / "replay_b.jsonl"
    analyses = {
        a: _analysis(2, {"same": 1, "missing": 1}, {"same": {"c1"}}, {}, {}),
        b: _analysis(1, {"same": 1}, {"same": {"c1"}}, {}, {}),
    }
    monkeypatch.setattr(corpus, "_cached_replay_file_analysis", lambda p: analyses[Path(p)])

    metrics, keys = corpus.analyze_replay_files([a, b])

    assert keys == {"same", "missing"}
    assert metrics["cross_file_repeated_state_count"] == 1
    assert metrics["cross_cycle_observed_state_count"] == 1
    assert metrics["cross_cycle_repeated_state_count"] == 0
    assert metrics["cross_cycle_unique_state_count"] == 1


def test_compact_uniform_cycles_expand_in_same_cycle_fallback(monkeypatch, tmp_path):
    """Two uniform shards sharing an id remain one observed cycle per state."""
    a = tmp_path / "replay_a.jsonl"
    b = tmp_path / "replay_b.jsonl"
    analyses = {
        a: corpus._ReplayFileAnalysis(
            identity=("a",), sha256="0" * 64, records=2,
            malformed_records=0, forced_move_count=0,
            state_counts={"same": 1, "only-a": 1}, state_cycles={},
            uniform_generation_cycle="c1", source_counts={}, game_sources={},
        ),
        b: corpus._ReplayFileAnalysis(
            identity=("b",), sha256="0" * 64, records=1,
            malformed_records=0, forced_move_count=0,
            state_counts={"same": 1}, state_cycles={},
            uniform_generation_cycle="c1", source_counts={}, game_sources={},
        ),
    }
    monkeypatch.setattr(
        corpus, "_cached_replay_file_analysis", lambda p: analyses[Path(p)])

    metrics, keys = corpus.analyze_replay_files([a, b])

    assert keys == {"same", "only-a"}
    assert metrics["cross_cycle_observed_state_count"] == 2
    assert metrics["cross_cycle_repeated_state_count"] == 0
    assert metrics["cross_cycle_unique_state_count"] == 2


def test_rolling_window_analysis_matches_full_merge_across_rotations(shards):
    """One-file rotations retain every metric from the unrestricted oracle."""

    manager = corpus.CorpusSnapshotManager(
        str(shards[0].parent), str(shards[0].parent / "snapshots"))
    identities = {path: corpus._replay_file_identity(path) for path in shards[:8]}
    previous = frozenset(
        corpus.canonical_state_key(_state(index)) for index in range(3))
    aggregate_id = None
    fresh_set_id = None
    for files in (shards[:6], shards[1:7], shards[2:8], shards[1:7]):
        expected, expected_keys = corpus.analyze_replay_files(
            files,
            previous,
            include_state_digest=False,
            _file_identities=identities,
        )
        actual, actual_keys, actual_new = manager._analyze_replay_window(
            files, previous, identities)
        assert actual_keys == expected_keys
        assert actual == expected
        assert actual_new == expected_keys - previous
        if aggregate_id is None:
            aggregate_id = id(manager._replay_window_analysis.state_counts)
            fresh_set_id = id(manager._replay_window_analysis.fresh_state_keys)
        else:
            assert id(manager._replay_window_analysis.state_counts) == aggregate_id
            assert id(manager._replay_window_analysis.fresh_state_keys) == fresh_set_id


def test_rolling_freshness_rebuilds_for_changed_or_mutable_predecessor(shards):
    """Only one exact immutable predecessor may drive incremental freshness."""

    manager = corpus.CorpusSnapshotManager(
        str(shards[0].parent), str(shards[0].parent / "snapshots"))
    files = shards[:6]
    identities = {path: corpus._replay_file_identity(path) for path in files}
    first = frozenset({corpus.canonical_state_key(_state(0))})
    second = frozenset({
        corpus.canonical_state_key(_state(0)),
        corpus.canonical_state_key(_state(1)),
    })

    _metrics, keys, fresh = manager._analyze_replay_window(
        files, first, identities)
    window = manager._replay_window_analysis
    assert window is not None
    first_fresh_id = id(window.fresh_state_keys)
    assert fresh == keys - first
    assert window.freshness_reference is first

    _metrics, keys, fresh = manager._analyze_replay_window(
        files, second, identities)
    assert fresh == keys - second
    assert id(window.fresh_state_keys) != first_fresh_id
    assert window.freshness_reference is second

    mutable = set(first)
    _metrics, keys, fresh = manager._analyze_replay_window(
        files, mutable, identities)
    assert fresh == keys - mutable
    assert window.freshness_reference is None
    mutable.add(corpus.canonical_state_key(_state(2)))
    _metrics, keys, fresh = manager._analyze_replay_window(
        files, mutable, identities)
    assert fresh == keys - mutable
    assert window.freshness_reference is None


def test_rolling_validation_overlap_updates_with_one_shard_rotation(
    shards, monkeypatch,
):
    """Immutable exclusions advance inside the existing rolling shard walk."""

    manager = corpus.CorpusSnapshotManager(
        str(shards[0].parent), str(shards[0].parent / "snapshots"))
    identities = {path: corpus._replay_file_identity(path) for path in shards[:7]}
    previous = frozenset(
        corpus.canonical_state_key(_state(index)) for index in range(3))
    _metrics, first_keys, first_fresh = manager._analyze_replay_window(
        shards[:6], previous, identities)
    ordered = sorted(first_keys)
    validation = frozenset(ordered[::3])
    external = frozenset(ordered[1::5])
    first_expected = (
        corpus._validation_overlap_state_count(
            first_keys, validation, external),
        corpus._validation_overlap_state_count(
            first_fresh, validation, external),
    )
    assert manager._rolling_validation_overlap_counts(
        validation, external) == first_expected

    def unexpected_rebuild(*_args, **_kwargs):
        raise AssertionError("one-shard rotation rebuilt full overlap counts")

    monkeypatch.setattr(
        corpus, "_validation_overlap_state_count", unexpected_rebuild)
    _metrics, second_keys, second_fresh = manager._analyze_replay_window(
        shards[1:7], previous, identities)
    actual = manager._rolling_validation_overlap_counts(validation, external)
    exclusion = validation | external
    assert actual == (
        len(second_keys & exclusion),
        len(second_fresh & exclusion),
    )


def test_rolling_validation_overlap_never_reuses_mutable_inputs(shards):
    """In-place exclusion edits always force an exact cardinality rebuild."""

    manager = corpus.CorpusSnapshotManager(
        str(shards[0].parent), str(shards[0].parent / "snapshots"))
    files = shards[:6]
    identities = {path: corpus._replay_file_identity(path) for path in files}
    previous = frozenset()
    _metrics, keys, fresh = manager._analyze_replay_window(
        files, previous, identities)
    ordered = sorted(keys)
    validation = {ordered[0]}
    external = {ordered[1]}

    assert manager._rolling_validation_overlap_counts(
        validation, external) == (2, 2)
    validation.add(ordered[2])
    external.add(ordered[3])
    assert manager._rolling_validation_overlap_counts(
        validation, external) == (4, 4)
    window = manager._replay_window_analysis
    assert window is not None
    assert window.exclusion_validation_reference is None
    assert window.exclusion_external_reference is None


def test_rolling_window_analysis_falls_back_for_duplicate_cycle_ids(
    monkeypatch, tmp_path,
):
    """Ambiguous cycle provenance clears the rolling aggregate and stays exact."""

    paths = [tmp_path / "replay_a.jsonl", tmp_path / "replay_b.jsonl"]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    identities = {path: corpus._replay_file_identity(path) for path in paths}
    analyses = {
        paths[0]: corpus._ReplayFileAnalysis(
            identity=identities[paths[0]].as_key(), sha256="0" * 64, records=2,
            malformed_records=0, forced_move_count=1,
            state_counts={"same": 1, "only-a": 1}, state_cycles={},
            uniform_generation_cycle="c1", source_counts={"algorithm": 2},
            game_sources={"g": "algorithm"},
        ),
        paths[1]: corpus._ReplayFileAnalysis(
            identity=identities[paths[1]].as_key(), sha256="1" * 64, records=1,
            malformed_records=0, forced_move_count=0,
            state_counts={"same": 1}, state_cycles={},
            uniform_generation_cycle="c1", source_counts={"current_model": 1},
            game_sources={"g": "current_model"},
        ),
    }
    monkeypatch.setattr(
        corpus,
        "_cached_replay_file_analysis_for_identity",
        lambda path, _identity: analyses[Path(path)],
    )
    manager = corpus.CorpusSnapshotManager(
        str(tmp_path), str(tmp_path / "snapshots"))

    expected, expected_keys = corpus.analyze_replay_files(
        paths, include_state_digest=False, _file_identities=identities)
    actual, actual_keys, actual_new = manager._analyze_replay_window(
        paths, set(), identities)

    assert actual_keys == expected_keys
    assert actual == expected
    assert actual_new == expected_keys
    assert manager._replay_window_analysis is None
