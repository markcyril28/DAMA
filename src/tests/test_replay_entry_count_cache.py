"""[Pass 181] ``ReplayBuffer.count_entries()`` must stay exact while never
re-reading an unchanged shard.

Every self-play cycle in the snapshot regime calls ``count_entries()`` for a
replay-buffer statistic.  Before the durable per-shard line-count cache it
re-read every shard not written by the current session (60 x ~14.5 MB on
drvfs, ~14 s of every ~19 s cycle).  These tests pin the contract: counts are
physical line counts, a shard is read at most once per (size, mtime_ns)
identity, the sidecar is fail-open, pruned with the window, and always
replaced atomically rather than rewritten through a possibly shared inode.
"""
import builtins
import fnmatch
import json
import os
import stat
import threading
import time
from datetime import datetime as real_datetime
from pathlib import Path

import pytest

from dama.ai.ml import replay as replay_mod
from dama.ai.ml.replay import _ENTRY_COUNT_SIDECAR_NAME, ReplayBuffer, ReplayEntry


def _entry(i):
    return {"state": {"i": i}, "legal_moves": [[0]], "chosen_index": 0, "result": 0}


def _write_shards(replay_dir, counts, max_files=0):
    buf = ReplayBuffer(str(replay_dir), max_files=max_files)
    paths = []
    for n in counts:
        paths.append(buf.start_new_file())
        buf.add_entry_dicts([_entry(i) for i in range(n)])
        time.sleep(0.01)  # distinct mtimes so rotation order is deterministic
    buf.close()
    return buf, paths


def _sidecar(replay_dir):
    return json.loads(
        (replay_dir / _ENTRY_COUNT_SIDECAR_NAME).read_text(encoding="utf-8"))


def _forbid_reads(monkeypatch):
    def _boom(path):
        raise AssertionError(f"count_entries re-read an unchanged shard: {path}")

    monkeypatch.setattr(replay_mod, "_count_replay_lines", _boom)


def _spy_reads(monkeypatch):
    calls = []
    real = replay_mod._count_replay_lines

    def _spy(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(replay_mod, "_count_replay_lines", _spy)
    return calls


def test_replay_namespace_is_committed_before_buffer_initialization(
    tmp_path, monkeypatch,
):
    replay_dir = tmp_path / "replay"
    calls = []

    def track_parent_sync(path):
        calls.append(Path(path))

    monkeypatch.setattr(replay_mod, "_fsync_directory", track_parent_sync)

    buffer = ReplayBuffer(str(replay_dir), max_files=0)

    assert buffer.replay_dir == replay_dir
    assert calls == [tmp_path]


def test_replay_parent_sync_failure_stops_before_publication(
    tmp_path, monkeypatch,
):
    replay_dir = tmp_path / "replay"

    def fail_parent_sync(path):
        assert Path(path) == tmp_path
        raise OSError(5, "simulated replay-parent fsync failure")

    monkeypatch.setattr(replay_mod, "_fsync_directory", fail_parent_sync)

    with pytest.raises(OSError, match="simulated replay-parent fsync failure"):
        ReplayBuffer(
            str(replay_dir), max_files=0, cache_written_entries=False)

    assert replay_dir.is_dir()
    assert list(replay_dir.iterdir()) == []


def test_count_entries_exact_and_durable_across_processes(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replay"
    buf, paths = _write_shards(replay_dir, [3, 5, 7])
    # Session-written shards are counted from the promoted file cache and
    # their exact line counts are made durable on the first call.
    assert buf.count_entries() == 15
    sidecar = _sidecar(replay_dir)
    assert sidecar["schema"] == 1
    assert set(sidecar["entries"]) == {p.name for p in paths}
    for p in paths:
        st = p.stat()
        size, mtime_ns, _count = sidecar["entries"][p.name]
        assert (size, mtime_ns) == (st.st_size, st.st_mtime_ns)
    assert sorted(sidecar["entries"][p.name][2] for p in paths) == [3, 5, 7]

    # Next session: one new shard, and the three inherited ones must answer
    # from the sidecar without a single read.
    _forbid_reads(monkeypatch)
    second = ReplayBuffer(str(replay_dir), max_files=0)
    extra = second.start_new_file()
    second.add_entry_dicts([_entry(i) for i in range(4)])
    second.close()
    assert second.count_entries() == 19
    assert second.count_entries() == 19
    assert set(_sidecar(replay_dir)["entries"]) == {p.name for p in paths} | {extra.name}

    # Third session: everything is inherited, nothing is read.
    third = ReplayBuffer(str(replay_dir), max_files=0)
    assert third.count_entries() == 19


def test_changed_shard_is_recounted_and_sidecar_updated(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replay"
    buf, paths = _write_shards(replay_dir, [3, 5, 7])
    assert buf.count_entries() == 15
    target = paths[0]
    with open(target, "a", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(_entry(99)) for _ in range(2)) + "\n")

    calls = _spy_reads(monkeypatch)
    fresh = ReplayBuffer(str(replay_dir), max_files=0)
    assert fresh.count_entries() == 17
    assert calls == [target]  # only the changed shard was read
    st = target.stat()
    assert _sidecar(replay_dir)["entries"][target.name] == [st.st_size, st.st_mtime_ns, 5]


def test_identity_mismatch_in_sidecar_forces_recount(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replay"
    buf, paths = _write_shards(replay_dir, [4])
    assert buf.count_entries() == 4
    sidecar = _sidecar(replay_dir)
    size, mtime_ns, _count = sidecar["entries"][paths[0].name]
    sidecar["entries"][paths[0].name] = [size, mtime_ns + 1, 999]
    (replay_dir / _ENTRY_COUNT_SIDECAR_NAME).write_text(json.dumps(sidecar), encoding="utf-8")

    calls = _spy_reads(monkeypatch)
    fresh = ReplayBuffer(str(replay_dir), max_files=0)
    assert fresh.count_entries() == 4
    assert calls == [paths[0]]
    assert _sidecar(replay_dir)["entries"][paths[0].name] == [size, mtime_ns, 4]


@pytest.mark.parametrize("payload", [
    "not json",
    json.dumps({"schema": 2, "entries": {}}),
    json.dumps({"schema": 1, "entries": []}),
    json.dumps({"schema": 1, "entries": {"x.jsonl": [1, 2]}}),
    json.dumps({"schema": 1, "entries": {"x.jsonl": [True, 2, 3]}}),
    json.dumps({"schema": 1, "entries": {"x.jsonl": [1, 2, -3]}}),
])
def test_unusable_sidecar_is_ignored_and_rewritten(tmp_path, payload):
    replay_dir = tmp_path / "replay"
    _buf, paths = _write_shards(replay_dir, [2, 4])
    (replay_dir / _ENTRY_COUNT_SIDECAR_NAME).write_text(payload, encoding="utf-8")
    fresh = ReplayBuffer(str(replay_dir), max_files=0)
    assert fresh.count_entries() == 6
    sidecar = _sidecar(replay_dir)
    assert sidecar["schema"] == 1
    assert set(sidecar["entries"]) == {p.name for p in paths}
    assert sorted(v[2] for v in sidecar["entries"].values()) == [2, 4]


def test_cleanup_prunes_deleted_shards_from_sidecar(tmp_path):
    replay_dir = tmp_path / "replay"
    buf, paths = _write_shards(replay_dir, [1, 2, 3], max_files=2)
    counts = dict(zip((p.name for p in paths), (1, 2, 3)))
    assert buf.count_entries() == 6
    assert set(_sidecar(replay_dir)["entries"]) == set(counts)

    assert buf.cleanup_old_files() == 1
    survivors = {p.name for p in paths if p.exists()}
    assert len(survivors) == 2
    assert buf.count_entries() == sum(counts[name] for name in survivors)
    assert set(_sidecar(replay_dir)["entries"]) == survivors


def test_cleanup_commits_pruned_names_before_corpus_handoff(
    tmp_path, monkeypatch,
):
    """The reduced replay window is durable before admission can consume it."""
    replay_dir = tmp_path / "replay"
    buf, paths = _write_shards(replay_dir, [1, 2, 3], max_files=2)
    for index, path in enumerate(paths):
        timestamp = 1_700_000_000 + index
        os.utime(path, (timestamp, timestamp))
    assert buf.get_buffer_state()[1] == 3

    events = []
    original_stage = buf._stage_cleanup_file_stats

    def track_sync(path):
        assert not paths[0].exists()
        events.append(("sync", Path(path)))

    def track_stage(file_stats):
        events.append(("handoff", [path for path, _stat in file_stats]))
        original_stage(file_stats)

    monkeypatch.setattr(replay_mod, "_fsync_directory", track_sync)
    monkeypatch.setattr(buf, "_stage_cleanup_file_stats", track_stage)

    assert buf.cleanup_old_files() == 1
    assert events == [
        ("sync", replay_dir),
        ("handoff", [paths[2], paths[1]]),
    ]


def test_cleanup_directory_sync_failure_is_not_acknowledged(
    tmp_path, monkeypatch,
):
    """A failed deletion commit must not authorize a corpus handoff."""
    replay_dir = tmp_path / "replay"
    buf, paths = _write_shards(replay_dir, [1, 2], max_files=1)
    for index, path in enumerate(paths):
        timestamp = 1_700_000_000 + index
        os.utime(path, (timestamp, timestamp))
    assert buf.get_buffer_state()[1] == 2

    def fail_sync(path):
        assert Path(path) == replay_dir
        raise OSError(5, "simulated replay cleanup directory fsync failure")

    monkeypatch.setattr(replay_mod, "_fsync_directory", fail_sync)

    with pytest.raises(
        OSError, match="simulated replay cleanup directory fsync failure",
    ):
        buf.cleanup_old_files()

    assert not paths[0].exists()
    assert paths[1].exists()
    assert buf.take_replay_file_stats_handoff() is None


def test_cleanup_without_deletions_does_not_sync_directory(
    tmp_path, monkeypatch,
):
    """A replay window already within its bound pays no new sync cost."""
    replay_dir = tmp_path / "replay"
    buf, _paths = _write_shards(replay_dir, [1], max_files=1)

    def unexpected_sync(_path):
        raise AssertionError("cleanup synced without deleting a shard")

    monkeypatch.setattr(replay_mod, "_fsync_directory", unexpected_sync)

    assert buf.cleanup_old_files() == 0


def test_replay_listing_reuses_captured_direntry_stats(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replay"
    buf, paths = _write_shards(replay_dir, [1, 2, 3], max_files=2)
    captured = [(path, path.stat()) for path in paths]

    observed = []

    def captured_stats(*, strict=False):
        observed.append(strict)
        return list(captured)

    monkeypatch.setattr(buf, "_replay_file_stats", captured_stats)

    def unexpected_path_stat(_path):
        raise AssertionError("get_replay_files repeated a pathname stat")

    monkeypatch.setattr(Path, "stat", unexpected_path_stat)
    expected = [
        path
        for path, _stat in sorted(
            captured, key=lambda item: item[1].st_mtime, reverse=True
        )
    ]
    assert buf.get_replay_files() == expected
    assert observed == [True]


def test_replay_listing_does_not_hide_directory_errors(tmp_path, monkeypatch):
    buf = ReplayBuffer(str(tmp_path / "replay"), max_files=0)

    def inaccessible_directory(_path):
        raise PermissionError("unreadable replay directory")

    monkeypatch.setattr(os, "scandir", inaccessible_directory)
    with pytest.raises(PermissionError, match="unreadable replay directory"):
        buf.get_replay_files()
    assert buf._replay_file_stats() == []


def test_discarded_cycle_shard_leaves_no_record(tmp_path):
    replay_dir = tmp_path / "replay"
    buf, paths = _write_shards(replay_dir, [2])
    assert buf.count_entries() == 2
    partial = buf.start_new_file()
    buf.add_entry_dicts([_entry(0), _entry(1), _entry(2)])
    assert buf.discard_current_file() == partial
    assert not partial.exists()
    assert buf.count_entries() == 2
    assert set(_sidecar(replay_dir)["entries"]) == {paths[0].name}


def test_discarded_cycle_is_quarantined_when_unlink_is_refused(
    tmp_path, monkeypatch,
):
    """An unlink failure must not leave an incomplete active replay shard."""
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)

    complete = buf.start_new_file()
    buf.add_entry_dicts([_entry(0), _entry(1)])
    buf.close()
    assert buf.count_entries() == 2

    partial = buf.start_new_file()
    buf.add_entry_dicts([_entry(2), _entry(3), _entry(4)])
    staging = buf._current_staging_file
    assert staging is not None
    real_unlink = Path.unlink

    def refuse_partial(path, *args, **kwargs):
        if path == staging:
            raise PermissionError("synthetic unlink refusal")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_partial)
    assert buf.discard_current_file() == partial

    quarantined = list(replay_dir.glob(".replay_*.jsonl.incomplete*"))
    assert not partial.exists()
    assert len(quarantined) == 1
    assert len(quarantined[0].read_text(encoding="utf-8").splitlines()) == 3
    assert buf.get_replay_files() == [complete]
    assert buf.count_entries() == 2
    assert buf._session_entry_counts == {}
    assert set(_sidecar(replay_dir)["entries"]) == {complete.name}


def test_concurrent_same_second_writers_claim_distinct_shards(
    tmp_path, monkeypatch,
):
    """Filename selection and creation must be one atomic operation."""
    replay_dir = tmp_path / "replay"
    initial_name = ".replay_20260907_090000.jsonl.pending"
    open_barrier = threading.Barrier(2)
    real_open = builtins.open
    first_open_by_thread = set()
    first_open_lock = threading.Lock()

    class FixedDateTime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 9, 7, 9, 0, 0)

    def synchronized_open(path, *args, **kwargs):
        path = Path(path)
        thread_id = threading.get_ident()
        if path.parent == replay_dir and path.name == initial_name:
            with first_open_lock:
                first_open = thread_id not in first_open_by_thread
                first_open_by_thread.add(thread_id)
            if first_open:
                open_barrier.wait(timeout=5)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(replay_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(builtins, "open", synchronized_open)

    paths = []
    errors = []
    result_lock = threading.Lock()

    def write_one(marker):
        try:
            buf = ReplayBuffer(
                str(replay_dir), max_files=0, cache_written_entries=False)
            path = buf.start_new_file()
            buf.add_entry_dicts([_entry(marker)])
            buf.close()
            with result_lock:
                paths.append(path)
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=write_one, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert len(set(paths)) == 2
    shards = sorted(replay_dir.glob("replay_*.jsonl"))
    assert shards == sorted(paths)
    assert sorted(
        len(path.read_text(encoding="utf-8").splitlines()) for path in shards
    ) == [1, 1]


def test_concurrent_sidecar_updates_merge_distinct_shard_counts(
    tmp_path, monkeypatch,
):
    """Whole-map replacement must not discard another writer's cache record."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    shards = []
    buffers = []
    for index in range(2):
        shard = replay_dir / f"replay_20260907_10000{index}.jsonl"
        shard.write_text("{}\n", encoding="utf-8")
        buffer = ReplayBuffer(str(replay_dir), max_files=0)
        buffer._remember_entry_count(shard.name, shard.stat(), 1)
        shards.append(shard)
        buffers.append(buffer)

    first_save_entered = threading.Event()
    second_persist_started = threading.Event()
    real_save = replay_mod._save_entry_count_sidecar
    save_calls = 0
    save_calls_lock = threading.Lock()

    def delay_first_save(replay_root, entries):
        nonlocal save_calls
        with save_calls_lock:
            save_calls += 1
            first_save = save_calls == 1
        if first_save:
            first_save_entered.set()
            assert second_persist_started.wait(timeout=5)
            # Without the sidecar lock, the second writer replaces the map in
            # this interval and the delayed first writer then loses its record.
            time.sleep(0.05)
        return real_save(replay_root, entries)

    monkeypatch.setattr(
        replay_mod, "_save_entry_count_sidecar", delay_first_save)
    live_names = [{shard.name} for shard in shards]
    errors = []

    def persist(buffer, names, started_event=None):
        if started_event is not None:
            started_event.set()
        try:
            buffer._persist_entry_counts(names)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(
        target=persist, args=(buffers[0], live_names[0]))
    first.start()
    assert first_save_entered.wait(timeout=5)
    second = threading.Thread(
        target=persist,
        args=(buffers[1], live_names[1], second_persist_started),
    )
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not errors
    assert not first.is_alive() and not second.is_alive()
    assert save_calls == 2
    assert set(_sidecar(replay_dir)["entries"]) == {
        shard.name for shard in shards
    }

    # A new process-equivalent buffer must use both durable records and avoid
    # reopening either immutable replay shard merely to recount its lines.
    _forbid_reads(monkeypatch)
    assert ReplayBuffer(str(replay_dir), max_files=0).count_entries() == 2


def test_sidecar_lock_failure_does_not_block_exact_count(
    tmp_path, monkeypatch,
):
    """The durable sidecar remains a fail-open performance cache."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    shard = replay_dir / "replay_20260907_100000.jsonl"
    shard.write_text("{}\n{}\n{}\n", encoding="utf-8")
    real_open = Path.open

    def refuse_lock(path, *args, **kwargs):
        if path.name == replay_mod._ENTRY_COUNT_SIDECAR_LOCK_NAME:
            raise PermissionError("synthetic lock refusal")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_lock)
    assert ReplayBuffer(str(replay_dir), max_files=0).count_entries() == 3
    assert not (replay_dir / _ENTRY_COUNT_SIDECAR_NAME).exists()


def test_stalled_sidecar_lock_is_bounded_and_publication_retries(
    tmp_path, monkeypatch,
):
    """A frozen cache writer must not stall an otherwise exact replay count."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    shard = replay_dir / "replay_20260907_110000.jsonl"
    shard.write_text("{}\n{}\n{}\n", encoding="utf-8")
    buffer = ReplayBuffer(str(replay_dir), max_files=0)

    holder_ready = threading.Event()
    release_holder = threading.Event()
    count_finished = threading.Event()
    outcome = {}

    def hold_lock():
        with replay_mod._entry_count_sidecar_lock(replay_dir):
            holder_ready.set()
            release_holder.wait(timeout=5)

    def count_entries():
        outcome["count"] = buffer.count_entries()
        count_finished.set()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert holder_ready.wait(timeout=2)
    counter = threading.Thread(target=count_entries)
    counter.start()
    completed_while_stalled = count_finished.wait(timeout=0.6)
    release_holder.set()
    holder.join(timeout=2)
    counter.join(timeout=2)

    assert completed_while_stalled
    assert not holder.is_alive() and not counter.is_alive()
    assert outcome["count"] == 3
    assert buffer._entry_count_dirty
    assert not (replay_dir / _ENTRY_COUNT_SIDECAR_NAME).exists()

    # Once the competing writer is gone, the still-dirty optional cache is
    # retried and becomes durable without reopening the immutable shard.
    _forbid_reads(monkeypatch)
    assert buffer.count_entries() == 3
    assert not buffer._entry_count_dirty
    assert _sidecar(replay_dir)["entries"][shard.name][2] == 3


def test_open_writer_shard_counted_but_not_persisted_until_close(tmp_path):
    replay_dir = tmp_path / "replay"
    buf, _paths = _write_shards(replay_dir, [3])
    assert buf.count_entries() == 3
    live = buf.start_new_file()
    buf.add_entry_dicts([_entry(i) for i in range(4)])
    assert buf.count_entries() == 7
    assert live.name not in _sidecar(replay_dir)["entries"]
    buf.close()
    assert buf.count_entries() == 7
    st = live.stat()
    assert _sidecar(replay_dir)["entries"][live.name] == [st.st_size, st.st_mtime_ns, 4]


def test_sidecar_rewrite_never_writes_through_a_shared_inode(tmp_path):
    """A hardlink copy of the replay dir (probes build those) shares the
    sidecar inode; a rewrite must replace the name, not the shared bytes."""
    replay_dir = tmp_path / "replay"
    buf, _paths = _write_shards(replay_dir, [2])
    assert buf.count_entries() == 2
    sidecar = replay_dir / _ENTRY_COUNT_SIDECAR_NAME
    before = sidecar.read_bytes()
    twin = tmp_path / "twin.json"
    os.link(sidecar, twin)

    extra = buf.start_new_file()
    buf.add_entry_dicts([_entry(0)])
    buf.close()
    assert buf.count_entries() == 3
    assert twin.read_bytes() == before
    assert sidecar.stat().st_ino != twin.stat().st_ino
    assert extra.name in _sidecar(replay_dir)["entries"]
    assert not list(replay_dir.glob(".entry_count_cache.*.tmp"))


def test_sidecar_commit_orders_file_sync_replace_and_directory_sync(
    tmp_path, monkeypatch,
):
    """A successful cache publication commits bytes, then its public name."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    entries = {"replay_20260908_120000.jsonl": (123, 456, 7)}
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
        assert path == replay_dir
        events.append("directory_fsync")

    monkeypatch.setattr(replay_mod.os, "fsync", tracking_fsync)
    monkeypatch.setattr(replay_mod.os, "replace", tracking_replace)
    monkeypatch.setattr(
        replay_mod, "_fsync_directory", tracking_directory_fsync)

    assert replay_mod._save_entry_count_sidecar(replay_dir, entries)
    assert events == ["file_fsync", "replace", "directory_fsync"]
    assert _sidecar(replay_dir)["entries"] == {
        "replay_20260908_120000.jsonl": [123, 456, 7],
    }
    assert not list(replay_dir.glob(".entry_count_cache.*.tmp"))


def test_sidecar_directory_sync_failure_retries_without_recount(
    tmp_path, monkeypatch,
):
    """A lost commit acknowledgement stays dirty and retries fail-open."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    shard = replay_dir / "replay_20260908_120000.jsonl"
    shard.write_text("{}\n{}\n{}\n", encoding="utf-8")
    buffer = ReplayBuffer(str(replay_dir), max_files=0)
    real_directory_fsync = replay_mod._fsync_directory

    def fail_directory_fsync(_path):
        raise OSError(5, "synthetic entry-count directory fsync failure")

    monkeypatch.setattr(
        replay_mod, "_fsync_directory", fail_directory_fsync)
    assert buffer.count_entries() == 3
    assert buffer._entry_count_dirty
    assert _sidecar(replay_dir)["entries"][shard.name][2] == 3
    assert not list(replay_dir.glob(".entry_count_cache.*.tmp"))

    monkeypatch.setattr(
        replay_mod, "_fsync_directory", real_directory_fsync)
    _forbid_reads(monkeypatch)
    assert buffer.count_entries() == 3
    assert not buffer._entry_count_dirty
    assert _sidecar(replay_dir)["entries"][shard.name][2] == 3


def test_count_matches_physical_lines_for_hand_written_shard(tmp_path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    shard = replay_dir / "replay_20260101_000000.jsonl"
    shard.write_text("{}\n\n{}", encoding="utf-8")  # blank + unterminated line
    buf = ReplayBuffer(str(replay_dir), max_files=0)
    assert buf.count_entries() == 3
    assert _sidecar(replay_dir)["entries"][shard.name][2] == 3


def test_sidecar_name_is_invisible_to_replay_globs():
    for pattern in ("replay_*.jsonl", "*.jsonl"):
        assert not fnmatch.fnmatch(_ENTRY_COUNT_SIDECAR_NAME, pattern)


def test_empty_directory_counts_zero_without_sidecar(tmp_path):
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(str(replay_dir), max_files=0)
    assert buf.count_entries() == 0
    assert buf.get_buffer_state() == (0, 0, 0)
    assert not (replay_dir / _ENTRY_COUNT_SIDECAR_NAME).exists()


def test_buffer_state_reuses_one_shard_identity_snapshot(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replay"
    buf, paths = _write_shards(replay_dir, [2, 4, 6])
    assert buf.count_entries() == 12

    fresh = ReplayBuffer(str(replay_dir), max_files=0)
    file_stats = fresh._replay_file_stats()
    expected_bytes = sum(stat.st_size for _, stat in file_stats)
    monkeypatch.setattr(fresh, "_replay_file_stats", lambda: file_stats)

    # A warm exact count must consume the captured stat results directly.  Any
    # Path.stat() below would be the duplicate drvfs transaction this API exists
    # to remove (uncached or changed shards still recheck after reading).
    def _unexpected_stat(*_args, **_kwargs):
        raise AssertionError("warm buffer state repeated a shard stat")

    monkeypatch.setattr(Path, "stat", _unexpected_stat)
    _forbid_reads(monkeypatch)
    assert fresh.get_buffer_state() == (12, len(paths), expected_bytes)
    assert fresh.count_entries() == 12


def test_cleanup_reuses_immediately_preceding_buffer_identity_scan(
    tmp_path, monkeypatch,
):
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=2, cache_written_entries=False)
    paths = []
    for count in [1, 2, 3]:
        paths.append(buf.start_new_file())
        buf.add_entry_dicts([_entry(i) for i in range(count)])
    buf.close()
    for index, path in enumerate(paths):
        timestamp = 1_700_000_000 + index
        os.utime(path, (timestamp, timestamp))

    assert buf.get_buffer_state()[1] == 3

    def repeated_identity_scan(*_args, **_kwargs):
        raise AssertionError("cleanup restatted an unchanged replay window")

    monkeypatch.setattr(buf, "_replay_file_stats", repeated_identity_scan)
    assert buf.cleanup_old_files() == 1
    assert not paths[0].exists()
    assert paths[1].exists()
    assert paths[2].exists()
    handoff = buf.take_replay_file_stats_handoff()
    assert handoff is not None
    _directory_identity, retained_stats = handoff
    assert [path for path, _stat in retained_stats] == [paths[2], paths[1]]
    assert buf.take_replay_file_stats_handoff() is None


def test_cleanup_rechecks_identities_after_concurrent_shard_publication(
    tmp_path, monkeypatch,
):
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=2, cache_written_entries=False)
    paths = []
    for count in [1, 2]:
        paths.append(buf.start_new_file())
        buf.add_entry_dicts([_entry(i) for i in range(count)])
    buf.close()
    for index, path in enumerate(paths):
        timestamp = 1_700_000_000 + index
        os.utime(path, (timestamp, timestamp))

    assert buf.get_buffer_state()[1] == 2
    added = replay_dir / "replay_20260101_000003.jsonl"
    added.write_text("{}\n", encoding="utf-8")
    os.utime(added, (1_700_000_003, 1_700_000_003))

    scans = 0
    original_scan = buf._replay_file_stats

    def track_strict_fallback(*, strict=False):
        nonlocal scans
        scans += 1
        assert strict
        return original_scan(strict=strict)

    monkeypatch.setattr(buf, "_replay_file_stats", track_strict_fallback)
    assert buf.cleanup_old_files() == 1
    assert scans == 1
    assert not paths[0].exists()
    assert paths[1].exists()
    assert added.exists()


def test_cleanup_handoff_preserves_directory_error_visibility(
    tmp_path, monkeypatch,
):
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=1, cache_written_entries=False)
    buf.start_new_file()
    buf.add_entry_dicts([_entry(0)])
    buf.close()
    assert buf.get_buffer_state()[1] == 1

    def inaccessible_directory(_path):
        raise PermissionError("unreadable replay directory")

    monkeypatch.setattr(os, "scandir", inaccessible_directory)
    with pytest.raises(PermissionError, match="unreadable replay directory"):
        buf.cleanup_old_files()


def test_cleanup_handoff_is_disabled_for_mutable_legacy_shards(
    tmp_path, monkeypatch,
):
    replay_dir = tmp_path / "replay"
    buf, _paths = _write_shards(replay_dir, [1], max_files=1)
    assert buf.get_buffer_state()[1] == 1

    scans = 0
    original_scan = buf._replay_file_stats

    def track_strict_scan(*, strict=False):
        nonlocal scans
        scans += 1
        assert strict
        return original_scan(strict=strict)

    monkeypatch.setattr(buf, "_replay_file_stats", track_strict_scan)
    assert buf.cleanup_old_files() == 0
    assert scans == 1


def test_snapshot_writer_releases_closed_entries_but_preserves_exact_reload(tmp_path):
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    shard = buf.start_new_file()
    records = [_entry(i) for i in range(7)]
    buf.add_entry_dicts(records)

    # An open shard remains exactly countable without retaining the nested raw
    # dictionaries, and closing it must not promote a parsed-entry cache.
    assert buf.count_entries() == 7
    assert buf._session_dicts == {}
    assert buf._session_entries == {}
    assert buf._session_entry_counts == {shard: 7}
    buf.close()
    assert buf._session_dicts == {}
    assert buf._session_entries == {}
    assert buf._session_entry_counts == {}
    assert buf._file_cache == {}
    assert buf.count_entries() == 7
    st = shard.stat()
    assert _sidecar(replay_dir)["entries"][shard.name] == [
        st.st_size, st.st_mtime_ns, 7]

    # The public legacy reader remains available on demand and reconstructs
    # the exact entries from the durable shard when a caller explicitly asks.
    loaded = buf.load_all_entries()
    assert [entry.to_dict() for entry in loaded] == records
    assert len(buf._file_cache) == 1


def test_snapshot_writer_buffers_batches_until_cycle_close(tmp_path):
    """Snapshot cycles remain hidden until close while counts stay exact."""

    snapshot_dir = tmp_path / "snapshot_replay"
    snapshot = ReplayBuffer(
        str(snapshot_dir), max_files=0, cache_written_entries=False)
    snapshot_shard = snapshot.start_new_file()
    records = [_entry(0), _entry(1)]
    snapshot.add_entry_dicts(records)

    # The active snapshot path keeps small batches in its bounded userspace
    # buffer under an ignored staging name. Integer-only session accounting
    # keeps it exactly countable without exposing a partial corpus shard.
    staging = snapshot._current_staging_file
    assert staging is not None
    assert not snapshot_shard.exists()
    assert staging.stat().st_size == 0
    assert snapshot.get_replay_files() == []
    assert snapshot.count_entries() == len(records)
    assert snapshot.get_buffer_state()[1] == 1
    assert snapshot._session_dicts == {}
    assert snapshot._session_entry_counts == {snapshot_shard: len(records)}
    snapshot.close()
    assert snapshot_shard.stat().st_size > 0
    assert not staging.exists()
    assert snapshot.get_replay_files() == [snapshot_shard]
    assert [entry.to_dict() for entry in snapshot.load_all_entries()] == records

    legacy_dir = tmp_path / "legacy_replay"
    legacy = ReplayBuffer(str(legacy_dir), max_files=0)
    legacy_shard = legacy.start_new_file()
    legacy.add_entry_dicts(records)

    # Default callers retain the established per-batch visibility contract.
    assert legacy_shard.stat().st_size > 0
    legacy.close()


def test_snapshot_writer_commits_data_before_publication(
    tmp_path, monkeypatch,
):
    """A public cycle appears only after file sync, then its name is synced."""
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    shard = buf.start_new_file()
    buf.add_entry_dicts([_entry(0), _entry(1)])
    staging = buf._current_staging_file
    assert staging is not None

    events = []
    real_fsync = os.fsync
    real_link = os.link
    real_unlink = Path.unlink

    def tracking_fsync(fd):
        mode = os.fstat(fd).st_mode
        events.append("directory_fsync" if stat.S_ISDIR(mode) else "file_fsync")
        return real_fsync(fd)

    def tracking_link(source, destination, *args, **kwargs):
        if Path(destination) == shard:
            events.append("public_link")
        return real_link(source, destination, *args, **kwargs)

    def tracking_unlink(path, *args, **kwargs):
        if path == staging:
            events.append("staging_unlink")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(replay_mod.os, "fsync", tracking_fsync)
    monkeypatch.setattr(replay_mod.os, "link", tracking_link)
    monkeypatch.setattr(Path, "unlink", tracking_unlink)

    buf.close()

    assert events == [
        "file_fsync", "public_link", "staging_unlink", "directory_fsync",
    ]
    assert not staging.exists()
    assert [entry.to_dict() for entry in buf.load_all_entries()] == [
        _entry(0), _entry(1),
    ]


def test_snapshot_writer_file_sync_failure_stays_hidden(
    tmp_path, monkeypatch,
):
    """An unsynced cycle cannot enter the public replay namespace."""
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    shard = buf.start_new_file()
    buf.add_entry_dicts([_entry(0)])
    staging = buf._current_staging_file
    assert staging is not None

    def fail_file_fsync(fd):
        assert not stat.S_ISDIR(os.fstat(fd).st_mode)
        raise OSError(5, "synthetic replay file fsync failure")

    monkeypatch.setattr(replay_mod.os, "fsync", fail_file_fsync)

    with pytest.raises(OSError, match="replay file fsync failure"):
        buf.close()

    assert not shard.exists()
    assert not staging.exists()
    assert buf._current_writer is None
    assert buf._current_file is None
    assert buf._current_staging_file is None
    assert buf._session_entry_counts == {}


def test_snapshot_writer_directory_sync_failure_finishes_close(
    tmp_path, monkeypatch,
):
    """A post-link sync error is reported without corrupting buffer state."""
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    shard = buf.start_new_file()
    records = [_entry(0), _entry(1)]
    buf.add_entry_dicts(records)
    staging = buf._current_staging_file
    assert staging is not None
    real_fsync = os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(5, "synthetic replay directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(replay_mod.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="replay directory fsync failure"):
        buf.close()

    assert not staging.exists()
    assert [entry.to_dict() for entry in buf.load_all_entries()] == records
    assert buf._current_writer is None
    assert buf._current_file is None
    assert buf._current_staging_file is None
    assert buf._session_entry_counts == {}
    assert buf._entry_count_cache[shard.name][2] == len(records)


def test_snapshot_publication_refuses_to_replace_a_colliding_shard(tmp_path):
    """Atomic close must preserve an unexpected pre-existing final name."""
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    shard = buf.start_new_file()
    buf.add_entry_dicts([_entry(0)])
    staging = buf._current_staging_file
    assert staging is not None

    shard.write_text("external-writer\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        buf.close()

    assert shard.read_text(encoding="utf-8") == "external-writer\n"
    assert not staging.exists()
    assert buf._current_file is None
    assert buf._current_staging_file is None
    assert buf._session_entry_counts == {}


def test_snapshot_writer_quarantines_deferred_close_failure(tmp_path):
    """A buffered close error cannot publish an incomplete snapshot shard."""

    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    shard = buf.start_new_file()
    buf.add_entry_dicts([_entry(0)])
    real_writer = buf._current_writer

    class FailingClose:
        def flush(self):
            real_writer.flush()

        def fileno(self):
            return real_writer.fileno()

        def close(self):
            real_writer.close()
            raise OSError("synthetic deferred flush failure")

    buf._current_writer = FailingClose()
    with pytest.raises(OSError, match="synthetic deferred flush failure"):
        buf.close()

    assert not shard.exists()
    assert buf._current_file is None
    assert buf._session_dicts == {}
    assert buf._session_entries == {}
    assert buf._session_entry_counts == {}


def test_snapshot_close_failure_is_hidden_when_unlink_is_refused(
    tmp_path, monkeypatch,
):
    """A failed deferred flush must never remain an active replay shard."""
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    shard = buf.start_new_file()
    buf.add_entry_dicts([_entry(0), _entry(1)])
    staging = buf._current_staging_file
    assert staging is not None
    real_writer = buf._current_writer
    real_unlink = Path.unlink

    class FailingClose:
        def flush(self):
            real_writer.flush()

        def fileno(self):
            return real_writer.fileno()

        def close(self):
            real_writer.close()
            raise OSError("synthetic deferred flush failure")

    def refuse_shard(path, *args, **kwargs):
        if path == staging:
            raise PermissionError("synthetic unlink refusal")
        return real_unlink(path, *args, **kwargs)

    buf._current_writer = FailingClose()
    monkeypatch.setattr(Path, "unlink", refuse_shard)
    with pytest.raises(OSError, match="synthetic deferred flush failure"):
        buf.close()

    quarantined = list(replay_dir.glob(".replay_*.jsonl.incomplete*"))
    assert not shard.exists()
    assert len(quarantined) == 1
    assert len(quarantined[0].read_text(encoding="utf-8").splitlines()) == 2
    assert buf.get_replay_files() == []
    assert buf.count_entries() == 0
    assert buf._current_file is None
    assert buf._session_dicts == {}
    assert buf._session_entries == {}
    assert buf._session_entry_counts == {}


def test_snapshot_writer_validates_dicts_before_buffering_them(tmp_path):
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    invalid = _entry(0)
    invalid["chosen_index"] = len(invalid["legal_moves"])
    with pytest.raises(ValueError, match="chosen_index"):
        buf.add_entry_dicts([invalid])
    assert buf._session_dicts == {}
    assert buf._session_entry_counts == {}
    buf.discard_current_file()


def test_snapshot_writer_validates_dicts_without_constructing_entries(
    tmp_path, monkeypatch,
):
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)

    def forbid_construction(cls, data):
        raise AssertionError("snapshot validation constructed a ReplayEntry")

    monkeypatch.setattr(
        ReplayEntry, "from_dict", classmethod(forbid_construction))
    buf.add_entry_dicts([_entry(0), _entry(1)])

    assert sum(buf._session_entry_counts.values()) == 2
    buf.close()


@pytest.mark.parametrize("case", [
    "missing_state",
    "missing_legal_moves",
    "missing_chosen_index",
    "chosen_index",
    "played_index",
    "sample_weight",
    "opening_plies",
])
def test_validate_dict_matches_from_dict_rejections(case):
    invalid = _entry(0)
    if case.startswith("missing_"):
        invalid.pop(case.removeprefix("missing_"))
    elif case in ("chosen_index", "played_index"):
        invalid[case] = len(invalid["legal_moves"])
    else:
        invalid[case] = "not-a-number"

    with pytest.raises(Exception) as constructor_error:
        ReplayEntry.from_dict(invalid)
    with pytest.raises(Exception) as validator_error:
        ReplayEntry.validate_dict(invalid)

    assert type(validator_error.value) is type(constructor_error.value)
    assert str(validator_error.value) == str(constructor_error.value)


def test_snapshot_writer_counts_typed_entries_without_retaining_them(tmp_path):
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    shard = buf.start_new_file()
    first = ReplayEntry.from_dict(_entry(0))
    remaining = [ReplayEntry.from_dict(_entry(i)) for i in (1, 2)]
    buf.add_entry(first)
    buf.add_entries(remaining)

    assert buf._session_entries == {}
    assert buf._session_entry_counts == {shard: 3}
    assert buf.count_entries() == 3
    buf.close()
    assert buf._session_entry_counts == {}
    assert buf.count_entries() == 3
    st = shard.stat()
    assert _sidecar(replay_dir)["entries"][shard.name] == [
        st.st_size, st.st_mtime_ns, 3]


def test_default_writer_retains_closed_entries_for_legacy_loader(tmp_path):
    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(str(replay_dir), max_files=0)
    shard = buf.start_new_file()
    buf.add_entry_dicts([_entry(0), _entry(1)])
    buf.close()
    assert shard in buf._file_cache
    assert len(buf._file_cache[shard][1]) == 2
