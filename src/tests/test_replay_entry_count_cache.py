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
import fnmatch
import json
import os
import time
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
    """Snapshot cycles coalesce writes while legacy readers still see flushes."""

    snapshot_dir = tmp_path / "snapshot_replay"
    snapshot = ReplayBuffer(
        str(snapshot_dir), max_files=0, cache_written_entries=False)
    snapshot_shard = snapshot.start_new_file()
    records = [_entry(0), _entry(1)]
    snapshot.add_entry_dicts(records)

    # The active snapshot path keeps small batches in its bounded userspace
    # buffer. Integer-only session accounting keeps it exactly countable.
    assert snapshot_shard.stat().st_size == 0
    assert snapshot.count_entries() == len(records)
    assert snapshot._session_dicts == {}
    assert snapshot._session_entry_counts == {snapshot_shard: len(records)}
    snapshot.close()
    assert snapshot_shard.stat().st_size > 0
    assert [entry.to_dict() for entry in snapshot.load_all_entries()] == records

    legacy_dir = tmp_path / "legacy_replay"
    legacy = ReplayBuffer(str(legacy_dir), max_files=0)
    legacy_shard = legacy.start_new_file()
    legacy.add_entry_dicts(records)

    # Default callers retain the established per-batch visibility contract.
    assert legacy_shard.stat().st_size > 0
    legacy.close()


def test_snapshot_writer_quarantines_deferred_close_failure(tmp_path):
    """A buffered close error cannot publish an incomplete snapshot shard."""

    replay_dir = tmp_path / "replay"
    buf = ReplayBuffer(
        str(replay_dir), max_files=0, cache_written_entries=False)
    shard = buf.start_new_file()
    buf.add_entry_dicts([_entry(0)])
    real_writer = buf._current_writer

    class FailingClose:
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
