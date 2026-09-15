"""Corpus publication must not lend its temporary writers to tensorizer forks."""

import gzip
import os
from pathlib import Path
import threading

import pytest

from dama.ai.ml import corpus, fork_writers


def _inherited_handles(identity):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            count = 0
            for name in os.listdir("/proc/self/fd"):
                try:
                    info = os.stat(f"/proc/self/fd/{name}")
                except OSError:
                    continue
                count += (info.st_dev, info.st_ino) == identity
            os.write(write_fd, str(count).encode("ascii"))
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        return int(os.read(read_fd, 128))
    finally:
        os.close(read_fd)
        os.waitpid(pid, 0)


@pytest.mark.skipif(
    not hasattr(os, "fork") or not Path("/proc/self/fd").is_dir(),
    reason="descriptor inheritance requires fork and /proc/self/fd",
)
@pytest.mark.parametrize("writer", ["text", "jsonl", "keys", "merge", "sidecar"])
def test_corpus_writer_is_not_inherited_by_concurrent_tensorizer(
    tmp_path, monkeypatch, writer,
):
    manager = corpus.CorpusSnapshotManager(
        str(tmp_path / "replay"), str(tmp_path / "snapshots"),
        trained_ledger_enabled=True,
    )
    manager.trained_ledger_dir.mkdir(parents=True)
    source_keys = {"0" * 64, "1" * 64}
    corpus._write_state_keys(manager._ledger_state_keys_path, source_keys)
    path = tmp_path / "published"
    if writer == "merge":
        path.write_bytes(gzip.compress(b"00\n22\n"))
    child_handles, failures = [], []
    real_fsync = os.fsync

    def fsync_with_concurrent_fork(descriptor):
        info = os.fstat(descriptor)
        if not child_handles:
            def fork_tensorizer():
                try:
                    child_handles.append(_inherited_handles((info.st_dev, info.st_ino)))
                except BaseException as exc:
                    failures.append(exc)

            worker = threading.Thread(target=fork_tensorizer)
            worker.start()
            worker.join(10)
            assert not worker.is_alive(), "concurrent fork did not complete"
        return real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(corpus.os, "fsync", fsync_with_concurrent_fork)
        if writer == "text":
            corpus._write_text_atomic(path, "snapshot_v000002/manifest.json\n")
        elif writer == "jsonl":
            corpus._write_jsonl_atomic(path, [{"name": "shard.jsonl"}])
        elif writer == "keys":
            corpus._write_state_keys(path, source_keys)
        elif writer == "merge":
            assert corpus._merge_state_keys_file(path, ["11", "22"]) == 1
        else:
            manager._write_ledger_fingerprint_sidecar({
                corpus._state_key_fingerprint(key) for key in source_keys
            })

    assert failures == []
    assert child_handles == [0]
    assert not fork_writers._FORK_CHILD_DROPPED_FDS
    if writer == "text":
        assert path.read_text() == "snapshot_v000002/manifest.json\n"
    elif writer == "jsonl":
        assert path.read_bytes() == b'{"name":"shard.jsonl"}\n'
    elif writer == "keys":
        assert corpus._read_state_keys(path) == source_keys
    elif writer == "merge":
        assert gzip.decompress(path.read_bytes()) == b"00\n11\n22\n"
    else:
        assert manager._load_ledger_fingerprint_sidecar() == {
            corpus._state_key_fingerprint(key) for key in source_keys
        }
    assert not list(tmp_path.rglob("*.tmp"))
