"""Concurrent self-play forks cannot retain shared JSON writer temporaries."""

import json
import os
from pathlib import Path
import threading

import pytest

from dama.ai.ml import fork_writers, run_status


pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork") or not Path("/proc/self/fd").is_dir(),
    reason="descriptor inheritance is observed through fork and /proc/self/fd",
)


def _identity(descriptor):
    info = os.fstat(descriptor)
    return info.st_dev, info.st_ino


def _inherited_writers(identity):
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


def test_shared_json_writer_temporary_is_not_inherited_during_serialization(
    tmp_path, monkeypatch,
):
    path = tmp_path / "pending_acceptance.json"
    child_handles = []
    real_dump = json.dump

    def dump_while_pool_forks(payload, handle, **kwargs):
        child_handles.append(_inherited_writers(_identity(handle.fileno())))
        return real_dump(payload, handle, **kwargs)

    monkeypatch.setattr(run_status.json, "dump", dump_while_pool_forks)
    run_status._write_json_atomic(path, {"pending": True})
    assert child_handles == [0]
    assert json.loads(path.read_text()) == {"pending": True}
    assert not fork_writers._FORK_CHILD_DROPPED_FDS


@pytest.mark.parametrize("boundary", ["open", "close"])
def test_shared_json_writer_protects_complete_raw_descriptor_lifetime(
    tmp_path, monkeypatch, boundary,
):
    path = tmp_path / "pending_acceptance.json"
    boundary_reached = threading.Event()
    release_boundary = threading.Event()
    fork_started = threading.Event()
    fork_finished = threading.Event()
    descriptors, identity, child_handles, failures = [], [], [], []
    real_mkstemp = fork_writers.tempfile.mkstemp
    real_close = os.close

    def pause_at_boundary(descriptor):
        identity[:] = [_identity(descriptor)]
        boundary_reached.set()
        assert release_boundary.wait(10), "writer boundary was never released"

    def mkstemp(**kwargs):
        descriptor, name = real_mkstemp(**kwargs)
        descriptors.append(descriptor)
        if boundary == "open":
            pause_at_boundary(descriptor)
        return descriptor, name

    def close(descriptor):
        if (
            boundary == "close" and descriptor in descriptors
            and threading.current_thread() is writer
        ):
            pause_at_boundary(descriptor)
        return real_close(descriptor)

    def write_json():
        try:
            run_status._write_json_atomic(path, {"pending": True})
        except BaseException as exc:
            failures.append(exc)

    def fork_pool():
        try:
            fork_started.set()
            child_handles.append(_inherited_writers(identity[0]))
        except BaseException as exc:
            failures.append(exc)
        finally:
            fork_finished.set()

    monkeypatch.setattr(fork_writers.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(run_status.os, "close", close)
    writer = threading.Thread(target=write_json)
    forker = threading.Thread(target=fork_pool)
    writer.start()
    try:
        assert boundary_reached.wait(10)
        forker.start()
        assert fork_started.wait(10)
        fork_finished.wait(0.25)
    finally:
        release_boundary.set()
        writer.join(10)
        if forker.ident is not None:
            forker.join(10)
    assert not writer.is_alive() and not forker.is_alive()
    assert failures == []
    assert child_handles == [0]
    assert json.loads(path.read_text()) == {"pending": True}
    assert not fork_writers._FORK_CHILD_DROPPED_FDS
