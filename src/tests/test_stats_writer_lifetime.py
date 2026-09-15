"""Granular stats writers retain raw ownership across forks and wrapper errors."""

import errno
import io
import os
from pathlib import Path
import threading

import pytest

from dama.ai.ml import fork_writers, stats_collector


def _is_open(descriptor):
    try:
        os.fstat(descriptor)
    except OSError as exc:
        if exc.errno != errno.EBADF:
            raise
        return False
    return True


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


@pytest.mark.skipif(
    not hasattr(os, "fork") or not Path("/proc/self/fd").is_dir(),
    reason="descriptor inheritance is observed through fork and /proc/self/fd",
)
@pytest.mark.parametrize("boundary", ["open", "body", "stream_close"])
def test_stats_writer_temporary_is_not_inherited_by_another_threads_fork(
    tmp_path, monkeypatch, boundary,
):
    path = tmp_path / "incremental.jsonl"
    boundary_reached = threading.Event()
    release_boundary = threading.Event()
    fork_started = threading.Event()
    fork_finished = threading.Event()
    identity, child_handles, failures = [], [], []
    real_mkstemp = fork_writers.tempfile.mkstemp
    real_fdopen = os.fdopen

    def pause_at_boundary(descriptor):
        info = os.fstat(descriptor)
        identity[:] = [(info.st_dev, info.st_ino)]
        boundary_reached.set()
        assert release_boundary.wait(10), "writer boundary was never released"

    def mkstemp(**kwargs):
        descriptor, name = real_mkstemp(**kwargs)
        if boundary == "open":
            pause_at_boundary(descriptor)
        return descriptor, name

    class StreamProxy:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            if boundary == "stream_close":
                pause_at_boundary(self.stream.fileno())
            return self.stream.close()

    def fdopen(descriptor, *args, **kwargs):
        return StreamProxy(real_fdopen(descriptor, *args, **kwargs))

    def write_stats():
        try:
            with stats_collector._atomic_text_writer(path) as handle:
                if boundary == "body":
                    pause_at_boundary(handle.fileno())
                handle.write('{"step": 7}\n')
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
    monkeypatch.setattr(stats_collector.os, "fdopen", fdopen)
    writer = threading.Thread(target=write_stats)
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
    assert path.read_bytes() == b'{"step": 7}\n'
    assert not fork_writers._FORK_CHILD_DROPPED_FDS


def test_stats_writer_partial_stream_unwind_preserves_reused_descriptor(
    tmp_path, monkeypatch,
):
    path = tmp_path / "session_report.json"
    prior = b'{"old": true}\n'
    path.write_bytes(prior)
    failure = MemoryError("partial stream construction failed")
    descriptors, unrelated = [], []
    real_close = os.close

    def partial_wrapper(descriptor, *args, **kwargs):
        descriptors.append(descriptor)
        raw = io.FileIO(descriptor, "wb", closefd=kwargs.get("closefd", True))
        raw.close()
        raw_still_open = _is_open(descriptor)
        replacement = os.open(tmp_path / "unrelated", os.O_CREAT | os.O_RDWR, 0o600)
        if not raw_still_open and replacement != descriptor:
            os.dup2(replacement, descriptor)
            real_close(replacement)
            replacement = descriptor
        unrelated.append(replacement)
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(stats_collector.os, "fdopen", partial_wrapper)
            with pytest.raises(MemoryError) as caught:
                with stats_collector._atomic_text_writer(path):
                    pytest.fail("partial wrapper must never enter the write body")
            assert caught.value is failure
        assert len(descriptors) == len(unrelated) == 1
        assert _is_open(unrelated[0])
        assert os.write(unrelated[0], b"still-owned") == len(b"still-owned")
        if descriptors[0] != unrelated[0]:
            assert not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        for descriptor in set(descriptors + unrelated):
            if _is_open(descriptor):
                real_close(descriptor)


@pytest.mark.parametrize(
    "failure_type", [OSError, RuntimeError, KeyboardInterrupt, SystemExit, None],
)
def test_stats_writer_close_preserves_primary_error(
    tmp_path, monkeypatch, failure_type,
):
    """A buffered close failure cannot hide a write failure or interruption."""
    path = tmp_path / "session_report.json"
    prior = b'{"old": true}\n'
    path.write_bytes(prior)
    original = failure_type("stats write failed") if failure_type else None
    close_failure = OSError("stats stream close failed")
    descriptors, closed = [], []
    real_fdopen = os.fdopen
    real_close = os.close

    class FailingCloseStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            self.close()

        def close(self):
            self.stream.close()
            raise close_failure

    def fdopen(descriptor, *args, **kwargs):
        descriptors.append(descriptor)
        return FailingCloseStream(real_fdopen(descriptor, *args, **kwargs))

    def close(descriptor):
        if descriptor in descriptors:
            closed.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(stats_collector.os, "fdopen", fdopen)
    monkeypatch.setattr(stats_collector.os, "close", close)
    expected = original if original is not None else close_failure
    with pytest.raises(type(expected)) as caught:
        with stats_collector._atomic_text_writer(path) as handle:
            handle.write('{"partial":')
            if original is not None:
                raise original

    assert caught.value is expected
    assert path.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))
    assert closed == descriptors
    assert not _is_open(descriptors[0])
