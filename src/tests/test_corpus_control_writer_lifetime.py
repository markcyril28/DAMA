"""Atomic corpus control writes keep ownership across partial stream creation."""

import errno
import io
import json
import os

import pytest

import dama.ai.ml.corpus as corpus


@pytest.fixture(params=["text", "jsonl"])
def writer_kind(request):
    return request.param


def _is_open(fd):
    try:
        os.fstat(fd)
    except OSError as exc:
        assert exc.errno == errno.EBADF
        return False
    return True


def _cleanup_descriptors(descriptors):
    # Keep failure controls from leaking descriptors into subsequent tests.
    for fd in set(descriptors):
        if _is_open(fd):
            os.close(fd)


def _prior_file(tmp_path, writer_kind):
    path = tmp_path / ("current.txt" if writer_kind == "text" else "ledger.jsonl")
    prior = b"old-snapshot\n" if writer_kind == "text" else b'{"old":true}'
    path.write_bytes(prior)
    return path, prior


def _publish(writer_kind, path):
    if writer_kind == "text":
        corpus._write_text_atomic(path, "new-snapshot\n\u2713\n")
    else:
        corpus._write_jsonl_atomic(
            path, [{"z": 2, "a": "\u2713"}, {"last": True}], append_existing=True,
        )


def _open_replacement(tmp_path, number, original_close):
    other = os.open(tmp_path / "unrelated", os.O_CREAT | os.O_RDWR, 0o600)
    if other != number:
        os.dup2(other, number)
        original_close(other)
    return number


@pytest.mark.parametrize(
    "error_type", [OSError, MemoryError, KeyboardInterrupt, SystemExit],
)
def test_wrapper_failure_releases_temporary_and_preserves_prior_file(
    tmp_path, monkeypatch, writer_kind, error_type,
):
    path, prior = _prior_file(tmp_path, writer_kind)
    failure = error_type("stream construction failed")
    descriptors = []

    def fail_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", fail_wrapper)
            with pytest.raises(error_type) as caught:
                _publish(writer_kind, path)
            assert caught.value is failure
        assert len(descriptors) == 1
        assert not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup_descriptors(descriptors)


def test_partial_stream_unwind_does_not_close_an_unrelated_reopened_file(
    tmp_path, monkeypatch, writer_kind,
):
    path, prior = _prior_file(tmp_path, writer_kind)
    failure = MemoryError("buffer allocation failed after FileIO creation")
    descriptors = []
    unrelated = []
    original_close = os.close

    def partial_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        raw = io.FileIO(fd, "wb", closefd=kwargs.get("closefd", True))
        raw.close()
        if _is_open(fd):
            # A concurrent opener gets another number while the writer owns fd.
            unrelated.append(os.open(
                tmp_path / "unrelated", os.O_CREAT | os.O_RDWR, 0o600,
            ))
        else:
            # Deterministically reproduce the concurrent close/reuse ordering.
            unrelated.append(_open_replacement(tmp_path, fd, original_close))
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", partial_wrapper)
            with pytest.raises(MemoryError) as caught:
                _publish(writer_kind, path)
            assert caught.value is failure
        assert len(descriptors) == len(unrelated) == 1
        assert _is_open(unrelated[0]), "constructor cleanup closed an unrelated file"
        assert os.write(unrelated[0], b"still-owned") == len(b"still-owned")
        assert not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup_descriptors(descriptors + unrelated)


def test_secondary_close_error_preserves_original_error_and_is_not_retried(
    tmp_path, monkeypatch, writer_kind,
):
    path, prior = _prior_file(tmp_path, writer_kind)
    failure = MemoryError("primary stream construction failure")
    descriptors = []
    closes = []
    unrelated = []
    original_close = os.close

    def fail_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        raise failure

    def fail_close(fd):
        closes.append(fd)
        original_close(fd)
        unrelated.append(_open_replacement(tmp_path, fd, original_close))
        raise OSError("secondary close failure after descriptor release")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", fail_wrapper)
            patch.setattr(corpus.os, "close", fail_close)
            with pytest.raises(MemoryError) as caught:
                _publish(writer_kind, path)
            assert caught.value is failure
        assert len(descriptors) == 1
        assert closes == descriptors
        assert unrelated == descriptors
        assert _is_open(unrelated[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup_descriptors(descriptors + unrelated)


@pytest.mark.parametrize("stage", ["write", "flush", "fsync", "stream_close", "replace"])
def test_failed_publication_releases_owned_descriptor_and_keeps_prior_file(
    tmp_path, monkeypatch, writer_kind, stage,
):
    path, prior = _prior_file(tmp_path, writer_kind)
    failure = KeyboardInterrupt(f"injected {stage} failure")
    descriptors = []
    wrappers = []
    unrelated = []
    original_fdopen = os.fdopen
    original_close = os.close

    class StreamProxy:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.close()

        def close(self):
            result = self.stream.close()
            if stage == "stream_close":
                raise failure
            return result

        def write(self, value):
            if stage == "write":
                raise failure
            return self.stream.write(value)

        def flush(self):
            if stage == "flush":
                raise failure
            return self.stream.flush()

    def record_wrapper(fd, *args, **kwargs):
        stream = original_fdopen(fd, *args, **kwargs)
        descriptors.append(fd)
        wrappers.append(stream)
        return StreamProxy(stream)

    def fail_operation(*args):
        if stage == "replace":
            assert wrappers[0].closed
            assert not _is_open(descriptors[0])
            unrelated.append(_open_replacement(tmp_path, descriptors[0], original_close))
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", record_wrapper)
            if stage == "fsync":
                patch.setattr(corpus.os, "fsync", fail_operation)
            elif stage == "replace":
                patch.setattr(corpus.os, "replace", fail_operation)
            with pytest.raises(KeyboardInterrupt) as caught:
                _publish(writer_kind, path)
            assert caught.value is failure
        assert len(descriptors) == 1
        assert wrappers[0].closed
        if unrelated:
            assert _is_open(unrelated[0])
        else:
            assert not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup_descriptors(descriptors + unrelated)


def test_raw_close_error_prevents_publication_without_retrying_reused_number(
    tmp_path, monkeypatch, writer_kind,
):
    path, prior = _prior_file(tmp_path, writer_kind)
    failure = OSError("raw close reported failure after release")
    closes = []
    unrelated = []
    original_close = os.close

    def fail_close(fd):
        closes.append(fd)
        original_close(fd)
        unrelated.append(_open_replacement(tmp_path, fd, original_close))
        raise failure

    def unexpected_replace(*args):
        raise AssertionError("close failure must prevent publication")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "close", fail_close)
            patch.setattr(corpus.os, "replace", unexpected_replace)
            with pytest.raises(OSError) as caught:
                _publish(writer_kind, path)
            assert caught.value is failure
        assert len(closes) == 1
        assert unrelated == closes
        assert _is_open(unrelated[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup_descriptors(closes + unrelated)


def test_healthy_write_publishes_complete_bytes_only_after_sync_and_close(
    tmp_path, monkeypatch, writer_kind,
):
    path, prior = _prior_file(tmp_path, writer_kind)
    expected = (
        "new-snapshot\n\u2713\n".encode("utf-8") if writer_kind == "text" else
        prior + b'\n{"a":"\\u2713","z":2}\n{"last":true}\n'
    )
    events = []
    descriptors = []
    original_fdopen = os.fdopen
    original_close = os.close
    original_fsync = os.fsync
    original_replace = os.replace

    class StreamProxy:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.close()

        def close(self):
            result = self.stream.close()
            events.append("stream_close")
            return result

        def flush(self):
            events.append("flush")
            return self.stream.flush()

    def record_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        return StreamProxy(original_fdopen(fd, *args, **kwargs))

    def record_fsync(fd):
        assert fd == descriptors[0]
        events.append("file_fsync")
        return original_fsync(fd)

    def record_close(fd):
        assert fd == descriptors[0]
        events.append("raw_close")
        return original_close(fd)

    def record_replace(source, destination):
        assert not _is_open(descriptors[0])
        assert type(path)(source).read_bytes() == expected
        events.append("replace")
        return original_replace(source, destination)

    def record_directory_fsync(directory):
        assert directory == path.parent
        assert path.read_bytes() == expected
        events.append("directory_fsync")

    with monkeypatch.context() as patch:
        patch.setattr(corpus.os, "fdopen", record_wrapper)
        patch.setattr(corpus.os, "fsync", record_fsync)
        patch.setattr(corpus.os, "close", record_close)
        patch.setattr(corpus.os, "replace", record_replace)
        patch.setattr(corpus.run_status, "_fsync_directory", record_directory_fsync)
        _publish(writer_kind, path)
    expected_events = ["flush", "file_fsync", "stream_close", "raw_close", "replace"]
    if writer_kind == "text":
        expected_events.append("directory_fsync")
    assert events == expected_events
    assert path.read_bytes() == expected
    if writer_kind == "jsonl":
        assert [json.loads(row) for row in path.read_text().splitlines()] == [
            {"old": True}, {"a": "\u2713", "z": 2}, {"last": True},
        ]
    assert not list(tmp_path.glob("*.tmp"))
