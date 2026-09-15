"""Canonical snapshot key publication owns its descriptor until explicit close."""

import errno
import gzip
import io
import os

import pytest

import dama.ai.ml.corpus as corpus


def _is_open(fd):
    try:
        os.fstat(fd)
    except OSError as exc:
        assert exc.errno == errno.EBADF
        return False
    return True


def _cleanup_descriptors(descriptors):
    # Baseline-failure controls must not leak into subsequent tests.
    for fd in set(descriptors):
        if _is_open(fd):
            os.close(fd)


def _prior_keys(tmp_path):
    path = tmp_path / "canonical_state_keys.txt.gz"
    corpus._write_state_keys(path, ["old-key"])
    return path, path.read_bytes()


@pytest.mark.parametrize(
    "error_type", [OSError, MemoryError, KeyboardInterrupt, SystemExit],
)
def test_key_wrapper_failure_releases_descriptor_and_temporary(
    tmp_path, monkeypatch, error_type,
):
    path, prior = _prior_keys(tmp_path)
    failure = error_type("key wrapper construction failed")
    descriptors = []
    closefd_arguments = []

    def fail_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        closefd_arguments.append(kwargs.get("closefd"))
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", fail_wrapper)
            with pytest.raises(error_type) as caught:
                corpus._write_state_keys(path, ["new-key"])
            assert caught.value is failure
        assert len(descriptors) == 1
        assert not _is_open(descriptors[0])
        assert closefd_arguments == [False]
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup_descriptors(descriptors)


def test_partial_key_stream_unwind_preserves_raw_ownership(tmp_path, monkeypatch):
    path, prior = _prior_keys(tmp_path)
    failure = MemoryError("buffer allocation failed after FileIO creation")
    descriptors = []
    survived_stream_close = []

    def partial_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        stream = io.FileIO(fd, "wb", closefd=kwargs.get("closefd", True))
        stream.close()
        survived_stream_close.append(_is_open(fd))
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", partial_wrapper)
            with pytest.raises(MemoryError) as caught:
                corpus._write_state_keys(path, ["new-key"])
            assert caught.value is failure
        assert survived_stream_close == [True]
        assert len(descriptors) == 1
        assert not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup_descriptors(descriptors)


def test_key_cleanup_failures_preserve_original_error(tmp_path, monkeypatch):
    path, prior = _prior_keys(tmp_path)
    failure = MemoryError("primary wrapper failure")
    descriptors = []
    closes = []
    original_close = os.close

    def fail_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        raise failure

    def fail_close(fd):
        closes.append(fd)
        # A reported close error can follow release; never retry this number.
        original_close(fd)
        raise OSError("secondary close failure")

    def fail_unlink(*args, **kwargs):
        raise OSError("secondary unlink failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", fail_wrapper)
            patch.setattr(corpus.os, "close", fail_close)
            patch.setattr(corpus.os, "unlink", fail_unlink)
            with pytest.raises(MemoryError) as caught:
                corpus._write_state_keys(path, ["new-key"])
            assert caught.value is failure
        assert len(descriptors) == 1
        assert closes == descriptors
        assert not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        # Deliberately refused unlink is best effort, not successful cleanup.
        assert len(list(tmp_path.glob("*.tmp"))) == 1
    finally:
        _cleanup_descriptors(descriptors)


@pytest.mark.parametrize(
    "stage", ["sorting", "write", "gzip_finalize", "flush", "fsync", "stream_close", "replace"],
)
def test_key_publication_failure_closes_once_without_touching_reused_fd(
    tmp_path, monkeypatch, stage,
):
    path, prior = _prior_keys(tmp_path)
    failure = KeyboardInterrupt(f"injected {stage} failure")
    descriptors = []
    wrappers = []
    closes = []
    reused = []
    original_fdopen = os.fdopen
    original_close = os.close
    original_gzip_open = gzip.open

    class RawProxy:
        def __init__(self, raw):
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def __enter__(self):
            self.raw.__enter__()
            return self

        def __exit__(self, *args):
            return self.close()

        def close(self):
            result = self.raw.close()
            if stage == "stream_close":
                raise failure
            return result

        def flush(self):
            if stage == "flush":
                raise failure
            return self.raw.flush()

    class GzipProxy:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.close()

        def close(self):
            result = self.handle.close()
            if stage == "gzip_finalize":
                raise failure
            return result

        def write(self, value):
            if stage == "write":
                raise failure
            return self.handle.write(value)

    def record_wrapper(fd, *args, **kwargs):
        raw = original_fdopen(fd, *args, **kwargs)
        descriptors.append(fd)
        wrappers.append(raw)
        return RawProxy(raw)

    def record_close(fd):
        closes.append(fd)
        return original_close(fd)

    def wrap_gzip(*args, **kwargs):
        return GzipProxy(original_gzip_open(*args, **kwargs))

    def fail_operation(*args):
        if stage == "replace":
            assert wrappers[0].closed
            assert not _is_open(descriptors[0])
            other = os.open(os.devnull, os.O_RDONLY)
            if other != descriptors[0]:
                os.dup2(other, descriptors[0])
                original_close(other)
            reused.append(descriptors[0])
        raise failure

    def fail_keys():
        yield "new-key"
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", record_wrapper)
            patch.setattr(corpus.os, "close", record_close)
            patch.setattr(corpus.gzip, "open", wrap_gzip)
            if stage == "fsync":
                patch.setattr(corpus.os, "fsync", fail_operation)
            elif stage == "replace":
                patch.setattr(corpus.os, "replace", fail_operation)
            with pytest.raises(KeyboardInterrupt) as caught:
                corpus._write_state_keys(
                    path, fail_keys() if stage == "sorting" else ["new-key"])
            assert caught.value is failure
        assert len(descriptors) == 1
        assert wrappers[0].closed
        assert closes == descriptors
        if stage == "replace":
            assert reused == descriptors
            assert _is_open(reused[0])
        else:
            assert not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup_descriptors(descriptors + reused)


def test_healthy_key_write_commits_complete_sorted_gzip_after_close(
    tmp_path, monkeypatch,
):
    path, _prior = _prior_keys(tmp_path)
    expected = b"a\na\nz\n"
    descriptors = []
    events = []
    original_fdopen = os.fdopen
    original_close = os.close
    original_fsync = os.fsync
    original_replace = os.replace

    def record_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        return original_fdopen(fd, *args, **kwargs)

    def record_fsync(fd):
        assert fd == descriptors[0]
        events.append("fsync")
        return original_fsync(fd)

    def record_close(fd):
        events.append("close")
        return original_close(fd)

    def record_replace(source, destination):
        assert not _is_open(descriptors[0])
        # Decompression verifies the CRC/size trailer exists at publication.
        with open(source, "rb") as handle:
            assert gzip.decompress(handle.read()) == expected
        events.append("replace")
        return original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(corpus.os, "fdopen", record_wrapper)
        patch.setattr(corpus.os, "close", record_close)
        patch.setattr(corpus.os, "fsync", record_fsync)
        patch.setattr(corpus.os, "replace", record_replace)
        corpus._write_state_keys(path, iter(["z", "a", "a"]))
    assert events == ["fsync", "close", "replace"]
    assert gzip.decompress(path.read_bytes()) == expected
    assert corpus._read_state_keys(path) == {"a", "z"}
    assert not list(tmp_path.glob("*.tmp"))


def test_healthy_key_close_failure_prevents_publication(tmp_path, monkeypatch):
    path, prior = _prior_keys(tmp_path)
    failure = OSError("raw close reported failure")
    descriptors = []
    original_close = os.close

    def fail_close(fd):
        descriptors.append(fd)
        original_close(fd)
        raise failure

    def unexpected_replace(*args):
        raise AssertionError("close failure must prevent publication")

    with monkeypatch.context() as patch:
        patch.setattr(corpus.os, "close", fail_close)
        patch.setattr(corpus.os, "replace", unexpected_replace)
        with pytest.raises(OSError) as caught:
            corpus._write_state_keys(path, ["new-key"])
        assert caught.value is failure
    assert len(descriptors) == 1
    assert not _is_open(descriptors[0])
    assert path.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))
