"""Shared JSON publication retains raw ownership through partial construction."""

import errno
import io
import json
import os
from pathlib import Path

import pytest

from dama.ai.ml import run_status


def _is_open(fd):
    try:
        os.fstat(fd)
    except OSError as exc:
        if exc.errno != errno.EBADF:
            raise
        return False
    return True


def _cleanup(descriptors):
    for fd in set(descriptors):
        if _is_open(fd):
            os.close(fd)


def _replacement(tmp_path, fd, close):
    other = os.open(tmp_path / "unrelated", os.O_CREAT | os.O_RDWR, 0o600)
    if other != fd:
        os.dup2(other, fd)
        close(other)
    return fd


def _prior(tmp_path):
    path = tmp_path / "authority.json"
    prior = b'{"old": true}\n'
    path.write_bytes(prior)
    return path, prior


@pytest.mark.parametrize("error_type", [OSError, MemoryError, KeyboardInterrupt, SystemExit])
def test_wrapper_failure_releases_raw_descriptor_and_temporary(
    tmp_path, monkeypatch, error_type,
):
    path, prior = _prior(tmp_path)
    failure = error_type("stream construction failed")
    descriptors = []

    def fail_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(run_status.os, "fdopen", fail_wrapper)
            with pytest.raises(error_type) as caught:
                run_status._write_json_atomic(path, {"new": True})
            assert caught.value is failure
        assert len(descriptors) == 1
        assert not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup(descriptors)


def test_partial_stream_unwind_preserves_unrelated_reopened_descriptor(
    tmp_path, monkeypatch,
):
    path, prior = _prior(tmp_path)
    failure = MemoryError("partial stream construction failed")
    descriptors, unrelated = [], []
    original_close = os.close

    def partial_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        raw = io.FileIO(fd, "wb", closefd=kwargs.get("closefd", True))
        raw.close()
        if _is_open(fd):
            unrelated.append(os.open(tmp_path / "unrelated", os.O_CREAT | os.O_RDWR, 0o600))
        else:
            unrelated.append(_replacement(tmp_path, fd, original_close))
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(run_status.os, "fdopen", partial_wrapper)
            with pytest.raises(MemoryError) as caught:
                run_status._write_json_atomic(path, {"new": True})
            assert caught.value is failure
        assert len(descriptors) == len(unrelated) == 1
        assert _is_open(unrelated[0])
        assert os.write(unrelated[0], b"still-owned") == len(b"still-owned")
        if descriptors[0] != unrelated[0]:
            assert not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup(descriptors + unrelated)


class _StreamProxy:
    def __init__(self, stream, stage=None, failure=None, events=None):
        self.stream, self.stage, self.failure, self.events = stream, stage, failure, events

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.close()

    def close(self):
        result = self.stream.close()
        if self.events is not None:
            self.events.append("stream_close")
        if self.stage == "stream_close":
            raise self.failure
        return result

    def write(self, value):
        if self.stage == "write":
            raise self.failure
        return self.stream.write(value)

    def flush(self):
        if self.events is not None:
            self.events.append("flush")
        if self.stage == "flush":
            raise self.failure
        return self.stream.flush()


@pytest.mark.parametrize("stage", ["encode", "write", "flush", "fsync", "stream_close", "replace"])
def test_interrupted_publication_preserves_prior_json_and_cleans_temporary(
    tmp_path, monkeypatch, stage,
):
    path, prior = _prior(tmp_path)
    failure = KeyboardInterrupt(f"injected {stage} interruption")
    descriptors, streams, unrelated = [], [], []
    original_fdopen, original_close = os.fdopen, os.close

    def wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        streams.append(original_fdopen(fd, *args, **kwargs))
        return _StreamProxy(streams[-1], stage, failure)

    def interrupt(*args, **kwargs):
        if stage == "replace":
            assert streams[0].closed and not _is_open(descriptors[0])
            unrelated.append(_replacement(tmp_path, descriptors[0], original_close))
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(run_status.os, "fdopen", wrapper)
            if stage == "encode":
                patch.setattr(run_status.json, "dump", interrupt)
            elif stage in ("fsync", "replace"):
                patch.setattr(run_status.os, stage, interrupt)
            with pytest.raises(KeyboardInterrupt) as caught:
                run_status._write_json_atomic(path, {"new": True})
            assert caught.value is failure
        assert len(descriptors) == 1 and streams[0].closed
        assert _is_open(unrelated[0]) if unrelated else not _is_open(descriptors[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup(descriptors + unrelated)


@pytest.mark.parametrize("primary_failure", [False, True])
def test_raw_close_error_preserves_primary_error_and_never_retries_reused_number(
    tmp_path, monkeypatch, primary_failure,
):
    path, prior = _prior(tmp_path)
    primary, secondary = MemoryError("primary construction error"), OSError("close after release")
    descriptors, closes, unrelated = [], [], []
    original_close = os.close

    def fail_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        raise primary

    def fail_close(fd):
        closes.append(fd)
        original_close(fd)
        unrelated.append(_replacement(tmp_path, fd, original_close))
        raise secondary

    def unexpected_replace(*args):
        raise AssertionError("raw close failure must prevent publication")

    try:
        with monkeypatch.context() as patch:
            if primary_failure:
                patch.setattr(run_status.os, "fdopen", fail_wrapper)
            patch.setattr(run_status.os, "close", fail_close)
            patch.setattr(run_status.os, "replace", unexpected_replace)
            with pytest.raises(MemoryError if primary_failure else OSError) as caught:
                run_status._write_json_atomic(path, {"new": True})
            assert caught.value is (primary if primary_failure else secondary)
        assert len(closes) == 1 and closes == unrelated
        assert _is_open(unrelated[0])
        assert path.read_bytes() == prior
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        _cleanup(descriptors + closes + unrelated)


@pytest.mark.parametrize("interrupt_directory_sync", [False, True])
def test_complete_json_is_closed_before_replace_and_survives_directory_sync_failure(
    tmp_path, monkeypatch, interrupt_directory_sync,
):
    path, _ = _prior(tmp_path)
    payload = {"z": 2, "a": "\u2713", "nested": {"ok": True}}
    expected = b'{\n  "a": "\\u2713",\n  "nested": {\n    "ok": true\n  },\n  "z": 2\n}\n'
    failure = KeyboardInterrupt("directory sync interrupted after publication")
    events, descriptors = [], []
    original_fdopen, original_close = os.fdopen, os.close
    original_fsync, original_replace = os.fsync, os.replace

    def wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        return _StreamProxy(original_fdopen(fd, *args, **kwargs), events=events)

    def fsync(fd):
        assert fd == descriptors[0]
        events.append("file_fsync")
        return original_fsync(fd)

    def close(fd):
        assert fd == descriptors[0]
        events.append("raw_close")
        return original_close(fd)

    def replace(source, destination):
        assert not _is_open(descriptors[0])
        assert Path(source).read_bytes() == expected
        events.append("replace")
        return original_replace(source, destination)

    def sync_directory(directory):
        assert directory == path.parent and path.read_bytes() == expected
        events.append("directory_fsync")
        if interrupt_directory_sync:
            raise failure

    with monkeypatch.context() as patch:
        patch.setattr(run_status.os, "fdopen", wrapper)
        patch.setattr(run_status.os, "fsync", fsync)
        patch.setattr(run_status.os, "close", close)
        patch.setattr(run_status.os, "replace", replace)
        patch.setattr(run_status, "_fsync_directory", sync_directory)
        if interrupt_directory_sync:
            with pytest.raises(KeyboardInterrupt) as caught:
                run_status._write_json_atomic(path, payload)
            assert caught.value is failure
        else:
            run_status._write_json_atomic(path, payload)
    assert events == ["flush", "file_fsync", "stream_close", "raw_close", "replace", "directory_fsync"]
    assert path.read_bytes() == expected and json.loads(path.read_bytes()) == payload
    assert not list(tmp_path.glob("*.tmp"))
