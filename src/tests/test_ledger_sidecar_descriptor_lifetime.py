"""Optional ledger cache writers own raw descriptors through stream teardown."""

import errno
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

import dama.ai.ml.corpus as corpus


def _fixture(tmp_path: Path):
    manager = corpus.CorpusSnapshotManager(
        str(tmp_path / "replay"),
        str(tmp_path / "snapshots"),
        trained_ledger_enabled=True,
    )
    manager.trained_ledger_dir.mkdir(parents=True)
    fingerprints = {0, 17, 1 << 63, (1 << 64) - 1}
    corpus._write_state_keys(
        manager._ledger_state_keys_path,
        (f"{value:016x}" + "0" * 48 for value in fingerprints),
    )
    manager._ledger_seed_path.write_text("{}", encoding="utf-8")
    manager._ledger_shards_path.write_text(
        json.dumps({"name": "historical.jsonl"}) + "\n", encoding="utf-8"
    )
    manager._write_ledger_fingerprint_sidecar(fingerprints)
    return manager, fingerprints


def _is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError as error:
        assert error.errno == errno.EBADF
        return False
    return True


def _clean_descriptors(descriptors):
    # Failed baseline controls must not contaminate subsequent tests. Every
    # assertion about the descriptor's lifetime happens before this cleanup.
    for descriptor in descriptors:
        if _is_open(descriptor):
            os.close(descriptor)


@pytest.mark.parametrize("error_type", [OSError, MemoryError, KeyboardInterrupt])
def test_wrapper_failure_closes_raw_sidecar_descriptor(
    tmp_path, monkeypatch, capsys, error_type,
):
    manager, fingerprints = _fixture(tmp_path)
    source_before = manager._ledger_state_keys_path.read_bytes()
    sidecar_before = manager._ledger_fingerprints_path.read_bytes()
    failure = error_type("injected sidecar wrapper failure")
    descriptors = []
    closefd_arguments = []

    def fail_wrapper(fd, *args, **kwargs):
        assert _is_open(fd)
        descriptors.append(fd)
        closefd_arguments.append(kwargs.get("closefd"))
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", fail_wrapper)
            if error_type is OSError:
                manager._write_ledger_fingerprint_sidecar(fingerprints)
                assert "Could not cache trained-ledger fingerprints" in (
                    capsys.readouterr().out)
            else:
                with pytest.raises(error_type) as raised:
                    manager._write_ledger_fingerprint_sidecar(fingerprints)
                assert raised.value is failure
        assert len(descriptors) == 1
        assert not _is_open(descriptors[0])
        assert closefd_arguments == [False]
        assert manager._ledger_state_keys_path.read_bytes() == source_before
        assert manager._ledger_fingerprints_path.read_bytes() == sidecar_before
        assert not list(manager.trained_ledger_dir.glob("*.tmp"))
    finally:
        _clean_descriptors(descriptors)


@pytest.mark.parametrize("error_type", [OSError, MemoryError, KeyboardInterrupt])
def test_raw_descriptor_cleanup_error_preserves_primary_failure(
    tmp_path, monkeypatch, capsys, error_type,
):
    manager, fingerprints = _fixture(tmp_path)
    source_before = manager._ledger_state_keys_path.read_bytes()
    sidecar_before = manager._ledger_fingerprints_path.read_bytes()
    failure = error_type("primary wrapper failure")
    descriptors = []
    close_calls = []
    closefd_arguments = []
    original_close = os.close

    def fail_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        closefd_arguments.append(kwargs.get("closefd"))
        raise failure

    def fail_close(fd):
        close_calls.append(fd)
        # Model an OS error reported after descriptor release. This verifies
        # error precedence without depending on unlinking an open file.
        original_close(fd)
        raise OSError("secondary close failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", fail_wrapper)
            patch.setattr(corpus.os, "close", fail_close)
            if error_type is OSError:
                manager._write_ledger_fingerprint_sidecar(fingerprints)
                warning = capsys.readouterr().out
                assert "primary wrapper failure" in warning
                assert "secondary close failure" not in warning
            else:
                with pytest.raises(error_type) as raised:
                    manager._write_ledger_fingerprint_sidecar(fingerprints)
                assert raised.value is failure
        assert len(descriptors) == 1
        assert close_calls == descriptors
        assert closefd_arguments == [False]
        assert not _is_open(descriptors[0])
        # Cleanup must preserve public artifacts and must not retry a close
        # that reported an error after releasing its descriptor.
        assert manager._ledger_state_keys_path.read_bytes() == source_before
        assert manager._ledger_fingerprints_path.read_bytes() == sidecar_before
        assert not list(manager.trained_ledger_dir.glob("*.tmp"))
    finally:
        _clean_descriptors(descriptors)


@pytest.mark.parametrize("error_type", [OSError, MemoryError, KeyboardInterrupt])
def test_partially_initialized_stream_cannot_close_writer_owned_descriptor(
    tmp_path, monkeypatch, capsys, error_type,
):
    manager, fingerprints = _fixture(tmp_path)
    source_before = manager._ledger_state_keys_path.read_bytes()
    sidecar_before = manager._ledger_fingerprints_path.read_bytes()
    descriptors = []
    raw_survived_stream_close = []
    closefd_arguments = []
    failure = error_type("partial stream initialization failed")

    def partial_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        closefd_arguments.append(kwargs.get("closefd"))
        # io.open can construct a FileIO before a later buffering allocation
        # fails. Its cleanup must not release the writer-owned raw descriptor.
        stream = io.FileIO(fd, "wb", closefd=kwargs.get("closefd", True))
        stream.close()
        raw_survived_stream_close.append(_is_open(fd))
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", partial_wrapper)
            if error_type is OSError:
                manager._write_ledger_fingerprint_sidecar(fingerprints)
                assert "partial stream initialization failed" in capsys.readouterr().out
            else:
                with pytest.raises(error_type) as raised:
                    manager._write_ledger_fingerprint_sidecar(fingerprints)
                assert raised.value is failure
        assert closefd_arguments == [False]
        assert raw_survived_stream_close == [True]
        assert len(descriptors) == 1
        assert not _is_open(descriptors[0])
        assert manager._ledger_state_keys_path.read_bytes() == source_before
        assert manager._ledger_fingerprints_path.read_bytes() == sidecar_before
        assert not list(manager.trained_ledger_dir.glob("*.tmp"))
    finally:
        _clean_descriptors(descriptors)


@pytest.mark.parametrize("stage", ["write", "fsync", "replace", "directory"])
def test_writer_closes_raw_descriptor_once_on_later_failure(
    tmp_path, monkeypatch, capsys, stage,
):
    manager, fingerprints = _fixture(tmp_path)
    source_before = manager._ledger_state_keys_path.read_bytes()
    sidecar_before = manager._ledger_fingerprints_path.read_bytes()
    failure = OSError(f"injected {stage} failure")
    descriptors = []
    wrappers = []
    close_calls = []
    reused_descriptors = []
    closefd_arguments = []
    original_fdopen = os.fdopen
    original_close = os.close

    class FailingWriter:
        def __init__(self, raw):
            self.raw = raw

        def __enter__(self):
            self.raw.__enter__()
            return self

        def __exit__(self, *args):
            return self.close()

        def close(self):
            return self.raw.close()

        def write(self, value):
            raise failure

    def record_wrapper(fd, *args, **kwargs):
        raw = original_fdopen(fd, *args, **kwargs)
        descriptors.append(fd)
        wrappers.append(raw)
        closefd_arguments.append(kwargs.get("closefd"))
        return FailingWriter(raw) if stage == "write" else raw

    def record_close(fd):
        close_calls.append(fd)
        return original_close(fd)

    def fail_operation(*args, **kwargs):
        if stage == "replace":
            # Stream teardown and the writer's explicit raw close must finish
            # before replace. Reuse exposes a second, unsafe raw os.close(fd).
            reused = os.open(os.devnull, os.O_RDONLY)
            reused_descriptors.append(reused)
            assert reused == descriptors[0]
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", record_wrapper)
            patch.setattr(corpus.os, "close", record_close)
            if stage == "fsync":
                patch.setattr(corpus.os, "fsync", fail_operation)
            elif stage == "replace":
                patch.setattr(corpus.os, "replace", fail_operation)
            elif stage == "directory":
                patch.setattr(corpus.run_status, "_fsync_directory", fail_operation)
            manager._write_ledger_fingerprint_sidecar(fingerprints)
        assert len(descriptors) == 1
        assert wrappers[0].closed
        assert closefd_arguments == [False]
        if stage == "replace":
            assert reused_descriptors == descriptors
            assert _is_open(reused_descriptors[0])
        else:
            assert not _is_open(descriptors[0])
        assert close_calls == descriptors
        assert f"injected {stage} failure" in capsys.readouterr().out
        assert manager._ledger_state_keys_path.read_bytes() == source_before
        assert manager._ledger_fingerprints_path.read_bytes() == sidecar_before
        assert not list(manager.trained_ledger_dir.glob("*.tmp"))
        assert manager._load_ledger_fingerprint_sidecar() == fingerprints
    finally:
        _clean_descriptors(set(descriptors + reused_descriptors))


def test_optional_wrapper_failure_keeps_canonical_fallback_authoritative(
    tmp_path, monkeypatch, capsys,
):
    manager, fingerprints = _fixture(tmp_path)
    source_before = manager._ledger_state_keys_path.read_bytes()
    manager._ledger_fingerprints_path.write_bytes(b"invalid optional cache")
    manager._trained_ledger_source_sha256 = None
    descriptors = []

    def fail_wrapper(fd, *args, **kwargs):
        descriptors.append(fd)
        raise OSError("optional cache unavailable")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", fail_wrapper)
            names, actual = manager._load_trained_ledger()
        assert names == {"historical.jsonl"}
        assert actual == fingerprints
        assert manager._trained_ledger_source_sha256 == hashlib.sha256(
            source_before).hexdigest()
        assert len(descriptors) == 1
        assert not _is_open(descriptors[0])
        assert manager._ledger_state_keys_path.read_bytes() == source_before
        assert manager._ledger_fingerprints_path.read_bytes() == (
            b"invalid optional cache")
        assert not list(manager.trained_ledger_dir.glob("*.tmp"))
        assert "optional cache unavailable" in capsys.readouterr().out
        # Once wrapping succeeds, the regular repair restores a verified hit.
        manager._write_ledger_fingerprint_sidecar(fingerprints)
        assert manager._load_ledger_fingerprint_sidecar() == fingerprints
    finally:
        _clean_descriptors(descriptors)


def test_healthy_raw_close_failure_keeps_previous_sidecar(
    tmp_path, monkeypatch, capsys,
):
    manager, fingerprints = _fixture(tmp_path)
    source_before = manager._ledger_state_keys_path.read_bytes()
    sidecar_before = manager._ledger_fingerprints_path.read_bytes()
    descriptors = []
    replace_calls = []
    original_close = os.close

    def fail_close(fd):
        assert _is_open(fd)
        descriptors.append(fd)
        # A close can release its descriptor before reporting an error. Avoid
        # relying on POSIX-only removal of an intentionally open temporary.
        original_close(fd)
        raise OSError("raw close failed after successful serialization")

    def unexpected_replace(*args):
        replace_calls.append(args)
        raise AssertionError("sidecar must not publish after a raw close error")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "close", fail_close)
            patch.setattr(corpus.os, "replace", unexpected_replace)
            manager._write_ledger_fingerprint_sidecar(fingerprints)
        assert len(descriptors) == 1
        assert not _is_open(descriptors[0])
        assert not replace_calls
        assert "raw close failed after successful serialization" in (
            capsys.readouterr().out)
        assert manager._ledger_state_keys_path.read_bytes() == source_before
        assert manager._ledger_fingerprints_path.read_bytes() == sidecar_before
        assert not list(manager.trained_ledger_dir.glob("*.tmp"))
        assert manager._load_ledger_fingerprint_sidecar() == fingerprints
    finally:
        _clean_descriptors(descriptors)
