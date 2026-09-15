"""An interrupted ledger rewrite releases its source without publishing output."""

import gzip
import io
import os
from pathlib import Path

import pytest

import dama.ai.ml.corpus as corpus


@pytest.mark.parametrize("failure_site", ["temporary_open", "stream_init", "gzip_init", "write"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_output_failure_closes_suspended_ledger_reader(
    tmp_path, monkeypatch, failure_site, error_type,
):
    path = tmp_path / "trained_state_keys.txt.gz"
    # Exceed a complete output batch, keeping the input generator suspended
    # when the first write fails. A tiny ledger would exhaust and close first.
    payload = gzip.compress(
        b"".join(f"{index:064x}\n".encode("ascii") for index in range(20000)),
        compresslevel=1,
    )
    path.write_bytes(payload)
    failure = error_type("injected ledger output failure")
    source_handles = []
    output_handles = []
    real_path_open = Path.open
    real_gzip_open = gzip.open
    real_mkstemp = corpus.tempfile.mkstemp
    real_fdopen = os.fdopen

    def fail_while_source_is_open():
        assert source_handles and all(not handle.closed for handle in source_handles)
        raise failure

    def tracking_path_open(candidate, mode="r", *args, **kwargs):
        handle = real_path_open(candidate, mode, *args, **kwargs)
        if candidate == path and mode == "rb":
            source_handles.append(handle)
        return handle

    def tracking_mkstemp(**kwargs):
        if failure_site == "temporary_open":
            fail_while_source_is_open()
        return real_mkstemp(**kwargs)

    def tracking_fdopen(descriptor, *args, **kwargs):
        if failure_site == "stream_init":
            fail_while_source_is_open()
        handle = real_fdopen(descriptor, *args, **kwargs)
        output_handles.append(handle)
        return handle

    class FailingWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def write(self, value):
            # Leave real partial output so cleanup must remove a written file.
            self.handle.write(value[:1])
            self.handle.flush()
            fail_while_source_is_open()

        def __exit__(self, exc_type, exc, traceback):
            return self.close()

        def close(self):
            return self.handle.close()

    def tracking_gzip_open(target, mode="rb", *args, **kwargs):
        if mode == "wt" and failure_site == "gzip_init":
            fail_while_source_is_open()
        handle = real_gzip_open(target, mode, *args, **kwargs)
        if mode == "rt":
            source_handles.append(handle)
        elif mode == "wt":
            output_handles.append(handle)
            if failure_site == "write":
                return FailingWriter(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_path_open)
    monkeypatch.setattr(corpus.tempfile, "mkstemp", tracking_mkstemp)
    monkeypatch.setattr(corpus.os, "fdopen", tracking_fdopen)
    monkeypatch.setattr(gzip, "open", tracking_gzip_open)
    with pytest.raises(error_type) as raised:
        corpus._merge_state_keys_file(path, ["f" * 64])

    assert raised.value is failure
    # Retaining both traceback and handles prevents reference counting from
    # disguising a source descriptor left open in a suspended generator.
    assert raised.value.__traceback__ is not None
    with real_path_open(path, "rb") as verification:
        assert verification.read() == payload
    assert not list(tmp_path.glob("*.tmp"))
    assert all(handle.closed for handle in output_handles)
    assert len(source_handles) == 2
    assert all(handle.closed for handle in source_handles)


@pytest.mark.parametrize("cleanup_error_type", [OSError, RuntimeError, KeyboardInterrupt])
def test_source_close_error_does_not_mask_output_failure(
    tmp_path, monkeypatch, cleanup_error_type,
):
    path = tmp_path / "trained_state_keys.txt.gz"
    payload = gzip.compress(b"00\n11\n22\n")
    path.write_bytes(payload)
    primary = OSError("injected output open failure")
    cleanup = cleanup_error_type("injected source close failure")
    source_handles = []
    close_attempts = []
    real_open = Path.open

    def tracking_open(candidate, mode="r", *args, **kwargs):
        handle = real_open(candidate, mode, *args, **kwargs)
        if candidate == path and mode == "rb":
            source_handles.append(handle)
            real_close = handle.close

            def close_then_fail():
                close_attempts.append(True)
                real_close()
                raise cleanup

            monkeypatch.setattr(handle, "close", close_then_fail)
        return handle

    def failing_mkstemp(**kwargs):
        raise primary

    monkeypatch.setattr(Path, "open", tracking_open)
    monkeypatch.setattr(corpus.tempfile, "mkstemp", failing_mkstemp)
    with pytest.raises(OSError) as raised:
        corpus._merge_state_keys_file(path, ["ff"])

    assert raised.value is primary
    assert raised.value.__traceback__ is not None
    with real_open(path, "rb") as verification:
        assert verification.read() == payload
    assert not list(tmp_path.glob("*.tmp"))
    assert close_attempts
    assert source_handles and all(handle.closed for handle in source_handles)


@pytest.mark.parametrize("existing", [False, True])
def test_successful_merge_closes_reader_and_preserves_union(tmp_path, monkeypatch, existing):
    path = tmp_path / "trained_state_keys.txt.gz"
    if existing:
        path.write_bytes(gzip.compress(b"00\n22\n"))
    source_handles = []
    real_open = Path.open

    def tracking_open(candidate, mode="r", *args, **kwargs):
        handle = real_open(candidate, mode, *args, **kwargs)
        if candidate == path and mode == "rb":
            source_handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)
    assert corpus._merge_state_keys_file(path, ["33", "22", "11", "33"]) == (2 if existing else 3)
    assert len(source_handles) == int(existing)
    assert all(handle.closed for handle in source_handles)
    expected = b"00\n11\n22\n33\n" if existing else b"11\n22\n33\n"
    assert gzip.decompress(path.read_bytes()) == expected
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("existing", [False, True])
def test_empty_additions_leave_ledger_untouched(tmp_path, monkeypatch, existing):
    path = tmp_path / "trained_state_keys.txt.gz"
    payload = gzip.compress(b"00\n22\n")
    if existing:
        path.write_bytes(payload)
    real_open = Path.open

    def reject_ledger_open(candidate, *args, **kwargs):
        if candidate in (path, path.with_suffix(".gz.tmp")):
            raise AssertionError("empty additions must not open a ledger stream")
        return real_open(candidate, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", reject_ledger_open)
        patch.setattr(corpus.tempfile, "mkstemp", lambda **kwargs: pytest.fail(
            "empty additions must not create a temporary"))
        assert corpus._merge_state_keys_file(path, iter(())) == 0
    assert path.exists() is existing
    if existing:
        assert path.read_bytes() == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_partial_merge_wrapper_keeps_raw_ownership(tmp_path, monkeypatch):
    path = tmp_path / "trained_state_keys.txt.gz"
    payload = gzip.compress(b"00\n22\n")
    path.write_bytes(payload)
    failure = MemoryError("partial merge stream construction")
    descriptors = []
    unrelated = []

    def partial_wrapper(descriptor, *args, **kwargs):
        descriptors.append(descriptor)
        raw = io.FileIO(descriptor, "wb", closefd=kwargs.get("closefd", True))
        raw.close()
        unrelated.append(os.open(tmp_path / "unrelated", os.O_CREAT | os.O_RDWR, 0o600))
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(corpus.os, "fdopen", partial_wrapper)
            with pytest.raises(MemoryError) as caught:
                corpus._merge_state_keys_file(path, ["11"])
        assert caught.value is failure
        assert len(descriptors) == len(unrelated) == 1
        assert unrelated[0] != descriptors[0]
        assert os.write(unrelated[0], b"owned") == 5
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
        assert path.read_bytes() == payload
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        for descriptor in set(descriptors + unrelated):
            try:
                os.close(descriptor)
            except OSError:
                pass
