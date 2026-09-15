"""Temporary-writer cleanup retains the original failure and owned resources."""

import os
from pathlib import Path

import pytest

from dama.ai.ml import fork_writers


class _FailingCloseTemporary:
    def __init__(self, handle, close_error):
        self.handle = handle
        self.close_error = close_error
        self.close_calls = 0

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def close(self):
        self.close_calls += 1
        self.handle.close()
        if self.close_error is not None:
            raise self.close_error


def _failing_temporaries(monkeypatch, close_error):
    real_temporary = fork_writers.tempfile.NamedTemporaryFile
    opened = []

    def temporary(**kwargs):
        handle = _FailingCloseTemporary(real_temporary(**kwargs), close_error)
        opened.append(handle)
        return handle

    monkeypatch.setattr(fork_writers.tempfile, "NamedTemporaryFile", temporary)
    return opened


@pytest.mark.parametrize(
    "primary_type", [OSError, MemoryError, KeyboardInterrupt, SystemExit, None],
)
def test_temporary_close_error_preserves_body_failure_and_refuses_publication(
    tmp_path, monkeypatch, primary_type,
):
    primary = primary_type("write interrupted") if primary_type else None
    secondary = OSError("close failed after releasing the descriptor")
    opened = _failing_temporaries(monkeypatch, secondary)
    target = tmp_path / "checkpoint.pt"
    prior = b"prior checkpoint"
    target.write_bytes(prior)
    temporary_name = None
    try:
        with pytest.raises(primary_type or OSError) as caught:
            with fork_writers.fork_safe_temporary_file(
                dir=tmp_path, delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(b"partial checkpoint")
                if primary is not None:
                    raise primary
            os.replace(temporary_name, target)
        assert caught.value is (primary if primary is not None else secondary)
        assert len(opened) == 1 and opened[0].handle.closed
        assert opened[0].close_calls == 1
        assert not fork_writers._FORK_CHILD_DROPPED_FDS
        assert target.read_bytes() == prior
    finally:
        # Once yielded, the existing publication callers own the temp name.
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


@pytest.mark.parametrize("close_fails", [False, True])
def test_failed_registration_cleans_name_not_yet_delivered_to_caller(
    tmp_path, monkeypatch, close_fails,
):
    primary = MemoryError("writer registration failed")
    secondary = OSError("close failed after releasing the descriptor")
    opened = _failing_temporaries(monkeypatch, secondary if close_fails else None)

    class FailedRegistration(set):
        def add(self, descriptor):
            raise primary

    monkeypatch.setattr(fork_writers, "_FORK_CHILD_DROPPED_FDS", FailedRegistration())

    with pytest.raises(MemoryError) as caught:
        with fork_writers.fork_safe_temporary_file(dir=tmp_path, delete=False):
            pytest.fail("failed registration must not enter the write body")

    assert caught.value is primary
    assert len(opened) == 1 and opened[0].handle.closed
    assert opened[0].close_calls == 1
    assert not fork_writers._FORK_CHILD_DROPPED_FDS
    assert list(tmp_path.iterdir()) == []
