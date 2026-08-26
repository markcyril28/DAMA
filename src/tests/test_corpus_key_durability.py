"""Proofread 2026-08-25 C2: validation key files must be durable and atomic.

``_write_state_keys`` gz output is neither fsynced nor atomically placed, while
the manifest commit is treated as "the single commit point".  Power loss can
leave the committed manifest pointing at a truncated gz, which is a permanent,
unrecoverable split failure.  The writer must follow the project-wide
temp + fsync + os.replace contract (run_status._write_json_atomic semantics).
"""

import gzip
import os
from pathlib import Path
from unittest import mock

import pytest

from dama.ai.ml.corpus import _read_state_keys, _write_state_keys


def test_write_state_keys_is_round_trip_stable(tmp_path: Path) -> None:
    path = tmp_path / "keys.txt.gz"

    _write_state_keys(path, ["k3", "k1", "k2"])

    assert _read_state_keys(path) == {"k1", "k2", "k3"}
    # Sorted on disk so lineage diffs are deterministic.
    raw = gzip.decompress(path.read_bytes()).decode("ascii")
    assert raw == "k1\nk2\nk3\n"


def test_write_state_keys_leaves_no_truncated_target_on_failure(
    tmp_path: Path,
) -> None:
    """A mid-write failure must not destroy or corrupt an existing key file."""
    path = tmp_path / "canonical_state_keys.txt.gz"
    _write_state_keys(path, ["old"])
    before = path.read_bytes()

    with mock.patch.object(
        gzip.GzipFile, "write", side_effect=OSError("disk full")
    ):
        with pytest.raises(OSError):
            _write_state_keys(path, ["new-a", "new-b"])

    assert path.read_bytes() == before


def test_write_state_keys_fsyncs_before_the_commit_point(
    tmp_path: Path,
) -> None:
    """The bytes must be flushed to the file, then fsynced, before rename."""
    path = tmp_path / "keys.txt.gz"
    events = []
    real_fsync = os.fsync

    def tracking_fsync(fd):
        events.append(("fsync", fd))
        return real_fsync(fd)

    with mock.patch.object(os, "fsync", side_effect=tracking_fsync), \
            mock.patch.object(os, "replace", side_effect=os.replace) as replace_mock:
        _write_state_keys(path, ["k1", "k2"])

    assert events, "gz payload was never fsynced"
    replace_mock.assert_called_once()
    committed_name = replace_mock.call_args.args[1]
    assert Path(committed_name) == path
