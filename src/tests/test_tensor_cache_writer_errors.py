"""Tensor-cache cleanup preserves the first error and the previous cache."""

import gzip
from pathlib import Path
import tempfile

import pytest
import torch

from dama.ai.ml import dataset as dataset_module
from dama.ai.ml.dataset import CachedTensorDataset
from dama.ai.ml.move_encoder import BOARD_PLANES, MOVE_FEATURE_SIZE


def _dataset():
    return CachedTensorDataset(
        torch.zeros(2, BOARD_PLANES, 8, 8),
        torch.zeros(2, 4, MOVE_FEATURE_SIZE),
        torch.full((2,), 4, dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
    )


@pytest.mark.parametrize("stage", ["raw_close", "gzip_close", "unlink"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_cache_cleanup_preserves_serialization_failure(
    tmp_path, monkeypatch, stage, error_type,
):
    target = tmp_path / "cache.pt"
    prior = b"previous complete cache"
    target.write_bytes(prior)
    primary = error_type("serialization failed first")
    secondary = OSError("cache cleanup failed second")
    real_temporary = tempfile.NamedTemporaryFile
    real_gzip = gzip.GzipFile
    real_unlink = Path.unlink
    streams = []
    cleanup_calls = []

    class FailingClose:
        def __init__(self, stream):
            self.stream = stream
            streams.append(stream)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def close(self):
            cleanup_calls.append(stage)
            self.stream.close()
            raise secondary

    def fail_serialization(_payload, stream):
        stream.write(b"partial cache")
        raise primary

    def fail_unlink(path, *args, **kwargs):
        cleanup_calls.append(stage)
        # Release the name before reporting failure, independently of the
        # filesystem's open-file unlink behavior.
        real_unlink(path, *args, **kwargs)
        raise secondary

    monkeypatch.setattr(dataset_module.torch, "save", fail_serialization)
    if stage == "raw_close":
        monkeypatch.setattr(
            tempfile, "NamedTemporaryFile",
            lambda **kwargs: FailingClose(real_temporary(**kwargs)),
        )
    elif stage == "gzip_close":
        monkeypatch.setattr(
            dataset_module.gzip, "GzipFile",
            lambda **kwargs: FailingClose(real_gzip(**kwargs)),
        )
    else:
        monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(error_type) as caught:
        _dataset().save(str(target), compress=stage == "gzip_close")

    assert caught.value is primary
    assert cleanup_calls == [stage]
    assert all(stream.closed for stream in streams)
    assert target.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("compressed", [False, True])
def test_cache_rejects_close_failure_after_successful_serialization(
    tmp_path, monkeypatch, compressed,
):
    target = tmp_path / "cache.pt"
    prior = b"previous complete cache"
    target.write_bytes(prior)
    secondary = OSError("healthy cache close failed")
    real_temporary = tempfile.NamedTemporaryFile
    real_gzip = gzip.GzipFile
    streams = []

    class FailingClose:
        def __init__(self, stream):
            self.stream = stream
            streams.append(stream)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def close(self):
            self.stream.close()
            raise secondary

    if compressed:
        monkeypatch.setattr(
            dataset_module.gzip, "GzipFile",
            lambda **kwargs: FailingClose(real_gzip(**kwargs)),
        )
    else:
        monkeypatch.setattr(
            tempfile, "NamedTemporaryFile",
            lambda **kwargs: FailingClose(real_temporary(**kwargs)),
        )

    with pytest.raises(OSError) as caught:
        _dataset().save(str(target), compress=compressed)

    assert caught.value is secondary
    assert all(stream.closed for stream in streams)
    assert target.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))
