"""Public model saves must preserve the last complete checkpoint on failure."""

import os
from pathlib import Path

import pytest
import torch

from dama.ai.ml import model as model_module
from dama.ai.ml import fork_writers
from dama.ai.ml.move_encoder import ENCODING_VERSION


@pytest.fixture
def network():
    return model_module.create_model(
        channels=2, num_blocks=0, embedding_size=4, hidden_size=3,
        value_head_enabled=True, value_head_hidden=2,
    ).eval()


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, KeyboardInterrupt])
def test_failed_serialization_preserves_existing_checkpoint(tmp_path, network, error_type):
    target = tmp_path / "latest.pt"
    prior = b"last complete checkpoint"
    target.write_bytes(prior)
    error = error_type("serialization failed")

    class InvalidMetadata:
        def __reduce__(self):
            raise error

    with pytest.raises(error_type) as caught:
        model_module.save_model(network, str(target), metadata=InvalidMetadata())

    assert caught.value is error
    assert target.read_bytes() == prior
    assert list(tmp_path.iterdir()) == [target]


def test_checkpoint_readers_keep_old_revision_until_publication(tmp_path, network, monkeypatch):
    target = tmp_path / "latest.pt"
    prior = b"last complete checkpoint"
    target.write_bytes(prior)
    real_save = torch.save

    def serialize(checkpoint, destination, *args, **kwargs):
        result = real_save(checkpoint, destination, *args, **kwargs)
        assert target.read_bytes() == prior
        return result

    monkeypatch.setattr(torch, "save", serialize)
    model_module.save_model(network, str(target), step=27)

    assert torch.load(target, weights_only=False)["step"] == 27
    assert list(tmp_path.iterdir()) == [target]


def test_saved_checkpoint_preserves_architecture_weights_and_metadata(tmp_path, network):
    target = tmp_path / "latest.pt"
    model_module.save_model(network, str(target), step=27, loss=0.125)
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    restored = model_module.load_model(str(target), "cpu")

    assert checkpoint["encoding_version"] == ENCODING_VERSION
    assert checkpoint["step"] == 27
    assert checkpoint["loss"] == 0.125
    assert restored.arch_params == network.arch_params
    for name, expected in network.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("failure", ["error", "interrupt", "short_write"])
def test_partial_write_preserves_checkpoint_and_cleans_temporary(
    tmp_path, network, monkeypatch, failure,
):
    target = tmp_path / "latest.pt"
    prior = b"last complete checkpoint"
    target.write_bytes(prior)
    error = KeyboardInterrupt("write interrupted") if failure == "interrupt" else OSError("write failed")
    real_temporary = fork_writers.tempfile.NamedTemporaryFile
    opened = []

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, data):
            assert self.stream.fileno() in fork_writers._FORK_CHILD_DROPPED_FDS
            self.stream.write(data[:16])
            assert target.read_bytes() == prior
            if failure == "short_write":
                return 16
            raise error

    def open_writer(*args, **kwargs):
        stream = real_temporary(*args, **kwargs)
        opened.append(stream)
        assert Path(stream.name).parent == target.parent
        return FailingWriter(stream)

    monkeypatch.setattr(fork_writers.tempfile, "NamedTemporaryFile", open_writer)
    with pytest.raises(type(error)) as caught:
        model_module.save_model(network, str(target))

    if failure != "short_write":
        assert caught.value is error
    else:
        assert "short checkpoint write" in str(caught.value)
    assert len(opened) == 1 and opened[0].closed
    assert not fork_writers._FORK_CHILD_DROPPED_FDS
    assert target.read_bytes() == prior
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_failed_checkpoint_publication_preserves_existing_file(
    tmp_path, network, monkeypatch, operation,
):
    target = tmp_path / "latest.pt"
    prior = b"last complete checkpoint"
    target.write_bytes(prior)
    error = OSError(f"{operation} failed")

    def fail(*args, **kwargs):
        assert target.read_bytes() == prior
        raise error

    monkeypatch.setattr(os, operation, fail)
    with pytest.raises(OSError) as caught:
        model_module.save_model(network, str(target))

    assert caught.value is error
    assert not fork_writers._FORK_CHILD_DROPPED_FDS
    assert target.read_bytes() == prior
    assert list(tmp_path.iterdir()) == [target]
