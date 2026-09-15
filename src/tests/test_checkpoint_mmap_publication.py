"""Checkpoint loading must keep metadata and tensor storage on one revision."""

import os
from pathlib import Path

import pytest
import torch

from dama.ai.ml import model as model_module


def _network(value):
    network = model_module.create_model(
        channels=2, num_blocks=0, embedding_size=4, hidden_size=3,
    ).eval()
    for tensor in network.state_dict().values():
        tensor.fill_(value)
    return network


def _assert_state_equal(actual, expected):
    assert actual.arch_params == expected.arch_params
    for name, tensor in actual.state_dict().items():
        torch.testing.assert_close(tensor, expected.state_dict()[name], rtol=0, atol=0)


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='Linux descriptor paths required')
def test_atomic_replacement_during_mapping_preserves_one_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / 'latest.pt'
    replacement = tmp_path / 'replacement.pt'
    first = _network(1)
    second = _network(2)
    model_module.save_model(first, str(target))
    # Different metadata moves tensor offsets while leaving architecture intact.
    model_module.save_model(second, str(replacement), padding='x' * 256)
    real_from_file = torch.UntypedStorage.from_file
    mapped_paths = []

    def replace_then_map(path, *args, **kwargs):
        mapped_paths.append(path)
        os.replace(replacement, target)
        return real_from_file(path, *args, **kwargs)

    monkeypatch.setattr(torch.UntypedStorage, 'from_file', replace_then_map)
    loaded = model_module.load_model(str(target), 'cpu')

    assert len(mapped_paths) == 1
    _assert_state_equal(loaded, first)
    # The published path has independently advanced to the next checkpoint.
    monkeypatch.setattr(torch.UntypedStorage, 'from_file', real_from_file)
    _assert_state_equal(model_module.load_model(str(target), 'cpu'), second)


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='Linux descriptor paths required')
def test_legacy_retry_keeps_opened_checkpoint_when_path_is_replaced(tmp_path, monkeypatch):
    target = tmp_path / 'legacy.pt'
    replacement = tmp_path / 'replacement.pt'
    first = _network(1)
    second = _network(2)
    model_module.save_model(first, str(target))
    checkpoint = torch.load(target, weights_only=False)
    torch.save(checkpoint, target, _use_new_zipfile_serialization=False)
    model_module.save_model(second, str(replacement))
    real_load = torch.load
    calls = []

    def replace_after_legacy_error(*args, **kwargs):
        calls.append(kwargs.get('mmap'))
        try:
            return real_load(*args, **kwargs)
        except RuntimeError as error:
            assert model_module._LEGACY_MMAP_ERROR in str(error)
            os.replace(replacement, target)
            raise

    monkeypatch.setattr(torch, 'load', replace_after_legacy_error)
    loaded = model_module.load_model(str(target), 'cpu')

    assert calls == [True, None]
    _assert_state_equal(loaded, first)


def _hide_procfs(monkeypatch):
    real_exists = Path.exists

    def exists(path):
        return False if str(path).startswith('/proc/self/fd/') else real_exists(path)

    monkeypatch.setattr(Path, 'exists', exists)


@pytest.mark.parametrize('replace_target', [
    False,
    pytest.param(True, marks=pytest.mark.skipif(
        os.name == 'nt', reason='Windows may deny replacement of an open checkpoint',
    )),
])
def test_without_procfs_loads_same_open_revision_eagerly(
    tmp_path, monkeypatch, replace_target,
):
    target = tmp_path / 'latest.pt'
    replacement = tmp_path / 'replacement.pt'
    first = _network(1)
    second = _network(2)
    model_module.save_model(first, str(target))
    model_module.save_model(second, str(replacement))
    _hide_procfs(monkeypatch)
    real_load = torch.load
    inputs = []

    def replace_then_load(source, **kwargs):
        inputs.append(source)
        assert kwargs.get('mmap') is None
        if replace_target:
            os.replace(replacement, target)
        return real_load(source, **kwargs)

    monkeypatch.setattr(torch, 'load', replace_then_load)
    loaded = model_module.load_model(str(target), 'cpu')

    _assert_state_equal(loaded, first)
    assert len(inputs) == 1 and inputs[0].closed


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='Linux descriptor paths required')
@pytest.mark.parametrize('load_mode', ['mmap', 'legacy', 'no_procfs'])
def test_failed_load_closes_pinned_descriptor_and_propagates_error(
    tmp_path, monkeypatch, load_mode,
):
    target = tmp_path / 'corrupt.pt'
    target.write_bytes(b'invalid checkpoint')
    if load_mode == 'no_procfs':
        _hide_procfs(monkeypatch)
    inputs = []
    error = RuntimeError('checkpoint payload is corrupt')

    def fail_load(source, **kwargs):
        inputs.append(source)
        if load_mode == 'legacy' and kwargs.get('mmap'):
            raise RuntimeError(model_module._LEGACY_MMAP_ERROR)
        raise error

    monkeypatch.setattr(torch, 'load', fail_load)
    with pytest.raises(RuntimeError) as caught:
        model_module.load_model(str(target), 'cpu')

    assert caught.value is error
    assert len(inputs) == (2 if load_mode == 'legacy' else 1)
    for source in inputs:
        if hasattr(source, 'closed'):
            assert source.closed
        else:
            assert not source.exists()
