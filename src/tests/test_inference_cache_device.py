"""Cached models must follow the concrete GPU requested by each caller."""

from types import SimpleNamespace

import pytest
import torch

from dama.ai.ml import inference


@pytest.fixture
def model_loader(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    current = SimpleNamespace(index=0, loads=[], folds=[])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current.index)

    def load(path, device):
        # Like real tensor placement, an unindexed CUDA request uses the
        # caller's current GPU when the model is loaded.
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", current.index)
        model = SimpleNamespace(device=device)
        current.loads.append(model)
        return model

    monkeypatch.setattr(inference, "load_model", load)
    monkeypatch.setattr(inference, "fold_batchnorm", current.folds.append)
    inference.clear_model_cache()
    yield str(checkpoint), current
    inference.clear_model_cache()


@pytest.mark.parametrize("device", [None, "cuda", torch.device("cuda")])
def test_current_cuda_device_change_selects_its_own_cached_model(model_loader, device):
    path, current = model_loader
    first = inference.get_model(path, device)
    current.index = 1
    second = inference.get_model(path, device)

    assert first.device == torch.device("cuda:0")
    assert second.device == torch.device("cuda:1")
    assert second is not first
    current.index = 0
    assert inference.get_model(path, device) is first
    assert len(current.loads) == 2


def test_indexed_and_current_cuda_requests_share_the_same_model(model_loader):
    path, current = model_loader
    current.index = 1
    explicit = inference.get_model(path, "cuda:1")
    assert inference.get_model(path, "cuda") is explicit
    current.index = 0
    assert inference.get_model(path, "cuda:1") is explicit
    assert len(current.loads) == 1


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
def test_cpu_cache_never_queries_cuda_current_device(model_loader, monkeypatch, device):
    path, current = model_loader
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unavailable():
        raise AssertionError("CPU inference must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", unavailable)
    model = inference.get_model(path, device)
    assert model.device == torch.device("cpu")
    assert inference.get_model(path, device) is model
    assert current.folds == [model]
