"""Prefetched tensors must join the consumer stream on their target device."""

import pytest
import torch

from dama.ai.ml.dataset import CUDAPrefetcher


class _DeviceStream:
    def __init__(self, device):
        self.device = device
        self.waits = []

    def wait_stream(self, stream):
        self.waits.append(stream)


class _CudaTensor(torch.Tensor):
    """Exercise stream ownership on CPU hosts without a second physical GPU."""

    @property
    def is_cuda(self):
        return True

    def record_stream(self, stream):
        self.recorded_stream = stream


@pytest.mark.parametrize("target,current", [(0, 0), (1, 0), (0, 1)])
@pytest.mark.parametrize("indexed", [False, True])
def test_prefetch_wait_and_storage_lifetime_use_target_device(
    monkeypatch, target, current, indexed,
):
    consumers = [_DeviceStream(torch.device(f"cuda:{index}")) for index in range(2)]
    producer = _DeviceStream(torch.device(f"cuda:{target}"))

    def current_stream(device=None):
        index = current if device is None else torch.device(device).index
        if index is None:
            index = current
        return consumers[index]

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    gpu_tensor = torch.ones(2).as_subclass(_CudaTensor)
    batch = (gpu_tensor, torch.zeros(2), "metadata")
    prefetcher = CUDAPrefetcher.__new__(CUDAPrefetcher)
    prefetcher.device = torch.device(f"cuda:{target}" if indexed else "cuda")
    prefetcher.stream = producer
    prefetcher._next_batch = batch
    advances = []
    monkeypatch.setattr(prefetcher, "_prefetch", lambda: advances.append(True))

    assert next(prefetcher) is batch
    assert consumers[target].waits == [producer]
    assert consumers[1 - target].waits == []
    assert gpu_tensor.recorded_stream is consumers[target]
    assert advances == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer integration")
def test_prefetch_preserves_batches_on_explicit_cuda_device():
    device = torch.device("cuda", torch.cuda.current_device())
    batches = [(torch.arange(6).reshape(2, 3).pin_memory(), "first"),
               (torch.arange(3).reshape(1, 3).pin_memory(), "last")]
    prefetcher = CUDAPrefetcher(batches, device)

    actual = list(prefetcher)

    assert len(actual) == len(prefetcher) == 2
    for (tensor, label), (expected, expected_label) in zip(actual, batches):
        assert tensor.device == device
        assert torch.equal(tensor.cpu(), expected)
        assert label == expected_label
