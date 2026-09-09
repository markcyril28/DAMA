"""Resident buffer allocation must fit its final storage plus upload headroom."""

import pytest
import torch

from dama.ai.ml.dataset import CachedTensorDataset, FastBatchIterator
from dama.ai.ml.move_encoder import BOARD_PLANES, MOVE_FEATURE_SIZE


_FIELDS = (
    "boards", "move_features", "move_counts", "targets",
    "reward_weights", "value_targets",
)


def _dataset(offset):
    return CachedTensorDataset(
        (torch.arange(8 * BOARD_PLANES * 64).reshape(8, BOARD_PLANES, 8, 8)
         % 2).float(),
        (torch.arange(8 * 32 * MOVE_FEATURE_SIZE).reshape(8, 32, MOVE_FEATURE_SIZE)
         % 8).float() / 8,
        torch.full((8,), 32, dtype=torch.int32),
        torch.arange(offset, offset + 8, dtype=torch.int32),
        torch.ones(8),
        torch.arange(8).float() % 3 - 1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocation contract")
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("mode", ["startup", "grow"])
def test_resident_buffer_allocation_has_no_full_board_scratch(amp, mode):
    original = _dataset(0)
    replacement = _dataset(8)
    device = torch.device("cuda")
    if mode == "grow":
        loader = FastBatchIterator(
            original, 4, shuffle=False, device=device, amp_enabled=amp)
        assert loader.on_gpu
    torch.cuda.synchronize()
    starting_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if mode == "startup":
        loader = FastBatchIterator(
            original, 4, shuffle=False, device=device,
            capacity=65536, amp_enabled=amp)
        expected = original
    else:
        loader.update_data(replacement, max_entries=60000)
        expected = original.concat(replacement)
    torch.cuda.synchronize()
    assert loader.on_gpu
    storage_bytes = sum(
        getattr(loader, "_" + field).numel()
        * getattr(loader, "_" + field).element_size()
        for field in _FIELDS
    )
    # Only eight new rows need upload staging. Allow allocator rounding and
    # that small upload, but never another full capacity-sized board tensor.
    peak_growth = torch.cuda.max_memory_allocated() - starting_bytes
    assert peak_growth <= storage_bytes + 2 * 1024**2
    assert loader._boards.is_contiguous(memory_format=torch.channels_last)
    assert loader.n == len(expected)
    for field in _FIELDS:
        actual = getattr(loader, "_" + field)[:loader.n].cpu()
        assert torch.equal(actual, getattr(expected, field).to(actual.dtype))
