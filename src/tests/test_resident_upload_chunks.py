"""Snapshot uploads bound staging memory while preserving resident contents."""

import pytest
import torch

import dama.ai.ml.dataset as dataset_module
from dama.ai.ml.dataset import CachedTensorDataset, FastBatchIterator
from dama.ai.ml.move_encoder import BOARD_PLANES, MOVE_FEATURE_SIZE


_FIELDS = (
    "boards", "move_features", "move_counts", "targets",
    "reward_weights", "value_targets",
)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
def test_chunked_copy_preserves_conversion_and_slice_boundaries(
    monkeypatch, count, dtype, layout,
):
    # Two rows fit, except in the explicit smaller-than-one-row case below.
    row_elements = BOARD_PLANES * 64
    monkeypatch.setattr(dataset_module, "_GPU_UPLOAD_CHUNK_BYTES", row_elements * 8)
    source = torch.arange(count * row_elements).reshape(count, BOARD_PLANES, 8, 8).float() / 31
    storage = torch.full((count + 2, BOARD_PLANES, 8, 8), -7, dtype=dtype)
    storage = storage.contiguous(memory_format=layout)
    dataset_module._copy_resident_tensor(storage[1:count + 1], source, layout)
    assert torch.equal(storage[1:count + 1], source.to(dtype))
    assert torch.all(storage[0] == -7) and torch.all(storage[-1] == -7)
    assert storage.is_contiguous(memory_format=layout)


def test_chunk_smaller_than_one_row_still_copies_strided_source(monkeypatch):
    monkeypatch.setattr(dataset_module, "_GPU_UPLOAD_CHUNK_BYTES", 1)
    source = torch.arange(24, dtype=torch.int32).reshape(4, 6)[:, ::2]
    target = torch.empty_like(source)
    dataset_module._copy_resident_tensor(target, source)
    assert torch.equal(target, source)


def _dataset(count):
    boards = torch.arange(BOARD_PLANES * 64).reshape(1, BOARD_PLANES, 8, 8).float() % 2
    features = torch.arange(32 * MOVE_FEATURE_SIZE).reshape(1, 32, MOVE_FEATURE_SIZE).float() / 641
    return CachedTensorDataset(
        boards.repeat(count, 1, 1, 1), features.repeat(count, 1, 1),
        torch.full((count,), 32, dtype=torch.int32),
        torch.arange(count, dtype=torch.int32) % 32,
        torch.arange(count).float() % 17 / 7,
        torch.arange(count).float() % 3 - 1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA upload allocation contract")
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("mode", ["replace", "grow"])
def test_full_upload_bounds_cuda_staging_and_preserves_all_rows(amp, mode):
    initial = _dataset(8)
    replacement = _dataset(131073)
    loader = FastBatchIterator(
        initial, 2048, device=torch.device("cuda"), amp_enabled=amp,
        capacity=len(replacement) if mode == "replace" else 0,
    )
    assert loader.on_gpu
    pointers = [getattr(loader, "_" + field).data_ptr() for field in _FIELDS]
    torch.cuda.synchronize()
    starting = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    loader.replace_data(replacement)
    torch.cuda.synchronize()
    resident = sum(getattr(loader, "_" + field).numel()
                   * getattr(loader, "_" + field).element_size() for field in _FIELDS)
    growth = resident if mode == "grow" else 0
    # Dtype/layout conversion can need source and destination scratch. Allow
    # both bounded chunks plus allocator rounding, never a full field copy.
    staging_allowance = 2 * dataset_module._GPU_UPLOAD_CHUNK_BYTES + 16 * 1024**2
    assert torch.cuda.max_memory_allocated() - starting <= growth + staging_allowance
    if mode == "replace":
        assert pointers == [getattr(loader, "_" + field).data_ptr() for field in _FIELDS]
    assert loader.n == len(replacement)
    assert loader._boards.is_contiguous(memory_format=torch.channels_last)
    for field in _FIELDS:
        actual = getattr(loader, "_" + field)[:loader.n].cpu()
        assert torch.equal(actual, getattr(replacement, field).to(actual.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA startup allocation contract")
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("count", [0, 1, 262145])
def test_preallocated_startup_bounds_staging_and_preserves_all_rows(amp, count):
    source = _dataset(count)
    capacity = count + 17
    torch.cuda.synchronize()
    starting = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    loader = FastBatchIterator(
        source, 2048, shuffle=False, device=torch.device("cuda"),
        capacity=capacity, amp_enabled=amp,
    )
    torch.cuda.synchronize()
    assert loader.on_gpu and loader.n == count
    assert loader.dataset is source
    resident = sum(getattr(loader, "_" + field).numel()
                   * getattr(loader, "_" + field).element_size() for field in _FIELDS)
    # Startup must fit the final buffers plus bounded conversion staging,
    # including when the initial snapshot already spans multiple chunks.
    staging_allowance = 2 * dataset_module._GPU_UPLOAD_CHUNK_BYTES + 16 * 1024**2
    assert torch.cuda.max_memory_allocated() - starting <= resident + staging_allowance
    assert loader._boards.is_contiguous(memory_format=torch.channels_last)
    for field in _FIELDS:
        storage = getattr(loader, "_" + field)
        assert len(storage) == capacity
        assert torch.equal(storage[:count].cpu(), getattr(source, field).to(storage.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA copy staging contract")
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_direct_upload_bounds_staging_and_orders_current_stream(pinned, dtype):
    count = 65537  # One full 64 MiB source chunk and a one-row tail.
    source = torch.arange(count * BOARD_PLANES * 64, dtype=torch.int32)
    source = source.remainder_(257).float().div_(251).reshape(count, BOARD_PLANES, 8, 8)
    if pinned:
        source = source.pin_memory()
    storage = torch.empty(
        count + 2, BOARD_PLANES, 8, 8, device="cuda", dtype=dtype,
        memory_format=torch.channels_last,
    ).fill_(-7)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    starting = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.stream(stream):
        dataset_module._copy_resident_tensor(storage[1:-1], source, torch.channels_last)
        # A consumer on the caller's stream must see both the first chunk and
        # the tail without any synchronization inside the upload helper.
        consumed = storage[1, 0, 0, 0] + storage[-2, -1, -1, -1]
    stream.synchronize()
    # Conversion may stage a source chunk, but must not additionally retain a
    # converted output chunk before copying into the existing destination.
    assert torch.cuda.max_memory_allocated() - starting <= (
        dataset_module._GPU_UPLOAD_CHUNK_BYTES + 8 * 1024**2)
    assert torch.equal(storage[1:-1].cpu(), source.to(dtype))
    expected = source[0, 0, 0, 0].to(dtype) + source[-1, -1, -1, -1].to(dtype)
    assert consumed.item() == expected.item()
    assert torch.all(storage[0] == -7) and torch.all(storage[-1] == -7)
    assert storage.is_contiguous(memory_format=torch.channels_last)
