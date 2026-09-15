"""Full resident growth must release obsolete storage before allocating its successor."""

import weakref

import pytest
import torch

import dama.ai.ml.dataset as dataset_module
from dama.ai.ml.dataset import CachedTensorDataset, FastBatchIterator
from dama.ai.ml.move_encoder import BOARD_PLANES, MOVE_FEATURE_SIZE


_FIELDS = (
    "boards", "move_features", "move_counts", "targets",
    "reward_weights", "value_targets",
)


def _dataset(count=7, offset=0):
    rows = torch.arange(offset, offset + count)
    return CachedTensorDataset(
        (rows[:, None, None, None] % 2).expand(
            count, BOARD_PLANES, 8, 8).float().clone(),
        (rows[:, None, None] / 32).expand(
            count, 4, MOVE_FEATURE_SIZE).clone(),
        torch.full((count,), 4, dtype=torch.int32),
        (rows % 4).to(torch.int64),
        rows.float() / 4 + 1,
        rows.double() / 8 - 1,
    )


def _loader(source, *, amp=False, capacity=0, shuffle=False):
    loader = FastBatchIterator(
        source, batch_size=4, shuffle=shuffle, drop_last=False,
    )
    # Exercise production resident allocation/copies on CPU without requiring
    # CUDA. These buffers are independent of the caller-owned CPU dataset.
    for field in _FIELDS:
        value = getattr(source, field)
        dtype = torch.float16 if amp and field in _FIELDS[:2] else value.dtype
        options = {"memory_format": torch.channels_last} if field == "boards" else {}
        buffer = torch.empty(
            (max(capacity, len(source)), *value.shape[1:]), dtype=dtype, **options)
        buffer[:len(source)].copy_(value)
        setattr(loader, "_" + field, buffer)
    loader.on_gpu = True
    loader._device = torch.device("cpu")
    loader._check_preshuffle_budget = lambda: False
    return loader


def _refs(loader):
    return [weakref.ref(getattr(loader, "_" + field)) for field in _FIELDS]


def _assert_contents(loader, expected, dtypes):
    assert loader.on_gpu
    assert loader.n == len(expected)
    assert loader.drop_last == (len(expected) > loader.batch_size)
    for field, dtype in zip(_FIELDS, dtypes):
        actual = getattr(loader, "_" + field)
        assert actual.dtype == dtype
        assert torch.equal(actual[:loader.n], getattr(expected, field).to(dtype))
    assert loader._boards.is_contiguous(memory_format=torch.channels_last)

    seed = 20260911
    order = (torch.randperm(len(expected), generator=torch.Generator().manual_seed(seed))
             if loader.shuffle else torch.arange(len(expected)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        batches = list(loader)
    count = len(expected)
    if loader.drop_last:
        count -= count % loader.batch_size
    assert len(batches) == len(loader)
    assert sum(len(batch[0]) for batch in batches) == count
    if count:
        for index, (field, dtype) in enumerate(zip(_FIELDS, dtypes)):
            actual = torch.cat([batch[index] for batch in batches])
            assert torch.equal(actual, getattr(expected, field)[order[:count]].to(dtype))
        assert all(batch[0].is_contiguous(memory_format=torch.channels_last)
                   for batch in batches)


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_full_growth_retires_all_old_buffers_before_first_allocation(
        monkeypatch, amp, shuffle):
    source, incoming = _dataset(), _dataset(11, offset=17)
    source_values = [getattr(source, field).clone() for field in _FIELDS]
    incoming_values = [getattr(incoming, field).clone() for field in _FIELDS]
    loader = _loader(source, amp=amp, shuffle=shuffle)
    dtypes = [getattr(loader, "_" + field).dtype for field in _FIELDS]
    old_refs = _refs(loader)
    real_empty = torch.empty
    calls = []

    def observe(*args, **kwargs):
        assert all(ref() is None for ref in old_refs)
        calls.append(kwargs["dtype"])
        return real_empty(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", observe)
        loader.replace_data(incoming)
    assert calls == dtypes
    assert len(loader._boards) == len(incoming)
    _assert_contents(loader, incoming, dtypes)
    for field, before, replacement_before in zip(_FIELDS, source_values, incoming_values):
        assert torch.equal(getattr(source, field), before)
        assert torch.equal(getattr(incoming, field), replacement_before)


@pytest.mark.parametrize("amp", [False, True])
def test_repeated_replacements_grow_shrink_and_empty_without_retaining_old_rows(amp):
    loader = _loader(_dataset(), amp=amp, capacity=9, shuffle=True)
    dtypes = [getattr(loader, "_" + field).dtype for field in _FIELDS]
    for index, count in enumerate((11, 3, 0, 15, 4, 17)):
        old_capacity = len(loader._boards)
        old_pointers = [getattr(loader, "_" + field).data_ptr() for field in _FIELDS]
        expected = _dataset(count, offset=19 + index)
        loader.replace_data(expected)
        _assert_contents(loader, expected, dtypes)
        if count <= old_capacity:
            assert [getattr(loader, "_" + field).data_ptr() for field in _FIELDS] == old_pointers
        else:
            assert len(loader._boards) == count


def test_retained_batch_views_survive_full_growth():
    loader = _loader(_dataset(), amp=True)
    iterator = iter(loader)
    batch = next(iterator)
    iterator.close()
    before = [tensor.clone() for tensor in batch]
    incoming = _dataset(11, offset=23)
    dtypes = [getattr(loader, "_" + field).dtype for field in _FIELDS]
    loader.replace_data(incoming)
    _assert_contents(loader, incoming, dtypes)
    for actual, expected in zip(batch, before):
        assert torch.equal(actual, expected)


def test_replacement_source_can_own_old_resident_storage():
    # A larger expanded source can legally alias an old single-row buffer.
    loader = _loader(_dataset(1), amp=True)
    incoming = CachedTensorDataset(*(
        getattr(loader, "_" + field).expand(9, *getattr(loader, "_" + field).shape[1:])
        for field in _FIELDS
    ))
    values = [getattr(incoming, field).clone() for field in _FIELDS]
    dtypes = [getattr(loader, "_" + field).dtype for field in _FIELDS]
    loader.replace_data(incoming)
    _assert_contents(loader, incoming, dtypes)
    for field, before in zip(_FIELDS, values):
        assert torch.equal(getattr(incoming, field), before)


@pytest.mark.parametrize("count,cap", [(3, 0), (11, 5)])
def test_incremental_growth_retains_old_buffers_through_allocation(monkeypatch, count, cap):
    source, incoming = _dataset(), _dataset(count, offset=19)
    loader = _loader(source, amp=True)
    dtypes = [getattr(loader, "_" + field).dtype for field in _FIELDS]
    old_refs = _refs(loader)
    real_empty = torch.empty
    calls = []

    def observe(*args, **kwargs):
        assert all(ref() is not None for ref in old_refs)
        calls.append(kwargs["dtype"])
        return real_empty(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", observe)
        loader.update_data(incoming, max_entries=cap)
    assert calls == dtypes
    _assert_contents(loader, source.concat(incoming, max_entries=cap), dtypes)


@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
@pytest.mark.parametrize("fail_at", [1, 6])
@pytest.mark.parametrize("operation", ["allocate", "upload"])
def test_full_growth_errors_propagate_after_retirement(
        monkeypatch, error_type, fail_at, operation):
    loader = _loader(_dataset())
    incoming = _dataset(11, offset=23)
    old_refs = _refs(loader)
    real_operation = (torch.empty if operation == "allocate" else
                      dataset_module._copy_resident_tensor)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        assert all(ref() is None for ref in old_refs)
        calls += 1
        if calls == fail_at:
            raise error_type("injected resident growth failure")
        return real_operation(*args, **kwargs)

    if operation == "allocate":
        monkeypatch.setattr(torch, "empty", fail)
    else:
        monkeypatch.setattr(dataset_module, "_copy_resident_tensor", fail)
    with pytest.raises(error_type, match="injected resident growth failure"):
        loader.replace_data(incoming)
    assert calls == fail_at


@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
def test_post_growth_budget_query_error_propagates_after_new_rows_are_installed(error_type):
    loader = _loader(_dataset(), amp=True)
    incoming = _dataset(11, offset=23)
    dtypes = [getattr(loader, "_" + field).dtype for field in _FIELDS]

    def fail_query():
        raise error_type("injected resident budget query failure")

    loader._check_preshuffle_budget = fail_query
    with pytest.raises(error_type, match="injected resident budget query failure"):
        loader.replace_data(incoming)
    _assert_contents(loader, incoming, dtypes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA resident growth lifetime")
def test_cuda_full_growth_releases_old_allocations_before_new_storage(monkeypatch):
    source, incoming = _dataset(), _dataset(11, offset=23)
    loader = FastBatchIterator(
        source, batch_size=4, shuffle=False, device=torch.device("cuda"),
        amp_enabled=True, capacity=9,
    )
    assert loader.on_gpu
    old_refs = _refs(loader)
    real_empty = torch.empty
    calls = []

    def observe(*args, **kwargs):
        assert all(ref() is None for ref in old_refs)
        calls.append(kwargs["dtype"])
        return real_empty(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", observe)
        loader.replace_data(incoming)
    assert len(calls) == 6
    torch.cuda.synchronize()
    for field in _FIELDS:
        actual = getattr(loader, "_" + field).cpu()
        assert torch.equal(actual, getattr(incoming, field).to(actual.dtype))
    assert loader._boards.is_contiguous(memory_format=torch.channels_last)
