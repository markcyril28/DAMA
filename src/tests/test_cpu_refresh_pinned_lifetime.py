"""CPU refresh must retire obsolete pins without invalidating their owners."""

import weakref

import pytest
import torch

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
        (rows % 4).to(torch.int32),
        rows.float() / 4 + 1,
        rows.float() / 8 - 1,
    )


def _simulate_pinning(monkeypatch):
    """Model pin_memory's independent copy and already-pinned identity on CPU."""
    pinned = {}

    def pin(tensor, *args, **kwargs):
        ref = pinned.get(id(tensor))
        if ref is not None and ref() is tensor:
            return tensor
        result = tensor.clone()
        pinned[id(result)] = weakref.ref(result)
        return result

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    return pin


def _loader(source, *, shuffle=False, pin_memory=True):
    return FastBatchIterator(
        source, batch_size=4, shuffle=shuffle, drop_last=False,
        pin_memory=pin_memory,
    )


def _refs(loader):
    return [weakref.ref(getattr(loader, "_" + field)) for field in _FIELDS]


def _assert_rows(loader, expected):
    assert not loader.on_gpu and loader._device is None
    assert not loader._can_preshuffle and not loader.pin_memory
    assert loader.n == len(expected)
    assert loader.drop_last == (len(expected) > loader.batch_size)
    for field in _FIELDS:
        actual, wanted = getattr(loader, "_" + field), getattr(expected, field)
        assert actual.device.type == "cpu" and actual.dtype == wanted.dtype
        assert torch.equal(actual, wanted)

    # Use a separate generator for the oracle, without changing epoch RNG work.
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
        for index, field in enumerate(_FIELDS):
            actual = torch.cat([batch[index] for batch in batches])
            assert torch.equal(actual, getattr(expected, field)[order[:count]])
        assert all(batch[0].is_contiguous(memory_format=torch.channels_last)
                   for batch in batches)


@pytest.mark.parametrize("operation,count,cap", [
    ("replace", 5, 0), ("replace", 0, 0),
    ("incremental", 3, 0), ("incremental", 4, 8),
    ("incremental", 9, 5),
])
def test_old_independent_pins_retire_before_replacement_allocation(
        monkeypatch, operation, count, cap):
    pin = _simulate_pinning(monkeypatch)
    source, incoming = _dataset(), _dataset(count, offset=21)
    source_values = [getattr(source, field).clone() for field in _FIELDS]
    expected = (incoming if operation == "replace" else
                source.concat(incoming, max_entries=cap))
    loader = _loader(source)
    old_refs = _refs(loader)
    calls = []

    def observe(tensor, *args, **kwargs):
        assert all(ref() is None for ref in old_refs)
        assert loader.dataset is not source or operation == "replace"
        calls.append(tensor)
        return pin(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "pin_memory", observe)
    if operation == "replace":
        loader.replace_data(incoming)
        assert loader.dataset is incoming
    else:
        loader.update_data(incoming, max_entries=cap)
    assert len(calls) == 6
    _assert_rows(loader, expected)
    for field, before in zip(_FIELDS, source_values):
        assert torch.equal(getattr(source, field), before)


def test_repeated_cpu_replacements_retire_each_previous_window(monkeypatch):
    pin = _simulate_pinning(monkeypatch)
    loader = _loader(_dataset())
    for count in (3, 8, 0, 7):
        old_refs = _refs(loader)
        expected = _dataset(count, offset=11 + count)
        calls = []

        def observe(tensor, *args, **kwargs):
            assert all(ref() is None for ref in old_refs)
            calls.append(tensor)
            return pin(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "pin_memory", observe)
        loader.replace_data(expected)
        assert len(calls) == 6
        _assert_rows(loader, expected)


@pytest.mark.parametrize("operation", ["replace", "incremental"])
def test_cpu_refresh_preserves_seeded_shuffle_and_field_alignment(monkeypatch, operation):
    _simulate_pinning(monkeypatch)
    source, incoming = _dataset(), _dataset(6, offset=19)
    loader = _loader(source, shuffle=True)
    if operation == "replace":
        expected = incoming
        loader.replace_data(incoming)
    else:
        expected = source.concat(incoming, max_entries=9)
        loader.update_data(incoming, max_entries=9)
    _assert_rows(loader, expected)


@pytest.mark.parametrize("operation", ["replace", "incremental"])
def test_cpu_refresh_without_cuda_keeps_direct_source_tensors(monkeypatch, operation):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_pin(*args, **kwargs):
        raise AssertionError("CPU-only refresh attempted pinning")

    monkeypatch.setattr(torch.Tensor, "pin_memory", unexpected_pin)
    source, incoming = _dataset(), _dataset(3, offset=11)
    loader = _loader(source)
    if operation == "replace":
        expected = incoming
        loader.replace_data(incoming)
    else:
        expected = source.concat(incoming, max_entries=8)
        loader.update_data(incoming, max_entries=8)
    _assert_rows(loader, expected)
    for field in _FIELDS:
        assert getattr(loader, "_" + field) is getattr(loader.dataset, field)


@pytest.mark.parametrize("replacement_kind", ["self", "already_pinned", "active_alias"])
def test_refresh_preserves_datasets_that_own_active_or_source_tensors(
        monkeypatch, replacement_kind):
    pin = _simulate_pinning(monkeypatch)
    source = _dataset()
    if replacement_kind == "already_pinned":
        source = CachedTensorDataset(*(pin(getattr(source, field)) for field in _FIELDS))
    loader = _loader(source)
    incoming = (CachedTensorDataset(*(getattr(loader, "_" + field) for field in _FIELDS))
                if replacement_kind == "active_alias" else source)
    owned = [getattr(incoming, field) for field in _FIELDS]
    values = [tensor.clone() for tensor in owned]
    loader.replace_data(incoming)
    assert loader.dataset is incoming
    _assert_rows(loader, incoming)
    for field, tensor, before in zip(_FIELDS, owned, values):
        assert getattr(incoming, field) is tensor
        assert torch.equal(tensor, before)
        if replacement_kind != "self":
            assert getattr(loader, "_" + field) is tensor


def test_retained_nonshuffle_batch_views_remain_valid_after_refresh(monkeypatch):
    _simulate_pinning(monkeypatch)
    source = _dataset()
    loader = _loader(source)
    iterator = iter(loader)
    batch = next(iterator)
    iterator.close()
    expected_batch = [tensor.clone() for tensor in batch]
    incoming = _dataset(3, offset=31)
    loader.replace_data(incoming)
    _assert_rows(loader, incoming)
    for actual, expected in zip(batch, expected_batch):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
def test_concat_failure_keeps_previous_buffers_and_dataset(monkeypatch, error_type):
    _simulate_pinning(monkeypatch)
    source = _dataset()
    loader = _loader(source)
    old_refs = _refs(loader)

    def fail_concat(*args, **kwargs):
        raise error_type("injected concat failure")

    monkeypatch.setattr(source, "concat", fail_concat)
    with pytest.raises(error_type, match="injected concat failure"):
        loader.update_data(_dataset(3, offset=19))
    assert loader.dataset is source and loader.n == len(source)
    for field, ref in zip(_FIELDS, old_refs):
        assert getattr(loader, "_" + field) is ref()
        assert torch.equal(ref(), getattr(source, field))


@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
@pytest.mark.parametrize("fail_at", [1, 6])
def test_pin_failure_propagates_after_obsolete_pins_are_retired(
        monkeypatch, error_type, fail_at):
    pin = _simulate_pinning(monkeypatch)
    source, incoming = _dataset(), _dataset(5, offset=19)
    loader = _loader(source)
    old_refs = _refs(loader)
    source_values = [getattr(incoming, field).clone() for field in _FIELDS]
    calls = []

    def fail_pin(tensor, *args, **kwargs):
        calls.append(tensor)
        if len(calls) == fail_at:
            raise error_type("injected pin failure")
        return pin(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "pin_memory", fail_pin)
    with pytest.raises(error_type, match="injected pin failure"):
        loader.replace_data(incoming)
    assert len(calls) == fail_at
    assert all(ref() is None for ref in old_refs)
    for field, expected in zip(_FIELDS, source_values):
        assert torch.equal(getattr(incoming, field), expected)


def test_empty_incremental_update_does_not_release_or_repin(monkeypatch):
    _simulate_pinning(monkeypatch)
    source = _dataset()
    loader = _loader(source)
    old_refs = _refs(loader)

    def unexpected_pin(*args, **kwargs):
        raise AssertionError("empty incremental update attempted pinning")

    monkeypatch.setattr(torch.Tensor, "pin_memory", unexpected_pin)
    loader.update_data(_dataset(0))
    assert loader.dataset is source and loader.n == len(source)
    for field, ref in zip(_FIELDS, old_refs):
        assert getattr(loader, "_" + field) is ref()


@pytest.mark.parametrize("operation", ["replace", "incremental"])
def test_refresh_respects_disabled_pinning_on_cuda_host(monkeypatch, operation):
    _simulate_pinning(monkeypatch)
    source, incoming = _dataset(), _dataset(3, offset=23)
    loader = _loader(source, pin_memory=False)
    assert all(getattr(loader, "_" + field) is getattr(source, field) for field in _FIELDS)

    def unexpected_pin(*args, **kwargs):
        raise RuntimeError("pinning was disabled for this loader")

    monkeypatch.setattr(torch.Tensor, "pin_memory", unexpected_pin)
    if operation == "replace":
        expected = incoming
        loader.replace_data(incoming)
    else:
        expected = source.concat(incoming, max_entries=8)
        loader.update_data(incoming, max_entries=8)
    _assert_rows(loader, expected)
    for field in _FIELDS:
        assert getattr(loader, "_" + field) is getattr(loader.dataset, field)


def test_cuda_availability_error_propagates_before_buffer_retirement(monkeypatch):
    _simulate_pinning(monkeypatch)
    loader = _loader(_dataset())
    old_refs = _refs(loader)

    def fail_available():
        raise RuntimeError("injected availability failure")

    monkeypatch.setattr(torch.cuda, "is_available", fail_available)
    with pytest.raises(RuntimeError, match="injected availability failure"):
        loader.replace_data(_dataset(3, offset=13))
    for field, ref in zip(_FIELDS, old_refs):
        assert getattr(loader, "_" + field) is ref()
