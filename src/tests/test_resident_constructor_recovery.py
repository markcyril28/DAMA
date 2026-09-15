"""Optional GPU residency must not make a valid CPU snapshot unusable."""

import weakref

import pytest
import torch

from dama.ai.ml.dataset import CachedTensorDataset, FastBatchIterator
from dama.ai.ml.move_encoder import BOARD_PLANES, MOVE_FEATURE_SIZE


_FIELDS = (
    "boards", "move_features", "move_counts", "targets",
    "reward_weights", "value_targets",
)
_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA resident constructor recovery")


def _dataset(count=11, offset=0):
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


def _inject_upload_failure(monkeypatch, operation, fail_at,
                           error_type=torch.cuda.OutOfMemoryError):
    real_empty = torch.empty
    real_copy = torch.Tensor.copy_
    real_to = torch.Tensor.to
    real_pin = torch.Tensor.pin_memory
    observed = {"calls": 0, "refs": [], "pins": 0, "failed": False}

    def maybe_fail():
        observed["calls"] += 1
        if observed["calls"] == fail_at:
            observed["failed"] = True
            # A retained exception would itself retain the failed operation's
            # GPU inputs through its traceback and invalidate the leak check.
            raise error_type("injected optional resident upload failure")

    def empty(*args, **kwargs):
        is_cuda = torch.device(kwargs.get("device", "cpu")).type == "cuda"
        if is_cuda and operation == "empty":
            maybe_fail()
        result = real_empty(*args, **kwargs)
        if is_cuda:
            observed["refs"].append(weakref.ref(result))
        return result

    def copy(tensor, source, *args, **kwargs):
        if tensor.is_cuda and not source.is_cuda and operation == "copy":
            maybe_fail()
        return real_copy(tensor, source, *args, **kwargs)

    def to(tensor, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        is_cuda = (isinstance(target, (str, torch.device))
                   and torch.device(target).type == "cuda")
        if is_cuda and not tensor.is_cuda and operation == "to":
            maybe_fail()
        result = real_to(tensor, *args, **kwargs)
        if is_cuda and not tensor.is_cuda:
            observed["refs"].append(weakref.ref(result))
        return result

    def pin(tensor, *args, **kwargs):
        # Release every partially assigned resident field before CPU pinning
        # starts, rather than gradually overwriting fields while both coexist.
        assert all(ref() is None for ref in observed["refs"])
        observed["pins"] += 1
        return real_pin(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    return observed


def _assert_cpu_rows(loader, source, order=None, pinned=False):
    assert loader.dataset is source
    assert not loader.on_gpu
    assert loader._device is None
    assert not loader._can_preshuffle
    assert loader.n == len(source)
    assert not loader.pin_memory
    for field in _FIELDS:
        actual = getattr(loader, "_" + field)
        expected = getattr(source, field)
        assert actual.device.type == "cpu"
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        if pinned and len(source):
            assert actual.is_pinned()

    batches = list(loader)
    assert len(batches) == len(loader)
    count = len(source)
    if loader.drop_last:
        count -= count % loader.batch_size
    assert sum(len(batch[0]) for batch in batches) == count
    if not count:
        return
    order = torch.arange(len(source)) if order is None else order
    for index, field in enumerate(_FIELDS):
        actual = torch.cat([batch[index] for batch in batches])
        expected = getattr(source, field).index_select(0, order[:count])
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    assert all(batch[0].is_contiguous(memory_format=torch.channels_last)
               for batch in batches)


@_CUDA
@pytest.mark.parametrize("operation", ["empty", "copy", "to"])
@pytest.mark.parametrize("fail_at", [1, 6])
@pytest.mark.parametrize("drop_last,pin_memory,amp", [
    (False, False, False), (False, True, True), (True, True, True),
])
def test_constructor_oom_releases_partial_upload_and_preserves_cpu_dataset(
        monkeypatch, operation, fail_at, drop_last, pin_memory, amp):
    source = _dataset()
    observed = _inject_upload_failure(monkeypatch, operation, fail_at)

    loader = FastBatchIterator(
        source, batch_size=4, shuffle=False, drop_last=drop_last,
        pin_memory=pin_memory, device=torch.device("cuda"),
        capacity=0 if operation == "to" else len(source) + 7,
        amp_enabled=amp,
    )

    assert observed["failed"] and observed["calls"] == fail_at
    assert observed["pins"] == (6 if pin_memory else 0)
    assert all(ref() is None for ref in observed["refs"])
    assert loader.drop_last is drop_last
    _assert_cpu_rows(loader, source, pinned=pin_memory)

    replacement = _dataset(3, offset=17)
    loader.replace_data(replacement)

    assert observed["calls"] == fail_at  # Replacement stays on the CPU path.
    assert not loader.drop_last
    _assert_cpu_rows(loader, replacement, pinned=pin_memory)
    if not pin_memory:
        assert observed["pins"] == 0
        for field in _FIELDS:
            assert getattr(loader, "_" + field) is getattr(replacement, field)


@_CUDA
@pytest.mark.parametrize("operation", ["empty", "copy", "to"])
def test_constructor_recovery_shuffles_cpu_rows_once(monkeypatch, operation):
    source = _dataset()
    _inject_upload_failure(monkeypatch, operation, 6)
    expected_order = torch.tensor([10, 1, 9, 3, 8, 5, 7, 0, 2, 4, 6])
    calls = []

    def randperm(count, *, device):
        assert count == len(source) and torch.device(device).type == "cpu"
        calls.append(count)
        return expected_order.clone()

    monkeypatch.setattr(torch, "randperm", randperm)
    loader = FastBatchIterator(
        source, batch_size=4, shuffle=True, drop_last=False,
        device=torch.device("cuda"), amp_enabled=True,
        capacity=0 if operation == "to" else 18,
    )

    _assert_cpu_rows(loader, source, order=expected_order)
    assert calls == [len(source)]


@_CUDA
@pytest.mark.parametrize("operation", ["empty", "to"])
@pytest.mark.parametrize("fail_at", [1, 6])
def test_empty_dataset_constructor_oom_falls_back(monkeypatch, operation, fail_at):
    source = _dataset(0)
    observed = _inject_upload_failure(monkeypatch, operation, fail_at)
    loader = FastBatchIterator(
        source, batch_size=4, shuffle=False, drop_last=False,
        device=torch.device("cuda"), pin_memory=True,
        capacity=7 if operation == "empty" else 0, amp_enabled=True,
    )

    assert observed["failed"]
    assert all(ref() is None for ref in observed["refs"])
    assert observed["pins"] == 6
    _assert_cpu_rows(loader, source)


@_CUDA
@pytest.mark.parametrize("operation", ["empty", "copy", "to"])
def test_constructor_non_oom_upload_error_propagates(monkeypatch, operation):
    observed = _inject_upload_failure(monkeypatch, operation, 6, RuntimeError)

    with pytest.raises(RuntimeError, match="injected optional resident upload failure"):
        FastBatchIterator(
            _dataset(), batch_size=4, pin_memory=True,
            device=torch.device("cuda"), capacity=0 if operation == "to" else 18,
        )

    assert observed["calls"] == 6 and observed["pins"] == 0


@_CUDA
@pytest.mark.parametrize("query", [
    "get_device_properties", "memory_allocated", "_check_preshuffle_budget",
])
@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
def test_constructor_budget_query_error_propagates(monkeypatch, query, error_type):
    calls = []

    def fail(*args, **kwargs):
        calls.append(query)
        raise error_type("injected budget query failure")

    def unexpected_pin(*args, **kwargs):
        pytest.fail("Budget-query errors must not trigger CPU recovery")

    target = FastBatchIterator if query == "_check_preshuffle_budget" else torch.cuda
    monkeypatch.setattr(target, query, fail)
    monkeypatch.setattr(torch.Tensor, "pin_memory", unexpected_pin)

    with pytest.raises(error_type, match="injected budget query failure"):
        FastBatchIterator(
            _dataset(), batch_size=4, pin_memory=True, device=torch.device("cuda"),
        )

    assert calls == [query]


@_CUDA
@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
def test_cpu_pinning_error_after_constructor_recovery_propagates(monkeypatch, error_type):
    observed = _inject_upload_failure(monkeypatch, "empty", 6)
    pins = []

    def fail_pin(tensor, *args, **kwargs):
        assert all(ref() is None for ref in observed["refs"])
        pins.append(tensor)
        raise error_type("injected CPU pinning failure")

    monkeypatch.setattr(torch.Tensor, "pin_memory", fail_pin)

    with pytest.raises(error_type, match="injected CPU pinning failure"):
        FastBatchIterator(
            _dataset(), batch_size=4, pin_memory=True,
            device=torch.device("cuda"), capacity=18,
        )

    assert len(pins) == 1 and observed["calls"] == 6


@pytest.mark.parametrize("device", [None, torch.device("cpu")])
@pytest.mark.parametrize("pin_memory", [False, True])
def test_cpu_constructor_does_not_attempt_gpu_residency(monkeypatch, device, pin_memory):
    source = _dataset()

    def unexpected_gpu_call(*args, **kwargs):
        pytest.fail("CPU construction must not consult CUDA residency budgets")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected_gpu_call)
    monkeypatch.setattr(torch.cuda, "memory_allocated", unexpected_gpu_call)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = FastBatchIterator(
        source, batch_size=4, shuffle=False, drop_last=False,
        pin_memory=pin_memory, device=device, capacity=18, amp_enabled=True,
    )

    assert loader._storage_dtype == torch.float32
    _assert_cpu_rows(loader, source)
    for field in _FIELDS:
        assert getattr(loader, "_" + field) is getattr(source, field)
