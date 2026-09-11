"""Optional epoch gathers must recover without changing the sample schedule."""

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
    not torch.cuda.is_available(), reason="CUDA resident gather recovery")


def _loader(device="cpu", drop_last=False):
    count = 11
    rows = torch.arange(count)
    dataset = CachedTensorDataset(
        (rows[:, None, None, None] % 2).expand(
            count, BOARD_PLANES, 8, 8).float().clone(),
        (rows[:, None, None] / 16).expand(
            count, 4, MOVE_FEATURE_SIZE).clone(),
        torch.full((count,), 4, dtype=torch.int32),
        (rows % 4).to(torch.int32),
        rows.float() / 4 + 1,
        rows.float() / 8 - 1,
    )
    loader = FastBatchIterator(
        dataset, batch_size=4, shuffle=True, drop_last=drop_last,
        device=torch.device(device) if device == "cuda" else None,
        amp_enabled=device == "cuda",
    )
    if device == "cuda":
        assert loader.on_gpu
    # The same gather and cleanup logic is portable; CPU cases keep the
    # behavioral contract covered even on hosts without a GPU.
    loader._can_preshuffle = True
    return loader


def _observe_gathers(monkeypatch, loader, fail_field=None, error=None,
                     fallback_error=None):
    sources = {field: getattr(loader, "_" + field) for field in _FIELDS}
    real_getitem = torch.Tensor.__getitem__
    real_contiguous = torch.Tensor.contiguous
    real_randperm = torch.randperm
    observed = {
        "full_refs": [], "full_fields": [], "fallback_fields": [],
        "permutations": [], "failed": False,
    }

    def getitem(tensor, index):
        field = next((name for name, source in sources.items()
                      if tensor is source), None)
        if field is not None and isinstance(index, torch.Tensor):
            if index.numel() == loader.n:
                observed["full_fields"].append(field)
                if field == fail_field and not observed["failed"]:
                    observed["failed"] = True
                    raise error
                result = real_getitem(tensor, index)
                observed["full_refs"].append(weakref.ref(result))
                return result
            # Failed full gathers must relinquish their storage before even
            # the first allocation of the per-batch recovery path.
            assert all(ref() is None for ref in observed["full_refs"])
            observed["fallback_fields"].append(field)
            if (fallback_error is not None
                    and len(observed["fallback_fields"]) == 8):
                # Six fields from the first batch have already been yielded;
                # fail during the second batch to detect accidental replay.
                raise fallback_error
        return real_getitem(tensor, index)

    def contiguous(tensor, *args, **kwargs):
        result = real_contiguous(tensor, *args, **kwargs)
        if tensor.ndim == 4 and tensor.shape[0] == loader.n:
            # Board layout conversion may allocate separately from indexing.
            observed["full_refs"].append(weakref.ref(result))
        return result

    def randperm(*args, **kwargs):
        permutation = real_randperm(*args, **kwargs)
        observed["permutations"].append(permutation.clone())
        return permutation

    monkeypatch.setattr(torch.Tensor, "__getitem__", getitem)
    monkeypatch.setattr(torch.Tensor, "contiguous", contiguous)
    monkeypatch.setattr(torch, "randperm", randperm)
    return observed


def _assert_epoch(loader, batches, permutation):
    count = loader.n
    if loader.drop_last:
        count -= count % loader.batch_size
    assert len(batches) == len(loader)
    assert sum(len(batch[0]) for batch in batches) == count
    for field_index, field in enumerate(_FIELDS):
        source = getattr(loader, "_" + field)
        expected = source.index_select(0, permutation[:count])
        actual = torch.cat([batch[field_index] for batch in batches])
        assert actual.dtype == source.dtype
        assert actual.device == source.device
        assert torch.equal(actual, expected)
    assert all(batch[0].is_contiguous(memory_format=torch.channels_last)
               for batch in batches)


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("device,fail_field", [
    *(("cpu", field) for field in _FIELDS),
    pytest.param("cuda", "boards", marks=_CUDA),
    pytest.param("cuda", "value_targets", marks=_CUDA),
])
def test_preshuffle_oom_reuses_permutation_and_releases_partial_gathers(
        monkeypatch, device, fail_field, drop_last):
    loader = _loader(device, drop_last)
    originals = [getattr(loader, "_" + field).clone() for field in _FIELDS]
    observed = _observe_gathers(
        monkeypatch, loader, fail_field,
        torch.cuda.OutOfMemoryError("injected optional gather allocation"))

    batches = list(loader)

    assert observed["failed"]
    assert not loader._can_preshuffle
    assert len(observed["permutations"]) == 1
    assert observed["full_fields"] == list(
        _FIELDS[:_FIELDS.index(fail_field) + 1])
    assert all(ref() is None for ref in observed["full_refs"])
    _assert_epoch(loader, batches, observed["permutations"][0])
    first_full_fields = list(observed["full_fields"])

    second_batches = list(loader)

    assert len(observed["permutations"]) == 2
    assert observed["full_fields"] == first_full_fields
    _assert_epoch(loader, second_batches, observed["permutations"][1])
    for field, original in zip(_FIELDS, originals):
        assert torch.equal(getattr(loader, "_" + field), original)


def test_preshuffle_non_oom_propagates_and_releases_partial_gathers(monkeypatch):
    loader = _loader()
    error = RuntimeError("injected invalid gather")
    observed = _observe_gathers(monkeypatch, loader, "value_targets", error)

    with pytest.raises(RuntimeError, match="injected invalid gather") as caught:
        next(iter(loader))

    assert caught.value is error
    assert loader._can_preshuffle
    assert not observed["fallback_fields"]
    assert all(ref() is None for ref in observed["full_refs"])


def test_board_layout_oom_releases_gather_before_recovery(monkeypatch):
    loader = _loader()
    observed = _observe_gathers(monkeypatch, loader)
    observed_contiguous = torch.Tensor.contiguous

    def contiguous(tensor, *args, **kwargs):
        if (tensor.ndim == 4 and tensor.shape[0] == loader.n
                and not observed["failed"]):
            observed["failed"] = True
            # Do not retain this exception: its traceback owns the already
            # gathered input until the production handler has unwound.
            raise torch.cuda.OutOfMemoryError("injected board layout allocation")
        return observed_contiguous(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", contiguous)

    batches = list(loader)

    assert observed["failed"]
    assert not loader._can_preshuffle
    assert observed["full_fields"] == ["boards"]
    assert len(observed["permutations"]) == 1
    assert all(ref() is None for ref in observed["full_refs"])
    _assert_epoch(loader, batches, observed["permutations"][0])


def test_preshuffle_randperm_oom_propagates_without_retry(monkeypatch):
    loader = _loader()
    observed = _observe_gathers(monkeypatch, loader)
    error = torch.cuda.OutOfMemoryError("injected permutation allocation")
    calls = []

    def randperm(*args, **kwargs):
        calls.append((args, kwargs))
        raise error

    monkeypatch.setattr(torch, "randperm", randperm)

    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        next(iter(loader))

    assert caught.value is error
    assert len(calls) == 1
    assert not observed["full_fields"]
    assert not observed["fallback_fields"]


def test_recovery_batch_oom_propagates_after_first_batch_without_retry(monkeypatch):
    loader = _loader()
    error = torch.cuda.OutOfMemoryError("injected per-batch allocation")
    observed = _observe_gathers(
        monkeypatch, loader, "value_targets",
        torch.cuda.OutOfMemoryError("injected optional gather allocation"),
        fallback_error=error)
    iterator = iter(loader)

    first_batch = next(iterator)

    assert len(first_batch[0]) == loader.batch_size
    for field_index, field in enumerate(_FIELDS):
        expected = getattr(loader, "_" + field).index_select(
            0, observed["permutations"][0][:loader.batch_size])
        assert torch.equal(first_batch[field_index], expected)
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        next(iterator)

    assert caught.value is error
    assert len(observed["permutations"]) == 1
    assert observed["fallback_fields"] == list(_FIELDS) + [
        "boards", "move_features"]
    assert all(ref() is None for ref in observed["full_refs"])
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize("termination", ["close", "oom"])
def test_successful_preshuffle_releases_copies_when_generator_ends(
        monkeypatch, termination):
    loader = _loader()
    observed = _observe_gathers(monkeypatch, loader)
    iterator = iter(loader)
    next(iterator)
    assert any(ref() is not None for ref in observed["full_refs"])

    if termination == "close":
        iterator.close()
    else:
        error = torch.cuda.OutOfMemoryError("injected consumer failure")
        with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
            iterator.throw(error)
        assert caught.value is error

    assert loader._can_preshuffle
    assert not observed["fallback_fields"]
    assert all(ref() is None for ref in observed["full_refs"])
