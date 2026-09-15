"""Late snapshot cancellation never publishes partial replacement tensors."""

import gc
import threading
import weakref
from types import SimpleNamespace

import psutil
import pytest
import torch

from dama.ai.ml.dataset import CachedTensorDataset
from .test_train_window_assembly import FIELDS, _assert_equal, _expected, window


def _replace_window(window, cap=4):
    holder, context, record, rows = window
    previous = holder._load_or_reuse_train_dataset(context)
    cache = holder._train_tensor_window
    saved = {field: getattr(previous, field).clone() for field in FIELDS}
    context.manifest = {"files": [
        record("replacement.jsonl", rows[8:]), context.manifest["files"][0],
    ]}
    context.max_train_entries = cap
    expected = _expected(holder, context)
    return holder, context, previous, cache, saved, expected


def _observe_stages(monkeypatch, holder, stopped, *, stop_at=None, fail_at=None,
                    available=64 * 1024**3):
    """Inject cancellation at actual work boundaries, independent of polling."""
    events = []
    private_refs = []
    cat_fields = iter(FIELDS)
    gather_fields = iter(FIELDS)
    encode = CachedTensorDataset.from_entries
    cat = torch.cat
    getitem = torch.Tensor.__getitem__
    sample = holder._snapshot_manager.train_cap_sample_indices
    memory = psutil.virtual_memory

    def finished(stage):
        events.append(stage)
        if stage == stop_at:
            stopped.set()
        if stage == fail_at:
            raise RuntimeError(f"injected failure: {stage}")

    def observe_encode(*args, **kwargs):
        result = encode(*args, **kwargs)
        private_refs.extend(weakref.ref(getattr(result, field)) for field in FIELDS)
        return result

    def observe_cat(*args, **kwargs):
        result = cat(*args, **kwargs)
        private_refs.append(weakref.ref(result))
        finished("cat:" + next(cat_fields))
        return result

    def observe_sample(*args, **kwargs):
        result = sample(*args, **kwargs)
        finished("sample")
        return result

    def observe_getitem(tensor, index):
        result = getitem(tensor, index)
        if isinstance(index, torch.Tensor) and index.dtype == torch.int64:
            private_refs.append(weakref.ref(result))
            finished("gather:" + next(gather_fields))
        return result

    def observe_memory():
        # Tensorization has its own RAM gate before assembly begins.
        if "sample" not in events:
            return memory()
        finished("memory")
        return SimpleNamespace(available=available)

    monkeypatch.setattr(CachedTensorDataset, "from_entries", observe_encode)
    monkeypatch.setattr(torch, "cat", observe_cat)
    monkeypatch.setattr(torch.Tensor, "__getitem__", observe_getitem)
    monkeypatch.setattr(holder._snapshot_manager, "train_cap_sample_indices", observe_sample)
    monkeypatch.setattr(psutil, "virtual_memory", observe_memory)
    return events, private_refs


def _stages(cap):
    return (["cat:" + field for field in FIELDS] + ["sample"]
            + (["gather:" + field for field in FIELDS] if cap else [])
            + ["memory"])


@pytest.mark.parametrize("stop_at,cap,available", [
    *(('cat:' + field, 4, 64 * 1024**3) for field in FIELDS),
    ("sample", 0, 64 * 1024**3),
    ("sample", 4, 64 * 1024**3),
    *(('gather:' + field, 4, 64 * 1024**3) for field in FIELDS),
    ("memory", 0, 64 * 1024**3),
    ("memory", 4, 64 * 1024**3),
    ("memory", 4, 0),
])
def test_late_stop_releases_private_tensors_without_replacing_cache(
    window, monkeypatch, stop_at, cap, available,
):
    holder, context, previous, cache, saved, expected = _replace_window(window, cap)
    stopped = threading.Event()
    with monkeypatch.context() as patch:
        events, private_refs = _observe_stages(
            patch, holder, stopped, stop_at=stop_at, available=available)
        actual = holder._load_or_reuse_train_dataset(context, stopped.is_set)

    assert stopped.is_set(), "The requested assembly stage was not reached"
    assert actual is None
    stages = _stages(cap)
    assert events == stages[:stages.index(stop_at) + 1]
    assert holder._train_tensor_window is cache
    assert cache["dataset"] is previous
    for field in FIELDS:
        assert torch.equal(getattr(previous, field), saved[field]), field
    gc.collect()
    assert private_refs
    assert all(ref() is None for ref in private_refs)
    stopped.clear()
    _assert_equal(holder._load_or_reuse_train_dataset(context, stopped.is_set), expected)


@pytest.mark.parametrize("fail_at", ["cat:targets", "sample", "gather:targets"])
def test_late_work_failure_propagates_and_leaves_previous_window_retryable(
    window, monkeypatch, fail_at,
):
    holder, context, previous, cache, saved, expected = _replace_window(window)
    stopped = threading.Event()
    with monkeypatch.context() as patch:
        events, private_refs = _observe_stages(
            patch, holder, stopped, fail_at=fail_at)
        with pytest.raises(RuntimeError, match=f"injected failure: {fail_at}"):
            holder._load_or_reuse_train_dataset(context, stopped.is_set)

    stages = _stages(4)
    assert events == stages[:stages.index(fail_at) + 1]
    assert holder._train_tensor_window is cache
    for field in FIELDS:
        assert torch.equal(getattr(previous, field), saved[field]), field
    gc.collect()
    assert all(ref() is None for ref in private_refs)
    _assert_equal(holder._load_or_reuse_train_dataset(context, stopped.is_set), expected)


@pytest.mark.parametrize("with_callback", [False, True])
@pytest.mark.parametrize("kind,cap", [("empty", 0), ("reuse", 0), ("reuse", 4),
                                      ("mixed", 4)])
def test_healthy_assembly_keeps_empty_reused_and_capped_results(
    window, kind, cap, with_callback,
):
    holder, context, record, rows = window
    holder._load_or_reuse_train_dataset(context)
    if kind == "empty":
        context.manifest = {"files": [context.manifest["files"][1]]}
    elif kind == "mixed":
        context.manifest = {"files": [
            record("replacement.jsonl", rows[8:]), context.manifest["files"][0],
        ]}
    context.max_train_entries = cap
    expected = _expected(holder, context)
    callback = threading.Event().is_set if with_callback else None
    _assert_equal(holder._load_or_reuse_train_dataset(context, callback), expected)
