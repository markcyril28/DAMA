"""Shared output storage is retired as each independent tensor becomes owned."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import weakref

import pytest
import torch

import dama.ai.ml.dataset as dataset
from dama.ai.ml.replay import ReplayEntry
from dama.game_state import GameState


_FIELDS = (
    "boards", "move_features", "move_counts", "targets",
    "reward_weights", "value_targets",
)


@pytest.fixture
def parallel_input(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("Shared-output preprocessing requires fork")
    state = GameState.initial()
    entries = []
    for index in range(32):
        moves = state.legal_moves()
        chosen = index % len(moves)
        entries.append(ReplayEntry(
            state=state.to_compact(),
            legal_moves=[move.to_dict() for move in moves],
            chosen_index=chosen, result=index % 3 - 1,
            score=float(index - 16), sample_weight=0.5 + index / 8,
        ))
        state = state.apply_move(moves[chosen])
    entries *= 64
    monkeypatch.setattr(dataset, "_default_preprocess_workers", lambda: 1)
    oracle = dataset.preprocess_entries_to_tensors(entries, show_progress=False)
    context = mp.get_context("fork")
    monkeypatch.setattr(mp, "get_start_method", lambda: "fork")
    monkeypatch.setattr(dataset, "ProcessPoolExecutor", partial(
        ProcessPoolExecutor, mp_context=context))
    monkeypatch.setattr(dataset, "_default_preprocess_workers", lambda: 2)
    monkeypatch.setattr(dataset, "get_available_ram_gb", lambda: 8.0)
    return entries, oracle


def _prepare(kind, entries):
    if kind == "entries":
        return dataset.preprocess_entries_to_tensors(entries, show_progress=False)
    result = dataset.CachedTensorDataset.from_dicts(
        [entry.to_dict() for entry in entries], show_progress=False)
    return tuple(getattr(result, name) for name in _FIELDS)


def _track_segments(monkeypatch, *, fail_allocation=0):
    original = shared_memory.SharedMemory
    segments = []
    parent = os.getpid()
    allocations = 0

    def allocate(*args, **kwargs):
        nonlocal allocations
        if kwargs.get("create") and os.getpid() == parent:
            allocations += 1
            if allocations == fail_allocation:
                raise OSError("injected shared allocation failure")
            result = original(*args, **kwargs)
            segments.append(result)
            return result
        return original(*args, **kwargs)

    monkeypatch.setattr(shared_memory, "SharedMemory", allocate)
    return segments, original


def _assert_outputs_and_cleanup(outputs, oracle, segments, original_shared):
    for actual, expected in zip(outputs, oracle):
        assert torch.equal(actual, expected)
    for segment in segments:
        assert segment.buf is None
        with pytest.raises(FileNotFoundError):
            original_shared(name=segment.name)


@pytest.mark.parametrize("kind", ["entries", "dicts"])
@pytest.mark.parametrize("compiled", [True, False])
def test_completed_shared_outputs_are_released_before_next_copy(
    monkeypatch, parallel_input, kind, compiled,
):
    entries, oracle = parallel_input
    if compiled:
        if not dataset._HAS_CYTHON or not dataset._HAS_CYTHON_DICTS:
            pytest.skip("Compiled encoders are unavailable")
    else:
        monkeypatch.setattr(dataset, "_HAS_CYTHON", False)
        monkeypatch.setattr(dataset, "_HAS_CYTHON_DICTS", False)
    segments, original_shared = _track_segments(monkeypatch)
    original_clone = torch.Tensor.clone
    live_segments_at_copy = []
    parent = os.getpid()

    def clone(tensor, *args, **kwargs):
        if os.getpid() == parent:
            live_segments_at_copy.append(sum(s.buf is not None for s in segments))
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    outputs = _prepare(kind, entries)
    _assert_outputs_and_cleanup(outputs, oracle, segments, original_shared)
    assert len(segments) == 6
    assert live_segments_at_copy == [6, 5, 4, 3, 2, 1]


@pytest.mark.parametrize("kind", ["entries", "dicts"])
@pytest.mark.parametrize("failed_copy", range(1, 7))
def test_partial_copy_failure_keeps_exact_fallback_and_cleans_segments(
    monkeypatch, parallel_input, kind, failed_copy,
):
    entries, oracle = parallel_input
    segments, original_shared = _track_segments(monkeypatch)
    original_clone = torch.Tensor.clone
    copies = 0
    parent = os.getpid()
    partial_outputs = []
    original_pool = dataset.ProcessPoolExecutor
    pool_calls = 0

    def clone(tensor, *args, **kwargs):
        nonlocal copies
        if os.getpid() == parent:
            copies += 1
            if copies == failed_copy:
                raise RuntimeError("injected tensor copy failure")
        result = original_clone(tensor, *args, **kwargs)
        if os.getpid() == parent:
            partial_outputs.append(weakref.ref(result))
        return result

    def pool(**kwargs):
        nonlocal pool_calls
        pool_calls += 1
        if pool_calls == 2:
            # Failed shared copies have no consumer. A memory-constrained
            # retry must not inherit them alongside its new output arrays.
            assert len(partial_outputs) == failed_copy - 1
            assert all(ref() is None for ref in partial_outputs)
            assert all(segment.buf is None for segment in segments)
        return original_pool(**kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    monkeypatch.setattr(dataset, "ProcessPoolExecutor", pool)
    outputs = _prepare(kind, entries)
    assert copies == failed_copy
    assert pool_calls == 2
    _assert_outputs_and_cleanup(outputs, oracle, segments, original_shared)


@pytest.mark.parametrize("kind", ["entries", "dicts"])
def test_partial_shared_allocation_failure_preserves_fallback(
    monkeypatch, parallel_input, kind,
):
    entries, oracle = parallel_input
    segments, original_shared = _track_segments(monkeypatch, fail_allocation=2)
    outputs = _prepare(kind, entries)
    assert len(segments) == 1
    _assert_outputs_and_cleanup(outputs, oracle, segments, original_shared)


@pytest.mark.parametrize("kind", ["entries", "dicts"])
def test_shared_unlink_failure_is_retried_at_final_cleanup(
    monkeypatch, parallel_input, kind,
):
    entries, oracle = parallel_input
    segments, original_shared = _track_segments(monkeypatch)
    original_unlink = original_shared.unlink
    attempts = {}

    def unlink(segment):
        attempts[segment.name] = attempts.get(segment.name, 0) + 1
        if segment is segments[0] and attempts[segment.name] == 1:
            raise OSError("injected transient unlink failure")
        return original_unlink(segment)

    monkeypatch.setattr(original_shared, "unlink", unlink)
    outputs = _prepare(kind, entries)
    _assert_outputs_and_cleanup(outputs, oracle, segments, original_shared)
    assert attempts[segments[0].name] == 2


@pytest.mark.parametrize("kind", ["entries", "dicts"])
@pytest.mark.parametrize("mode", ["fallback", "spawn"])
def test_combined_worker_outputs_release_consumed_source_fields(
    monkeypatch, parallel_input, kind, mode,
):
    entries, oracle = parallel_input
    # Exercise the real spawn threshold as well as fork's recovery path.
    entries = entries * 3
    oracle = tuple(torch.cat([tensor] * 3) for tensor in oracle)
    if mode == "spawn":
        monkeypatch.setattr(mp, "get_start_method", lambda: "spawn")
        monkeypatch.setattr(dataset, "ProcessPoolExecutor", partial(
            ProcessPoolExecutor, mp_context=mp.get_context("spawn")))
    else:
        _track_segments(monkeypatch, fail_allocation=1)

    original_concatenate = dataset.np.concatenate
    consumed = []
    calls = 0

    def concatenate(arrays, *args, **kwargs):
        nonlocal calls
        # Completed fields have independent combined outputs. Their worker
        # arrays must be gone before allocating the next combined field.
        assert all(ref() is None for ref in consumed)
        result = original_concatenate(arrays, *args, **kwargs)
        consumed.extend(weakref.ref(array) for array in arrays)
        calls += 1
        return result

    monkeypatch.setattr(dataset.np, "concatenate", concatenate)
    outputs = _prepare(kind, entries)
    assert calls == 6
    assert all(ref() is None for ref in consumed)
    for actual, expected in zip(outputs, oracle):
        assert torch.equal(actual, expected)
