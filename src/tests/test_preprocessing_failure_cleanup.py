"""Failed fork preprocessing releases its input window before a retry."""

from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
import gc
import multiprocessing as mp
from multiprocessing import shared_memory
import weakref

import pytest
import torch

from dama.ai.ml import dataset
from dama.ai.ml.replay import ReplayEntry
from dama.game_state import GameState


@pytest.fixture
def fork_preprocessing(monkeypatch):
    monkeypatch.setattr(mp, "get_start_method", lambda: "fork")
    monkeypatch.setattr(dataset, "_default_preprocess_workers", lambda: 2)
    monkeypatch.setattr(dataset, "get_available_ram_gb", lambda: 8.0)
    # Undo stale state even when this regression test fails against old code.
    for name in ("_fork_entries", "_fork_total_n", "_fork_shm_names", "_fork_max_moves"):
        monkeypatch.setattr(dataset, name, getattr(dataset, name))


def _entries():
    state = GameState.initial()
    entry = ReplayEntry(
        state=state.to_compact(),
        legal_moves=[move.to_dict() for move in state.legal_moves()],
        chosen_index=1, result=1, score=7.0, sample_weight=1.5,
    )
    return [entry] * 2000


@pytest.mark.parametrize("failure_site", ["enter", "map", "concatenate"])
def test_failed_fork_fallback_releases_inputs_and_allows_exact_retry(
    monkeypatch, fork_preprocessing, failure_site,
):
    original_shared = shared_memory.SharedMemory
    segments = []
    calls = []
    entry_ref = None
    rows_id = None

    def allocate(*args, **kwargs):
        segment = original_shared(*args, **kwargs)
        if kwargs.get("create"):
            segments.append(segment)
        return segment

    @contextmanager
    def pool(**kwargs):
        assert id(dataset._fork_entries) == rows_id
        calls.append("start")
        try:
            if failure_site == "enter":
                raise BrokenProcessPool("failed worker startup")
            yield worker
        finally:
            # The inputs must remain available until the pool has shut down.
            assert id(dataset._fork_entries) == rows_id
            calls.append("closed")

    class Worker:
        def map(self, function, args):
            if len(calls) == 1 or failure_site == "map":
                raise BrokenProcessPool("failed worker results")
            return map(function, args)
    worker = Worker()

    def fail_concatenate(*args, **kwargs):
        raise MemoryError("failed fallback output allocation")

    monkeypatch.setattr(shared_memory, "SharedMemory", allocate)
    with monkeypatch.context() as failing:
        failing.setattr(dataset, "ProcessPoolExecutor", pool)
        if failure_site == "concatenate":
            failing.setattr(dataset.np, "concatenate", fail_concatenate)

        def prepare():
            nonlocal entry_ref, rows_id
            rows = _entries()
            rows_id = id(rows)
            entry_ref = weakref.ref(rows[0])
            dataset.preprocess_entries_to_tensors(rows, show_progress=False)

        with pytest.raises((BrokenProcessPool, MemoryError)):
            prepare()

    gc.collect()
    assert calls == ["start", "closed", "start", "closed"]
    assert dataset._fork_entries is None
    assert dataset._fork_total_n == 0
    assert dataset._fork_shm_names is None
    assert entry_ref() is None
    assert len(segments) == 6
    for segment in segments:
        assert segment.buf is None
        with pytest.raises(FileNotFoundError):
            original_shared(name=segment.name)

    # A later real fork call must encode only the new caller's rows. Compare
    # all six outputs with the sequential encoder, including reward weights.
    if "fork" not in mp.get_all_start_methods():
        return  # The fault/ownership contract above is platform independent.
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    monkeypatch.setattr(dataset, "ProcessPoolExecutor", partial(
        ProcessPoolExecutor, mp_context=mp.get_context("fork")))
    fresh = _entries()
    fresh[0] = ReplayEntry(
        state=fresh[0].state, legal_moves=fresh[0].legal_moves,
        chosen_index=2, result=-1, score=-3.0, sample_weight=.75,
    )
    actual = dataset.preprocess_entries_to_tensors(fresh, show_progress=False)
    monkeypatch.setattr(dataset, "_default_preprocess_workers", lambda: 1)
    oracle = dataset.preprocess_entries_to_tensors(fresh, show_progress=False)
    assert all(torch.equal(got, want) for got, want in zip(actual, oracle))


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_interrupted_shared_preprocessing_releases_inputs_and_segments(
    monkeypatch, fork_preprocessing, exception,
):
    original_shared = shared_memory.SharedMemory
    segments = []

    def allocate(*args, **kwargs):
        segment = original_shared(*args, **kwargs)
        if kwargs.get("create"):
            segments.append(segment)
        return segment

    @contextmanager
    def pool(**kwargs):
        yield worker

    class Worker:
        def map(self, *args):
            raise exception("injected interruption")
    worker = Worker()
    monkeypatch.setattr(shared_memory, "SharedMemory", allocate)
    monkeypatch.setattr(dataset, "ProcessPoolExecutor", pool)
    rows = _entries()
    caller_contents = list(rows)
    with pytest.raises(exception, match="injected interruption"):
        dataset.preprocess_entries_to_tensors(rows, show_progress=False)
    assert rows == caller_contents
    assert dataset._fork_entries is None
    assert dataset._fork_total_n == 0
    assert dataset._fork_shm_names is None
    for segment in segments:
        assert segment.buf is None
        with pytest.raises(FileNotFoundError):
            original_shared(name=segment.name)
