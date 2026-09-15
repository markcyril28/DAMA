"""Concurrent fork preprocessing must encode only each caller's own rows."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing as mp
import os
import threading
import time

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
def fork_setup(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("Fork preprocessing is unavailable")
    monkeypatch.setattr(mp, "get_start_method", lambda: "fork")
    monkeypatch.setattr(dataset, "get_available_ram_gb", lambda: 8.0)
    # Undo stale staging even when a regression fails against old code.
    for name in ("_fork_entries", "_fork_total_n", "_fork_shm_names", "_fork_max_moves"):
        monkeypatch.setattr(dataset, name, getattr(dataset, name))


def _rows(offset, repeats):
    """Distinct playouts per offset, so a mixed-up caller changes every field."""
    state = GameState.initial()
    pattern = []
    for index in range(32):
        moves = state.legal_moves()
        if not moves:
            state = GameState.initial()
            moves = state.legal_moves()
        chosen = (index + offset) % len(moves)
        pattern.append(ReplayEntry(
            state=state.to_compact(),
            legal_moves=[move.to_dict() for move in moves],
            chosen_index=chosen, result=(index + offset) % 3 - 1,
            score=float((3 * index + offset) % 11 - 5),
            sample_weight=0.5 + (index + offset) % 7 / 8,
        ))
        state = state.apply_move(moves[chosen])
    return pattern * repeats


def _prepare(kind, rows):
    if kind == "entries":
        return dataset.preprocess_entries_to_tensors(rows, show_progress=False)
    result = dataset.CachedTensorDataset.from_dicts(
        [row.to_dict() for row in rows], show_progress=False)
    return tuple(getattr(result, name) for name in _FIELDS)


@pytest.mark.parametrize("first_kind,second_kind", [
    ("entries", "entries"), ("dicts", "dicts"),
    ("entries", "dicts"), ("dicts", "entries"),
])
def test_concurrent_fork_preprocessing_keeps_each_callers_rows(
    monkeypatch, fork_setup, first_kind, second_kind,
):
    # Both callers exceed the fork threshold, like the 5,000-state frozen
    # suite at a checkpoint racing a background snapshot tensorization.
    first_rows, second_rows = _rows(1, 64), _rows(6, 66)
    monkeypatch.setattr(dataset, "_default_preprocess_workers", lambda: 1)
    oracles = {
        "first": dataset.preprocess_entries_to_tensors(first_rows, show_progress=False),
        "second": dataset.preprocess_entries_to_tensors(second_rows, show_progress=False),
    }
    monkeypatch.setattr(dataset, "_default_preprocess_workers", lambda: 2)

    first_staged = threading.Event()
    events = []
    threads = {}

    class InterleavingPool(ProcessPoolExecutor):
        def __init__(self, **kwargs):
            super().__init__(mp_context=mp.get_context("fork"), **kwargs)

        def map(self, function, *iterables, **kwargs):
            owner = threading.current_thread().name
            events.append((owner, "dispatch"))
            if owner == "first" and not first_staged.is_set():
                staged_names = dataset._fork_shm_names
                first_staged.set()
                # Let the second caller stage before this pool forks. With
                # serialized staging it cannot, and the bounded wait expires.
                deadline = time.monotonic() + 0.75
                while (dataset._fork_shm_names is staged_names
                       and time.monotonic() < deadline):
                    time.sleep(0.005)
            return super().map(function, *iterables, **kwargs)

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                events.append((threading.current_thread().name, "exit"))

    monkeypatch.setattr(dataset, "ProcessPoolExecutor", InterleavingPool)
    results = {}

    def run(name, kind, rows):
        try:
            results[name] = _prepare(kind, rows)
        except BaseException as exc:  # Reported by the assertions below.
            results[name] = exc

    threads["first"] = threading.Thread(
        target=run, name="first", args=("first", first_kind, first_rows))
    threads["second"] = threading.Thread(
        target=run, name="second", args=("second", second_kind, second_rows))
    threads["first"].start()
    assert first_staged.wait(30), "first caller never dispatched its pool"
    threads["second"].start()
    for thread in threads.values():
        thread.join(120)
    assert not any(thread.is_alive() for thread in threads.values())

    for name, oracle in oracles.items():
        outputs = results[name]
        assert not isinstance(outputs, BaseException), f"{name}: {outputs!r}"
        for field, actual, expected in zip(_FIELDS, outputs, oracle):
            assert torch.equal(actual, expected), f"{name} {field} used another caller's rows"
    # Every pool of the first caller exits before the second dispatches.
    owners = [owner for owner, _ in events]
    assert owners == sorted(owners, key=("first", "second").index)
    assert dataset._fork_entries is None
    assert dataset._fork_total_n == 0
    assert dataset._fork_shm_names is None


@pytest.mark.parametrize("kind", ["entries", "dicts"])
def test_interrupted_fork_preprocessing_releases_staging_lock(
    monkeypatch, fork_setup, kind,
):
    monkeypatch.setattr(dataset, "_default_preprocess_workers", lambda: 2)

    class Worker:
        def map(self, *args):
            raise KeyboardInterrupt("injected interruption")

    @contextmanager
    def pool(**kwargs):
        yield Worker()

    monkeypatch.setattr(dataset, "ProcessPoolExecutor", pool)
    with pytest.raises(KeyboardInterrupt, match="injected interruption"):
        _prepare(kind, _rows(2, 64))
    # A leaked lock would stall the next checkpoint's teacher evaluation.
    assert dataset._fork_preprocess_lock.acquire(blocking=False)
    dataset._fork_preprocess_lock.release()
    assert dataset._fork_entries is None
    assert dataset._fork_total_n == 0
    assert dataset._fork_shm_names is None


def _acquire_inherited_lock_or_fail():
    os._exit(0 if dataset._fork_preprocess_lock.acquire(timeout=5) else 3)


@pytest.mark.skipif(
    not hasattr(os, "register_at_fork") or "fork" not in mp.get_all_start_methods(),
    reason="Fork handlers are unavailable",
)
def test_child_forked_during_preprocessing_gets_an_unheld_lock():
    # Self-play or preprocessing workers can fork while another thread holds
    # the staging lock; no holder exists in the child to release it.
    child = None
    with dataset._fork_preprocess_lock:
        child = mp.get_context("fork").Process(target=_acquire_inherited_lock_or_fail)
        child.start()
        child.join(30)
    try:
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.kill()
            child.join()
