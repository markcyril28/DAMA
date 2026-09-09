"""Full snapshot handoffs must replace, rather than extend, training data."""

from types import SimpleNamespace

import pytest
import torch

import dama.ai.ml.trainer as trainer_module
from dama.ai.ml.dataset import CachedTensorDataset, FastBatchIterator
from dama.ai.ml.replay import ReplayEntry
from dama.game_state import GameState


_FIELDS = (
    "boards", "move_features", "move_counts", "targets",
    "reward_weights", "value_targets",
)


def _dataset(start, count):
    state = GameState.initial()
    entries = []
    for index in range(start + count):
        moves = state.legal_moves()
        chosen = index % len(moves)
        if index >= start:
            entries.append(ReplayEntry(
                state=state.to_compact(),
                legal_moves=[move.to_dict() for move in moves],
                chosen_index=chosen, result=index % 3 - 1,
                sample_weight=1.0 + index / 4,
            ))
        state = state.apply_move(moves[chosen])
    return CachedTensorDataset.from_entries(
        entries, max_moves_per_sample=32, show_progress=False)


def _holder(cap=100):
    holder = object.__new__(trainer_module.Trainer)
    holder.config = SimpleNamespace(replay_max_entries=cap)
    holder.stats = trainer_module.TrainingStats()
    holder.stats_collector = None
    holder._bg_selfplay_lock = trainer_module.threading.Lock()
    holder._bg_selfplay_dataset = holder._bg_selfplay_incremental = None
    holder._bg_snapshot_manifest = holder._bg_validation_entries = None
    return holder


def _loader(dataset, storage, capacity=0, amp=False):
    loader = FastBatchIterator(
        dataset, batch_size=2, shuffle=False,
        device=torch.device("cuda") if storage == "cuda" else None,
        capacity=capacity, amp_enabled=amp,
    )
    if storage == "resident_cpu":
        # Exercise the same resident-buffer copies on hosts without CUDA.
        for field in _FIELDS:
            value = getattr(loader, "_" + field)
            buffer = torch.empty((max(capacity, loader.n), *value.shape[1:]),
                                 dtype=value.dtype)
            buffer[:loader.n] = value
            setattr(loader, "_" + field, buffer)
        loader.on_gpu = True
        loader._device = torch.device("cpu")
        loader._check_preshuffle_budget = lambda: False
    elif storage == "cuda":
        assert loader.on_gpu
    return loader


def _assert_contents(loader, expected):
    assert loader.n == len(expected)
    assert loader.drop_last == (len(expected) > loader.batch_size)
    for field in _FIELDS:
        actual = getattr(loader, "_" + field)[:loader.n].cpu()
        assert torch.equal(actual, getattr(expected, field).to(actual.dtype))
    # Read the public iterator as well, including the empty snapshot case.
    batches = list(loader)
    presented = sum(len(batch[0]) for batch in batches)
    expected_count = len(expected)
    if loader.drop_last:
        expected_count -= expected_count % loader.batch_size
    assert presented == expected_count
    for batch_index, batch in enumerate(batches):
        start = batch_index * loader.batch_size
        for actual, field in zip(batch, _FIELDS):
            assert torch.equal(
                actual.cpu(),
                getattr(expected, field)[start:start + len(actual)].to(actual.dtype),
            )


@pytest.mark.parametrize("storage", ["cpu", "resident_cpu"])
@pytest.mark.parametrize("capacity", [0, 12])
def test_full_snapshot_refresh_replaces_every_previous_row(storage, capacity):
    holder = _holder()
    loader = _loader(_dataset(0, 5), storage, capacity)
    # Both shrink and grow below the replay cap. An append hidden by a cap
    # equal to the snapshot length would miss the production defect.
    for version, count in enumerate((3, 7, 2, 0, 4), start=2):
        expected = _dataset(version, count)
        holder._bg_selfplay_dataset = expected
        holder._bg_snapshot_manifest = {"fingerprint": f"snapshot-{version}"}
        holder._bg_validation_entries = []
        dataset, incremental = holder._collect_background_selfplay()
        refreshed, on_gpu = holder._refresh_dataloader(
            loader, dataset, incremental, effective_workers=0)
        assert refreshed is loader and on_gpu == loader.on_gpu
        assert holder.stats.dataset_fingerprint == f"snapshot-{version}"
        _assert_contents(loader, expected)


@pytest.mark.parametrize("storage", ["cpu", "resident_cpu"])
@pytest.mark.parametrize("include_full_dataset", [False, True])
def test_incremental_refresh_still_appends_and_trims(storage, include_full_dataset):
    holder = _holder(cap=7)
    initial, incremental = _dataset(0, 5), _dataset(5, 3)
    expected = initial.concat(incremental, max_entries=7)
    loader = _loader(initial, storage, capacity=12)
    holder._refresh_dataloader(
        loader, expected if include_full_dataset else None,
        incremental, effective_workers=0)
    _assert_contents(loader, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("capacity", [0, 12])
def test_cuda_full_snapshot_replacement_reuses_capacity_and_dtype(amp, capacity):
    holder = _holder()
    loader = _loader(_dataset(0, 5), "cuda", capacity=capacity, amp=amp)
    for count in (3, 7, 0, 4):
        old_capacity = len(loader._boards)
        pointers = {field: getattr(loader, "_" + field).data_ptr() for field in _FIELDS}
        expected = _dataset(5, count)
        holder._refresh_dataloader(loader, expected, None, effective_workers=0)
        _assert_contents(loader, expected)
        if count <= old_capacity:
            assert {field: getattr(loader, "_" + field).data_ptr() for field in _FIELDS} == pointers
        else:
            # A full replacement only needs its new size, not replay-cap slack.
            assert len(loader._boards) == count
        assert loader._boards.dtype == (torch.float16 if amp else torch.float32)
        assert loader._boards.is_contiguous(memory_format=torch.channels_last)
