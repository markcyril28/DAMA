"""Training-loop ownership at background dataset handoff boundaries."""

import weakref
from types import SimpleNamespace

import pytest
import torch

import dama.ai.ml.trainer as trainer_module


_FIELDS = (
    "boards", "move_features", "move_counts", "targets",
    "reward_weights", "value_targets",
)


def _dataset():
    from dama.ai.ml.replay import ReplayEntry
    from dama.game_state import GameState

    state = GameState.initial()
    entries = [
        ReplayEntry(
            state=state.to_compact(),
            legal_moves=[move.to_dict() for move in state.legal_moves()],
            chosen_index=index, result=1, score=2.5,
        )
        for index in range(3)
    ]
    return trainer_module.CachedTensorDataset.from_entries(
        entries, max_moves_per_sample=32, show_progress=False)


def _exercise_refresh(
    monkeypatch, *, delivery, payload, storage, dataset_factory=_dataset,
    module=trainer_module, observe=None,
):
    """Run the real loop and refresh with two epochs and no persistent writes.

    ``copied_cpu`` exercises the resident-buffer copy path on CPU for bounded
    memory probes. ``cuda`` tests the actual asynchronous H2D handoff too.
    """
    initial = dataset_factory()
    entry_count = len(initial)
    # A missed refresh must not pass by reading an identical startup batch.
    initial.targets = (initial.targets + 1) % initial.move_counts
    loader = module.FastBatchIterator(
        initial, batch_size=entry_count, shuffle=False, drop_last=False,
        device=torch.device("cuda") if storage == "cuda" else None,
    )
    if storage == "copied_cpu":
        for field in _FIELDS:
            setattr(loader, "_" + field, getattr(loader, "_" + field).clone())
        loader.on_gpu = True
        loader._device = torch.device("cpu")
        loader._check_preshuffle_budget = lambda: False
    if storage == "cuda":
        assert loader.on_gpu
    if storage == "standard_cpu":
        loader = torch.utils.data.DataLoader(initial, batch_size=entry_count)

    holder = object.__new__(module.Trainer)
    holder.config = module.TrainingConfig(
        batch_size=entry_count, train_steps=2, pipeline_mode="simultaneous",
        max_stale_epochs=1, replay_max_entries=entry_count,
        test_vs_algo=False, thermal_enabled=False, dataloader_workers=0,
        policy_stage="policy_only", reward_mode="none",
    )
    holder.model = torch.nn.Linear(1, 1)
    holder.device = torch.device("cpu")
    holder.stats = module.TrainingStats()
    holder.stats_collector = None
    holder.step = holder.epoch = 0
    holder.scheduler = None
    holder._stopped = holder._paused = False
    holder._snapshot_manager = None
    holder._acceptance_thread = None
    holder.replay_buffer = SimpleNamespace(count_entries=lambda: entry_count * 10)
    holder._preloaded_snapshot_dataset = initial
    holder._preloaded_snapshot_cache_metadata = None
    holder._preloaded_snapshot_cache_checked = True
    holder._preloaded_validation_dataset = None
    holder._preloaded_validation_cache_metadata = None
    holder._prepare_training_split = lambda: ([], [])
    holder._ensure_frozen_teacher_suite = lambda: None
    holder._service_control_queue = lambda: None
    holder._save_progress_report_if_due = lambda: None
    holder._stop_background_selfplay = lambda: None
    holder._save_checkpoint = lambda *_: None
    holder._wait_for_checkpoint_writer = lambda: None
    holder._bg_selfplay_lock = module.threading.Lock()
    holder._bg_selfplay_thread = SimpleNamespace(is_alive=lambda: True)
    holder._bg_selfplay_dataset = holder._bg_selfplay_incremental = None
    holder._bg_snapshot_manifest = holder._bg_validation_entries = None
    sources = {}
    observations = []

    def publish():
        for kind in ("dataset", "incremental"):
            if payload not in (kind, "both"):
                continue
            dataset = dataset_factory()
            sources[kind] = {
                "dataset": weakref.ref(dataset),
                "tensors": [weakref.ref(getattr(dataset, field)) for field in _FIELDS],
                "bytes": sum(getattr(dataset, field).nbytes for field in _FIELDS),
            }
            setattr(holder, "_bg_selfplay_" + kind, dataset)

    def start(_games):
        if delivery == "epoch":
            publish()

    def unexpected_wait(**_kwargs):
        raise AssertionError("The prepared dataset should wake the stale-data loop")

    holder._start_background_selfplay = start
    holder._data_ready_event = SimpleNamespace(
        clear=publish, wait=unexpected_wait,
    )

    def create_loader(dataset, **_kwargs):
        if dataset is initial:
            return loader
        return module.FastBatchIterator(
            dataset, batch_size=entry_count, shuffle=False, drop_last=False)

    monkeypatch.setattr(module, "create_dataloader_from_dataset", create_loader)

    def epoch(current_loader, **_kwargs):
        if holder.step == 1:
            assert sources
            if observe is not None:
                observe(holder, current_loader, sources)
            observations.append({
                "alive_datasets": sum(s["dataset"]() is not None for s in sources.values()),
                "alive_tensors": sum(ref() is not None for s in sources.values() for ref in s["tensors"]),
                "source_bytes": sum(s["bytes"] for s in sources.values()),
                "batch": tuple(t.cpu().clone() for t in next(iter(current_loader))),
            })
            holder._stopped = True
        holder.step += 1
        return 1.0

    holder.train_epoch = epoch
    holder._run_training()
    assert holder.step == 2 and len(observations) == 1
    return observations[0]


@pytest.mark.parametrize("delivery", ["epoch", "wait"])
@pytest.mark.parametrize("payload", ["dataset", "incremental", "both"])
@pytest.mark.parametrize("storage", ["copied_cpu", "fast_cpu", "standard_cpu"])
def test_training_loop_releases_background_tensor_sources(
    monkeypatch, delivery, payload, storage,
):
    result = _exercise_refresh(
        monkeypatch, delivery=delivery, payload=payload, storage=storage)
    if storage == "copied_cpu":
        assert result["alive_datasets"] == result["alive_tensors"] == 0
    # CPU fallbacks may retain the source or a merged copy. Reading every
    # field here verifies that the necessary data survives either way.
    expected = _dataset()
    for actual, field in zip(result["batch"], _FIELDS):
        assert torch.equal(actual, getattr(expected, field))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("delivery", ["epoch", "wait"])
def test_training_loop_released_sources_finish_cuda_upload(monkeypatch, delivery):
    result = _exercise_refresh(
        monkeypatch, delivery=delivery, payload="both", storage="cuda")
    assert result["alive_datasets"] == result["alive_tensors"] == 0
    expected = _dataset()
    for actual, field in zip(result["batch"], _FIELDS):
        assert torch.equal(actual, getattr(expected, field))
