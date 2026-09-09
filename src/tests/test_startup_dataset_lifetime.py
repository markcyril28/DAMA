"""Verified startup caches should outlive only their actual consumers."""

import weakref
from types import SimpleNamespace

import pytest
import torch

import dama.ai.ml.trainer as trainer_module
from dama.ai.ml.replay import ReplayEntry
from dama.game_state import GameState


_FIELDS = (
    "boards", "move_features", "move_counts", "targets",
    "reward_weights", "value_targets",
)


def _entries(version):
    state = GameState.initial()
    return [ReplayEntry(
        state=state.to_compact(),
        legal_moves=[move.to_dict() for move in state.legal_moves()],
        chosen_index=(version + index) % len(state.legal_moves()),
        result=version % 3 - 1, sample_weight=1.0 + version / 4,
    ) for index in range(3)]


def _exercise_startup_caches(
    monkeypatch, *, storage, cached_training=True, cached_validation=True,
    entries_factory=_entries, module=trainer_module, observe=None,
    snapshots=False,
):
    """Exercise startup and one full refresh without games or artifact writes."""
    holder = object.__new__(module.Trainer)
    holder.config = module.TrainingConfig(
        batch_size=3, train_steps=2, pipeline_mode="simultaneous",
        max_stale_epochs=1, replay_max_entries=1000000,
        test_vs_algo=False, thermal_enabled=False, dataloader_workers=0,
        policy_stage="policy_only", reward_mode="none",
    )
    holder.model = torch.nn.Linear(1, 1)
    holder.device = torch.device("cpu")
    holder.stats = module.TrainingStats()
    holder.stats_collector = holder.scheduler = None
    holder.step = holder.epoch = 0
    holder._stopped = holder._paused = False
    holder._snapshot_manager = holder._acceptance_thread = None
    if snapshots:
        holder._snapshot_manager = SimpleNamespace(
            eligible_replay_files=lambda: (["first.jsonl", "second.jsonl"], []))
        monkeypatch.setattr(module, "analyze_replay_files", lambda _files:
                            ({"records": 1000000}, set()))
    holder.replay_buffer = SimpleNamespace(count_entries=lambda: 1000000)
    holder._preloaded_snapshot_dataset = None
    holder._preloaded_snapshot_cache_metadata = None
    holder._preloaded_snapshot_cache_checked = True
    holder._preloaded_validation_dataset = None
    holder._preloaded_validation_cache_metadata = None
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
    sources, observations = {}, []

    def tensorize(entries):
        return module.CachedTensorDataset.from_entries(
            entries, max_moves_per_sample=32, show_progress=False)

    def track(name, dataset):
        sources[name] = {
            "dataset": weakref.ref(dataset),
            "tensors": [weakref.ref(getattr(dataset, field)) for field in _FIELDS],
            "bytes": sum(getattr(dataset, field).nbytes for field in _FIELDS),
        }

    def prepare_split():
        train, validation = entries_factory(0), entries_factory(1)
        if cached_training:
            holder._preloaded_snapshot_dataset = tensorize(train)
            track("training", holder._preloaded_snapshot_dataset)
            train = []
        if cached_validation:
            holder._preloaded_validation_dataset = tensorize(validation)
            track("validation", holder._preloaded_validation_dataset)
            validation = []
        return train, validation

    holder._prepare_training_split = prepare_split

    def create_loader(dataset, **_kwargs):
        if storage == "standard_cpu":
            return torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)
        loader = module.FastBatchIterator(
            dataset, batch_size=3, shuffle=False, drop_last=False,
            device=torch.device("cuda") if storage == "cuda" else None,
        )
        if storage == "copied_cpu":
            # Follow resident-buffer ownership without claiming CPU copies
            # measure GPU upload performance. Real CUDA has a separate test.
            for field in _FIELDS:
                setattr(loader, "_" + field, getattr(loader, "_" + field).clone())
            loader.on_gpu = True
            loader._device = torch.device("cpu")
            loader._check_preshuffle_budget = lambda: False
        elif storage == "cuda":
            assert loader.on_gpu
        return loader

    monkeypatch.setattr(module, "create_dataloader_from_dataset", create_loader)
    monkeypatch.setattr(module, "create_dataloader", lambda entries, **kwargs:
                        create_loader(tensorize(entries), **kwargs))

    def publish(_games):
        holder._bg_selfplay_dataset = tensorize(entries_factory(2))
        holder._bg_validation_entries = entries_factory(3)
        holder._bg_snapshot_manifest = {"fingerprint": "replacement"}

    holder._start_background_selfplay = publish
    holder._data_ready_event = SimpleNamespace(set=lambda: None)

    def epoch(loader, **_kwargs):
        validation = holder._validation_dataloader
        if observe is not None:
            observe(holder, loader, sources)
        observations.append({
            "alive_datasets": sum(s["dataset"]() is not None for s in sources.values()),
            "alive_tensors": sum(ref() is not None for s in sources.values() for ref in s["tensors"]),
            "source_bytes": sum(s["bytes"] for s in sources.values()),
            "batch": tuple(t.cpu().clone() for t in next(iter(loader))),
            "validation": tuple(getattr(validation, field)[:3].cpu().clone() for field in _FIELDS),
        })
        if holder.step == 1:
            holder._stopped = True
        holder.step += 1
        return 1.0

    holder.train_epoch = epoch
    holder._run_training()
    assert holder.step == 2 and len(observations) == 2
    return observations


@pytest.mark.parametrize("storage", ["copied_cpu", "fast_cpu", "standard_cpu"])
@pytest.mark.parametrize("cached_training,cached_validation", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_startup_cache_sources_release_after_replacement(
    monkeypatch, storage, cached_training, cached_validation,
):
    before, after = _exercise_startup_caches(
        monkeypatch, storage=storage, cached_training=cached_training,
        cached_validation=cached_validation)
    assert before["alive_datasets"] == cached_training + cached_validation
    for epoch_index, result in enumerate((before, after)):
        for key, version in (("batch", epoch_index * 2),
                             ("validation", epoch_index * 2 + 1)):
            expected = trainer_module.CachedTensorDataset.from_entries(
                _entries(version), max_moves_per_sample=32, show_progress=False)
            for actual, field in zip(result[key], _FIELDS):
                assert torch.equal(actual, getattr(expected, field))
    assert after["alive_datasets"] == after["alive_tensors"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_startup_cache_release_preserves_real_cuda_batches(monkeypatch):
    before, after = _exercise_startup_caches(monkeypatch, storage="cuda")
    assert before["alive_datasets"] == 2
    assert after["alive_datasets"] == after["alive_tensors"] == 0
    for epoch_index, result in enumerate((before, after)):
        for key, version in (("batch", epoch_index * 2),
                             ("validation", epoch_index * 2 + 1)):
            expected = trainer_module.CachedTensorDataset.from_entries(
                _entries(version), max_moves_per_sample=32, show_progress=False)
            for actual, field in zip(result[key], _FIELDS):
                assert torch.equal(actual, getattr(expected, field))
