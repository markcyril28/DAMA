"""Real checkpoint round trips retain generation progress without a sidecar."""

import threading

import pytest
import torch

from dama.ai.ml import trainer as trainer_module
from dama.ai.ml.trainer import Trainer


@pytest.mark.parametrize(
    "sidecar_step, sidecar_cycles, expected_cycles",
    [(0, 0, 7), (1500, 3, 7), (3000, 99, 7), (2000, 11, 11)],
    ids=["missing-sidecar", "older-sidecar", "rewound-sidecar", "newer-cycle-count"],
)
def test_checkpoint_restores_captured_generation_progress(
    monkeypatch, tmp_path, sidecar_step, sidecar_cycles, expected_cycles,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    holder = object.__new__(Trainer)
    holder.config = trainer_module.TrainingConfig(
        checkpoint_dir=str(checkpoint_dir), latest_path=str(tmp_path / "latest.pt"))
    holder.model = torch.nn.Linear(2, 1)
    holder.optimizer = torch.optim.Adam(holder.model.parameters(), lr=0.01)
    holder.stats = trainer_module.TrainingStats(generation_cycles_completed=7)
    holder.step = 2000
    holder.epoch = 1
    holder.scheduler = None
    holder.scaler = None
    holder.stats_collector = None
    holder.log_file = str(tmp_path / "train.jsonl")
    holder.device = torch.device("cpu")
    holder._checkpoint_thread = None
    holder._active_snapshot_manifest = {}
    holder._evaluate_validation_loss = lambda: None
    holder._evaluate_teacher_promotion = lambda _path: None
    holder._live_optimizer_context = lambda: {}
    holder._snapshot_stats = lambda: {}
    holder._save_stats = lambda **_kwargs: None
    holder._put_status = lambda _message: None
    holder._prune_old_checkpoints = lambda _path: []
    holder._has_non_finite_tensors = lambda: False
    holder._durable_generation_cycle_max = lambda: -1

    writer_started = threading.Event()
    release_writer = threading.Event()
    real_save = torch.save

    def delayed_save(*args, **kwargs):
        writer_started.set()
        assert release_writer.wait(10), "checkpoint writer was not released"
        return real_save(*args, **kwargs)

    monkeypatch.setattr(trainer_module.torch, "save", delayed_save)
    try:
        checkpoint_path = holder._save_checkpoint(loss=0.5)
        assert writer_started.wait(10)
        # A completed producer cycle must not alter the captured revision.
        holder.stats.generation_cycles_completed = 8
    finally:
        release_writer.set()
        holder._wait_for_checkpoint_writer(timeout=30)

    holder.stats = trainer_module.TrainingStats(
        total_steps=sidecar_step, generation_cycles_completed=sidecar_cycles)
    holder._load_checkpoint(checkpoint_path)
    assert holder.stats.generation_cycles_completed == expected_cycles
    # With no replay history left to recover, checkpoint progress is the floor
    # that prevents reusing an already completed generation's identifier.
    assert holder._allocate_generation_cycle_id() == expected_cycles
