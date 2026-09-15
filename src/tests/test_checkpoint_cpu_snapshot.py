"""CPU training must serialize the checkpoint's captured optimizer revision."""

import threading

import torch

from dama.ai.ml import trainer as trainer_module
from dama.ai.ml.trainer import Trainer


def test_cpu_checkpoint_does_not_alias_live_model_or_optimizer(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    holder = object.__new__(Trainer)
    holder.config = trainer_module.TrainingConfig(
        checkpoint_dir=str(checkpoint_dir), latest_path=str(tmp_path / "latest.pt"))
    holder.model = torch.nn.Linear(2, 1)
    holder.optimizer = torch.optim.Adam(holder.model.parameters(), lr=0.01)
    holder.model(torch.ones(1, 2)).sum().backward()
    holder.optimizer.step()
    expected_model = {
        key: value.clone() for key, value in holder.model.state_dict().items()}
    expected_optimizer = {
        key: {name: value.clone() for name, value in state.items()}
        for key, state in holder.optimizer.state_dict()["state"].items()}

    holder.stats = trainer_module.TrainingStats()
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

    writer_started = threading.Event()
    release_writer = threading.Event()
    real_save = torch.save

    def delayed_save(*args, **kwargs):
        writer_started.set()
        assert release_writer.wait(10), "checkpoint writer was not released"
        return real_save(*args, **kwargs)

    monkeypatch.setattr(trainer_module.torch, "save", delayed_save)
    try:
        Trainer._save_checkpoint(holder, loss=0.5)
        assert writer_started.wait(10)
        # Another ordinary CPU optimizer step occurs while disk serialization
        # is queued. It must not change the already captured checkpoint.
        holder.optimizer.zero_grad()
        holder.model(torch.full((1, 2), 3.0)).sum().backward()
        holder.optimizer.step()
    finally:
        release_writer.set()
        Trainer._wait_for_checkpoint_writer(holder, timeout=30)

    checkpoint = torch.load(
        checkpoint_dir / "model_step_002000.pt", map_location="cpu", weights_only=False)
    mismatches = []
    for key, expected in expected_model.items():
        if not torch.equal(checkpoint["model_state_dict"][key], expected):
            mismatches.append(f"model.{key}")
    for key, state in expected_optimizer.items():
        for name, expected in state.items():
            if not torch.equal(checkpoint["optimizer_state_dict"]["state"][key][name], expected):
                mismatches.append(f"optimizer.{key}.{name}")
    assert mismatches == []
