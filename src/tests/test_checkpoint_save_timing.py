"""Checkpoint telemetry measures the completed asynchronous writer work."""

import json
import time
from types import SimpleNamespace

import pytest
import torch

from dama.ai.ml import trainer as trainer_module
from dama.ai.ml.stats_collector import StatsCollector
from dama.ai.ml.trainer import Trainer


def _checkpoint_holder(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    holder = object.__new__(Trainer)
    holder.config = trainer_module.TrainingConfig(
        checkpoint_dir=str(checkpoint_dir), latest_path=str(tmp_path / "latest.pt"))
    holder.model = torch.nn.Linear(2, 1)
    holder.optimizer = torch.optim.Adam(holder.model.parameters())
    holder.stats = trainer_module.TrainingStats()
    holder.step = 2000
    holder.epoch = 1
    holder.scheduler = None
    holder.scaler = None
    holder.stats_collector = StatsCollector(
        output_dir=str(tmp_path / "stats"), session_id="checkpoint")
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
    return holder


def _enable_promotion(holder, tmp_path):
    holder.config.promoted_path = str(tmp_path / "promoted.pt")
    holder._evaluate_teacher_promotion = lambda _path: {
        "promotion": {"promoted": True}}
    holder._promotion_registry = SimpleNamespace(persist=lambda _decision: None)
    holder._enqueue_checkpoint_acceptance = lambda *_args: None


def test_checkpoint_duration_reaches_incremental_log(monkeypatch, tmp_path):
    holder = _checkpoint_holder(tmp_path)
    clock = [10.0]
    monkeypatch.setattr(trainer_module, "time", SimpleNamespace(
        perf_counter=lambda: clock[0], time=time.time,
        monotonic=time.monotonic, sleep=time.sleep))
    real_save = torch.save
    real_publish = holder._publish_checkpoint_alias

    def save(*args, **kwargs):
        result = real_save(*args, **kwargs)
        clock[0] += 1.25
        return result

    def publish(*args, **kwargs):
        result = real_publish(*args, **kwargs)
        clock[0] += 0.75
        return result

    def save_stats(**_kwargs):
        clock[0] += 0.5

    def prune(_path):
        clock[0] += 0.25
        return []

    monkeypatch.setattr(trainer_module.torch, "save", save)
    holder._publish_checkpoint_alias = publish
    holder._save_stats = save_stats
    holder._prune_old_checkpoints = prune
    checkpoint_path = Trainer._save_checkpoint(holder, loss=0.5)
    Trainer._wait_for_checkpoint_writer(holder, timeout=30)
    holder.stats_collector.flush_incremental()

    row = json.loads((tmp_path / "stats" / "incremental_checkpoint.jsonl").read_text())
    assert row["session_summary"]["checkpoints_recorded"] == 1
    recorded = row["latest_records"]["checkpoint"]
    assert recorded["path"] == checkpoint_path
    assert recorded["save_time_sec"] == pytest.approx(2.75)
    assert recorded["file_size_mb"] > 0


@pytest.mark.parametrize("failed_stage", ["serialization", "publication", "statistics"])
def test_failed_writer_does_not_record_completed_duration(
        monkeypatch, tmp_path, failed_stage):
    holder = _checkpoint_holder(tmp_path)

    def fail(*_args, **_kwargs):
        raise OSError(f"controlled {failed_stage} failure")

    if failed_stage == "serialization":
        monkeypatch.setattr(trainer_module.torch, "save", fail)
    elif failed_stage == "publication":
        holder._publish_checkpoint_alias = fail
    else:
        holder._save_stats = fail
    Trainer._save_checkpoint(holder, loss=0.5)
    with pytest.raises(RuntimeError, match=f"controlled {failed_stage} failure"):
        Trainer._wait_for_checkpoint_writer(holder, timeout=30)
    assert holder.stats_collector.checkpoint_records == []


def test_promoted_checkpoint_duration_includes_alias_and_task(monkeypatch, tmp_path):
    holder = _checkpoint_holder(tmp_path)
    _enable_promotion(holder, tmp_path)
    clock = [10.0]
    monkeypatch.setattr(trainer_module, "time", SimpleNamespace(
        perf_counter=lambda: clock[0]))
    real_publish = holder._publish_checkpoint_alias

    def publish(source, destination):
        result = real_publish(source, destination)
        if str(destination) == holder.config.promoted_path:
            clock[0] += 2.0
        return result

    def enqueue(*_args):
        clock[0] += 3.0

    holder._publish_checkpoint_alias = publish
    holder._enqueue_checkpoint_acceptance = enqueue
    Trainer._save_checkpoint(holder, loss=0.5)
    Trainer._wait_for_checkpoint_writer(holder, timeout=30)
    record = holder.stats_collector.checkpoint_records[0]
    assert record['save_time_sec'] == pytest.approx(5.0)


@pytest.mark.parametrize("failed_stage", ["promoted_alias", "acceptance_task"])
def test_failed_promotion_does_not_record_completed_checkpoint(
        tmp_path, failed_stage):
    holder = _checkpoint_holder(tmp_path)
    _enable_promotion(holder, tmp_path)
    real_publish = holder._publish_checkpoint_alias

    def fail(*_args):
        raise OSError(f"controlled {failed_stage} failure")

    def publish(source, destination):
        if str(destination) == holder.config.promoted_path:
            fail()
        return real_publish(source, destination)

    if failed_stage == "promoted_alias":
        holder._publish_checkpoint_alias = publish
    else:
        holder._enqueue_checkpoint_acceptance = fail
    Trainer._save_checkpoint(holder, loss=0.5)
    with pytest.raises(RuntimeError, match=f"controlled {failed_stage} failure"):
        Trainer._wait_for_checkpoint_writer(holder, timeout=30)
    assert holder.stats_collector.checkpoint_records == []
