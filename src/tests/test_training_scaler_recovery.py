"""Recover an unusable GradScaler without interrupting finite-scale warm-up."""

from types import SimpleNamespace

import pytest
import torch

from dama.ai.ml import trainer as trainer_module
from .test_training_dataset_lifetime import _dataset
from .test_training_scaled_step_accounting import _batch, _trainer


def _run_loop(monkeypatch, tmp_path, *, initial_scale, accumulation=1,
              corrupt_gradient=None, empty=False):
    holder = _trainer(monkeypatch, accumulation, clip=False)
    holder.config.batch_size = 2
    holder.config.checkpoint_every = 100
    holder.config.max_stale_epochs = 0
    holder.config.test_vs_algo = False
    holder.config.dataloader_workers = 0
    holder.config.reward_mode = "none"
    holder.stats = trainer_module.TrainingStats()
    holder.stats_collector = None
    holder.scaler = (None if initial_scale is None else
                     torch.amp.GradScaler("cpu", init_scale=initial_scale))
    holder._snapshot_manager = holder._acceptance_thread = None
    holder.replay_buffer = SimpleNamespace(count_entries=lambda: 100)
    dataset = _dataset()
    holder._preloaded_snapshot_dataset = dataset
    holder._preloaded_snapshot_cache_metadata = None
    holder._preloaded_snapshot_cache_checked = True
    holder._preloaded_validation_dataset = None
    holder._preloaded_validation_cache_metadata = None
    holder._prepare_training_split = lambda: ([], [])
    holder._ensure_frozen_teacher_suite = lambda: None
    holder._set_validation_entries = lambda *args, **kwargs: None
    holder._commit_validation_tensor_identity = lambda *args: None
    holder._record_epoch_loss = lambda *args: None
    holder._save_progress_report_if_due = lambda: None
    holder._start_background_selfplay = lambda *args: None
    holder._collect_background_selfplay = lambda: (None, None)
    holder._stop_background_selfplay = lambda: None
    holder._wait_for_checkpoint_writer = lambda: None

    class Loader(list):
        pass

    loader = Loader([] if empty else [_batch()])
    loader.dataset = dataset
    monkeypatch.setattr(trainer_module, "create_dataloader_from_dataset",
                        lambda *args, **kwargs: loader)
    monkeypatch.setattr(trainer_module, "GradScaler",
                        lambda **kwargs: torch.amp.GradScaler("cpu", **kwargs))
    observations = []
    original_epoch = holder.train_epoch

    def train_epoch(*args, **kwargs):
        loss = original_epoch(*args, **kwargs)
        observations.append((holder.step, holder._last_epoch_batches,
                             None if holder.scaler is None else holder.scaler.get_scale()))
        return loss

    holder.train_epoch = train_epoch

    def service_control():
        # Bound the pre-fix failure; no checkpoint, self-play or wait is live.
        if holder.epoch >= 12:
            holder._stopped = True

    holder._service_control_queue = service_control
    hook = (None if corrupt_gradient is None else
            holder.model.weight.register_hook(corrupt_gradient))
    checkpoint = tmp_path / "model_step_000000.pt"
    checkpoint.touch()
    holder._rollback_checkpoint_candidates = lambda pattern: [checkpoint]
    holder._has_non_finite_tensors = lambda: False
    rollbacks = []

    def load_checkpoint(path):
        rollbacks.append((holder.epoch, holder.step, path))
        if hook is not None:
            hook.remove()
        if empty:
            holder._stopped = True

    # Exercise the existing rollback selector and conservative scaler reset;
    # the persisted model read is outside this numerical-progress control.
    holder._load_checkpoint = load_checkpoint
    before = holder.model.weight.detach().clone()
    holder._run_training()
    return holder, before, rollbacks, observations


@pytest.mark.parametrize("initial_scale", [1e-45, 0.0, float("nan"), float("inf")])
@pytest.mark.parametrize("accumulation", [1, 3])
def test_unusable_scaler_enters_existing_recovery_after_three_epochs(
    monkeypatch, tmp_path, initial_scale, accumulation, capsys,
):
    holder, before, rollbacks, observations = _run_loop(
        monkeypatch, tmp_path, initial_scale=initial_scale,
        accumulation=accumulation,
        corrupt_gradient=lambda gradient: torch.full_like(gradient, float("inf")),
    )

    assert [entry[:2] for entry in rollbacks] == [(3, 0)]
    assert [entry[:2] for entry in observations[:3]] == [(0, 1)] * 3
    assert holder.step == holder.scheduler.last_epoch == 1
    assert holder.epoch == 4
    assert holder.scaler.get_scale() == 1024
    assert torch.isfinite(holder.model.weight).all()
    assert not torch.equal(holder.model.weight, before)
    assert "GradScaler scale is unusable" in capsys.readouterr().out


@pytest.mark.parametrize("accumulation", [1, 3])
def test_positive_finite_scale_warmup_can_reject_more_than_three_epochs(
    monkeypatch, tmp_path, accumulation,
):
    holder, before, rollbacks, observations = _run_loop(
        monkeypatch, tmp_path, initial_scale=2 ** 24, accumulation=accumulation,
        corrupt_gradient=lambda gradient: gradient.to(torch.float16).to(gradient.dtype),
    )

    assert rollbacks == []
    assert all(entry[:2] == (0, 1) for entry in observations[:3])
    assert holder.step == 1
    assert not torch.equal(holder.model.weight, before)
    assert holder.scaler.get_scale() > 0


def test_cpu_without_scaler_retains_ordinary_progress(monkeypatch, tmp_path):
    holder, before, rollbacks, _ = _run_loop(
        monkeypatch, tmp_path, initial_scale=None)
    assert rollbacks == []
    assert holder.step == holder.epoch == 1
    assert not torch.equal(holder.model.weight, before)


def test_empty_epochs_retain_existing_three_epoch_recovery(monkeypatch, tmp_path):
    holder, _, rollbacks, observations = _run_loop(
        monkeypatch, tmp_path, initial_scale=None, empty=True)
    assert [entry[:2] for entry in rollbacks] == [(3, 0)]
    assert observations == [(0, 0, None)] * 3
    assert holder.step == 0
