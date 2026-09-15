"""Rejected AMP updates must not consume the optimizer-step budget."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from dama.ai.ml import trainer as trainer_module


def _trainer(monkeypatch, accum_steps, clip):
    holder = object.__new__(trainer_module.Trainer)
    holder.config = trainer_module.TrainingConfig(
        amp=True, train_steps=1, learning_rate=0.1,
        gradient_accumulation_steps=accum_steps,
        grad_clip_norm=5.0 if clip else None,
        stats_record_every=1, stats_score_dist_every=1,
        checkpoint_every=1, thermal_enabled=False,
    )
    holder.model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        holder.model.weight.copy_(torch.tensor([[0.2, 0.1], [-0.1, 0.3]]))
    holder.model.forward_padded = (
        lambda boards, features, counts: holder.model(features.squeeze(-1)))
    holder.device = torch.device("cpu")
    holder.optimizer = torch.optim.SGD(holder.model.parameters(), lr=0.1)
    holder.scheduler = torch.optim.lr_scheduler.StepLR(
        holder.optimizer, step_size=1, gamma=0.9)
    monkeypatch.setattr(trainer_module, "autocast", lambda **kwargs: nullcontext())
    holder.scaler = torch.amp.GradScaler("cpu")
    holder.amp_dtype = torch.float16
    holder._use_padded = True
    holder._compiled_fwd_loss = None
    holder._control_queue = None
    holder._stopped = holder._paused = False
    holder.step = holder.epoch = 0
    holder.records, holder.events, holder.checkpoints = [], [], []
    holder.stats_collector = SimpleNamespace(
        record_training_step=lambda **record: holder.records.append(record),
        record_epoch=lambda **record: None,
        record_non_finite_event=lambda *event: holder.events.append(event),
    )
    holder._record_step_stats = lambda *args: None
    holder._update_process_title = lambda *args: None
    holder._save_checkpoint = lambda *args, **kwargs: holder.checkpoints.append(holder.step)
    return holder


def _batch():
    return (
        torch.zeros(2, 1),
        torch.tensor([[[1.0], [0.2]], [[0.1], [1.0]]]),
        torch.tensor([2, 2]), torch.tensor([0, 1]),
        torch.ones(2), torch.zeros(2),
    )


@pytest.mark.parametrize("accum_steps,batch_count", [(1, 1), (3, 1), (3, 3)])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("clip", [False, True])
def test_rejected_scaled_update_does_not_advance_progress(
    monkeypatch, accum_steps, batch_count, invalid, clip, capsys,
):
    holder = _trainer(monkeypatch, accum_steps, clip)
    before = holder.model.weight.detach().clone()
    initial_scale = holder.scaler.get_scale()
    holder.model.weight.register_hook(lambda gradient: torch.full_like(gradient, invalid))

    holder.train_epoch([_batch()] * batch_count)

    assert torch.equal(holder.model.weight, before)
    assert holder.scaler.get_scale() == initial_scale / 2
    assert holder.step == holder.scheduler.last_epoch == 0
    assert holder.records == holder.checkpoints == []
    assert len(holder.events) == 1
    assert holder.events[0][0:2] == (0, "grad_scaler")
    assert "skipped" in holder.events[0][2]
    assert "skipped" in capsys.readouterr().out
    assert not holder.optimizer._optimizer_step_post_hooks
    if clip:
        assert "grad_norm=" in holder.events[0][2]


@pytest.mark.parametrize("accum_steps,batch_count", [(1, 2), (3, 4), (3, 6)])
@pytest.mark.parametrize("clip", [False, True])
def test_valid_window_after_rejection_reaches_the_step_budget(
    monkeypatch, accum_steps, batch_count, clip,
):
    holder = _trainer(monkeypatch, accum_steps, clip)
    reference = torch.nn.Linear(2, 2, bias=False)
    reference.load_state_dict(holder.model.state_dict())
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    batch = _batch()
    torch.nn.functional.cross_entropy(reference(batch[1].squeeze(-1)), batch[3]).backward()
    optimizer.step()
    backward_calls = 0

    def corrupt_first_window(gradient):
        nonlocal backward_calls
        backward_calls += 1
        if backward_calls <= accum_steps:
            return torch.full_like(gradient, float("inf"))
        return gradient

    holder.model.weight.register_hook(corrupt_first_window)
    holder.train_epoch([batch] * batch_count)

    assert backward_calls == batch_count
    torch.testing.assert_close(holder.model.weight, reference.weight)
    assert holder.step == holder.scheduler.last_epoch == 1
    assert holder.checkpoints == [1]
    assert len(holder.records) == len(holder.events) == 1
    assert holder.records[0]["batch_size"] == 2 * (batch_count - accum_steps)
    assert not holder.optimizer._optimizer_step_post_hooks


@pytest.mark.parametrize("enabled", [False, True])
def test_valid_scaled_update_counts_when_scale_grows_or_scaler_is_disabled(monkeypatch, enabled):
    holder = _trainer(monkeypatch, 3, False)
    holder.scaler = torch.amp.GradScaler("cpu", growth_interval=1, enabled=enabled)
    initial_scale = holder.scaler.get_scale()

    holder.train_epoch([_batch()])

    assert holder.step == holder.scheduler.last_epoch == 1
    assert holder.checkpoints == [1]
    assert len(holder.records) == 1
    assert holder.events == []
    assert holder.scaler.get_scale() == initial_scale * (2 if enabled else 1)


def test_rejections_still_count_correctly_after_scaler_underflow(monkeypatch):
    holder = _trainer(monkeypatch, 1, False)
    holder.scaler = torch.amp.GradScaler("cpu", init_scale=1e-45)
    holder.model.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf")))

    holder.train_epoch([_batch(), _batch()])

    assert holder.scaler.get_scale() == 0
    assert holder.step == holder.scheduler.last_epoch == 0
    assert holder.records == holder.checkpoints == []
    assert len(holder.events) == 2


@pytest.mark.parametrize("failure_at", ["step", "update"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_optimizer_observer_is_removed_when_scaler_fails(monkeypatch, failure_at, error_type):
    holder = _trainer(monkeypatch, 1, False)

    def fail(*args, **kwargs):
        raise error_type("injected scaler failure")

    monkeypatch.setattr(holder.scaler, failure_at, fail)
    initial_hooks = dict(holder.optimizer._optimizer_step_post_hooks)
    with pytest.raises(error_type, match="injected scaler failure"):
        holder.train_epoch([_batch()])
    assert dict(holder.optimizer._optimizer_step_post_hooks) == initial_hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("accum_steps", [1, 3])
def test_real_cuda_scaler_rejects_then_completes_a_valid_window(monkeypatch, accum_steps):
    holder = _trainer(monkeypatch, accum_steps, True)
    monkeypatch.setattr(trainer_module, "autocast", torch.amp.autocast)
    holder.device = torch.device("cuda")
    holder.model.to(holder.device)
    holder.optimizer = torch.optim.AdamW(holder.model.parameters(), lr=0.1, fused=False)
    holder.scheduler = torch.optim.lr_scheduler.StepLR(
        holder.optimizer, step_size=1, gamma=0.9)
    holder.scaler = torch.amp.GradScaler("cuda", init_scale=128)
    before = holder.model.weight.detach().clone()
    backward_calls = 0

    def corrupt_first_window(gradient):
        nonlocal backward_calls
        backward_calls += 1
        if backward_calls <= accum_steps:
            return torch.full_like(gradient, float("inf"))
        return gradient

    holder.model.weight.register_hook(corrupt_first_window)
    holder.train_epoch([_batch()] * (accum_steps + 1))

    assert backward_calls == accum_steps + 1
    assert holder.scaler.get_scale() == 64
    assert not torch.equal(holder.model.weight, before)
    assert torch.isfinite(holder.model.weight).all()
    assert holder.step == holder.scheduler.last_epoch == 1
    assert holder.checkpoints == [1]
    assert len(holder.records) == len(holder.events) == 1
    assert holder.records[0]["batch_size"] == 2
