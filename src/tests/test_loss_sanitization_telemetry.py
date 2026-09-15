"""Loss telemetry distinguishes measured zero from sanitized invalid values."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import dama.ai.ml.trainer as trainer_module


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward_padded(self, boards, move_features, move_counts):
        scores = self.scale * move_features.squeeze(-1)
        valid = torch.arange(scores.shape[1])[None, :] < move_counts[:, None]
        return scores.masked_fill(~valid, float("-inf"))

    def forward_padded_with_value(self, boards, move_features, move_counts):
        return (
            self.forward_padded(boards, move_features, move_counts),
            self.scale.expand(boards.shape[0]) * 0.0,
        )


class _CapturingStats:
    def __init__(self):
        self.events = []
        self.steps = []

    def record_non_finite_event(self, step, source, message):
        self.events.append((step, source, message))

    def record_training_step(self, **record):
        self.steps.append(record)

    def record_epoch(self, **record):
        pass


def _holder(monkeypatch, mode):
    holder = object.__new__(trainer_module.Trainer)
    holder.config = SimpleNamespace(
        gradient_accumulation_steps=1,
        stats_record_every=1,
        stats_score_dist_every=100,
        stats_system_every=100,
        stats_model_health_every=100,
        checkpoint_every=100,
        train_steps=1,
        grad_clip_norm=None,
        amp=mode == "amp",
        value_head_enabled=mode == "compiled_value",
        value_weight=0.15,
        policy_stage="policy_only",
        batch_size=2,
        learning_rate=0.1,
        thermal_enabled=False,
    )
    holder.device = torch.device("cpu")
    holder.model = _TinyPolicy()
    holder.optimizer = torch.optim.SGD(holder.model.parameters(), lr=0.1)
    holder.scheduler = None
    holder.scaler = None
    holder.amp_dtype = torch.bfloat16
    holder.stats_collector = _CapturingStats()
    holder._use_padded = True
    holder._compiled_fwd_loss = None
    holder._control_queue = None
    holder._stopped = False
    holder._paused = False
    holder.step = 0
    holder.epoch = 0
    holder._data_refreshed_pending = False
    holder._record_step_stats = lambda *args, **kwargs: None
    holder._update_process_title = lambda *args, **kwargs: None
    holder.repairs = []
    holder._repair_batchnorm_stats = lambda: holder.repairs.append(True)

    if mode == "amp":
        # Exercise AMP's trainer branch without requiring CUDA in CPU tests.
        monkeypatch.setattr(trainer_module, "autocast", lambda **kwargs: nullcontext())
    elif mode.startswith("compiled"):
        compile_function = torch.compile
        monkeypatch.setattr(
            torch, "compile",
            lambda function, **kwargs: compile_function(
                function, backend="eager", fullgraph=True),
        )
        if mode == "compiled_value":
            holder._compiled_fwd_loss = trainer_module._make_compiled_fwd_loss_value(
                holder.model, "default", holder.config.value_weight)
        else:
            holder._compiled_fwd_loss = trainer_module._make_compiled_fwd_loss(
                holder.model, "default")
    return holder


def _batch(case):
    counts = torch.tensor([1, 1]) if case == "forced" else torch.tensor([2, 2])
    logits = [100.0, 0.0] if case == "saturated" else [1.0, 0.0]
    weights = torch.zeros(2) if case == "zero_weights" else torch.ones(2)
    if case == "nonfinite_weights":
        weights.fill_(float("nan"))
    return (
        torch.zeros(2, 1),
        torch.tensor([[logits], [logits]]).transpose(1, 2),
        counts,
        torch.zeros(2, dtype=torch.long),
        weights,
        torch.zeros(2),
    )


@pytest.mark.parametrize("mode", ["eager", "amp", "compiled", "compiled_value"])
@pytest.mark.parametrize("case", ["forced", "saturated", "zero_weights"])
def test_legitimate_zero_loss_is_not_reported_as_nonfinite(monkeypatch, mode, case):
    holder = _holder(monkeypatch, mode)

    average = holder.train_epoch([_batch(case)])

    assert average == 0.0
    assert holder.stats_collector.steps[0]["loss"] == 0.0
    assert holder.stats_collector.events == []
    assert holder.repairs == []


@pytest.mark.parametrize("mode", ["eager", "amp"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_sanitized_nonfinite_loss_is_reported(monkeypatch, mode, invalid):
    holder = _holder(monkeypatch, mode)
    # Finite forward outputs and gradients isolate the scalar-loss telemetry.
    holder._compute_loss_padded = lambda scores, *args: scores.sum() * 0.0 + invalid

    average = holder.train_epoch([_batch("ordinary")])

    assert average == 0.0
    assert holder.stats_collector.steps[0]["loss"] == 0.0
    assert len(holder.stats_collector.events) == 1
    assert holder.stats_collector.events[0][:2] == (1, "nan_to_num")
    assert f"raw_loss={invalid!r}" in holder.stats_collector.events[0][2]
    assert holder.repairs == [True]


@pytest.mark.parametrize("mode", ["compiled", "compiled_value"])
def test_compiled_nonfinite_loss_is_reported(monkeypatch, mode):
    holder = _holder(monkeypatch, mode)

    average = holder.train_epoch([_batch("nonfinite_weights")])

    assert average == 0.0
    assert len(holder.stats_collector.events) == 1
    assert holder.stats_collector.events[0][:2] == (1, "nan_to_num")
    assert holder.repairs == [True]


@pytest.mark.parametrize("mode", ["eager", "amp", "compiled", "compiled_value"])
def test_finite_positive_loss_remains_ordinary_telemetry(monkeypatch, mode):
    holder = _holder(monkeypatch, mode)

    average = holder.train_epoch([_batch("ordinary")])

    assert average > 0.0
    assert holder.stats_collector.events == []
    assert holder.repairs == []


@pytest.mark.parametrize("mode", ["eager", "compiled"])
@pytest.mark.parametrize("invalid", [False, True])
def test_accumulated_loss_telemetry_at_checkpoint_only_boundary(
    monkeypatch, mode, invalid,
):
    holder = _holder(monkeypatch, mode)
    holder.config.gradient_accumulation_steps = 2
    holder.config.stats_record_every = 100
    holder.config.checkpoint_every = 1
    checkpoints = []
    holder._save_checkpoint = lambda loss, **kwargs: checkpoints.append(loss)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch = _batch("nonfinite_weights" if invalid else "forced")

    average = holder.train_epoch([batch, batch])

    assert average == 0.0
    assert holder.step == 1
    assert holder._last_epoch_batches == 2
    assert checkpoints == [0.0]
    assert holder.stats_collector.steps == []
    assert len(holder.stats_collector.events) == int(invalid)
    assert len(holder.repairs) == int(invalid)
