"""An epoch's final valid microbatches must produce a normalized update."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from dama.ai.ml import trainer as trainer_module


def _trainer(accum_steps, mode, monkeypatch):
    holder = object.__new__(trainer_module.Trainer)
    holder.config = trainer_module.TrainingConfig(
        amp=mode == "scaled", train_steps=100, learning_rate=0.1,
        gradient_accumulation_steps=accum_steps, grad_clip_norm=None,
        stats_record_every=1, stats_score_dist_every=1,
        checkpoint_every=100, thermal_enabled=False,
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
    holder.scaler = None
    if mode == "scaled":
        monkeypatch.setattr(trainer_module, "autocast", lambda **kwargs: nullcontext())
        holder.scaler = torch.amp.GradScaler("cpu")
    holder.amp_dtype = torch.bfloat16
    holder._use_padded = True
    holder._compiled_fwd_loss = None
    if mode == "compiled":
        compile_function = torch.compile
        monkeypatch.setattr(
            torch, "compile", lambda function, **kwargs:
            compile_function(function, backend="eager", fullgraph=True))
        holder._compiled_fwd_loss = trainer_module._make_compiled_fwd_loss(
            holder.model, None)
    if mode == "soft":
        holder.config.policy_stage = "enhanced"
    holder._control_queue = None
    holder._stopped = holder._paused = False
    holder.step = holder.epoch = 0
    holder.records = []
    holder.events = []
    holder.stats_collector = SimpleNamespace(
        record_training_step=lambda **record: holder.records.append(record),
        record_epoch=lambda **record: None,
        record_non_finite_event=lambda *event: holder.events.append(event),
    )
    holder._record_step_stats = lambda *args: None
    holder._update_process_title = lambda *args: None
    return holder


def _batch(index, valid=True, soft=False):
    boards = torch.zeros(2, 1) if valid else torch.full((2, 1), float("nan"))
    features = torch.tensor([[[1.0], [0.2 + index * 0.1]], [[0.1], [1.0]]])
    batch = (boards, features, torch.tensor([2, 2]), torch.tensor([0, 1]),
             torch.ones(2), torch.zeros(2))
    return batch + (torch.eye(2),) if soft else batch


@pytest.mark.parametrize("mode", ["eager", "scaled", "compiled", "soft"])
@pytest.mark.parametrize("accum_steps,validity", [
    (1, (True,)),
    (2, (True,)),
    (3, (True, True)),
    (2, (True, True, True)),
    (3, (True, False, False)),
    (3, (False, True, False, True, False)),
    (2, (False, False)),
])
def test_final_accumulation_window_matches_valid_microbatch_mean(
    monkeypatch, mode, accum_steps, validity,
):
    holder = _trainer(accum_steps, mode, monkeypatch)
    # Exercise rejection at every test step rather than relying on production's
    # much sparser numerical-health sampling cadence.
    holder._SANITY_CHECK_INTERVAL = 1
    batches = [_batch(index, valid, mode == "soft")
               for index, valid in enumerate(validity)]
    valid_batches = [batch for batch, valid in zip(batches, validity) if valid]
    reference = torch.nn.Linear(2, 2, bias=False)
    reference.load_state_dict(holder.model.state_dict())
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    expected_sizes = []
    expected_scores = []
    for start in range(0, len(valid_batches), accum_steps):
        group = valid_batches[start:start + accum_steps]
        optimizer.zero_grad(set_to_none=True)
        for batch in group:
            scores = reference(batch[1].squeeze(-1))
            (torch.nn.functional.cross_entropy(scores, batch[3]) / len(group)).backward()
        optimizer.step()
        scheduler.step()
        expected_sizes.append(sum(batch[0].shape[0] for batch in group))
        expected_scores.append(trainer_module.StatsCollector.compute_score_stats_padded(
            scores.detach(), batch[2]))

    holder.train_epoch(batches)

    assert holder.step == len(expected_sizes)
    assert holder._last_epoch_batches == len(valid_batches)
    assert [record["batch_size"] for record in holder.records] == expected_sizes
    assert holder.scheduler.last_epoch == scheduler.last_epoch
    torch.testing.assert_close(holder.model.weight, reference.weight)
    assert all(record["step_time"] > 0 for record in holder.records)
    for record, expected in zip(holder.records, expected_scores):
        assert record["score_stats"] == pytest.approx(expected)


@pytest.mark.parametrize("stop", [False, True])
def test_exhausted_loader_services_stop_before_pending_update(monkeypatch, stop):
    holder = _trainer(2, "eager", monkeypatch)
    before = holder.model.weight.detach().clone()
    holder._control_queue = object()

    class Loader:
        def __len__(self):
            return 1

        def __iter__(self):
            yield _batch(0)
            holder.stop_is_pending = stop

    holder.stop_is_pending = False
    holder._service_control_queue = lambda: setattr(
        holder, "_stopped", holder.stop_is_pending)
    holder.train_epoch(Loader())
    assert holder.step == int(not stop)
    assert torch.equal(holder.model.weight, before) is stop


def test_final_update_reaches_checkpoint_and_step_limit(monkeypatch):
    holder = _trainer(3, "eager", monkeypatch)
    holder.config.train_steps = holder.config.checkpoint_every = 1
    checkpoints = []
    holder._save_checkpoint = lambda loss, **kwargs: checkpoints.append(holder.step)
    holder.train_epoch([_batch(0), _batch(1)])
    assert holder.step == 1
    assert checkpoints == [1]


def test_full_window_at_step_limit_does_not_start_a_tail(monkeypatch):
    holder = _trainer(2, "eager", monkeypatch)
    holder.config.train_steps = 1
    holder.train_epoch([_batch(0), _batch(1), _batch(2)])
    assert holder.step == 1
    assert holder._last_epoch_batches == 2
    assert len(holder.records) == 1


@pytest.mark.parametrize("control", ["pause", "thermal-stop"])
def test_final_window_obeys_wait_controls(monkeypatch, control):
    holder = _trainer(2, "eager", monkeypatch)
    waits = []

    class Loader:
        def __len__(self):
            return 1

        def __iter__(self):
            yield _batch(0)
            if control == "pause":
                holder._paused = True
            else:
                holder.finish_requested = True

    def end_pause(seconds):
        assert holder.step == 0
        waits.append(seconds)
        holder._paused = False

    holder.finish_requested = False
    holder.config.thermal_enabled = control == "thermal-stop"
    holder._check_thermal_and_rest = lambda: setattr(
        holder, "_stopped", holder.finish_requested)
    monkeypatch.setattr(trainer_module.time, "sleep", end_pause)
    holder.train_epoch(Loader())
    assert holder.step == int(control == "pause")
    assert waits == ([0.1] if control == "pause" else [])


def test_rejected_tail_preserves_last_successful_raw_loss_evidence(monkeypatch):
    holder = _trainer(2, "eager", monkeypatch)
    holder._SANITY_CHECK_INTERVAL = 1
    holder._repair_batchnorm_stats = lambda: False

    def forward(boards, features, counts, targets, weights):
        scores = holder.model.forward_padded(boards, features, counts)
        return scores.sum() * 0, scores.detach(), torch.tensor(float("nan"))

    holder._compiled_fwd_loss = forward
    holder.train_epoch([_batch(0), _batch(1, valid=False)])
    assert holder.step == 1
    replacements = [event for event in holder.events if event[1] == "nan_to_num"]
    assert len(replacements) == 1
    assert "raw_loss=nan" in replacements[0][2]


@pytest.mark.parametrize("mode", ["eager", "scaled"])
def test_short_window_is_normalized_before_gradient_clipping(monkeypatch, mode):
    holder = _trainer(3, mode, monkeypatch)
    holder.config.grad_clip_norm = 0.05
    reference = torch.nn.Linear(2, 2, bias=False)
    reference.load_state_dict(holder.model.state_dict())
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    batch = _batch(0)
    torch.nn.functional.cross_entropy(
        reference(batch[1].squeeze(-1)), batch[3]).backward()
    unclipped = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.05)
    assert unclipped > 0.05
    optimizer.step()

    holder.train_epoch([batch])

    assert holder.step == 1
    torch.testing.assert_close(holder.model.weight, reference.weight)
    assert holder.records[0]["grad_norm"] == pytest.approx(unclipped.item())
