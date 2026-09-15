"""Stop requests received during a wait must not start another training batch."""

from queue import SimpleQueue
from threading import Lock

import pytest
import torch

from dama.ai.ml import trainer as trainer_module


def _trainer():
    holder = object.__new__(trainer_module.Trainer)
    holder.config = trainer_module.TrainingConfig(
        amp=False, train_steps=1, learning_rate=0.1, thermal_enabled=False,
    )
    holder.model = torch.nn.Linear(2, 2, bias=False)
    holder.forwards = []

    def forward(boards, features, counts):
        holder.forwards.append(True)
        return holder.model(features.squeeze(-1))

    holder.model.forward_padded = forward
    holder.device = torch.device("cpu")
    holder.optimizer = torch.optim.SGD(holder.model.parameters(), lr=0.1)
    holder.scheduler = holder.scaler = holder.stats_collector = None
    holder.amp_dtype = torch.bfloat16
    holder._use_padded = True
    holder._compiled_fwd_loss = None
    holder._control_queue = SimpleQueue()
    holder._control_lock = Lock()
    holder._next_control_poll = holder._next_status_push = 0
    holder._CONTROL_POLL_INTERVAL = 0
    holder._push_status_reply = lambda: None
    holder._stopped = holder._paused = False
    holder.step = holder.epoch = 0
    return holder


def _batch():
    return (
        torch.zeros(2, 1), torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]]),
        torch.tensor([2, 2]), torch.tensor([0, 1]), torch.ones(2), torch.zeros(2),
    )


@pytest.mark.parametrize("wait_kind", ["pause", "thermal"])
@pytest.mark.parametrize("stop", [True, False])
def test_wait_exit_rechecks_stop_before_training(monkeypatch, wait_kind, stop):
    holder = _trainer()
    before = {name: value.clone() for name, value in holder.model.state_dict().items()}

    def end_wait(*args):
        holder._control_queue.put({
            "type": trainer_module.MSG_STOP if stop else trainer_module.MSG_RESUME,
        })
        holder._service_control_queue()

    if wait_kind == "pause":
        holder.pause()
        monkeypatch.setattr(trainer_module.time, "sleep", end_wait)
    else:
        holder.config.thermal_enabled = True
        holder._check_thermal_and_rest = end_wait

    holder.train_epoch([_batch()])

    assert holder.is_stopped is stop
    assert holder.step == int(not stop)
    assert holder._last_epoch_batches == int(not stop)
    assert len(holder.forwards) == int(not stop)
    if stop:
        for name, value in holder.model.state_dict().items():
            assert torch.equal(value, before[name])
        assert holder.optimizer.state == {}
    else:
        assert any(not torch.equal(value, before[name])
                   for name, value in holder.model.state_dict().items())


def test_pause_received_during_thermal_wait_blocks_next_batch(monkeypatch):
    holder = _trainer()
    holder.config.thermal_enabled = True
    resumed = []

    def thermal_wait():
        holder._control_queue.put({"type": trainer_module.MSG_PAUSE})
        holder._service_control_queue()

    def resume_wait(seconds):
        assert holder.forwards == []
        resumed.append(True)
        holder._control_queue.put({"type": trainer_module.MSG_RESUME})
        holder._service_control_queue()

    holder._check_thermal_and_rest = thermal_wait
    monkeypatch.setattr(trainer_module.time, "sleep", resume_wait)
    holder.train_epoch([_batch()])

    assert resumed == [True]
    assert holder.step == 1
    assert holder._last_epoch_batches == 1


def test_stop_while_paused_does_not_finish_accumulation_window(monkeypatch):
    holder = _trainer()
    holder.config.gradient_accumulation_steps = 2
    before = {name: value.clone() for name, value in holder.model.state_dict().items()}

    class PausingLoader:
        def __len__(self):
            return 2

        def __iter__(self):
            yield _batch()
            holder.pause()
            yield _batch()

    monkeypatch.setattr(trainer_module.time, "sleep", lambda seconds: holder.stop())
    holder.train_epoch(PausingLoader())

    assert holder.step == 0
    assert holder._last_epoch_batches == 1
    assert holder.forwards == [True]
    for name, value in holder.model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("stop_source", ["queue", "direct", "headless", "pause", "none"])
def test_thermal_rest_services_controls_and_reports_interruption(
    monkeypatch, capsys, stop_source,
):
    holder = _trainer()
    holder.config.thermal_enabled = True
    holder.config.thermal_rest_seconds = 60
    holder._last_thermal_check = 0
    holder._get_gpu_temperature = lambda: 100
    holder._get_cpu_temperature = lambda: 20
    if stop_source == "headless":
        holder._control_queue = None
    now = [1000.0]
    sleeps = []

    def advance(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        if len(sleeps) == 1:
            if stop_source == "queue":
                holder._control_queue.put({"type": trainer_module.MSG_STOP})
            elif stop_source == "pause":
                holder._control_queue.put({"type": trainer_module.MSG_PAUSE})
            elif stop_source in ("direct", "headless"):
                holder.stop()

    monkeypatch.setattr(trainer_module.time, "time", lambda: now[0])
    monkeypatch.setattr(trainer_module.time, "sleep", advance)
    holder._check_thermal_and_rest()

    output = capsys.readouterr().out
    if stop_source in ("none", "pause"):
        assert not holder.is_stopped
        assert sum(sleeps) == 60
        assert holder.is_paused is (stop_source == "pause")
        assert ("Resuming training..." in output) is (stop_source == "none")
    else:
        assert holder.is_stopped
        assert sum(sleeps) <= 5
        assert "Resuming training..." not in output
