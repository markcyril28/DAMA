"""Background preparation honors duration expiry without an operator stop."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import dama.ai.ml.trainer as trainer_module
from dama.ai.ml.trainer import Trainer
from .test_snapshot_preparation_shutdown import _PreparationHarness


def _clock(monkeypatch):
    clock = SimpleNamespace(now=datetime(2026, 9, 15, 12))

    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now

    monkeypatch.setattr(trainer_module, "datetime", ClockDateTime)
    return clock


@pytest.mark.parametrize("route", ["per_file", "fallback"])
@pytest.mark.parametrize("expiry_stage", ["generate", "consider", "prepare", "validation"])
def test_snapshot_deadline_preserves_previous_handoff(
    monkeypatch, capsys, route, expiry_stage,
):
    harness = _PreparationHarness(monkeypatch, route)
    holder = harness.holder
    clock = _clock(monkeypatch)
    holder.config.stop_time = clock.now + timedelta(seconds=1)
    targets = {
        "generate": (holder, "run_selfplay"),
        "consider": (holder._snapshot_manager, "consider_snapshot"),
        "prepare": (holder._snapshot_manager, "prepare_split"),
        "validation": (holder, "_load_or_reuse_validation_entries"),
    }
    target, name = targets[expiry_stage]
    original = getattr(target, name)

    def finish_at_deadline(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now = holder.config.stop_time
        return result

    monkeypatch.setattr(target, name, finish_at_deadline)
    try:
        harness.start()
        holder._bg_selfplay_thread.join(timeout=1)
        assert not holder._bg_selfplay_thread.is_alive(), (
            "expired producer began another phase or generation cycle")
        assert not holder._stopped
        assert not holder._bg_selfplay_stop_event.is_set()
    finally:
        harness.close()

    expected = ["generate"]
    if expiry_stage != "generate":
        expected.append("consider")
    if expiry_stage in ("prepare", "validation"):
        expected.append(("prepare", harness.source_path, 1_000_000))
    if expiry_stage == "validation":
        expected.append(("validation", harness.context))
    assert harness.calls == expected
    for field, value in harness.old_handoff.items():
        assert getattr(holder, field) is value
    assert holder._train_tensor_window is harness.old_train_window
    assert holder._validation_tensor_identity is harness.old_validation_identity
    assert not holder._data_ready_event.is_set()
    captured = capsys.readouterr()
    assert "Background snapshot self-play error" not in captured.out
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("route", ["per_file", "fallback"])
@pytest.mark.parametrize("deadline", ["none", "future"])
def test_nonexpired_snapshot_deadline_keeps_exact_handoff(
    monkeypatch, route, deadline,
):
    harness = _PreparationHarness(monkeypatch, route)
    holder = harness.holder
    clock = _clock(monkeypatch)
    holder.config.stop_time = (
        None if deadline == "none" else clock.now + timedelta(hours=1))
    try:
        harness.start()
        assert holder._data_ready_event.wait(timeout=1)
        with holder._bg_selfplay_lock:
            assert holder._bg_selfplay_entries is None
            assert holder._bg_selfplay_dataset is harness.dataset
            assert holder._bg_selfplay_incremental is None
            assert holder._bg_snapshot_manifest is harness.context.manifest
            assert holder._bg_validation_entries is harness.validation_entries
            assert holder._bg_validation_identity is harness.validation_identity
        assert not holder._stopped
        assert not holder._bg_selfplay_stop_event.is_set()
    finally:
        harness.close()

    expected = harness.expected_prefix() + [("validation", harness.context)]
    if route == "per_file":
        expected.append(("per_file", harness.context))
    else:
        expected.extend([
            ("train", harness.context),
            ("tensorize", harness.train_entries, {
                "max_moves_per_sample": 32, "show_progress": True,
            }),
        ])
    assert harness.calls == expected


def _disk_holder(clock, minimum_gb):
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        selfplay_min_free_disk_gb=minimum_gb,
        replay_dir="/unused-replay",
        stop_time=clock.now + timedelta(seconds=1),
    )
    holder._stopped = False
    holder._bg_selfplay_stop_event = trainer_module.threading.Event()
    return holder


@pytest.mark.parametrize("minimum_gb", [0.0, 10.0])
def test_expired_disk_wait_does_not_authorize_generation(monkeypatch, minimum_gb):
    clock = _clock(monkeypatch)
    holder = _disk_holder(clock, minimum_gb)
    clock.now = holder.config.stop_time
    measurements = []

    def disk_usage(path):
        measurements.append(path)
        return SimpleNamespace(free=11 * 1024 ** 3)

    monkeypatch.setattr(trainer_module.shutil, "disk_usage", disk_usage)

    assert holder._wait_for_selfplay_disk_headroom() is False
    assert measurements == []
    assert not holder._stopped
    assert not holder._bg_selfplay_stop_event.is_set()


def test_storage_wait_exits_at_deadline_without_stop_event(monkeypatch):
    clock = _clock(monkeypatch)
    holder = _disk_holder(clock, 10.0)
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        assert len(waits) == 1, "disk wait repeated after deadline expiry"
        clock.now += timedelta(seconds=timeout)
        return False

    monkeypatch.setattr(holder._bg_selfplay_stop_event, "wait", wait)
    monkeypatch.setattr(
        trainer_module.shutil, "disk_usage",
        lambda path: SimpleNamespace(free=9 * 1024 ** 3),
    )

    assert holder._wait_for_selfplay_disk_headroom() is False
    assert waits == [trainer_module._SELFPLAY_DISK_HEADROOM_POLL_SECONDS]
    assert not holder._stopped
    assert not holder._bg_selfplay_stop_event.is_set()


def test_deadline_during_disk_measurement_refuses_ready_space(monkeypatch):
    clock = _clock(monkeypatch)
    holder = _disk_holder(clock, 10.0)
    measurements = []

    def disk_usage(path):
        measurements.append(path)
        clock.now = holder.config.stop_time
        return SimpleNamespace(free=11 * 1024 ** 3)

    monkeypatch.setattr(trainer_module.shutil, "disk_usage", disk_usage)

    assert holder._wait_for_selfplay_disk_headroom() is False
    assert len(measurements) == 1
    assert not holder._stopped
    assert not holder._bg_selfplay_stop_event.is_set()


@pytest.mark.parametrize("deadline", ["none", "future"])
def test_nonexpired_disk_wait_recovers_headroom(monkeypatch, deadline):
    clock = _clock(monkeypatch)
    holder = _disk_holder(clock, 10.0)
    holder.config.stop_time = (
        None if deadline == "none" else clock.now + timedelta(hours=1))
    free_values = iter((9 * 1024 ** 3, 11 * 1024 ** 3))
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        clock.now += timedelta(seconds=timeout)
        return False

    monkeypatch.setattr(holder._bg_selfplay_stop_event, "wait", wait)
    monkeypatch.setattr(
        trainer_module.shutil, "disk_usage",
        lambda path: SimpleNamespace(free=next(free_values)),
    )

    assert holder._wait_for_selfplay_disk_headroom() is True
    assert waits == [trainer_module._SELFPLAY_DISK_HEADROOM_POLL_SECONDS]
    assert not holder._stopped
    assert not holder._bg_selfplay_stop_event.is_set()


@pytest.mark.parametrize("snapshot_mode", [True, False])
def test_paused_producer_exits_at_deadline_without_resume(monkeypatch, snapshot_mode):
    harness = _PreparationHarness(monkeypatch, "per_file")
    holder = harness.holder
    clock = _clock(monkeypatch)
    holder.config.stop_time = clock.now + timedelta(seconds=1)
    holder._paused = True
    if not snapshot_mode:
        holder._snapshot_manager = None
    waits = []
    real_wait = holder._bg_selfplay_stop_event.wait

    def wait(timeout=None):
        waits.append(timeout)
        clock.now = holder.config.stop_time
        # The short real wait gives cleanup a bounded escape on the baseline.
        return real_wait(timeout=0.01)

    monkeypatch.setattr(holder._bg_selfplay_stop_event, "wait", wait)
    try:
        harness.start()
        holder._bg_selfplay_thread.join(timeout=1)
        assert not holder._bg_selfplay_thread.is_alive(), (
            "paused producer ignored deadline expiry")
        assert waits == [0.5]
        assert holder._paused
        assert not holder._stopped
        assert not holder._bg_selfplay_stop_event.is_set()
    finally:
        harness.close()

    assert harness.calls == []
    assert not holder._data_ready_event.is_set()
    for field, value in harness.old_handoff.items():
        assert getattr(holder, field) is value
