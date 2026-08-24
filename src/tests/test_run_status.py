"""Every trainer exit must record a terminal reason, or leave proof it could not.

Merged audit Suggestion 5 / "Trainer stability": two WSL logs ended without a
terminal marker and one Windows run exited non-zero after repeated
closed-handle errors.  Console output alone cannot answer "why did it stop?" --
a hard kill discards whatever Python had buffered, and no in-process handler
runs at all for SIGKILL.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import dama.ai.ml.trainer as trainer_module
from dama.ai.ml import run_status
from dama.ai.ml.trainer import Trainer


def _read(log_dir: Path) -> dict:
    return json.loads(
        (log_dir / run_status.RUN_STATUS_FILENAME).read_text(encoding="utf-8"))


def test_begin_run_opens_a_running_marker_before_any_training(tmp_path: Path) -> None:
    assert run_status.begin_run(tmp_path, pid=4321) is None
    record = _read(tmp_path)
    assert record["status"] == "running"
    assert record["reason"] is None
    assert record["pid"] == 4321
    assert record["started_at"]
    assert record["ended_at"] is None


@pytest.mark.parametrize("reason", sorted(run_status.TERMINAL_REASONS))
def test_each_declared_terminal_reason_closes_the_marker(
    tmp_path: Path, reason: str
) -> None:
    run_status.begin_run(tmp_path, pid=1)
    run_status.record_terminal_reason(tmp_path, reason, context={"step": 7})
    record = _read(tmp_path)
    assert record["status"] == "terminated"
    assert record["reason"] == reason
    assert record["ended_at"]
    assert record["context"]["step"] == 7


def test_an_undeclared_reason_is_refused(tmp_path: Path) -> None:
    run_status.begin_run(tmp_path, pid=1)
    with pytest.raises(ValueError, match="Unknown terminal reason"):
        run_status.record_terminal_reason(tmp_path, "just because")


def test_a_run_that_was_killed_is_reported_and_preserved_by_the_next_start(
    tmp_path: Path,
) -> None:
    """The SIGKILL/OOM case: absence of a terminal record IS the record."""
    run_status.begin_run(tmp_path, pid=99, context={"step": 174000})
    # No record_terminal_reason() -- this models a process that simply vanished.

    unterminated = run_status.begin_run(tmp_path, pid=100)
    assert unterminated is not None
    assert unterminated["status"] == "unterminated"
    assert unterminated["pid"] == 99
    preserved = Path(unterminated["preserved_path"])
    assert preserved.is_file()
    assert "174000" in run_status.describe_unterminated(unterminated)

    # The live marker now belongs to the new run, and the old evidence remains.
    assert _read(tmp_path)["pid"] == 100
    assert run_status.begin_run(tmp_path, pid=101) is not None
    assert len(list(tmp_path.glob(
        f"{run_status.UNTERMINATED_PREFIX}*.json"))) >= 1


def test_a_cleanly_terminated_run_is_not_reported_as_killed(tmp_path: Path) -> None:
    run_status.begin_run(tmp_path, pid=1)
    run_status.record_terminal_reason(tmp_path, run_status.REASON_COMPLETED)
    assert run_status.begin_run(tmp_path, pid=2) is None


# ---------------------------------------------------------------------------
# Trainer.train() wrapper
# ---------------------------------------------------------------------------

def _holder(tmp_path: Path, **overrides) -> SimpleNamespace:
    config = SimpleNamespace(
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        policy_stage="policy_only",
        resume="models/checkpoints/model_step_174000.pt",
        stop_time=None,
    )
    for key, value in overrides.pop("config", {}).items():
        setattr(config, key, value)
    holder = SimpleNamespace(
        config=config,
        _stopped=False,
        step=174000,
        epoch=12,
        _run_training=lambda: None,
    )
    for key, value in overrides.items():
        setattr(holder, key, value)
    return holder


def test_normal_completion_records_completed(tmp_path: Path) -> None:
    holder = _holder(tmp_path)
    Trainer.train(holder)
    assert _read(Path(holder.config.log_dir))["reason"] == (
        run_status.REASON_COMPLETED)


def test_a_stop_request_is_distinguished_from_completion(tmp_path: Path) -> None:
    holder = _holder(tmp_path)

    def _stop() -> None:
        holder._stopped = True

    holder._run_training = _stop
    Trainer.train(holder)
    assert _read(Path(holder.config.log_dir))["reason"] == (
        run_status.REASON_STOP_REQUESTED)


def test_reaching_the_time_limit_is_distinguished_from_completion(
    tmp_path: Path,
) -> None:
    holder = _holder(
        tmp_path,
        config={"stop_time": datetime.now() - timedelta(seconds=1)},
    )
    Trainer.train(holder)
    assert _read(Path(holder.config.log_dir))["reason"] == (
        run_status.REASON_TIME_LIMIT)


def test_an_unhandled_exception_records_its_type_and_traceback(
    tmp_path: Path,
) -> None:
    """The Windows broken-pool run exited non-zero and explained nothing."""
    def _boom() -> None:
        raise OSError("[WinError 6] The handle is invalid")

    holder = _holder(tmp_path, _run_training=_boom)
    with pytest.raises(OSError):
        Trainer.train(holder)
    record = _read(Path(holder.config.log_dir))
    assert record["reason"] == run_status.REASON_EXCEPTION
    assert "OSError" in record["detail"]
    assert "handle is invalid" in record["detail"]
    assert "Traceback" in record["traceback"]
    assert record["context"]["step"] == 174000


def test_a_keyboard_interrupt_is_recorded_and_still_propagates(
    tmp_path: Path,
) -> None:
    def _interrupt() -> None:
        raise KeyboardInterrupt

    holder = _holder(tmp_path, _run_training=_interrupt)
    with pytest.raises(KeyboardInterrupt):
        Trainer.train(holder)
    assert _read(Path(holder.config.log_dir))["reason"] == (
        run_status.REASON_INTERRUPTED)


def test_a_previous_unterminated_run_is_announced_at_startup(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    holder = _holder(tmp_path)
    run_status.begin_run(holder.config.log_dir, pid=555, context={"step": 3})
    Trainer.train(holder)
    output = capsys.readouterr().out
    assert "without recording a terminal reason" in output
    assert "pid 555" in output


def test_marker_failure_never_prevents_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = []
    holder = _holder(tmp_path, _run_training=lambda: ran.append(True))

    def _explode(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(run_status, "begin_run", _explode)
    monkeypatch.setattr(run_status, "record_terminal_reason", _explode)
    Trainer.train(holder)
    assert ran == [True]


# ---------------------------------------------------------------------------
# Windows closed-handle teardown
# ---------------------------------------------------------------------------

def test_pool_teardown_survives_repeated_closed_handle_errors() -> None:
    """The signature of the 2026-08-23 Windows failure, reproduced portably.

    A broken pool leaves worker handles already closed, so every probe and
    every escalation raises.  Teardown must still complete: the caller owns
    unfinished batches it has to re-run sequentially, and an exception here
    would lose them and take the whole run down with it.
    """
    calls = {"is_alive": 0, "terminate": 0, "kill": 0, "join": 0}

    class _ClosedHandleProcess:
        def is_alive(self):
            calls["is_alive"] += 1
            raise OSError("handle is closed")

        def terminate(self):
            calls["terminate"] += 1
            raise ValueError("process object is closed")

        def kill(self):
            calls["kill"] += 1
            raise OSError("handle is closed")

        def join(self, timeout=None):
            calls["join"] += 1
            raise ValueError("process object is closed")

    class _BrokenExecutor:
        def __init__(self) -> None:
            self._processes = {0: _ClosedHandleProcess()}
            self._executor_manager_thread = None

        def shutdown(self, wait=True, cancel_futures=False):
            raise OSError("handle is closed")

    trainer_module._shutdown_selfplay_executor(_BrokenExecutor(), timeout=0)
    assert calls["is_alive"] >= 1
    assert calls["join"] >= 1


def test_pool_teardown_survives_a_shutdown_without_cancel_futures() -> None:
    """Older/narrower executor doubles must not break the bounded teardown."""
    seen = []

    class _NarrowExecutor:
        _processes: dict = {}
        _executor_manager_thread = None

        def shutdown(self, wait=True):
            seen.append(wait)

    trainer_module._shutdown_selfplay_executor(_NarrowExecutor(), timeout=0)
    assert seen == [False]
