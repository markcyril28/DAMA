"""Every trainer exit must record a terminal reason, or leave proof it could not.

Merged audit Suggestion 5 / "Trainer stability": two WSL logs ended without a
terminal marker and one Windows run exited non-zero after repeated
closed-handle errors.  Console output alone cannot answer "why did it stop?" --
a hard kill discards whatever Python had buffered, and no in-process handler
runs at all for SIGKILL.
"""

import json
import os
import stat
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


def test_atomic_status_write_fsyncs_file_and_directory_around_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete bytes and then their public pathname become crash-durable."""
    path = tmp_path / run_status.RUN_STATUS_FILENAME
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd):
        mode = os.fstat(fd).st_mode
        events.append("directory_fsync" if stat.S_ISDIR(mode) else "file_fsync")
        return real_fsync(fd)

    def tracking_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(run_status.os, "fsync", tracking_fsync)
    monkeypatch.setattr(run_status.os, "replace", tracking_replace)

    run_status._write_json_atomic(path, {"status": "running"})

    assert events == ["file_fsync", "replace", "directory_fsync"]
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "running"}


def test_atomic_status_write_reports_directory_fsync_failure_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed directory commit is visible to callers and strands no temp."""
    path = tmp_path / run_status.RUN_STATUS_FILENAME
    path.write_text('{"status": "old"}\n', encoding="utf-8")
    real_fsync = os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(5, "simulated status directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(run_status.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="status directory fsync failure"):
        run_status._write_json_atomic(path, {"status": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "new"}
    assert not list(tmp_path.glob("*.tmp"))


def test_status_directory_fsync_keeps_native_windows_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native Windows keeps atomic replacement without directory descriptors."""

    def fail_open(*_args, **_kwargs):
        raise AssertionError("native Windows must not open a directory fd")

    monkeypatch.setattr(run_status.os, "name", "nt")
    monkeypatch.setattr(run_status.os, "open", fail_open)

    run_status._fsync_directory(tmp_path)


def test_begin_run_opens_a_running_marker_before_any_training(tmp_path: Path) -> None:
    assert run_status.begin_run(tmp_path, pid=4321) is None
    record = _read(tmp_path)
    assert record["status"] == "running"
    assert record["reason"] is None
    assert record["pid"] == 4321
    assert record["started_at"]
    assert record["ended_at"] is None


def test_begin_run_commits_log_directory_before_running_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first-run marker cannot outlive only a volatile directory name."""
    parent = tmp_path / "logs"
    parent.mkdir()
    log_dir = parent / "new_namespace"
    synced = []
    real_sync = run_status._fsync_directory

    def tracking_sync(path: Path) -> None:
        synced.append(Path(path))
        real_sync(Path(path))

    monkeypatch.setattr(run_status, "_fsync_directory", tracking_sync)

    run_status.begin_run(log_dir, pid=4321)

    assert synced == [parent, log_dir]
    assert _read(log_dir)["status"] == "running"


def test_begin_run_reports_parent_directory_commit_failure_before_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not acknowledge a running marker beneath an uncommitted log dir."""
    parent = tmp_path / "logs"
    parent.mkdir()
    log_dir = parent / "new_namespace"

    def fail_parent_sync(path: Path) -> None:
        if Path(path) == parent:
            raise OSError(5, "simulated log-directory commit failure")
        raise AssertionError("marker publication must not begin")

    monkeypatch.setattr(run_status, "_fsync_directory", fail_parent_sync)

    with pytest.raises(OSError, match="log-directory commit failure"):
        run_status.begin_run(log_dir, pid=4321)

    assert log_dir.is_dir()
    assert not (log_dir / run_status.RUN_STATUS_FILENAME).exists()
    assert not list(log_dir.glob("*.tmp"))


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
        # Stubbed so the suite does not leave real SIGTERM/SIGHUP handlers
        # installed process-wide; the handler itself is exercised directly by
        # the signal tests at the bottom of this file.
        _install_termination_handler=lambda _log_dir: None,
    )
    # Real implementations, not stubs: these startup helpers only read config
    # and write diagnostics, so binding them keeps train()'s startup sequence
    # exercised rather than stubbed past.
    for _name in (
        "_capture_run_identity",
        "_capture_prelaunch_free_ram",
        "_warn_about_checkpoints_above_resume_point",
        "_warn_about_memory_headroom",
    ):
        setattr(holder, _name, getattr(Trainer, _name).__get__(holder))
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


# ---------------------------------------------------------------------------
# Startup failures, which happen before Trainer.train() opens the marker
# ---------------------------------------------------------------------------

def test_a_failure_before_the_training_loop_still_records_a_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-08-24 test run crashed here and left nothing behind.

    ``Trainer.__init__`` loads the resume checkpoint, opens the corpus and
    reaches the GPU -- all before ``train()`` opens the marker. A crash in any
    of them used to leave no ``run_status.json`` at all, which is
    indistinguishable from a launch that never happened.
    """
    log_dir = tmp_path / "logs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "training:\n"
        "  stage: \"policy_only\"\n"
        "paths:\n"
        f"  log_dir: \"{log_dir.as_posix()}\"\n",
        encoding="utf-8",
    )

    class _Exploding:
        def __init__(self, _config):
            raise RuntimeError("RNG state must be a torch.ByteTensor")

    monkeypatch.setattr(trainer_module, "Trainer", _Exploding)
    monkeypatch.setattr(
        trainer_module.sys, "argv",
        ["trainer", "--config", str(config_path)])

    with pytest.raises(RuntimeError):
        trainer_module.main()

    record = _read(log_dir)
    assert record["status"] == "terminated"
    assert record["reason"] == run_status.REASON_EXCEPTION
    assert "RNG state must be a torch.ByteTensor" in record["detail"]
    assert "Traceback" in record["traceback"]
    assert record["context"]["phase"] == "startup"


def test_an_interrupt_during_startup_is_not_reported_as_a_crash(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(log_dir=str(tmp_path), resume="")
    trainer_module._record_startup_failure(
        config, run_status.REASON_INTERRUPTED)

    record = _read(tmp_path)
    assert record["reason"] == run_status.REASON_INTERRUPTED
    assert "detail" not in record


def test_an_unwritable_log_dir_never_masks_the_startup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original exception must reach the operator, not a bookkeeping one."""
    def _explode(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(run_status, "record_terminal_reason", _explode)
    trainer_module._record_startup_failure(
        SimpleNamespace(log_dir=str(tmp_path), resume=""),
        run_status.REASON_EXCEPTION,
        detail="boom",
    )


# ---------------------------------------------------------------------------
# SIGTERM: how stop_training.sh actually ends a run
# ---------------------------------------------------------------------------

def test_sigterm_records_a_stop_instead_of_looking_like_a_kill(
    tmp_path: Path,
) -> None:
    """`bash stop_training.sh` SIGTERMs the trainer.

    Python's default SIGTERM disposition terminates without unwinding, so the
    ``finally`` in train() never ran and the marker stayed ``running`` -- the
    state this module reserves for a process killed outright, OOM named as the
    prime suspect. An ordinary operator stop produced exactly the false
    diagnosis the marker exists to prevent.
    """
    import signal

    run_status.begin_run(tmp_path, pid=os.getpid())
    holder = SimpleNamespace(_stopped=False, step=176000, epoch=3520)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        Trainer._install_termination_handler(holder, str(tmp_path))
        os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGHUP, signal.SIG_DFL)

    record = _read(tmp_path)
    assert record["status"] == "terminated"
    assert record["reason"] == run_status.REASON_STOP_REQUESTED
    assert record["detail"] == "received SIGTERM"
    assert record["context"]["step"] == 176000
    # The cooperative flag is set too, so a run between epochs still gets to
    # shut its pools down rather than waiting for the SIGKILL three seconds on.
    assert holder._stopped is True


def test_a_forked_self_play_worker_does_not_touch_the_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers inherit both the handler and the matched process title."""
    import signal

    installed: dict = {}
    monkeypatch.setattr(
        signal, "signal",
        lambda sig, handler: installed.setdefault(sig, handler))

    holder = SimpleNamespace(_stopped=False, step=1, epoch=1)
    Trainer._install_termination_handler(holder, str(tmp_path))
    handler = installed[signal.SIGTERM]

    reraised: list = []
    monkeypatch.setattr(os, "getpid", lambda: 999999)   # a forked child
    monkeypatch.setattr(os, "kill", lambda pid, sig: reraised.append((pid, sig)))

    handler(signal.SIGTERM, None)

    assert reraised == [(999999, signal.SIGTERM)]
    assert holder._stopped is False
    assert not (tmp_path / run_status.RUN_STATUS_FILENAME).exists()


def test_a_pool_dying_from_the_same_stop_signal_is_not_filed_as_a_crash(
    tmp_path: Path,
) -> None:
    """Workers share the process title, so they are SIGTERMed too.

    Whichever dies first decides what the trainer sees: an operator stop very
    often arrives as a BrokenProcessPool raised out of the training loop. That
    is still a stop, not a defect.
    """
    def _pool_died():
        raise RuntimeError("A process in the process pool was terminated")

    holder = _holder(tmp_path, _run_training=_pool_died, _stopped=True)
    with pytest.raises(RuntimeError):
        Trainer.train(holder)

    record = _read(tmp_path / "logs")
    assert record["reason"] == run_status.REASON_STOP_REQUESTED
    # Nothing is discarded: the raised error is still on the record.
    assert "BrokenProcessPool" in record["detail"] or "pool" in record["detail"]
    assert "Traceback" in record["traceback"]


# ---------------------------------------------------------------------------
# A running marker whose pid is a live trainer refuses a second writer.
# Preserving it as "unterminated" would be a false verdict (the run did not
# die), and proceeding would hand two trainers one namespace: same checkpoint
# files, same replay shards, same corpus snapshots.
# ---------------------------------------------------------------------------

_HAS_PROCFS = Path("/proc").is_dir()


@pytest.mark.skipif(not _HAS_PROCFS, reason="live-trainer evidence is procfs-based")
def test_begin_run_refuses_to_stomp_a_live_trainer(tmp_path: Path) -> None:
    import subprocess
    import time

    fake = subprocess.Popen(["bash", "-c", "exec -a micro-trainer sleep 60"])
    try:
        deadline = time.time() + 5.0
        trainer_argv_ready = False
        while time.time() < deadline:
            cmdline = Path(f"/proc/{fake.pid}/cmdline").read_bytes()
            # The pre-exec Bash argv contains the literal script text
            # ``exec -a micro-trainer ...`` too. Wait for argv[0] itself to
            # change, otherwise the next read can land in exec's transient
            # empty-cmdline window and manufacture an active-guard failure.
            if cmdline.split(b"\0", 1)[0] == b"micro-trainer":
                trainer_argv_ready = True
                break
            time.sleep(0.01)
        assert trainer_argv_ready

        run_status.begin_run(tmp_path, pid=fake.pid)
        marker = tmp_path / run_status.RUN_STATUS_FILENAME
        before = marker.read_bytes()

        with pytest.raises(run_status.ActiveRunError) as excinfo:
            run_status.check_no_active_run(tmp_path)
        assert excinfo.value.pid == fake.pid
        assert "micro-trainer" in (excinfo.value.cmdline or "")

        with pytest.raises(run_status.ActiveRunError):
            run_status.begin_run(tmp_path, pid=os.getpid())

        # Zero writes on refusal: the live run's marker is untouched and no
        # false "unterminated" record was preserved beside it.
        assert marker.read_bytes() == before
        assert not list(tmp_path.glob(run_status.UNTERMINATED_PREFIX + "*"))
    finally:
        fake.kill()
        fake.wait()


@pytest.mark.skipif(not _HAS_PROCFS, reason="live-trainer evidence is procfs-based")
def test_begin_run_still_preserves_when_the_live_pid_is_not_a_trainer(
    tmp_path: Path,
) -> None:
    """Pid reuse by an unrelated process must not block the next start."""
    import subprocess

    bystander = subprocess.Popen(["sleep", "60"])
    try:
        run_status.begin_run(tmp_path, pid=bystander.pid)
        unterminated = run_status.begin_run(tmp_path, pid=os.getpid())
        assert unterminated is not None
        assert unterminated["status"] == "unterminated"
        assert list(tmp_path.glob(run_status.UNTERMINATED_PREFIX + "*"))
        assert _read(tmp_path)["pid"] == os.getpid()
    finally:
        bystander.kill()
        bystander.wait()


@pytest.mark.skipif(not _HAS_PROCFS, reason="live-trainer evidence is procfs-based")
def test_begin_run_never_refuses_over_its_own_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relaunch that recycles the dead trainer's pid must not see itself.

    The marker pid here IS this process, and the marker patterns are widened
    to match this process's cmdline, so only the self-pid exclusion lets the
    start proceed.
    """
    monkeypatch.setattr(
        run_status, "ACTIVE_RUN_CMDLINE_MARKERS", ("python",))
    run_status.begin_run(tmp_path, pid=os.getpid())
    unterminated = run_status.begin_run(tmp_path, pid=os.getpid())
    assert unterminated is not None
    assert unterminated["status"] == "unterminated"


def test_check_no_active_run_passes_missing_and_terminated_markers(
    tmp_path: Path,
) -> None:
    assert run_status.check_no_active_run(tmp_path) is None
    run_status.begin_run(tmp_path, pid=1)
    run_status.record_terminal_reason(tmp_path, run_status.REASON_COMPLETED)
    record = run_status.check_no_active_run(tmp_path)
    assert record is not None
    assert record["status"] == "terminated"


def test_gui_checks_for_a_live_run_before_constructing_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GUI must refuse before checkpoint load or a VRAM claim."""
    pytest.importorskip("PyQt6")
    from dama.ui import training_panel

    config = SimpleNamespace(
        log_dir=str(tmp_path),
        recovery_enforced=False,
        resume=None,
    )
    monkeypatch.setattr(
        trainer_module, "load_config_from_yaml", lambda _path: {})
    monkeypatch.setattr(
        trainer_module, "config_from_yaml", lambda _payload: config)
    monkeypatch.setattr(
        trainer_module, "validate_recovery_experiment_config",
        lambda _config: None,
    )

    checked = []
    constructed = []

    def _refuse(log_dir):
        checked.append(log_dir)
        raise run_status.ActiveRunError("live trainer owns this namespace")

    class _MustNotConstruct:
        def __init__(self, _config):
            constructed.append(True)
            raise AssertionError("Trainer construction preceded the guard")

    monkeypatch.setattr(run_status, "check_no_active_run", _refuse)
    monkeypatch.setattr(trainer_module, "Trainer", _MustNotConstruct)

    messages = []
    status_queue = SimpleNamespace(put=messages.append)
    training_panel._trainer_process(None, status_queue, {
        "config_path": "unused.yaml",
        "resume": None,
    })

    assert checked == [str(tmp_path)]
    assert constructed == []
    assert messages == [{
        "type": training_panel.MSG_ERROR,
        "message": "live trainer owns this namespace",
    }]
