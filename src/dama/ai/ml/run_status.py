"""Durable terminal-reason marker for one trainer run.

Audit Suggestion 5: two WSL logs and one Windows log ended with no terminal
marker at all, so "why did it stop?" was unanswerable after the fact.  Console
output cannot answer it either -- a hard kill (OOM, SIGHUP on terminal close)
discards whatever the process had buffered.

The contract here is deliberately small and has one property that matters more
than the others: **absence is itself a verdict.**  A marker left in
``running`` state is written before training starts and is only ever replaced
by a terminal record, so a run that dies in a way no in-process handler can
observe -- SIGKILL from the OOM killer, a power loss, a hypervisor reset --
leaves behind a record saying exactly that.  The next start finds it, preserves
it under its own filename, and reports it.

No torch, no config object: this must keep working when everything heavier has
already failed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional


RUN_STATUS_SCHEMA_VERSION = 1
RUN_STATUS_FILENAME = "run_status.json"
UNTERMINATED_PREFIX = "run_status_unterminated_"

# Terminal reasons. Every exit path from Trainer.train() must map onto one.
REASON_COMPLETED = "completed"
REASON_TIME_LIMIT = "time_limit_reached"
REASON_STOP_REQUESTED = "stop_requested"
REASON_INTERRUPTED = "interrupted"
REASON_EXCEPTION = "exception"
TERMINAL_REASONS = frozenset({
    REASON_COMPLETED,
    REASON_TIME_LIMIT,
    REASON_STOP_REQUESTED,
    REASON_INTERRUPTED,
    REASON_EXCEPTION,
})


def run_status_path(log_dir: str | Path) -> Path:
    return Path(log_dir) / RUN_STATUS_FILENAME


def read_run_status(log_dir: str | Path) -> Optional[dict]:
    path = run_status_path(log_dir)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _safe_stamp(value: Any) -> str:
    """Filename-safe form of an ISO timestamp (Windows forbids ':')."""
    return "".join(
        char if char.isalnum() else "-" for char in str(value)
    )[:64] or "unknown"


# Command-line substrings that identify this project's trainer process: the
# dynamic process title the launchers set ("micro-trainer | step=N loss=L" /
# "micro_trainer ..."), and the module invocation used when setproctitle is
# absent.  The title truncates comm to 15 bytes ("micro-trainer |"), so
# name-exact matching (pgrep -x) cannot detect a live trainer; cmdline can.
ACTIVE_RUN_CMDLINE_MARKERS = ("micro-trainer", "micro_trainer", "dama.ai.ml.trainer")


class ActiveRunError(RuntimeError):
    """A running marker's pid is a live trainer process: refuse a second writer.

    Raised instead of preserving the marker as "unterminated", because that
    verdict would be false (the run did not die) and proceeding would put two
    trainers into one namespace: same checkpoint files, same replay shards,
    same corpus snapshots, silent corruption.
    """

    def __init__(self, message: str, *, pid: Optional[int] = None,
                 cmdline: Optional[str] = None,
                 started_at: Optional[str] = None) -> None:
        super().__init__(message)
        self.pid = pid
        self.cmdline = cmdline
        self.started_at = started_at


def _live_trainer_cmdline(pid: Any) -> Optional[str]:
    """Cmdline of ``pid`` if it is a live trainer process other than us.

    Evidence is cmdline-only and procfs-based: on hosts without ``/proc``
    (native Windows) this returns None and the caller keeps today's behavior.
    The self-pid exclusion covers the recycled-pid case where the new trainer
    inherits the dead one's pid and would otherwise refuse over its own
    reflection.  A pid recycled into a trainer from a *different* namespace can
    still false-positive; the remedy is deleting the stale marker, and the
    ActiveRunError message says so.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return None
    if pid_int <= 0 or pid_int == os.getpid():
        return None
    proc_dir = Path("/proc") / str(pid_int)
    if not proc_dir.is_dir():
        return None
    try:
        raw = (proc_dir / "cmdline").read_bytes()
    except OSError:
        return None
    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    if any(marker in cmdline for marker in ACTIVE_RUN_CMDLINE_MARKERS):
        return cmdline
    return None


def check_no_active_run(log_dir: str | Path) -> Optional[dict]:
    """Raise ActiveRunError if this namespace's marker belongs to a live trainer.

    Returns the parsed marker (or None) otherwise, so ``begin_run`` can reuse
    the read.  Call this before constructing a Trainer: construction already
    loads the resume checkpoint, claims VRAM, and can run foreground corpus
    repair, all of which are wrong beside a live run even before any marker
    write.
    """
    previous = read_run_status(log_dir)
    if previous is None or previous.get("status") != "running":
        return previous
    cmdline = _live_trainer_cmdline(previous.get("pid"))
    if cmdline is None:
        return previous
    raise ActiveRunError(
        f"run_status.json in {log_dir} records a run (pid {previous.get('pid')}, "
        f"started {previous.get('started_at')}) that is still alive: "
        f"cmdline is '{cmdline[:120]}'. Refusing to start a second writer into "
        "this namespace. Stop it with 'bash stop_training.sh', or delete the "
        "marker file only if you are certain it is stale (e.g. pid reuse).",
        pid=previous.get("pid"),
        cmdline=cmdline,
        started_at=previous.get("started_at"),
    )


def begin_run(
    log_dir: str | Path,
    *,
    pid: Optional[int] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Optional[dict]:
    """Open a run marker; return the prior run's record if it never terminated.

    The prior record is preserved beside the new marker under its own name
    rather than being overwritten, so a sequence of silent deaths accumulates
    evidence instead of erasing it.
    """
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # The marker's own directory fsync below commits run_status.json, but it
    # cannot commit this directory's name in its parent.  On the first launch
    # of a namespace, make the log-directory entry durable before relying on
    # it to preserve evidence of a later hard kill or host failure.  Repeat
    # the parent sync on every session so a prior failed sync is retried even
    # though mkdir() now observes an existing directory.
    _fsync_directory(directory.parent)
    # Raises ActiveRunError before anything is written when the marker's pid
    # is a live trainer: the "unterminated" verdict below would be false and
    # overwriting the marker would hand two writers one namespace.
    previous = check_no_active_run(directory)
    unterminated = None
    if previous is not None and previous.get("status") == "running":
        unterminated = dict(previous)
        unterminated["detected_at"] = datetime.now(timezone.utc).isoformat()
        unterminated["status"] = "unterminated"
        unterminated["reason"] = None
        unterminated["note"] = (
            "The process disappeared without recording a terminal reason. No "
            "in-process handler can run for SIGKILL (OOM killer), a power "
            "loss, or a hypervisor reset, so this is the record of that class "
            "of exit."
        )
        preserved = directory / (
            f"{UNTERMINATED_PREFIX}"
            f"{_safe_stamp(previous.get('started_at'))}.json"
        )
        _write_json_atomic(preserved, unterminated)
        unterminated["preserved_path"] = str(preserved)

    _write_json_atomic(directory / RUN_STATUS_FILENAME, {
        "schema_version": RUN_STATUS_SCHEMA_VERSION,
        "status": "running",
        "reason": None,
        "pid": int(pid if pid is not None else os.getpid()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "context": dict(context or {}),
    })
    return unterminated


def record_terminal_reason(
    log_dir: str | Path,
    reason: str,
    *,
    detail: Optional[str] = None,
    traceback_text: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Close the marker with one of the declared terminal reasons."""
    if reason not in TERMINAL_REASONS:
        raise ValueError(f"Unknown terminal reason: {reason!r}")
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    existing = read_run_status(directory) or {}
    payload = {
        "schema_version": RUN_STATUS_SCHEMA_VERSION,
        "status": "terminated",
        "reason": reason,
        "pid": existing.get("pid", os.getpid()),
        "started_at": existing.get("started_at"),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "context": {**dict(existing.get("context") or {}), **dict(context or {})},
    }
    if detail:
        payload["detail"] = str(detail)
    if traceback_text:
        payload["traceback"] = str(traceback_text)
    path = directory / RUN_STATUS_FILENAME
    _write_json_atomic(path, payload)
    return path


def describe_unterminated(record: Mapping[str, Any]) -> str:
    """One-line operator-facing summary of a run that vanished."""
    started = record.get("started_at") or "an unknown time"
    pid = record.get("pid")
    context = record.get("context") or {}
    step = context.get("step")
    where = f" at step {step}" if step is not None else ""
    return (
        f"Previous training run (pid {pid}, started {started}) ended{where} "
        "without recording a terminal reason -- it was killed rather than "
        "exiting. On this box the usual cause is the OOM killer."
    )


def _fsync_directory(path: Path) -> None:
    """Persist a completed rename when the platform exposes directory fds."""

    if os.name == "nt":
        # Native Windows cannot open a directory through os.open(). Atomic
        # replacement remains the supported fallback there.
        return
    directory_fd = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` with ``payload`` as pretty-printed JSON.

    Shared implementation for every durable JSON artifact this project writes
    (run markers, acceptance reports, corpus manifests): temp file in the
    destination directory, file fsync, ``os.replace``, then directory fsync.
    Before replacement, a failure removes the temp and leaves the destination
    untouched. After replacement, a directory-fsync failure is reported even
    though the new complete file may already be visible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
