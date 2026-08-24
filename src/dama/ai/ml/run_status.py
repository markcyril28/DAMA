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
    previous = read_run_status(directory)
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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
