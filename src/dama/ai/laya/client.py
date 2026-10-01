"""Dama side of the Laya bridge: one laya worker process spoken to over JSON lines.

Stdlib only. The worker (``_bridge_worker.py``) runs under the interpreter of the
environment that has laya installed, so nothing from laya or torch is imported here.

Deadlock rule: QThread.terminate() can end a caller anywhere, including during the
~15 s first load. The bridge lock is held only for short bookkeeping (spawn,
register a pending request, write one line) and is always acquired with a
timeout; waiting for the ready event or a reply happens with no lock held, so a
terminated waiter never blocks later calls. A caller terminated inside the
bookkeeping itself (Popen, a pipe write) can still orphan the lock or corrupt
the interpreter, so the GUI never terminates a thread that uses the bridge.
"""

import atexit
import collections
import itertools
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

from .errors import LayaBudgetError, LayaUnavailableError
from .spec import SUPPORTED_LAYA_VERSIONS, BridgeChoice, BridgeSpec

_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bridge_worker.py")
_LOCK_TIMEOUT_SEC = 10.0  # bookkeeping takes milliseconds; longer means a holder was killed
_STDERR_KEEP_LINES = 200
_STDERR_QUOTE_LINES = 25  # lines of the kept tail quoted in error messages
_STDERR_QUOTE_CHARS = 4000
_EXIT_GRACE_SEC = 2.0  # wait for the exit code and the last stderr lines after EOF
_DEVICES = ("auto", "cuda", "cpu")
_SETTINGS_HINT = "Settings > Laya AI > Python interpreter"


class _Slot:
    """One in-flight request waiting for its reply."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reply: Optional[dict] = None
        self.error: Optional[str] = None


class _Worker:
    """One worker process plus what its reader threads have seen."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.spawned_at = time.monotonic()
        self.last_activity = self.spawned_at
        self.ready = threading.Event()  # set on ready, fatal or exit
        self.ready_info: Optional[dict] = None
        self.fatal: Optional[str] = None
        self.dead = False
        self.exit_reason = ""
        self.pending: Dict[int, _Slot] = {}
        self.stderr_tail: Deque[str] = collections.deque(maxlen=_STDERR_KEEP_LINES)
        self.stderr_done = threading.Event()
        self.stopped = threading.Event()  # process gone or detached; ends the idle watcher

    def usable(self) -> bool:
        """True while the process runs and its stdout is open."""
        return not self.dead and self.proc.poll() is None

    def tail(self) -> str:
        """Last stderr lines, formatted for an error message."""
        lines = list(self.stderr_tail)[-_STDERR_QUOTE_LINES:]
        if not lines:
            return ""
        text = "\n".join(lines)
        if len(text) > _STDERR_QUOTE_CHARS:
            text = "..." + text[-_STDERR_QUOTE_CHARS:]
        return "\n--- Laya worker stderr (last %d lines) ---\n%s" % (len(lines), text)


def _no_limit(seconds: float) -> Optional[float]:
    """Wait limit for Event.wait: non-positive settings mean no limit."""
    return seconds if seconds and seconds > 0 else None


class LayaBridge:
    """Client for one laya worker process, respawned on demand."""

    def __init__(self, spec: BridgeSpec, *, worker_path: Optional[str] = None) -> None:
        self._spec = spec
        self._worker_path = worker_path or _WORKER_PATH
        self._lock = threading.Lock()
        self._worker: Optional[_Worker] = None
        self._retired = False  # set by retire(); read under the lock before every spawn
        self._ids = itertools.count(1)

    # --- public API -------------------------------------------------------

    @property
    def spec(self) -> BridgeSpec:
        """Current spec (timeouts may be updated in place by get_bridge)."""
        return self._spec

    @property
    def alive(self) -> bool:
        """True while a worker process is running."""
        w = self._worker
        return w is not None and w.usable()

    @property
    def ready_info(self) -> Optional[dict]:
        """Ready event of the running worker, or None."""
        w = self._worker
        if w is None or not w.usable() or w.ready_info is None:
            return None
        return dict(w.ready_info)

    @property
    def pid(self) -> Optional[int]:
        """Process id of the running worker, or None."""
        w = self._worker
        return w.proc.pid if w is not None and w.usable() else None

    def start(self) -> dict:
        """Start the worker if needed and wait until it is ready; returns the ready event."""
        for _ in range(2):
            w = self._ensure_worker()
            info = self._await_ready(w)
            if info is not None:
                return info
        raise LayaUnavailableError("Laya worker exited right after it became ready%s" % w.tail())

    def choose(self, state: Union[str, dict], instructions: str, options: Sequence[str]) -> BridgeChoice:
        """Ask the worker which option is best; probabilities align to options."""
        opts = list(options)
        reply = self._request({"op": "choose", "state": state, "instructions": instructions,
                               "options": opts})
        return self._choice_from_reply(reply, len(opts))

    def ping(self) -> None:
        """Round trip a no-op request; raises LayaUnavailableError on failure."""
        self._request({"op": "ping"})

    def close(self, timeout: float = 5.0) -> None:
        """Stop the worker (idempotent); the next call respawns it."""
        self._stop(timeout, retire=False)

    def retire(self, timeout: float = 5.0) -> None:
        """Stop the worker for good (idempotent); later calls raise instead of respawning."""
        self._stop(timeout, retire=True)

    def _stop(self, timeout: float, retire: bool) -> None:
        """Detach and stop the current worker, optionally refusing later spawns."""
        acquired = self._lock.acquire(timeout=_LOCK_TIMEOUT_SEC)
        try:
            # Under the lock: a spawn already registered is stopped below, and
            # every later _ensure_worker sees the flag.
            if retire:
                self._retired = True
            w, self._worker = self._worker, None
        finally:
            if acquired:
                self._lock.release()
        if w is not None:
            reason = "the Laya bridge was shut down" if retire else "the Laya bridge was closed"
            self._shutdown_worker(w, timeout, reason)

    # --- process management -----------------------------------------------

    def _acquire(self) -> threading.Lock:
        """Take the bookkeeping lock or raise; never blocks indefinitely."""
        lock = self._lock
        if not lock.acquire(timeout=_LOCK_TIMEOUT_SEC):
            raise LayaUnavailableError(
                "Laya bridge is busy: its lock was not released within %.0f s" % _LOCK_TIMEOUT_SEC)
        return lock

    def _ensure_worker(self) -> _Worker:
        """Return the current worker, spawning one if there is none or it died."""
        stale = None
        lock = self._acquire()
        try:
            if self._retired:
                # A holder of a replaced or shut-down bridge would otherwise start a
                # worker that no registry, shutdown or atexit hook can reach.
                raise LayaUnavailableError(
                    "The Laya bridge was shut down (its settings changed or no side plays Laya)")
            w = self._worker
            if w is None or not w.usable():
                stale = w
                w = self._spawn()
                self._worker = w
        finally:
            lock.release()
        if stale is not None:
            self._kill(stale, "replaced by a new worker")
        return w

    def _command(self, spec: BridgeSpec) -> List[str]:
        """argv of the worker process."""
        return [spec.python, "-I", "-u", self._worker_path,
                "--model", spec.model, "--subfolder", spec.subfolder, "--device", spec.device,
                "--expected-laya-version", ",".join(SUPPORTED_LAYA_VERSIONS)]

    @staticmethod
    def _environment(spec: BridgeSpec) -> Dict[str, str]:
        """Environment of the worker process."""
        env = dict(os.environ)
        env.update(USE_TF="0", TOKENIZERS_PARALLELISM="false", TRANSFORMERS_VERBOSITY="error",
                   PYTHONNOUSERSITE="1")
        if spec.hf_home:
            env["HF_HOME"] = os.path.expanduser(spec.hf_home)
        if spec.offline:
            env["HF_HUB_OFFLINE"] = "1"
        else:
            env.pop("HF_HUB_OFFLINE", None)
        return env

    def _spawn(self) -> _Worker:
        """Start a worker process and its reader threads (called with the lock held)."""
        spec = self._spec
        if spec.device not in _DEVICES:
            raise LayaUnavailableError("Unsupported Laya device %r (use auto, cuda or cpu)" % spec.device)
        if not spec.python:
            raise LayaUnavailableError("No Python interpreter configured for Laya (%s)" % _SETTINGS_HINT)
        try:
            # errors="replace": a strict decode error would kill the stderr drain and block the child.
            proc = subprocess.Popen(
                self._command(spec), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=self._environment(spec), start_new_session=(os.name == "posix"), close_fds=True)
        except OSError as exc:
            raise LayaUnavailableError("Cannot start the Laya worker with %s: %s (%s)"
                                       % (spec.python, exc, _SETTINGS_HINT)) from exc
        w = _Worker(proc)
        for target, name in ((self._read_stdout, "stdout"), (self._drain_stderr, "stderr"),
                             (self._watch_idle, "idle")):
            threading.Thread(target=target, args=(w,), name="laya-bridge-%s-%d" % (name, proc.pid),
                             daemon=True).start()
        return w

    def _await_ready(self, w: _Worker) -> Optional[dict]:
        """Wait (no lock held) for w's ready event; None if it died after becoming ready."""
        spec = self._spec
        limit = _no_limit(spec.startup_timeout_sec)
        remaining = None if limit is None else max(0.0, w.spawned_at + limit - time.monotonic())
        if not w.ready.wait(remaining):
            self._kill(w, "startup timeout")
            raise LayaUnavailableError(
                "Laya worker was not ready within %.0f s (startup_timeout_sec); it was stopped%s"
                % (spec.startup_timeout_sec, w.tail()))
        if w.fatal is not None:
            w.stderr_done.wait(_EXIT_GRACE_SEC)
            self._kill(w, "fatal startup error")
            raise LayaUnavailableError("Laya worker failed to start: %s%s" % (w.fatal, w.tail()))
        if w.ready_info is None:
            w.stderr_done.wait(_EXIT_GRACE_SEC)
            raise LayaUnavailableError("Laya worker stopped before it was ready: %s%s"
                                       % (w.exit_reason or "exit code %s" % w.proc.poll(), w.tail()))
        if not w.usable():
            return None
        device = w.ready_info.get("device")
        if spec.device == "cuda" and device != "cuda":
            self._kill(w, "device mismatch")
            raise LayaUnavailableError("Laya was asked for cuda but loaded on %s; it was stopped" % device)
        return dict(w.ready_info)

    def _kill(self, w: _Worker, reason: str) -> None:
        """Detach and kill w now, then wake everything waiting on it."""
        self._detach(w)
        if not w.exit_reason:
            w.exit_reason = reason
        w.stopped.set()
        self._terminate_process(w.proc)
        self._mark_dead(w, "Laya worker was stopped (%s)" % reason)

    def _shutdown_worker(self, w: _Worker, timeout: float, reason: str) -> None:
        """Ask w to exit, kill it after timeout, then wake everything waiting on it."""
        if not w.exit_reason:
            w.exit_reason = reason
        w.stopped.set()
        if w.ready.is_set() and w.proc.poll() is None:
            try:
                w.proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                w.proc.stdin.flush()
                w.proc.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                w.proc.wait(timeout=max(0.0, timeout))
            except subprocess.TimeoutExpired:
                pass
        self._terminate_process(w.proc)  # a worker still loading never reads stdin
        self._mark_dead(w, "Laya worker was stopped (%s)" % reason)

    @staticmethod
    def _terminate_process(proc: subprocess.Popen) -> None:
        """Kill proc and its process group if it still runs, then reap it."""
        if proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)  # own session: pgid == pid
                else:
                    proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=_EXIT_GRACE_SEC)
        except subprocess.TimeoutExpired:
            pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()  # the reader threads close stdout and stderr themselves
        except (OSError, ValueError):
            pass

    def _detach(self, w: _Worker) -> None:
        """Forget w as the current worker if it still is."""
        acquired = self._lock.acquire(timeout=_LOCK_TIMEOUT_SEC)
        try:
            if self._worker is w:
                self._worker = None
        finally:
            if acquired:
                self._lock.release()

    @staticmethod
    def _mark_dead(w: _Worker, message: str) -> None:
        """Mark w dead and wake the ready waiters and every pending request."""
        w.dead = True  # before waking: a request registered after this sees dead itself
        w.stopped.set()
        w.ready.set()
        for rid in list(w.pending):
            slot = w.pending.pop(rid, None)
            if slot is not None:
                slot.error = message
                slot.event.set()

    # --- reader threads ---------------------------------------------------

    def _read_stdout(self, w: _Worker) -> None:
        """Dispatch protocol lines until EOF, then wake every waiter."""
        try:
            for line in w.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    w.stderr_tail.append("[stdout] " + line[:500])
                    continue
                if not isinstance(msg, dict):
                    continue
                event = msg.get("event")
                if event == "ready":
                    w.ready_info = msg
                    w.last_activity = time.monotonic()
                    w.ready.set()
                elif event == "fatal":
                    w.fatal = str(msg.get("error") or "unknown error")
                    w.ready.set()
                else:
                    rid = msg.get("id")
                    slot = w.pending.pop(rid, None) if type(rid) is int else None
                    if slot is None:
                        continue  # stale or unknown id: its waiter gave up or never existed
                    w.last_activity = time.monotonic()
                    slot.reply = msg
                    slot.event.set()
        except (OSError, ValueError):
            pass
        finally:
            try:
                w.proc.stdout.close()
            except (OSError, ValueError):
                pass
            self._on_stdout_eof(w)

    def _on_stdout_eof(self, w: _Worker) -> None:
        """The worker closed its protocol stream: reap it and fail every waiter."""
        w.dead = True
        try:
            w.proc.wait(timeout=_EXIT_GRACE_SEC)
        except subprocess.TimeoutExpired:
            pass
        w.stderr_done.wait(_EXIT_GRACE_SEC)
        self._detach(w)
        self._terminate_process(w.proc)  # detached first, so no writer still uses its stdin
        reason = w.exit_reason or "exit code %s" % w.proc.poll()
        self._mark_dead(w, "Laya worker exited (%s)%s" % (reason, w.tail()))

    @staticmethod
    def _drain_stderr(w: _Worker) -> None:
        """Keep the last stderr lines; an undrained pipe would block the worker."""
        try:
            for line in w.proc.stderr:
                w.stderr_tail.append(line.rstrip("\n"))
        except (OSError, ValueError):
            pass
        finally:
            try:
                w.proc.stderr.close()
            except (OSError, ValueError):
                pass
            w.stderr_done.set()

    def _watch_idle(self, w: _Worker) -> None:
        """Close w after idle_shutdown_sec with no request in flight."""
        while True:
            idle = self._spec.idle_shutdown_sec
            poll = max(0.02, min(30.0, idle / 4.0)) if idle > 0 else 1.0
            if w.stopped.wait(poll):
                return
            idle = self._spec.idle_shutdown_sec
            if (idle <= 0 or not w.ready.is_set() or w.ready_info is None or w.pending
                    or time.monotonic() - w.last_activity < idle):
                continue
            if not self._lock.acquire(timeout=1.0):
                continue
            try:
                current = (self._worker is w and not w.pending
                           and time.monotonic() - w.last_activity >= idle)
                if current:
                    self._worker = None
            finally:
                self._lock.release()
            if current:
                self._shutdown_worker(w, 5.0, "idle for %.0f s" % idle)
                return

    # --- requests -----------------------------------------------------------

    def _request(self, payload: dict) -> dict:
        """Send one request and wait (no lock held) for its successful reply."""
        rid = next(self._ids)
        line = json.dumps(dict(payload, id=rid)) + "\n"
        for _ in range(2):
            w = self._ensure_worker()
            if self._await_ready(w) is None:
                continue
            slot = _Slot()
            lock = self._acquire()
            try:
                if self._worker is not w or not w.usable():
                    slot = None  # closed by the idle watcher or died since it was ready
                else:
                    w.pending[rid] = slot
                    w.last_activity = time.monotonic()
                    try:
                        w.proc.stdin.write(line)
                        w.proc.stdin.flush()
                    except (OSError, ValueError) as exc:
                        w.pending.pop(rid, None)
                        slot.error = "cannot write to the Laya worker: %r" % exc
                        slot.event.set()
            finally:
                lock.release()
            if slot is not None:
                return self._await_reply(w, rid, slot)
        raise LayaUnavailableError("Laya worker stopped before the request could be sent")

    def _await_reply(self, w: _Worker, rid: int, slot: _Slot) -> dict:
        """Wait for one reply and map worker errors to exceptions."""
        spec = self._spec
        if not slot.event.wait(_no_limit(spec.request_timeout_sec)):
            w.pending.pop(rid, None)
            self._kill(w, "request timeout")
            raise LayaUnavailableError(
                "Laya worker did not answer within %.0f s (request_timeout_sec); it was stopped%s"
                % (spec.request_timeout_sec, w.tail()))
        if slot.error is not None:
            if not w.usable():
                self._kill(w, "request failed")
            raise LayaUnavailableError(slot.error)
        reply = slot.reply or {}
        if reply.get("ok") is True:
            device = reply.get("device")
            if spec.device == "cuda" and device != "cuda":
                self._kill(w, "device mismatch")
                raise LayaUnavailableError("Laya was asked for cuda but answered on %s; it was stopped"
                                           % device)
            return reply
        code = reply.get("code")
        message = str(reply.get("error") or "unknown error")
        if code == "budget":
            raise LayaBudgetError(message)
        if code == "device":
            self._kill(w, "device fallback")
            raise LayaUnavailableError("Laya left the requested device: %s; it was stopped" % message)
        raise LayaUnavailableError("Laya worker error: %s" % message)

    @staticmethod
    def _choice_from_reply(reply: dict, n_options: int) -> BridgeChoice:
        """Validate a choose reply against the options that were sent."""
        try:
            probabilities = tuple(float(p) for p in reply["probabilities"])
            index = reply["choice_index"]
            choice = BridgeChoice(
                probabilities=probabilities, choice_index=int(index),
                confidence=float(reply.get("confidence", 0.0)), device=str(reply.get("device", "")),
                elapsed_ms=float(reply.get("elapsed_ms", 0.0)),
                head_max_len=int(reply.get("head_max_len", 0)))
        except (KeyError, TypeError, ValueError) as exc:
            raise LayaUnavailableError("Malformed Laya reply: %r" % exc) from exc
        if (len(probabilities) != n_options or type(index) is not int
                or not 0 <= index < n_options or not all(math.isfinite(p) for p in probabilities)):
            raise LayaUnavailableError("Laya reply does not match the %d options sent: %r"
                                       % (n_options, reply))
        return choice


# --- interpreter discovery --------------------------------------------------

def _is_executable(path: str) -> bool:
    """True for an existing executable regular file."""
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _conda_bases() -> List[str]:
    """Conda installation roots worth searching, most specific first."""
    bases = []
    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe:
        bases.append(os.path.dirname(os.path.dirname(conda_exe)))
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        parent = os.path.dirname(os.path.normpath(prefix))
        if os.path.basename(parent) == "envs":
            bases.append(os.path.dirname(parent))
        bases.append(prefix)
    for name in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        bases.append(os.path.join(os.path.expanduser("~"), name))
    return bases


def resolve_python(python: str = "", conda_env: str = "laya-gpu") -> str:
    """Interpreter for the worker: an explicit path, else conda_env's python."""
    if python:
        path = os.path.abspath(os.path.expanduser(python))
        if not _is_executable(path):
            raise LayaUnavailableError(
                "Laya Python interpreter %r is not an executable file (%s)" % (python, _SETTINGS_HINT))
        return path
    if not conda_env:
        raise LayaUnavailableError("No Laya Python interpreter or conda env configured (%s)"
                                   % _SETTINGS_HINT)
    tail = ("envs", conda_env, "python.exe") if os.name == "nt" else ("envs", conda_env, "bin", "python")
    candidates = []
    for base in _conda_bases():
        candidate = os.path.join(base, *tail)
        if base and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if _is_executable(candidate):
            return candidate
    raise LayaUnavailableError(
        "No Python interpreter found for conda env %r; looked for:\n  %s\n"
        "Create the env with laya installed, or set the interpreter path in %s."
        % (conda_env, "\n  ".join(candidates), _SETTINGS_HINT))


# --- process-wide bridge ----------------------------------------------------

_bridge_lock = threading.Lock()
_bridge: Optional[LayaBridge] = None


def _acquire_module_lock() -> None:
    """Take the module lock or raise; it is held only for attribute swaps."""
    if not _bridge_lock.acquire(timeout=_LOCK_TIMEOUT_SEC):
        raise LayaUnavailableError("Laya bridge registry is busy")


def get_bridge(spec: BridgeSpec) -> LayaBridge:
    """Process-wide bridge for spec; a different process_key replaces the old one."""
    global _bridge
    _acquire_module_lock()
    try:
        old = _bridge
        if old is not None and old.spec.process_key() == spec.process_key():
            old._spec = spec  # timeouts and other non-process fields apply from the next call
            return old
        _bridge = new = LayaBridge(spec)
    finally:
        _bridge_lock.release()
    if old is not None:
        old.retire()
    return new


def current_bridge() -> Optional[LayaBridge]:
    """The process-wide bridge, if one was created."""
    return _bridge


def shutdown_bridge() -> None:
    """Retire the process-wide bridge (idempotent); get_bridge then builds a new one."""
    global _bridge
    _acquire_module_lock()
    try:
        old, _bridge = _bridge, None
    finally:
        _bridge_lock.release()
    if old is not None:
        old.retire()


atexit.register(shutdown_bridge)
