"""Run the fixed, balanced game-strength protocol for promoted checkpoints."""

from __future__ import annotations

import json
import hashlib
import hmac
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .acceptance import (
    ACCEPTANCE_GAMES_PER_OPPONENT,
    evaluate_acceptance_gates,
)
from .model_vs_algo import ModelVsAlgoTester


PENDING_TASK_SCHEMA_VERSION = 1
PENDING_TASK_PREFIX = "pending_acceptance_"
FAILURE_REPORT_PREFIX = "acceptance_failure_"


def acceptance_task_id(checkpoint_path: str, step: int) -> str:
    """Return a stable identifier for one promoted checkpoint evaluation."""
    normalized_path = os.path.normcase(
        str(Path(checkpoint_path).expanduser().resolve(strict=False)))
    identity = f"{int(step)}\n{normalized_path}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:16]
    return f"step-{int(step):06d}-{digest}"


def make_pending_acceptance_task(
    checkpoint_path: str,
    *,
    step: int,
    teacher_agreement: float,
    opening_plies: Sequence[int],
    opening_seed: int,
    inference_depth: int,
    max_moves: int,
    num_workers: int,
    training_stage: str,
    checkpoint_sha256: Optional[str] = None,
    suite_fingerprint: Optional[str] = None,
    teacher_correct_states: Optional[int] = None,
    teacher_total_states: Optional[int] = None,
) -> dict[str, Any]:
    """Build the complete durable input for one acceptance evaluation."""
    durable_checkpoint_path = str(
        Path(checkpoint_path).expanduser().resolve(strict=False))
    task = {
        "schema_version": PENDING_TASK_SCHEMA_VERSION,
        "status": "pending",
        "task_id": acceptance_task_id(durable_checkpoint_path, step),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": durable_checkpoint_path,
        "step": int(step),
        "teacher_agreement": float(teacher_agreement),
        "opening_plies": [int(value) for value in opening_plies],
        "opening_seed": int(opening_seed),
        "inference_depth": int(inference_depth),
        "max_moves": int(max_moves),
        "num_workers": max(1, int(num_workers)),
        "training_stage": str(training_stage),
    }
    if checkpoint_sha256:
        task["checkpoint_sha256"] = str(checkpoint_sha256).upper()
    if suite_fingerprint:
        task["suite_fingerprint"] = str(suite_fingerprint)
    if teacher_correct_states is not None:
        task["teacher_correct_states"] = int(teacher_correct_states)
    if teacher_total_states is not None:
        task["teacher_total_states"] = int(teacher_total_states)
    return _validate_pending_acceptance_task(task)


def pending_acceptance_task_path(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> Path:
    task_id = str(task["task_id"])
    return Path(output_dir) / f"{PENDING_TASK_PREFIX}{task_id}.json"


def failure_acceptance_report_path(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> Path:
    task_id = str(task["task_id"])
    return Path(output_dir) / f"{FAILURE_REPORT_PREFIX}{task_id}.json"


def persist_pending_acceptance_task(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> Path:
    """Atomically persist a pending task before it enters the memory queue."""
    normalized = _validate_pending_acceptance_task(task)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    path = pending_acceptance_task_path(output_root, normalized)
    if path.exists():
        existing = load_pending_acceptance_task(path)
        comparable_keys = set(normalized) - {"created_at"}
        if any(existing.get(key) != normalized.get(key) for key in comparable_keys):
            raise RuntimeError(f"Pending acceptance task conflicts with {path}")
        return path
    _write_json_atomic(path, normalized)
    return path


def load_pending_acceptance_task(path: str | Path) -> dict[str, Any]:
    pending_path = Path(path)
    with pending_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    normalized = _validate_pending_acceptance_task(payload)
    expected = pending_acceptance_task_path(pending_path.parent, normalized)
    if pending_path.name != expected.name:
        raise ValueError(
            f"Pending acceptance task filename does not match task_id: {pending_path}")
    return normalized


def discover_pending_acceptance_tasks(
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Load valid pending tasks in stable step and task-id order."""
    output_root = Path(output_dir)
    if not output_root.exists():
        return []
    tasks = []
    for path in sorted(output_root.glob(f"{PENDING_TASK_PREFIX}*.json")):
        try:
            tasks.append(load_pending_acceptance_task(path))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ignoring unreadable pending acceptance task {path}: {exc}")
    tasks.sort(key=lambda task: (int(task["step"]), str(task["task_id"])))
    return tasks


def successful_acceptance_report_path(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> Optional[Path]:
    """Return the matching durable success report, if one already exists."""
    path = Path(output_dir) / f"acceptance_step_{int(task['step']):06d}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        report_checkpoint = os.path.normcase(str(
            Path(report["checkpoint_path"]).expanduser().resolve(strict=False)))
        task_checkpoint = os.path.normcase(str(
            Path(task["checkpoint_path"]).expanduser().resolve(strict=False)))
        if (int(report.get("step", -1)) == int(task["step"])
                and report_checkpoint == task_checkpoint):
            return path
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return None


def terminal_acceptance_report_path(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> Optional[Path]:
    """Return a matching success or failure report for a durable task."""
    success = successful_acceptance_report_path(output_dir, task)
    if success is not None:
        return success
    failure = failure_acceptance_report_path(output_dir, task)
    if not failure.exists():
        return None
    try:
        with failure.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if str(report.get("task_id")) == str(task["task_id"]):
            return failure
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return None


def write_acceptance_failure_report(
    output_dir: str | Path,
    task: Mapping[str, Any],
    error: BaseException,
) -> Path:
    """Atomically record a terminal evaluation error for a pending task."""
    normalized = _validate_pending_acceptance_task(task)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "passed": False,
        "task_id": normalized["task_id"],
        "step": normalized["step"],
        "checkpoint_path": normalized["checkpoint_path"],
        "error_type": type(error).__name__,
        "error": str(error),
        "task": normalized,
    }
    path = failure_acceptance_report_path(output_root, normalized)
    _write_json_atomic(path, report)
    return path


def verify_pending_acceptance_checkpoint(
    task: Mapping[str, Any],
) -> Optional[str]:
    """Verify a recorded checkpoint digest immediately before evaluation."""
    expected = task.get("checkpoint_sha256")
    if not expected:
        return None
    checkpoint_path = Path(str(task["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pending acceptance checkpoint is missing: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest().upper()
    expected = str(expected).upper()
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError(
            f"Pending acceptance checkpoint SHA-256 mismatch for "
            f"{checkpoint_path}: expected {expected}, got {actual}")
    return actual


def remove_pending_acceptance_task(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> None:
    pending_acceptance_task_path(output_dir, task).unlink(missing_ok=True)


def _validate_pending_acceptance_task(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "task_id",
        "created_at",
        "checkpoint_path",
        "step",
        "teacher_agreement",
        "opening_plies",
        "opening_seed",
        "inference_depth",
        "max_moves",
        "num_workers",
        "training_stage",
    }
    missing = sorted(required - set(task))
    if missing:
        raise ValueError(f"Pending acceptance task is missing fields: {missing}")
    if int(task["schema_version"]) != PENDING_TASK_SCHEMA_VERSION:
        raise ValueError("Unsupported pending acceptance task schema_version")
    if task["status"] != "pending":
        raise ValueError("Pending acceptance task status must be 'pending'")
    checkpoint_path = str(task["checkpoint_path"])
    step = int(task["step"])
    expected_id = acceptance_task_id(checkpoint_path, step)
    if str(task["task_id"]) != expected_id:
        raise ValueError("Pending acceptance task_id does not match checkpoint and step")
    if step < 0:
        raise ValueError("Pending acceptance step must be non-negative")
    opening_plies = [int(value) for value in task["opening_plies"]]
    if any(value < 0 for value in opening_plies):
        raise ValueError("Pending acceptance opening plies must be non-negative")
    normalized = dict(task)
    normalized.update({
        "schema_version": PENDING_TASK_SCHEMA_VERSION,
        "status": "pending",
        "task_id": expected_id,
        "created_at": str(task["created_at"]),
        "checkpoint_path": checkpoint_path,
        "step": step,
        "teacher_agreement": float(task["teacher_agreement"]),
        "opening_plies": opening_plies,
        "opening_seed": int(task["opening_seed"]),
        "inference_depth": int(task["inference_depth"]),
        "max_moves": int(task["max_moves"]),
        "num_workers": max(1, int(task["num_workers"])),
        "training_stage": str(task["training_stage"]),
    })
    if normalized.get("checkpoint_sha256"):
        checkpoint_sha256 = str(normalized["checkpoint_sha256"]).upper()
        if (len(checkpoint_sha256) != 64
                or any(char not in "0123456789ABCDEF" for char in checkpoint_sha256)):
            raise ValueError(
                "Pending acceptance checkpoint_sha256 must be 64 hexadecimal characters")
        normalized["checkpoint_sha256"] = checkpoint_sha256
    if normalized.get("suite_fingerprint") is not None:
        normalized["suite_fingerprint"] = str(normalized["suite_fingerprint"])
    for key in ("teacher_correct_states", "teacher_total_states"):
        if normalized.get(key) is not None:
            normalized[key] = int(normalized[key])
            if normalized[key] < 0:
                raise ValueError(f"Pending acceptance {key} must be non-negative")
    if (
        normalized.get("teacher_correct_states") is not None
        and normalized.get("teacher_total_states") is not None
        and normalized["teacher_correct_states"] > normalized["teacher_total_states"]
    ):
        raise ValueError(
            "Pending acceptance teacher_correct_states exceeds total states")
    return normalized


def run_checkpoint_acceptance(
    checkpoint_path: str,
    *,
    step: int,
    teacher_agreement: float,
    opening_plies: Sequence[int],
    opening_seed: int,
    inference_depth: int,
    max_moves: int,
    num_workers: int,
    output_dir: str,
    training_stage: str,
    task_id: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    suite_fingerprint: Optional[str] = None,
    teacher_correct_states: Optional[int] = None,
    teacher_total_states: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate random first, then easy, and atomically persist one report."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    common = {
        "model_path": checkpoint_path,
        "num_workers": max(1, int(num_workers)),
        "max_moves": int(max_moves),
        "opening_plies": tuple(int(value) for value in opening_plies),
        "opening_seed": int(opening_seed),
        "ml_inference_depth": int(inference_depth),
    }

    random_stats = ModelVsAlgoTester(
        algo_difficulty="easy",
        opponent_type="random",
        stats_dir=str(output_root / "random_details"),
        **common,
    ).run_tests(num_games=ACCEPTANCE_GAMES_PER_OPPONENT)
    random_record = random_stats.to_dict()
    easy_stats = ModelVsAlgoTester(
        algo_difficulty="easy",
        opponent_type="algorithm",
        stats_dir=str(output_root / "easy_details"),
        **common,
    ).run_tests(num_games=ACCEPTANCE_GAMES_PER_OPPONENT)

    easy_record = easy_stats.to_dict()
    if random_record.get("opening_suite_id") != easy_record.get("opening_suite_id"):
        raise RuntimeError("Random and easy evaluations used different opening suites")

    decision = evaluate_acceptance_gates(
        teacher_agreement, random_record, easy_record)
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(Path(checkpoint_path)),
        "step": int(step),
        "training_stage": str(training_stage),
        "task_id": task_id,
        "checkpoint_sha256": (
            str(checkpoint_sha256).upper() if checkpoint_sha256 else None),
        "frozen_suite_fingerprint": suite_fingerprint,
        "teacher_agreement_counts": {
            "correct_states": teacher_correct_states,
            "total_states": teacher_total_states,
        },
        "selection_sequence": [
            "held_out_teacher_agreement",
            "random_game_strength",
            "easy_game_strength",
        ],
        "opening_seed": int(opening_seed),
        "opening_plies": [int(value) for value in opening_plies],
        "opening_suite_id": random_record.get("opening_suite_id"),
        "inference_depth": int(inference_depth),
        "max_moves": int(max_moves),
        "num_workers": max(1, int(num_workers)),
        "random": random_record,
        "easy": easy_record,
        **decision.to_dict(),
    }
    report_path = output_root / f"acceptance_step_{int(step):06d}.json"
    _write_json_atomic(report_path, report)
    report["report_path"] = str(report_path)
    return report


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
