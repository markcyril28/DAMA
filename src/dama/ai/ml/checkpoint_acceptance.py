"""Run the fixed, balanced game-strength protocol for promoted checkpoints."""

from __future__ import annotations

import json
import hashlib
import hmac
import math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .acceptance import (
    ACCEPTANCE_GAMES_PER_OPPONENT,
    evaluate_acceptance_gates,
)
from .model_vs_algo import (
    ModelVsAlgoTester,
    _evaluation_worker_context,
    _evaluation_worker_init,
    opening_suite_identity,
)


PENDING_TASK_SCHEMA_VERSION = 1
PENDING_TASK_PREFIX = "pending_acceptance_"
FAILURE_REPORT_PREFIX = "acceptance_failure_"
FROZEN_TEACHER_SUITE_SIZE = 5000
_ACCEPTANCE_SELECTION_SEQUENCE = [
    "held_out_teacher_agreement",
    "random_game_strength",
    "easy_game_strength",
]


def _normalize_acceptance_step(step: int) -> int:
    """Return an exact non-negative checkpoint step."""
    if type(step) is not int or step < 0:
        raise ValueError(
            "Acceptance step must be a non-negative integer")
    return step


def acceptance_task_id(checkpoint_path: str, step: int) -> str:
    """Return a stable identifier for one promoted checkpoint evaluation."""
    step = _normalize_acceptance_step(step)
    normalized_path = os.path.normcase(
        str(Path(checkpoint_path).expanduser().resolve(strict=False)))
    identity = f"{step}\n{normalized_path}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:16]
    return f"step-{step:06d}-{digest}"


def _normalize_acceptance_opening_plies(
    opening_plies: Sequence[int],
) -> tuple[int, ...]:
    """Return the exact positive-integer opening schedule for acceptance."""
    try:
        normalized = tuple(opening_plies)
    except TypeError as exc:
        raise ValueError(
            "Acceptance opening plies must be non-empty and positive integers"
        ) from exc
    if (
        not normalized
        or any(type(value) is not int or value <= 0 for value in normalized)
    ):
        raise ValueError(
            "Acceptance opening plies must be non-empty and positive integers")
    return normalized


def _normalize_acceptance_opening_seed(opening_seed: int) -> int:
    """Return an exact integer seed for the paired opening schedule."""
    if type(opening_seed) is not int:
        raise ValueError("Acceptance opening_seed must be an integer")
    return opening_seed


def _normalize_acceptance_game_bounds(
    inference_depth: int,
    max_moves: int,
) -> tuple[int, int]:
    """Return the supported inference depth and a positive move limit."""
    if type(inference_depth) is not int or inference_depth not in (1, 2, 3):
        raise ValueError(
            "Acceptance inference depth must be one of 1, 2, or 3")
    if type(max_moves) is not int or max_moves <= 0:
        raise ValueError("Acceptance max_moves must be a positive integer")
    return inference_depth, max_moves


def _normalize_acceptance_num_workers(num_workers: int) -> int:
    """Return an exact positive worker count for acceptance evaluation."""
    if type(num_workers) is not int or num_workers <= 0:
        raise ValueError(
            "Acceptance num_workers must be a positive integer")
    return num_workers


def _normalize_sha256(value: Any, *, field: str) -> str:
    """Return one exact hexadecimal SHA-256 provenance value."""
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise ValueError(
            f"Pending acceptance {field} must be 64 hexadecimal characters")
    return value


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
    step = _normalize_acceptance_step(step)
    inference_depth, max_moves = _normalize_acceptance_game_bounds(
        inference_depth, max_moves)
    num_workers = _normalize_acceptance_num_workers(num_workers)
    checkpoint_sha256 = _normalize_sha256(
        checkpoint_sha256, field="checkpoint_sha256").upper()
    suite_fingerprint = _normalize_sha256(
        suite_fingerprint, field="suite_fingerprint")
    task = {
        "schema_version": PENDING_TASK_SCHEMA_VERSION,
        "status": "pending",
        "task_id": acceptance_task_id(durable_checkpoint_path, step),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": durable_checkpoint_path,
        "step": step,
        "teacher_agreement": teacher_agreement,
        "opening_plies": list(
            _normalize_acceptance_opening_plies(opening_plies)),
        "opening_seed": _normalize_acceptance_opening_seed(opening_seed),
        "inference_depth": inference_depth,
        "max_moves": max_moves,
        "num_workers": num_workers,
        "training_stage": str(training_stage),
        "checkpoint_sha256": checkpoint_sha256,
        "suite_fingerprint": suite_fingerprint,
    }
    if teacher_correct_states is not None:
        task["teacher_correct_states"] = teacher_correct_states
    if teacher_total_states is not None:
        task["teacher_total_states"] = teacher_total_states
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
        try:
            existing = load_pending_acceptance_task(path)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            # The append-only promotion registry can reconstruct this exact
            # task after a crash, but an invalid file at the deterministic
            # pathname would otherwise block every relaunch. Preserve its
            # bytes for diagnosis outside the discovery glob, then let the
            # normal atomic writer restore the validated task below. I/O
            # failures still propagate rather than treating a transient read
            # problem as corrupt content.
            quarantine = _quarantine_unreadable_pending_task(path)
            print(
                f"Quarantined unreadable pending acceptance task {path} "
                f"as {quarantine}: {exc}"
            )
        else:
            if not _pending_acceptance_tasks_match(existing, normalized):
                raise RuntimeError(f"Pending acceptance task conflicts with {path}")
            return path
    _write_json_atomic(path, normalized)
    return path


def _quarantine_unreadable_pending_task(path: Path) -> Path:
    """Move one invalid pending task outside the active discovery pattern."""
    quarantine = path.with_name(f".{path.name}.corrupt")
    suffix = 0
    while quarantine.exists():
        suffix += 1
        quarantine = path.with_name(f".{path.name}.corrupt.{suffix}")
    path.rename(quarantine)
    return quarantine


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
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            json.JSONDecodeError,
        ) as exc:
            print(f"Ignoring unreadable pending acceptance task {path}: {exc}")
    tasks.sort(key=lambda task: (int(task["step"]), str(task["task_id"])))
    return tasks


def successful_acceptance_report_path(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> Optional[Path]:
    """Return the matching durable success report, if one already exists."""
    completed = load_completed_acceptance_report(output_dir, task)
    return completed[0] if completed is not None else None


def load_completed_acceptance_report(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> Optional[tuple[Path, dict[str, Any]]]:
    """Load one validated, durable game-protocol report and its path."""
    try:
        normalized = _validate_pending_acceptance_task(task)
        path = (
            Path(output_dir)
            / f"acceptance_step_{normalized['step']:06d}.json"
        )
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        metrics = report.get("metrics")
        agreement_counts = report.get("teacher_agreement_counts")
        if (
            not isinstance(metrics, Mapping)
            or not isinstance(agreement_counts, Mapping)
        ):
            return None
        report_checkpoint = os.path.normcase(str(
            Path(report["checkpoint_path"]).expanduser().resolve(strict=False)))
        task_checkpoint = os.path.normcase(str(
            Path(normalized["checkpoint_path"]).expanduser().resolve(
                strict=False)))
        if (type(report.get("step")) is not int
                or report["step"] != normalized["step"]
                or report_checkpoint != task_checkpoint):
            return None
        # A report is terminal only for the exact durable protocol input.
        # This prevents an old report for the same checkpoint path/step from
        # clearing a task whose teacher evidence, digest, suite, openings, or
        # runtime changed.
        provenance_pairs = (
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("suite_fingerprint", "frozen_suite_fingerprint"),
            ("opening_seed", "opening_seed"),
            ("opening_plies", "opening_plies"),
            ("inference_depth", "inference_depth"),
            ("max_moves", "max_moves"),
            ("num_workers", "num_workers"),
            ("training_stage", "training_stage"),
            ("task_id", "task_id"),
        )
        for task_key, report_key in provenance_pairs:
            if report.get(report_key) != normalized.get(task_key):
                return None
        if metrics.get("teacher_agreement") != normalized["teacher_agreement"]:
            return None
        count_pairs = (
            ("teacher_correct_states", "correct_states"),
            ("teacher_total_states", "total_states"),
        )
        for task_key, report_key in count_pairs:
            if agreement_counts.get(report_key) != normalized.get(task_key):
                return None
        if not _completed_acceptance_report_matches_task(report, normalized):
            return None
        return path, report
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        json.JSONDecodeError,
    ):
        return None
    return None


def _completed_acceptance_report_matches_task(
    report: Mapping[str, Any],
    task: Mapping[str, Any],
) -> bool:
    """Return whether a completed report proves its fixed game protocol.

    Input provenance alone cannot make a report terminal. Startup recovery may
    publish the accepted alias directly from this file, so the raw game counts,
    protocol metadata, and recorded gate decision must still agree with one
    another before the report is allowed to suppress a replacement evaluation.
    """
    if (
        type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("selection_sequence") != _ACCEPTANCE_SELECTION_SEQUENCE
    ):
        return False

    expected_suite_id = opening_suite_identity(
        int(task["opening_seed"]),
        task["opening_plies"],
        ACCEPTANCE_GAMES_PER_OPPONENT // 2,
    )
    if report.get("opening_suite_id") != expected_suite_id:
        return False

    task_checkpoint = os.path.normcase(str(
        Path(task["checkpoint_path"]).expanduser().resolve(strict=False)))
    records = (
        (report.get("random"), "random"),
        (report.get("easy"), "algorithm"),
    )
    for record, opponent_type in records:
        if not isinstance(record, Mapping):
            return False
        record_checkpoint = os.path.normcase(str(
            Path(record.get("model_path", "")).expanduser().resolve(
                strict=False)))
        expected_metadata = {
            "algo_difficulty": "easy",
            "opponent_type": opponent_type,
            "opening_seed": task["opening_seed"],
            "opening_plies": task["opening_plies"],
            "opening_suite_id": expected_suite_id,
            "opening_suite_size": ACCEPTANCE_GAMES_PER_OPPONENT // 2,
            "ml_inference_depth": task["inference_depth"],
        }
        if (
            record_checkpoint != task_checkpoint
            or any(record.get(key) != value
                   for key, value in expected_metadata.items())
            or not _completed_game_records_match_task(
                record, task, opponent_type)
        ):
            return False

    random_record, _ = records[0]
    easy_record, _ = records[1]
    decision = evaluate_acceptance_gates(
        float(task["teacher_agreement"]), random_record, easy_record)
    decision_payload = decision.to_dict()
    if (
        not decision.checks["random_exact_balanced_100_games"]
        or not decision.checks["easy_exact_balanced_100_games"]
        or report.get("passed") is not decision.passed
    ):
        return False
    return all(
        report.get(key) == decision_payload[key]
        for key in ("checks", "metrics", "thresholds", "ci_method")
    )


def _completed_game_records_match_task(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    opponent_type: str,
) -> bool:
    """Verify aggregate results against the complete paired-game evidence."""
    games = record.get("games")
    if not isinstance(games, list) or len(games) != ACCEPTANCE_GAMES_PER_OPPONENT:
        return False

    opening_plies = tuple(task["opening_plies"])
    games_per_side = ACCEPTANCE_GAMES_PER_OPPONENT // 2
    expected_specs = {
        (
            player,
            int(task["opening_seed"]) + index,
            opening_plies[index % len(opening_plies)],
        )
        for player in (1, 2)
        for index in range(games_per_side)
    }
    seen_specs: set[tuple[int, int, int]] = set()
    counts = {
        "total_games": 0,
        "ml_wins": 0,
        "draws": 0,
        "algo_wins": 0,
        "ml_as_p1_wins": 0,
        "ml_as_p1_draws": 0,
        "ml_as_p1_losses": 0,
        "ml_as_p2_wins": 0,
        "ml_as_p2_draws": 0,
        "ml_as_p2_losses": 0,
    }

    for game in games:
        if not isinstance(game, Mapping):
            return False
        player = game.get("ml_player")
        opening_seed = game.get("opening_seed")
        opening_length = game.get("opening_plies")
        if (
            type(player) is not int
            or player not in (1, 2)
            or type(opening_seed) is not int
            or type(opening_length) is not int
            or game.get("opponent_type") != opponent_type
            or game.get("ml_inference_depth") != task["inference_depth"]
        ):
            return False

        spec = (player, opening_seed, opening_length)
        if spec not in expected_specs or spec in seen_specs:
            return False
        seen_specs.add(spec)

        result = game.get("result")
        winner = game.get("winner")
        side = "p1" if player == 1 else "p2"
        if result == "ml_win" and winner == player:
            counts["ml_wins"] += 1
            counts[f"ml_as_{side}_wins"] += 1
        elif result == "algo_win" and winner == 3 - player:
            counts["algo_wins"] += 1
            counts[f"ml_as_{side}_losses"] += 1
        elif result == "draw" and winner is None:
            counts["draws"] += 1
            counts[f"ml_as_{side}_draws"] += 1
        else:
            return False
        counts["total_games"] += 1

    return seen_specs == expected_specs and all(
        type(record.get(key)) is int and record[key] == value
        for key, value in counts.items()
    )


def terminal_acceptance_report_path(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> Optional[Path]:
    """Return a matching success or failure report for a durable task."""
    terminal = load_terminal_acceptance_report(output_dir, task)
    return terminal[0] if terminal is not None else None


def load_terminal_acceptance_report(
    output_dir: str | Path,
    task: Mapping[str, Any],
) -> Optional[tuple[Path, dict[str, Any]]]:
    """Load one validated terminal report and its path."""
    completed = load_completed_acceptance_report(output_dir, task)
    if completed is not None:
        return completed
    failure = failure_acceptance_report_path(output_dir, task)
    if not failure.exists():
        return None
    try:
        with failure.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        report_task = report.get("task")
        if not isinstance(report_task, Mapping):
            return None
        normalized = _validate_pending_acceptance_task(task)
        if (
            type(report.get("schema_version")) is not int
            or report["schema_version"] != 1
            or report.get("status") != "error"
            or report.get("passed") is not False
            or str(report.get("task_id")) != normalized["task_id"]
            or type(report.get("step")) is not int
            or report["step"] != normalized["step"]
            or not _pending_acceptance_tasks_match(report_task, normalized)
        ):
            return None
        report_checkpoint = os.path.normcase(str(
            Path(report.get("checkpoint_path", "")).expanduser().resolve(
                strict=False)))
        task_checkpoint = os.path.normcase(str(
            Path(normalized["checkpoint_path"]).expanduser().resolve(
                strict=False)))
        if report_checkpoint == task_checkpoint:
            return failure, report
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        json.JSONDecodeError,
    ):
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
    expected = _normalize_sha256(
        task.get("checkpoint_sha256"), field="checkpoint_sha256").upper()
    checkpoint_path = Path(str(task["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pending acceptance checkpoint is missing: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest().upper()
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
        "checkpoint_sha256",
        "suite_fingerprint",
    }
    missing = sorted(required - set(task))
    if missing:
        raise ValueError(f"Pending acceptance task is missing fields: {missing}")
    if (
        type(task["schema_version"]) is not int
        or task["schema_version"] != PENDING_TASK_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported pending acceptance task schema_version")
    if task["status"] != "pending":
        raise ValueError("Pending acceptance task status must be 'pending'")
    checkpoint_path = str(task["checkpoint_path"])
    step = _normalize_acceptance_step(task["step"])
    expected_id = acceptance_task_id(checkpoint_path, step)
    if str(task["task_id"]) != expected_id:
        raise ValueError("Pending acceptance task_id does not match checkpoint and step")
    opening_plies = list(
        _normalize_acceptance_opening_plies(task["opening_plies"]))
    opening_seed = _normalize_acceptance_opening_seed(task["opening_seed"])
    inference_depth, max_moves = _normalize_acceptance_game_bounds(
        task["inference_depth"], task["max_moves"])
    num_workers = _normalize_acceptance_num_workers(task["num_workers"])
    (
        teacher_agreement,
        teacher_correct_states,
        teacher_total_states,
    ) = _validate_teacher_evidence(
        task["teacher_agreement"],
        task.get("teacher_correct_states"),
        task.get("teacher_total_states"),
    )
    normalized = dict(task)
    normalized.update({
        "schema_version": PENDING_TASK_SCHEMA_VERSION,
        "status": "pending",
        "task_id": expected_id,
        "created_at": str(task["created_at"]),
        "checkpoint_path": checkpoint_path,
        "step": step,
        "teacher_agreement": teacher_agreement,
        "opening_plies": opening_plies,
        "opening_seed": opening_seed,
        "inference_depth": inference_depth,
        "max_moves": max_moves,
        "num_workers": num_workers,
        "training_stage": str(task["training_stage"]),
        "teacher_correct_states": teacher_correct_states,
        "teacher_total_states": teacher_total_states,
        "checkpoint_sha256": _normalize_sha256(
            task["checkpoint_sha256"], field="checkpoint_sha256").upper(),
        "suite_fingerprint": _normalize_sha256(
            task["suite_fingerprint"], field="suite_fingerprint"),
    })
    return normalized


def _pending_acceptance_tasks_match(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    """Return whether two tasks name the same durable protocol input."""
    first_normalized = _validate_pending_acceptance_task(first)
    second_normalized = _validate_pending_acceptance_task(second)
    comparable_keys = (
        set(first_normalized) | set(second_normalized)
    ) - {"created_at"}
    return all(
        first_normalized.get(key) == second_normalized.get(key)
        for key in comparable_keys
    )


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
    # Reject corrupt durable evidence before creating a worker pool and running
    # the fixed 200-game protocol. The same helper guards task persistence and
    # recovery, while this boundary also protects standalone evaluator calls.
    (
        teacher_agreement,
        teacher_correct_states,
        teacher_total_states,
    ) = _validate_teacher_evidence(
        teacher_agreement,
        teacher_correct_states,
        teacher_total_states,
    )
    step = _normalize_acceptance_step(step)
    inference_depth, max_moves = _normalize_acceptance_game_bounds(
        inference_depth, max_moves)
    # Direct callers do not necessarily pass through durable task creation.
    # Enforce the same randomized-opening contract before creating the report
    # directory or worker pool so invalid protocols cannot spend 200 games or
    # become terminal evidence.
    opening_plies = _normalize_acceptance_opening_plies(opening_plies)
    opening_seed = _normalize_acceptance_opening_seed(opening_seed)
    num_workers = _normalize_acceptance_num_workers(num_workers)
    checkpoint_sha256 = _normalize_sha256(
        checkpoint_sha256, field="checkpoint_sha256").upper()
    suite_fingerprint = _normalize_sha256(
        suite_fingerprint, field="suite_fingerprint")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    common = {
        "model_path": checkpoint_path,
        "num_workers": num_workers,
        "max_moves": max_moves,
        "opening_plies": opening_plies,
        "opening_seed": opening_seed,
        "ml_inference_depth": inference_depth,
    }

    # Both phases use the same checkpoint and run sequentially. Reuse one
    # isolated pool so each worker loads and folds the CPU model only once while
    # preserving the CUDA-safe process boundary and the declared gate order.
    worker_ctx = _evaluation_worker_context()
    with ProcessPoolExecutor(
        max_workers=common["num_workers"],
        mp_context=worker_ctx,
        initializer=_evaluation_worker_init,
    ) as executor:
        random_stats = ModelVsAlgoTester(
            algo_difficulty="easy",
            opponent_type="random",
            stats_dir=str(output_root / "random_details"),
            executor=executor,
            **common,
        ).run_tests(num_games=ACCEPTANCE_GAMES_PER_OPPONENT)
        random_record = random_stats.to_dict()
        easy_stats = ModelVsAlgoTester(
            algo_difficulty="easy",
            opponent_type="algorithm",
            stats_dir=str(output_root / "easy_details"),
            executor=executor,
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
        "step": step,
        "training_stage": str(training_stage),
        "task_id": task_id,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_suite_fingerprint": suite_fingerprint,
        "teacher_agreement_counts": {
            "correct_states": teacher_correct_states,
            "total_states": teacher_total_states,
        },
        "selection_sequence": list(_ACCEPTANCE_SELECTION_SEQUENCE),
        "opening_seed": opening_seed,
        "opening_plies": [int(value) for value in opening_plies],
        "opening_suite_id": random_record.get("opening_suite_id"),
        "inference_depth": inference_depth,
        "max_moves": max_moves,
        "num_workers": num_workers,
        "random": random_record,
        "easy": easy_record,
        **decision.to_dict(),
    }
    report_path = output_root / f"acceptance_step_{step:06d}.json"
    _write_json_atomic(report_path, report)
    report["report_path"] = str(report_path)
    return report


def _validate_teacher_agreement(value: Any) -> float:
    """Return finite teacher agreement within the protocol's probability range."""
    if type(value) not in (int, float):
        raise ValueError(
            "Pending acceptance teacher_agreement must be a real number")
    agreement = float(value)
    if not math.isfinite(agreement) or not 0.0 <= agreement <= 1.0:
        raise ValueError(
            "Pending acceptance teacher_agreement must be finite and within [0, 1]")
    return agreement


def _validate_teacher_evidence(
    agreement: Any,
    correct_states: Any,
    total_states: Any,
) -> tuple[float, int, int]:
    """Validate the complete frozen-suite measurement behind acceptance."""
    normalized_agreement = _validate_teacher_agreement(agreement)
    if correct_states is None or total_states is None:
        raise ValueError(
            "Pending acceptance has no held-out teacher-agreement counts")
    if type(correct_states) is not int or type(total_states) is not int:
        raise ValueError(
            "Pending acceptance teacher-agreement counts must be integers")
    normalized_correct = correct_states
    normalized_total = total_states
    if normalized_total != FROZEN_TEACHER_SUITE_SIZE:
        raise ValueError(
            "Pending acceptance teacher agreement was measured on "
            f"{normalized_total} held-out state(s), not the required "
            f"{FROZEN_TEACHER_SUITE_SIZE}")
    if normalized_correct < 0:
        raise ValueError(
            "Pending acceptance teacher_correct_states must be non-negative")
    if normalized_correct > normalized_total:
        raise ValueError(
            "Pending acceptance teacher_correct_states exceeds total states")
    if normalized_correct / normalized_total != normalized_agreement:
        raise ValueError(
            "Pending acceptance teacher agreement is not the quotient of its "
            "recorded counts")
    return normalized_agreement, normalized_correct, normalized_total


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    # Thin delegate: one atomic-JSON implementation project-wide (see
    # run_status._write_json_atomic for the temp+fsync+replace contract).
    # The module-level name stays importable for tests and callers.
    from .run_status import _write_json_atomic as _shared
    _shared(path, payload)
