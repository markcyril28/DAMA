"""Durability tests for queued checkpoint acceptance work."""

import hashlib
import json
import os
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

import dama.ai.ml.trainer as trainer_module
from dama.ai.ml import checkpoint_acceptance
from dama.ai.ml import run_status
from dama.ai.ml.model_vs_algo import opening_suite_identity
from dama.ai.ml.trainer import Trainer, TrainingStats


def _make_task(tmp_path: Path, step: int = 140000) -> dict:
    checkpoint = tmp_path / f"model_step_{step:06d}.pt"
    checkpoint.write_bytes(f"checkpoint-{step}".encode("ascii"))
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return checkpoint_acceptance.make_pending_acceptance_task(
        str(checkpoint),
        step=step,
        teacher_agreement=0.55,
        opening_plies=(2, 4, 6, 8),
        opening_seed=20260819,
        inference_depth=1,
        max_moves=200,
        num_workers=2,
        training_stage="policy_only",
        checkpoint_sha256=checkpoint_sha256,
        suite_fingerprint="b" * 64,
        teacher_correct_states=2750,
        teacher_total_states=5000,
    )


def _holder(tmp_path: Path):
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        acceptance_dir=str(tmp_path / "acceptance"),
        accepted_path=str(tmp_path / "accepted.pt"),
    )
    holder._acceptance_thread = None
    holder._acceptance_queue = Queue()
    holder._acceptance_task_lock = threading.Lock()
    holder._acceptance_task_ids = set()
    holder._stopped = False
    holder._paused = False
    holder.stats = TrainingStats()
    holder._save_stats = lambda **_kwargs: True
    holder._put_status = lambda payload: None
    holder._publish_checkpoint_alias = lambda source, destination: None
    return holder


def _write_success_report(
    output_dir: Path,
    task: dict,
    *,
    passed: bool,
) -> Path:
    if passed:
        random_wdl = (90, 0, 10, 45, 0, 5, 45, 0, 5)
        easy_wdl = (70, 0, 30, 35, 0, 15, 35, 0, 15)
    else:
        random_wdl = easy_wdl = (0, 0, 100, 0, 0, 50, 0, 0, 50)
    suite_id = opening_suite_identity(
        task["opening_seed"], task["opening_plies"], 50)

    def _record(opponent_type: str, values: tuple[int, ...]) -> dict:
        (wins, draws, losses, p1_wins, p1_draws, p1_losses,
         p2_wins, p2_draws, p2_losses) = values
        games = []
        for player, side_wdl in (
            (1, (p1_wins, p1_draws, p1_losses)),
            (2, (p2_wins, p2_draws, p2_losses)),
        ):
            side_wins, side_draws, side_losses = side_wdl
            results = (
                ["ml_win"] * side_wins
                + ["draw"] * side_draws
                + ["algo_win"] * side_losses
            )
            assert len(results) == 50
            for index, result in enumerate(results):
                games.append({
                    "result": result,
                    "ml_player": player,
                    "winner": (
                        player if result == "ml_win"
                        else 3 - player if result == "algo_win"
                        else None
                    ),
                    "num_moves": 1,
                    "ml_moves": 1,
                    "algo_moves": 0,
                    "game_time_ms": 1.0,
                    "opponent_type": opponent_type,
                    "opening_plies": task["opening_plies"][
                        index % len(task["opening_plies"])
                    ],
                    "opening_seed": task["opening_seed"] + index,
                    "ml_inference_depth": task["inference_depth"],
                })
        return {
            "total_games": 100,
            "model_path": task["checkpoint_path"],
            "algo_difficulty": "easy",
            "opponent_type": opponent_type,
            "opening_seed": task["opening_seed"],
            "opening_plies": task["opening_plies"],
            "opening_suite_id": suite_id,
            "opening_suite_size": 50,
            "ml_inference_depth": task["inference_depth"],
            "ml_wins": wins,
            "draws": draws,
            "algo_wins": losses,
            "ml_as_p1_wins": p1_wins,
            "ml_as_p1_draws": p1_draws,
            "ml_as_p1_losses": p1_losses,
            "ml_as_p2_wins": p2_wins,
            "ml_as_p2_draws": p2_draws,
            "ml_as_p2_losses": p2_losses,
            "games": games,
        }

    random_record = _record("random", random_wdl)
    easy_record = _record("algorithm", easy_wdl)
    decision = checkpoint_acceptance.evaluate_acceptance_gates(
        task["teacher_agreement"], random_record, easy_record)
    assert decision.passed is passed
    report_path = output_dir / f"acceptance_step_{task['step']:06d}.json"
    checkpoint_acceptance._write_json_atomic(report_path, {
        "schema_version": 1,
        "checkpoint_path": task["checkpoint_path"],
        "step": task["step"],
        "task_id": task["task_id"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "frozen_suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_agreement_counts": {
            "correct_states": task.get("teacher_correct_states"),
            "total_states": task.get("teacher_total_states"),
        },
        "selection_sequence": [
            "held_out_teacher_agreement",
            "random_game_strength",
            "easy_game_strength",
        ],
        "opening_seed": task["opening_seed"],
        "opening_plies": task["opening_plies"],
        "opening_suite_id": suite_id,
        "inference_depth": task["inference_depth"],
        "max_moves": task["max_moves"],
        "num_workers": task["num_workers"],
        "training_stage": task["training_stage"],
        "random": random_record,
        "easy": easy_record,
        **decision.to_dict(),
    })
    return report_path


def test_registry_task_replaces_modified_pending_teacher_evidence(
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    modified = dict(task)
    modified["teacher_agreement"] = 1.0
    modified["teacher_correct_states"] = 5000
    output_dir = Path(_holder(tmp_path).config.acceptance_dir)
    pending = checkpoint_acceptance.persist_pending_acceptance_task(
        output_dir, modified)

    holder = _holder(tmp_path)
    holder._promotion_registry = SimpleNamespace(
        records=lambda: ({"promoted": True},))
    holder._acceptance_task_from_promotion = lambda _promotion: dict(task)
    queued = {}

    def _capture(candidate, *, persist):
        task_id = candidate["task_id"]
        if task_id in queued:
            return False
        if persist:
            checkpoint_acceptance.persist_pending_acceptance_task(
                output_dir, candidate)
        queued[task_id] = dict(candidate)
        return True

    holder._queue_checkpoint_acceptance_task = _capture

    assert holder._recover_pending_checkpoint_acceptance() == 1
    assert list(queued.values()) == [task]
    assert checkpoint_acceptance.load_pending_acceptance_task(pending) == task
    quarantined = list(output_dir.glob(f".{pending.name}.corrupt*"))
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text(encoding="utf-8")) == modified


def test_pending_task_is_atomic_and_discoverable(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"

    path = checkpoint_acceptance.persist_pending_acceptance_task(output_dir, task)

    assert path.exists()
    assert checkpoint_acceptance.load_pending_acceptance_task(path) == task
    assert checkpoint_acceptance.discover_pending_acceptance_tasks(output_dir) == [task]
    assert not list(output_dir.glob("*.tmp"))
    assert checkpoint_acceptance.persist_pending_acceptance_task(
        output_dir, task) == path


def test_pending_task_removal_commits_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    pending = checkpoint_acceptance.persist_pending_acceptance_task(
        output_dir, task)
    synced = []
    real_sync = run_status._fsync_directory

    def tracking_sync(path: Path) -> None:
        synced.append(Path(path))
        real_sync(path)

    monkeypatch.setattr(run_status, "_fsync_directory", tracking_sync)

    checkpoint_acceptance.remove_pending_acceptance_task(output_dir, task)

    assert not pending.exists()
    assert synced == [output_dir]


def test_missing_pending_task_removal_skips_directory_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"

    def unexpected_sync(_path: Path) -> None:
        raise AssertionError("missing pending task must not trigger a sync")

    monkeypatch.setattr(run_status, "_fsync_directory", unexpected_sync)

    checkpoint_acceptance.remove_pending_acceptance_task(output_dir, task)


def test_pending_task_removal_reports_directory_sync_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    pending = checkpoint_acceptance.persist_pending_acceptance_task(
        output_dir, task)

    def fail_sync(path: Path) -> None:
        assert Path(path) == output_dir
        raise OSError("simulated pending cleanup directory sync failure")

    monkeypatch.setattr(run_status, "_fsync_directory", fail_sync)

    with pytest.raises(OSError, match="pending cleanup directory sync failure"):
        checkpoint_acceptance.remove_pending_acceptance_task(output_dir, task)

    assert not pending.exists()


def test_pending_task_commits_acceptance_namespace_before_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    synced = []
    real_sync = run_status._fsync_directory

    def tracking_sync(path: Path) -> None:
        synced.append(Path(path))
        real_sync(path)

    monkeypatch.setattr(run_status, "_fsync_directory", tracking_sync)

    pending = checkpoint_acceptance.persist_pending_acceptance_task(
        output_dir, task)

    assert pending.is_file()
    assert synced == [output_dir.parent, output_dir]


def test_acceptance_namespace_sync_failure_prevents_pending_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"

    def fail_parent_sync(path: Path) -> None:
        assert Path(path) == output_dir.parent
        raise OSError("simulated acceptance namespace sync failure")

    def unexpected_write(_path: Path, _payload: dict) -> None:
        raise AssertionError("pending task published before namespace commit")

    monkeypatch.setattr(run_status, "_fsync_directory", fail_parent_sync)
    monkeypatch.setattr(
        checkpoint_acceptance, "_write_json_atomic", unexpected_write)

    with pytest.raises(OSError, match="namespace sync failure"):
        checkpoint_acceptance.persist_pending_acceptance_task(
            output_dir, task)

    assert output_dir.is_dir()
    assert not list(output_dir.iterdir())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("inference_depth", 0, "inference depth must be one of"),
        ("max_moves", 0, "max_moves must be a positive integer"),
    ),
)
def test_persisted_task_rejects_invalid_game_bounds(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    task = _make_task(tmp_path)
    task[field] = value
    path = checkpoint_acceptance.pending_acceptance_task_path(
        tmp_path / "acceptance", task)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        checkpoint_acceptance.load_pending_acceptance_task(path)


@pytest.mark.parametrize("checkpoint_path", (True, 1, ["checkpoint.pt"]))
def test_persisted_task_rejects_coerced_checkpoint_path(
    tmp_path: Path,
    checkpoint_path,
) -> None:
    task = _make_task(tmp_path)
    task["checkpoint_path"] = checkpoint_path
    task["task_id"] = checkpoint_acceptance.acceptance_task_id(
        str(checkpoint_path), task["step"])
    path = checkpoint_acceptance.pending_acceptance_task_path(
        tmp_path / "acceptance", task)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_path must be"):
        checkpoint_acceptance.load_pending_acceptance_task(path)


@pytest.mark.parametrize("opening_plies", (
    [True, 4, 6, 8],
    [2.5, 4, 6, 8],
    ["2", 4, 6, 8],
))
def test_persisted_task_rejects_coerced_opening_plies(
    tmp_path: Path,
    opening_plies: list,
) -> None:
    task = _make_task(tmp_path)
    task["opening_plies"] = opening_plies
    path = checkpoint_acceptance.pending_acceptance_task_path(
        tmp_path / "acceptance", task)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match="positive integers"):
        checkpoint_acceptance.load_pending_acceptance_task(path)


@pytest.mark.parametrize("opening_seed", (True, 20260819.5, "20260819"))
def test_persisted_task_rejects_coerced_opening_seed(
    tmp_path: Path,
    opening_seed,
) -> None:
    task = _make_task(tmp_path)
    task["opening_seed"] = opening_seed
    path = checkpoint_acceptance.pending_acceptance_task_path(
        tmp_path / "acceptance", task)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match="opening_seed must be an integer"):
        checkpoint_acceptance.load_pending_acceptance_task(path)


@pytest.mark.parametrize("num_workers", (True, False, "2", 2.5, 0, -1))
def test_persisted_task_rejects_invalid_num_workers(
    tmp_path: Path,
    num_workers,
) -> None:
    task = _make_task(tmp_path)
    task["num_workers"] = num_workers
    path = checkpoint_acceptance.pending_acceptance_task_path(
        tmp_path / "acceptance", task)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match="num_workers must be a positive integer"):
        checkpoint_acceptance.load_pending_acceptance_task(path)


@pytest.mark.parametrize(
    "training_stage", (True, 1, ["policy_only"], "unknown", ""))
def test_persisted_task_rejects_invalid_training_stage(
    tmp_path: Path,
    training_stage,
) -> None:
    task = _make_task(tmp_path)
    task["training_stage"] = training_stage
    path = checkpoint_acceptance.pending_acceptance_task_path(
        tmp_path / "acceptance", task)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match="training_stage must be"):
        checkpoint_acceptance.load_pending_acceptance_task(path)


@pytest.mark.parametrize("step", (True, "140000", 140000.5, -1))
def test_persisted_task_rejects_invalid_step(
    tmp_path: Path,
    step,
) -> None:
    task = _make_task(tmp_path)
    task["step"] = step
    path = checkpoint_acceptance.pending_acceptance_task_path(
        tmp_path / "acceptance", task)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match="step must be a non-negative integer"):
        checkpoint_acceptance.load_pending_acceptance_task(path)


def test_existing_pending_task_rejects_removed_protocol_provenance(
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    task_with_suite = {**task, "suite_fingerprint": "c" * 64}
    output_dir = tmp_path / "acceptance"
    checkpoint_acceptance.persist_pending_acceptance_task(
        output_dir, task_with_suite)

    with pytest.raises(RuntimeError, match="Pending acceptance task conflicts"):
        checkpoint_acceptance.persist_pending_acceptance_task(output_dir, task)


def test_stale_success_report_cannot_clear_changed_protocol(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    output_dir.mkdir()
    report_path = output_dir / f"acceptance_step_{task['step']:06d}.json"
    report_path.write_text(json.dumps({
        "checkpoint_path": task["checkpoint_path"],
        "step": task["step"],
        "task_id": task["task_id"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "teacher_agreement_counts": {
            "correct_states": task.get("teacher_correct_states"),
            "total_states": task.get("teacher_total_states"),
        },
        "metrics": {
            "teacher_agreement": task["teacher_agreement"],
        },
        "opening_seed": task["opening_seed"],
        "opening_plies": [2, 4],  # stale protocol, task uses four openings
    }), encoding="utf-8")
    assert checkpoint_acceptance.successful_acceptance_report_path(
        output_dir, task) is None


@pytest.mark.parametrize("step", (True, "140000", 140000.5))
def test_success_report_rejects_coerced_step(
    tmp_path: Path,
    step,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    output_dir.mkdir()
    report_path = _write_success_report(output_dir, task, passed=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["step"] = step
    checkpoint_acceptance._write_json_atomic(report_path, report)

    assert checkpoint_acceptance.load_completed_acceptance_report(
        output_dir, task) is None


@pytest.mark.parametrize("step", (True, "140000", 140000.5))
def test_failure_report_rejects_coerced_step(
    tmp_path: Path,
    step,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    report_path = checkpoint_acceptance.write_acceptance_failure_report(
        output_dir, task, RuntimeError("evaluation failed"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["step"] = step
    checkpoint_acceptance._write_json_atomic(report_path, report)

    assert checkpoint_acceptance.load_terminal_acceptance_report(
        output_dir, task) is None


@pytest.mark.parametrize("schema_version", (True, 1.0, "1", None))
def test_failure_report_rejects_invalid_schema_version(
    tmp_path: Path,
    schema_version,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    report_path = checkpoint_acceptance.write_acceptance_failure_report(
        output_dir, task, RuntimeError("evaluation failed"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if schema_version is None:
        report.pop("schema_version")
    else:
        report["schema_version"] = schema_version
    checkpoint_acceptance._write_json_atomic(report_path, report)

    assert checkpoint_acceptance.load_terminal_acceptance_report(
        output_dir, task) is None


@pytest.mark.parametrize(
    ("field", "replacement_value", "paired_field", "paired_value"), (
        ("teacher_agreement", 0.49, "teacher_correct_states", 2450),
        ("teacher_correct_states", 2450, "teacher_agreement", 0.49),
))
def test_stale_success_report_cannot_clear_changed_teacher_evidence(
    tmp_path: Path,
    field: str,
    replacement_value,
    paired_field: str,
    paired_value,
) -> None:
    old_task = _make_task(tmp_path)
    replacement = {
        **old_task,
        "created_at": "2026-09-07T12:00:00+00:00",
        field: replacement_value,
        paired_field: paired_value,
    }
    output_dir = tmp_path / "acceptance"
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        output_dir, replacement)
    _write_success_report(output_dir, old_task, passed=True)
    holder = _holder(tmp_path)
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 1

    assert pending_path.exists()
    assert holder._acceptance_queue.qsize() == 1
    assert holder._acceptance_queue.get_nowait()[field] == replacement_value
    assert holder.stats.acceptance_history == []
    assert not Path(holder.config.accepted_path).exists()


def test_forged_passing_decision_cannot_publish_accepted_alias(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    report_path = _write_success_report(
        Path(holder.config.acceptance_dir), task, passed=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["passed"] = True
    checkpoint_acceptance._write_json_atomic(report_path, report)
    holder._ensure_checkpoint_acceptance_worker = lambda: None
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias

    assert holder._recover_pending_checkpoint_acceptance() == 1

    assert pending_path.exists()
    assert holder._acceptance_queue.qsize() == 1
    assert holder.stats.acceptance_history == []
    assert not Path(holder.config.accepted_path).exists()


@pytest.mark.parametrize("corruption", (
    "missing_games",
    "aggregate_mismatch",
    "unpaired_opening",
    "wrong_opponent",
    "inconsistent_winner",
))
def test_success_report_requires_exact_paired_game_evidence(
    tmp_path: Path,
    corruption: str,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    report_path = _write_success_report(output_dir, task, passed=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if corruption == "missing_games":
        report["random"].pop("games")
    elif corruption == "aggregate_mismatch":
        report["random"]["games"][0]["result"] = "algo_win"
        report["random"]["games"][0]["winner"] = 2
    elif corruption == "unpaired_opening":
        report["random"]["games"][0]["opening_seed"] += 1
    elif corruption == "wrong_opponent":
        report["random"]["games"][0]["opponent_type"] = "algorithm"
    else:
        report["random"]["games"][0]["winner"] = 2
    checkpoint_acceptance._write_json_atomic(report_path, report)

    assert checkpoint_acceptance.successful_acceptance_report_path(
        output_dir, task) is None


@pytest.mark.parametrize("corruption", (
    "boolean_inference_depth",
    "floating_opening_schedule",
    "boolean_game_inference_depth",
    "numeric_gate_check",
    "floating_teacher_count",
    "boolean_winner",
))
def test_success_report_rejects_lossy_json_numeric_types(
    tmp_path: Path,
    corruption: str,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    report_path = _write_success_report(output_dir, task, passed=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if corruption == "boolean_inference_depth":
        report["inference_depth"] = True
    elif corruption == "floating_opening_schedule":
        report["random"]["opening_plies"] = [2.0, 4.0, 6.0, 8.0]
    elif corruption == "boolean_game_inference_depth":
        report["random"]["games"][0]["ml_inference_depth"] = True
    elif corruption == "numeric_gate_check":
        report["checks"]["random_exact_balanced_100_games"] = 1
    elif corruption == "floating_teacher_count":
        report["teacher_agreement_counts"]["correct_states"] = 2750.0
    else:
        game = next(
            game for game in report["random"]["games"]
            if game["ml_player"] == 1 and game["result"] == "ml_win"
        )
        game["winner"] = True
    checkpoint_acceptance._write_json_atomic(report_path, report)

    assert checkpoint_acceptance.load_completed_acceptance_report(
        output_dir, task) is None


def test_incomplete_game_evidence_cannot_publish_accepted_alias(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    report_path = _write_success_report(
        Path(holder.config.acceptance_dir), task, passed=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["random"].pop("games")
    checkpoint_acceptance._write_json_atomic(report_path, report)
    holder._ensure_checkpoint_acceptance_worker = lambda: None
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias

    assert holder._recover_pending_checkpoint_acceptance() == 1

    assert pending_path.exists()
    assert holder._acceptance_queue.qsize() == 1
    assert holder.stats.acceptance_history == []
    assert not Path(holder.config.accepted_path).exists()


@pytest.mark.parametrize("invalid_passed", ("true", 1, None))
def test_non_boolean_passing_decision_is_not_terminal(
    tmp_path: Path,
    invalid_passed,
) -> None:
    task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    report_path = _write_success_report(output_dir, task, passed=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["passed"] = invalid_passed
    checkpoint_acceptance._write_json_atomic(report_path, report)

    assert checkpoint_acceptance.successful_acceptance_report_path(
        output_dir, task) is None


def test_stale_failure_report_cannot_clear_changed_protocol(
    tmp_path: Path,
) -> None:
    old_task = _make_task(tmp_path)
    output_dir = tmp_path / "acceptance"
    checkpoint_acceptance.write_acceptance_failure_report(
        output_dir, old_task, RuntimeError("old protocol failed"))

    changed_task = {
        **old_task,
        "opening_plies": [2, 4],
    }
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        output_dir, changed_task)
    holder = _holder(tmp_path)
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 1
    assert pending_path.exists()
    assert holder._acceptance_queue.qsize() == 1
    assert holder.stats.acceptance_history == []


def test_queue_persists_before_memory_enqueue_and_deduplicates(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.pending_acceptance_task_path(
        holder.config.acceptance_dir, task)

    def assert_pending_before_worker_start() -> None:
        assert pending_path.exists()

    holder._ensure_checkpoint_acceptance_worker = assert_pending_before_worker_start

    assert holder._queue_checkpoint_acceptance_task(task, persist=True) is True
    assert holder._queue_checkpoint_acceptance_task(task, persist=True) is False
    assert holder._acceptance_queue.qsize() == 1
    assert holder._acceptance_queue.get_nowait()["task_id"] == task["task_id"]


def test_startup_requeues_pending_once_without_duplicates(tmp_path: Path) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 1
    assert holder._recover_pending_checkpoint_acceptance() == 0
    assert holder._acceptance_queue.qsize() == 1
    assert holder._acceptance_task_ids == {task["task_id"]}


def test_registry_recovery_quarantines_and_rebuilds_corrupt_pending_task(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    holder._promotion_registry = SimpleNamespace(records=lambda: ({
        "promoted": True,
        "step": task["step"],
        "teacher_agreement": task["teacher_agreement"],
        "training_stage": task["training_stage"],
        "checkpoint_path": task["checkpoint_path"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": task.get("teacher_correct_states"),
        "teacher_total_states": task.get("teacher_total_states"),
    },))
    holder._ensure_checkpoint_acceptance_worker = lambda: None
    pending_path = checkpoint_acceptance.pending_acceptance_task_path(
        holder.config.acceptance_dir, task)
    pending_path.parent.mkdir(parents=True)
    corrupt_bytes = b'{"schema_version":'
    pending_path.write_bytes(corrupt_bytes)

    assert holder._recover_pending_checkpoint_acceptance() == 1

    recovered_task = checkpoint_acceptance.load_pending_acceptance_task(
        pending_path)
    assert checkpoint_acceptance._pending_acceptance_tasks_match(
        recovered_task, task)
    assert holder._acceptance_queue.qsize() == 1
    assert checkpoint_acceptance._pending_acceptance_tasks_match(
        holder._acceptance_queue.get_nowait(), task)
    quarantines = list(
        pending_path.parent.glob(f".{pending_path.name}.corrupt*"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == corrupt_bytes

    assert holder._recover_pending_checkpoint_acceptance() == 0
    assert len(list(
        pending_path.parent.glob(f".{pending_path.name}.corrupt*"))) == 1


def test_registry_recovery_rejects_nonfinite_pending_teacher_evidence(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    holder._promotion_registry = SimpleNamespace(records=lambda: ({
        "promoted": True,
        "step": task["step"],
        "teacher_agreement": task["teacher_agreement"],
        "training_stage": task["training_stage"],
        "checkpoint_path": task["checkpoint_path"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": task.get("teacher_correct_states"),
        "teacher_total_states": task.get("teacher_total_states"),
    },))
    holder._ensure_checkpoint_acceptance_worker = lambda: None
    pending_path = checkpoint_acceptance.pending_acceptance_task_path(
        holder.config.acceptance_dir, task)
    pending_path.parent.mkdir(parents=True)
    malformed = {**task, "teacher_agreement": float("nan")}
    pending_path.write_text(json.dumps(malformed), encoding="utf-8")

    assert holder._recover_pending_checkpoint_acceptance() == 1

    recovered_task = checkpoint_acceptance.load_pending_acceptance_task(
        pending_path)
    assert checkpoint_acceptance._pending_acceptance_tasks_match(
        recovered_task, task)
    assert holder._acceptance_queue.qsize() == 1
    assert checkpoint_acceptance._pending_acceptance_tasks_match(
        holder._acceptance_queue.get_nowait(), task)
    quarantines = list(
        pending_path.parent.glob(f".{pending_path.name}.corrupt*"))
    assert len(quarantines) == 1
    quarantined = json.loads(quarantines[0].read_text(encoding="utf-8"))
    assert str(quarantined["teacher_agreement"]) == "nan"


@pytest.mark.parametrize("promoted", (1, "false"))
def test_registry_recovery_rejects_non_boolean_promotion_decision(
    tmp_path: Path,
    capsys,
    promoted,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    holder._promotion_registry = SimpleNamespace(records=lambda: ({
        "promoted": promoted,
        "step": task["step"],
        "teacher_agreement": task["teacher_agreement"],
        "training_stage": task["training_stage"],
        "checkpoint_path": task["checkpoint_path"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": task.get("teacher_correct_states"),
        "teacher_total_states": task.get("teacher_total_states"),
    },))
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 0

    assert holder._acceptance_queue.empty()
    assert not Path(holder.config.acceptance_dir).exists()
    assert "promoted decision must be a boolean" in capsys.readouterr().out


@pytest.mark.parametrize("step", (True, "140000", 140000.5, -1))
def test_registry_recovery_rejects_invalid_promotion_step(
    tmp_path: Path,
    capsys,
    step,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    holder._promotion_registry = SimpleNamespace(records=lambda: ({
        "promoted": True,
        "step": step,
        "teacher_agreement": task["teacher_agreement"],
        "training_stage": task["training_stage"],
        "checkpoint_path": task["checkpoint_path"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": task.get("teacher_correct_states"),
        "teacher_total_states": task.get("teacher_total_states"),
    },))
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 0

    assert holder._acceptance_queue.empty()
    assert not Path(holder.config.acceptance_dir).exists()
    assert "Acceptance step must be a non-negative integer" in capsys.readouterr().out


@pytest.mark.parametrize("training_stage", (True, 1, ["policy_only"], "unknown"))
def test_registry_recovery_rejects_invalid_promotion_training_stage(
    tmp_path: Path,
    capsys,
    training_stage,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    holder._promotion_registry = SimpleNamespace(records=lambda: ({
        "promoted": True,
        "step": task["step"],
        "teacher_agreement": task["teacher_agreement"],
        "training_stage": training_stage,
        "checkpoint_path": task["checkpoint_path"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": task.get("teacher_correct_states"),
        "teacher_total_states": task.get("teacher_total_states"),
    },))
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 0

    assert holder._acceptance_queue.empty()
    assert not Path(holder.config.acceptance_dir).exists()
    assert "training_stage must be" in capsys.readouterr().out


@pytest.mark.parametrize("checkpoint_path", (True, 1, ["checkpoint.pt"]))
def test_registry_recovery_rejects_coerced_checkpoint_path(
    tmp_path: Path,
    capsys,
    checkpoint_path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    holder._promotion_registry = SimpleNamespace(records=lambda: ({
        "promoted": True,
        "step": task["step"],
        "teacher_agreement": task["teacher_agreement"],
        "training_stage": task["training_stage"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": task["checkpoint_sha256"],
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": task.get("teacher_correct_states"),
        "teacher_total_states": task.get("teacher_total_states"),
    },))
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 0

    assert holder._acceptance_queue.empty()
    assert not Path(holder.config.acceptance_dir).exists()
    assert "checkpoint_path must be" in capsys.readouterr().out


def test_registry_recovery_rejects_coerced_checkpoint_digest(
    tmp_path: Path,
    capsys,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    holder._promotion_registry = SimpleNamespace(records=lambda: ({
        "promoted": True,
        "step": task["step"],
        "teacher_agreement": task["teacher_agreement"],
        "training_stage": task["training_stage"],
        "checkpoint_path": task["checkpoint_path"],
        "checkpoint_sha256": int("1" * 64),
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": task.get("teacher_correct_states"),
        "teacher_total_states": task.get("teacher_total_states"),
    },))
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 0

    assert holder._acceptance_queue.empty()
    assert not Path(holder.config.acceptance_dir).exists()
    assert "checkpoint_sha256 must be 64 hexadecimal" in capsys.readouterr().out


@pytest.mark.parametrize("teacher_agreement", (
    float("nan"),
    float("inf"),
    float("-inf"),
    -0.01,
    1.01,
))
def test_pending_task_rejects_invalid_teacher_agreement(
    tmp_path: Path,
    teacher_agreement: float,
) -> None:
    task = _make_task(tmp_path)
    task["teacher_agreement"] = teacher_agreement

    with pytest.raises(
        ValueError,
        match="teacher_agreement must be finite and within",
    ):
        checkpoint_acceptance.persist_pending_acceptance_task(tmp_path, task)


@pytest.mark.parametrize(("teacher_agreement", "correct_states"), (
    ("0.55", 2750),
    (True, 5000),
    (False, 0),
))
def test_pending_task_rejects_coerced_teacher_agreement(
    tmp_path: Path,
    teacher_agreement,
    correct_states: int,
) -> None:
    task = _make_task(tmp_path)
    task["teacher_agreement"] = teacher_agreement
    task["teacher_correct_states"] = correct_states

    with pytest.raises(ValueError, match="must be a real number"):
        checkpoint_acceptance.persist_pending_acceptance_task(tmp_path, task)


@pytest.mark.parametrize(("teacher_agreement", "correct_states"), (
    ("0.55", 2750),
    (True, 5000),
    (False, 0),
))
def test_registry_recovery_rejects_coerced_promotion_agreement(
    tmp_path: Path,
    capsys,
    teacher_agreement,
    correct_states: int,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    holder._promotion_registry = SimpleNamespace(records=lambda: ({
        "promoted": True,
        "step": task["step"],
        "teacher_agreement": teacher_agreement,
        "training_stage": task["training_stage"],
        "checkpoint_path": task["checkpoint_path"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": correct_states,
        "teacher_total_states": 5000,
    },))
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 0

    assert holder._acceptance_queue.empty()
    assert not Path(holder.config.acceptance_dir).exists()
    assert "teacher_agreement must be a real number" in capsys.readouterr().out


@pytest.mark.parametrize(("correct_states", "total_states", "message"), (
    (None, 5000, "no held-out teacher-agreement counts"),
    (True, 5000, "counts must be integers"),
    (2750.5, 5000, "counts must be integers"),
    (55, 100, "not the required 5000"),
    (2751, 5000, "quotient of its recorded counts"),
))
def test_pending_task_rejects_invalid_teacher_agreement_counts(
    tmp_path: Path,
    correct_states,
    total_states,
    message: str,
) -> None:
    task = _make_task(tmp_path)
    task["teacher_correct_states"] = correct_states
    task["teacher_total_states"] = total_states

    with pytest.raises(ValueError, match=message):
        checkpoint_acceptance.persist_pending_acceptance_task(tmp_path, task)


def test_startup_finalizes_completed_pending_without_requeue(tmp_path: Path) -> None:
    holder = _holder(tmp_path)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    _write_success_report(
        Path(holder.config.acceptance_dir), task, passed=True)
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 1
    assert not pending_path.exists()
    assert holder._acceptance_queue.empty()
    assert holder.stats.acceptance_history[-1]["task_id"] == task["task_id"]


def test_startup_repairs_completed_acceptance_without_pending_marker(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias
    task = _make_task(tmp_path)
    report_path = _write_success_report(
        Path(holder.config.acceptance_dir), task, passed=True)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    promotion = {
        "promoted": True,
        "step": task["step"],
        "teacher_agreement": task["teacher_agreement"],
        "training_stage": task["training_stage"],
        "checkpoint_path": task["checkpoint_path"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": task.get("teacher_correct_states"),
        "teacher_total_states": task.get("teacher_total_states"),
    }
    holder._promotion_registry = SimpleNamespace(records=lambda: (promotion,))

    assert holder._recover_pending_checkpoint_acceptance() == 1

    accepted = Path(holder.config.accepted_path)
    assert accepted.read_bytes() == Path(task["checkpoint_path"]).read_bytes()
    assert not list(
        Path(holder.config.acceptance_dir).glob("pending_acceptance_*.json"))
    assert holder.stats.acceptance_history[-1]["task_id"] == task["task_id"]
    assert holder.stats.acceptance_history[-1]["report_path"] == str(report_path)

    stale = accepted.with_name("stale-accepted.pt")
    stale.write_bytes(b"stale accepted checkpoint")
    os.replace(stale, accepted)
    assert holder._recover_pending_checkpoint_acceptance() == 1
    assert accepted.read_bytes() == Path(task["checkpoint_path"]).read_bytes()
    assert len(holder.stats.acceptance_history) == 1

    assert holder._recover_pending_checkpoint_acceptance() == 0
    assert len(holder.stats.acceptance_history) == 1


def test_startup_restores_every_passing_history_without_pending_markers(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias
    tasks = [_make_task(tmp_path, step) for step in (140000, 142000)]
    promotions = []
    for task in tasks:
        _write_success_report(
            Path(holder.config.acceptance_dir), task, passed=True)
        promotions.append({
            "promoted": True,
            "step": task["step"],
            "teacher_agreement": task["teacher_agreement"],
            "training_stage": task["training_stage"],
            "checkpoint_path": task["checkpoint_path"],
            "checkpoint_sha256": task["checkpoint_sha256"],
            "suite_fingerprint": task.get("suite_fingerprint"),
            "teacher_correct_states": task.get("teacher_correct_states"),
            "teacher_total_states": task.get("teacher_total_states"),
        })

    exemplar = tasks[0]
    holder.config.test_opening_plies = exemplar["opening_plies"]
    holder.config.test_opening_seed = exemplar["opening_seed"]
    holder.config.inference_depth = exemplar["inference_depth"]
    holder.config.selfplay_max_moves = exemplar["max_moves"]
    holder.config.cpu_workers = exemplar["num_workers"]
    holder.config.policy_stage = exemplar["training_stage"]
    holder._promotion_registry = SimpleNamespace(
        records=lambda: tuple(promotions))

    assert holder._recover_pending_checkpoint_acceptance() == 2
    assert [
        record["step"] for record in holder.stats.acceptance_history
    ] == [140000, 142000]
    assert Path(holder.config.accepted_path).read_bytes() == Path(
        tasks[-1]["checkpoint_path"]).read_bytes()

    # A partial statistics rollback must restore the older row in order, not
    # append it after the newer verdict consumed by status and UI callers.
    holder.stats.acceptance_history.pop(0)
    assert holder._recover_pending_checkpoint_acceptance() == 1
    assert [
        record["step"] for record in holder.stats.acceptance_history
    ] == [140000, 142000]

    assert holder._recover_pending_checkpoint_acceptance() == 0
    assert len(holder.stats.acceptance_history) == 2


@pytest.mark.parametrize("terminal_kind", ("gate_failure", "evaluator_failure"))
def test_startup_restores_failed_terminal_history_without_pending_marker(
    tmp_path: Path,
    terminal_kind: str,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    holder.config.test_opening_plies = task["opening_plies"]
    holder.config.test_opening_seed = task["opening_seed"]
    holder.config.inference_depth = task["inference_depth"]
    holder.config.selfplay_max_moves = task["max_moves"]
    holder.config.cpu_workers = task["num_workers"]
    holder.config.policy_stage = task["training_stage"]
    promotion = {
        "promoted": True,
        "step": task["step"],
        "teacher_agreement": task["teacher_agreement"],
        "training_stage": task["training_stage"],
        "checkpoint_path": task["checkpoint_path"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "suite_fingerprint": task.get("suite_fingerprint"),
        "teacher_correct_states": task.get("teacher_correct_states"),
        "teacher_total_states": task.get("teacher_total_states"),
    }
    holder._promotion_registry = SimpleNamespace(records=lambda: (promotion,))

    if terminal_kind == "gate_failure":
        report_path = _write_success_report(
            Path(holder.config.acceptance_dir), task, passed=False)
    else:
        report_path = checkpoint_acceptance.write_acceptance_failure_report(
            holder.config.acceptance_dir,
            task,
            RuntimeError("evaluator process failed"),
        )

    assert holder._recover_pending_checkpoint_acceptance() == 1
    assert holder._acceptance_queue.empty()
    assert not list(
        Path(holder.config.acceptance_dir).glob("pending_acceptance_*.json"))
    assert len(holder.stats.acceptance_history) == 1
    assert holder.stats.acceptance_history[0]["passed"] is False
    assert holder.stats.acceptance_history[0]["report_path"] == str(report_path)

    assert holder._recover_pending_checkpoint_acceptance() == 0
    assert len(holder.stats.acceptance_history) == 1


def test_stale_history_cannot_suppress_replacement_protocol(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    old_task = _make_task(tmp_path)
    old_task["opening_plies"] = [2, 4]
    replacement = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, replacement)
    old_report_path = _write_success_report(
        Path(holder.config.acceptance_dir), old_task, passed=False)
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    old_report["report_path"] = str(old_report_path)
    holder.stats.acceptance_history.append(old_report)
    _write_success_report(
        Path(holder.config.acceptance_dir), replacement, passed=True)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias

    assert holder._recover_pending_checkpoint_acceptance() == 1

    accepted = Path(holder.config.accepted_path)
    assert accepted.read_bytes() == Path(replacement["checkpoint_path"]).read_bytes()
    assert not pending_path.exists()
    assert len(holder.stats.acceptance_history) == 2
    assert holder.stats.acceptance_history[-1]["opening_plies"] == [2, 4, 6, 8]
    assert holder.stats.acceptance_history[-1]["passed"] is True


def test_exact_history_retry_remains_idempotent(tmp_path: Path) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    report_path = _write_success_report(
        Path(holder.config.acceptance_dir), task, passed=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report_path"] = str(report_path)
    holder.stats.acceptance_history.append(report)

    assert holder._recover_pending_checkpoint_acceptance() == 1

    assert not pending_path.exists()
    assert holder.stats.acceptance_history == [report]


def test_exact_passing_history_retry_repairs_accepted_alias(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    report_path = _write_success_report(
        Path(holder.config.acceptance_dir), task, passed=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report_path"] = str(report_path)
    holder.stats.acceptance_history.append(report)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias

    assert holder._recover_pending_checkpoint_acceptance() == 1

    accepted = Path(holder.config.accepted_path)
    assert accepted.read_bytes() == Path(task["checkpoint_path"]).read_bytes()
    assert not pending_path.exists()
    assert holder.stats.acceptance_history == [report]


def test_startup_repairs_accepted_alias_before_clearing_pending(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    _write_success_report(
        Path(holder.config.acceptance_dir), task, passed=True)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias

    assert holder._recover_pending_checkpoint_acceptance() == 1

    accepted = Path(holder.config.accepted_path)
    assert accepted.read_bytes() == Path(task["checkpoint_path"]).read_bytes()
    assert not pending_path.exists()


def test_startup_keeps_pending_when_accepted_alias_repair_fails(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    _write_success_report(
        Path(holder.config.acceptance_dir), task, passed=True)

    def fail_alias(_source: Path, _destination: Path) -> None:
        raise OSError("accepted alias unavailable")

    holder._publish_checkpoint_alias = fail_alias

    assert holder._recover_pending_checkpoint_acceptance() == 0
    assert pending_path.exists()
    assert holder.stats.acceptance_history == []


def test_startup_records_durable_evaluator_failure_before_cleanup(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    failure_path = checkpoint_acceptance.write_acceptance_failure_report(
        holder.config.acceptance_dir,
        task,
        RuntimeError("evaluator process failed"),
    )

    assert holder._recover_pending_checkpoint_acceptance() == 1

    assert not pending_path.exists()
    assert holder._acceptance_queue.empty()
    assert holder.stats.acceptance_history[-1]["task_id"] == task["task_id"]
    assert holder.stats.acceptance_history[-1]["passed"] is False
    assert holder.stats.acceptance_history[-1]["status"] == "error"
    assert holder.stats.acceptance_history[-1]["report_path"] == str(failure_path)


def test_startup_keeps_pending_when_evaluator_failure_finalization_fails(
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    checkpoint_acceptance.write_acceptance_failure_report(
        holder.config.acceptance_dir,
        task,
        RuntimeError("evaluator process failed"),
    )

    def fail_finalization(*_args) -> None:
        raise OSError("statistics unavailable")

    holder._finalize_checkpoint_acceptance_report = fail_finalization

    assert holder._recover_pending_checkpoint_acceptance() == 0
    assert pending_path.exists()


def test_success_removes_pending_only_after_durable_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)

    def successful_run(checkpoint_path: str, **kwargs) -> dict:
        assert checkpoint_path == task["checkpoint_path"]
        report_path = _write_success_report(
            Path(kwargs["output_dir"]), task, passed=False)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return {**report, "report_path": str(report_path)}

    monkeypatch.setattr(
        checkpoint_acceptance, "run_checkpoint_acceptance", successful_run)

    holder._process_checkpoint_acceptance_task(task)

    assert not pending_path.exists()
    assert holder.stats.acceptance_history[-1]["step"] == task["step"]
    assert checkpoint_acceptance.successful_acceptance_report_path(
        holder.config.acceptance_dir, task) is not None


def test_evaluator_return_cannot_override_durable_acceptance_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)

    def contradictory_run(*_args, **_kwargs) -> dict:
        report_path = _write_success_report(
            Path(holder.config.acceptance_dir), task, passed=False)
        durable_report = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            **durable_report,
            "passed": True,
            "report_path": str(report_path),
        }

    monkeypatch.setattr(
        checkpoint_acceptance, "run_checkpoint_acceptance", contradictory_run)

    holder._process_checkpoint_acceptance_task(task)

    assert not pending_path.exists()
    assert not Path(holder.config.accepted_path).exists()
    assert holder.stats.acceptance_history[-1]["passed"] is False


def test_passed_report_keeps_pending_until_accepted_alias_is_published(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)

    def successful_run(*_args, **_kwargs) -> dict:
        report_path = _write_success_report(
            Path(holder.config.acceptance_dir), task, passed=True)
        return {
            **json.loads(report_path.read_text(encoding="utf-8")),
            "report_path": str(report_path),
        }

    def fail_alias(_source: Path, _destination: Path) -> None:
        raise OSError("accepted alias unavailable")

    monkeypatch.setattr(
        checkpoint_acceptance, "run_checkpoint_acceptance", successful_run)
    holder._publish_checkpoint_alias = fail_alias

    holder._process_checkpoint_acceptance_task(task)

    assert pending_path.exists()
    assert holder.stats.acceptance_history == []
    assert not checkpoint_acceptance.failure_acceptance_report_path(
        holder.config.acceptance_dir, task).exists()


def test_checkpoint_replaced_during_evaluation_cannot_publish_accepted_alias(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias
    task = _make_task(tmp_path)
    accepted_path = Path(holder.config.accepted_path)
    accepted_path.write_bytes(b"previous accepted checkpoint")
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)

    def replace_checkpoint_after_report(*_args, **_kwargs) -> dict:
        report_path = _write_success_report(
            Path(holder.config.acceptance_dir), task, passed=True)
        Path(task["checkpoint_path"]).write_bytes(b"replaced checkpoint")
        return json.loads(report_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        checkpoint_acceptance,
        "run_checkpoint_acceptance",
        replace_checkpoint_after_report,
    )

    holder._process_checkpoint_acceptance_task(task)

    assert pending_path.exists()
    assert accepted_path.read_bytes() == b"previous accepted checkpoint"
    assert holder.stats.acceptance_history == []
    assert not checkpoint_acceptance.failure_acceptance_report_path(
        holder.config.acceptance_dir, task).exists()


def test_checkpoint_replaced_after_final_hash_publishes_verified_inode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias
    task = _make_task(tmp_path)
    checkpoint = Path(task["checkpoint_path"])
    original_bytes = checkpoint.read_bytes()
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    _write_success_report(
        Path(holder.config.acceptance_dir), task, passed=True)
    original_verify = checkpoint_acceptance.verify_pending_acceptance_checkpoint

    def replace_source_after_verify(candidate: dict) -> str | None:
        digest = original_verify(candidate)
        replacement = checkpoint.with_name(checkpoint.name + ".replacement")
        replacement.write_bytes(b"replacement checkpoint")
        os.replace(replacement, checkpoint)
        return digest

    monkeypatch.setattr(
        checkpoint_acceptance,
        "verify_pending_acceptance_checkpoint",
        replace_source_after_verify,
    )

    assert holder._recover_pending_checkpoint_acceptance() == 1

    assert checkpoint.read_bytes() == b"replacement checkpoint"
    assert Path(holder.config.accepted_path).read_bytes() == original_bytes
    assert not pending_path.exists()
    assert holder.stats.acceptance_history[-1]["task_id"] == task["task_id"]


def test_verified_accepted_alias_commits_final_public_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The verified staging rename and final accepted rename are both committed."""
    holder = _holder(tmp_path)
    holder._publish_checkpoint_alias = Trainer._publish_checkpoint_alias
    task = _make_task(tmp_path)
    destination = Path(holder.config.accepted_path)
    events = []
    real_sync = trainer_module._fsync_directory

    def tracking_sync(path: Path) -> None:
        events.append(Path(path))
        real_sync(Path(path))

    monkeypatch.setattr(trainer_module, "_fsync_directory", tracking_sync)

    holder._publish_verified_checkpoint_alias(
        Path(task["checkpoint_path"]),
        destination,
        task["checkpoint_sha256"],
    )

    assert events == [destination.parent, destination.parent]
    assert destination.read_bytes() == Path(task["checkpoint_path"]).read_bytes()


def test_unreported_failure_keeps_pending_for_next_startup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)

    def failed_run(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    def failed_report(*args, **kwargs):
        raise OSError("report disk unavailable")

    monkeypatch.setattr(
        checkpoint_acceptance, "run_checkpoint_acceptance", failed_run)
    monkeypatch.setattr(
        checkpoint_acceptance, "write_acceptance_failure_report", failed_report)

    holder._process_checkpoint_acceptance_task(task)

    assert pending_path.exists()
    assert checkpoint_acceptance.terminal_acceptance_report_path(
        holder.config.acceptance_dir, task) is None


def test_written_failure_allows_pending_cleanup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)

    def failed_run(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(
        checkpoint_acceptance, "run_checkpoint_acceptance", failed_run)

    holder._process_checkpoint_acceptance_task(task)

    failure_path = checkpoint_acceptance.failure_acceptance_report_path(
        holder.config.acceptance_dir, task)
    assert failure_path.exists()
    assert not pending_path.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["error"] == "evaluation failed"
    assert holder.stats.acceptance_history[-1]["status"] == "error"


def test_written_failure_keeps_pending_until_stats_are_durable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    evaluator_calls = 0

    def failed_run(*_args, **_kwargs):
        nonlocal evaluator_calls
        evaluator_calls += 1
        raise RuntimeError("evaluation failed")

    def failed_stats_save(**kwargs) -> bool:
        assert kwargs == {"_raise_on_error": True}
        raise OSError("statistics unavailable")

    monkeypatch.setattr(
        checkpoint_acceptance, "run_checkpoint_acceptance", failed_run)
    holder._save_stats = failed_stats_save

    holder._process_checkpoint_acceptance_task(task)

    failure_path = checkpoint_acceptance.failure_acceptance_report_path(
        holder.config.acceptance_dir, task)
    assert evaluator_calls == 1
    assert failure_path.exists()
    assert pending_path.exists()

    restarted = _holder(tmp_path)
    assert restarted._recover_pending_checkpoint_acceptance() == 1
    assert evaluator_calls == 1
    assert not pending_path.exists()
    assert restarted.stats.acceptance_history[-1]["task_id"] == task["task_id"]
    assert restarted.stats.acceptance_history[-1]["status"] == "error"


def test_stats_writer_can_surface_a_durable_finalization_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(stats_file=str(tmp_path / "stats.json"))
    holder.stats = TrainingStats()
    holder.step = 140000
    holder._stats_write_lock = threading.RLock()
    holder._stats_snapshot_generation = 0
    holder._stats_persisted_generation = -1

    def failed_temporary_file(*_args, **_kwargs):
        raise OSError("statistics unavailable")

    monkeypatch.setattr(
        "dama.ai.ml.trainer.tempfile.NamedTemporaryFile",
        failed_temporary_file,
    )

    assert holder._save_stats() is False
    with pytest.raises(OSError, match="statistics unavailable"):
        holder._save_stats(_raise_on_error=True)


def test_checkpoint_hash_mismatch_writes_failure_without_running_evaluator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    Path(task["checkpoint_path"]).write_bytes(b"tampered checkpoint")
    worker_pool_called = False

    def unexpected_worker_pool(**_kwargs):
        nonlocal worker_pool_called
        worker_pool_called = True
        raise AssertionError(
            "worker pool must not run after a digest mismatch")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)

    holder._process_checkpoint_acceptance_task(task)

    failure_path = checkpoint_acceptance.failure_acceptance_report_path(
        holder.config.acceptance_dir, task)
    assert worker_pool_called is False
    assert failure_path.exists()
    assert not pending_path.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert "SHA-256 mismatch" in failure["error"]


def test_manual_stop_does_not_discard_pending_work(tmp_path: Path) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    holder._ensure_checkpoint_acceptance_worker = lambda: None
    holder._queue_checkpoint_acceptance_task(task, persist=False)

    Trainer.stop(holder)

    assert holder._stopped is True
    assert pending_path.exists()
    assert holder._acceptance_queue.qsize() == 1
