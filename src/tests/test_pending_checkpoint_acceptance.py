"""Durability tests for queued checkpoint acceptance work."""

import hashlib
import json
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

from dama.ai.ml import checkpoint_acceptance
from dama.ai.ml.trainer import Trainer, TrainingStats


def _make_task(tmp_path: Path, step: int = 140000) -> dict:
    checkpoint = tmp_path / f"model_step_{step:06d}.pt"
    checkpoint.write_bytes(b"checkpoint")
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
    holder._save_stats = lambda: None
    holder._put_status = lambda payload: None
    holder._publish_checkpoint_alias = lambda source, destination: None
    return holder


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
        "opening_seed": task["opening_seed"],
        "opening_plies": [2, 4],  # stale protocol, task uses four openings
    }), encoding="utf-8")
    assert checkpoint_acceptance.successful_acceptance_report_path(
        output_dir, task) is None


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


def test_startup_cleans_completed_pending_without_requeue(tmp_path: Path) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    report_path = (
        Path(holder.config.acceptance_dir)
        / f"acceptance_step_{task['step']:06d}.json"
    )
    report_path.write_text(json.dumps({
        "checkpoint_path": task["checkpoint_path"],
        "step": task["step"],
        "passed": True,
        "task_id": task["task_id"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "frozen_suite_fingerprint": task.get("suite_fingerprint"),
        "opening_seed": task["opening_seed"],
        "opening_plies": task["opening_plies"],
        "inference_depth": task["inference_depth"],
        "max_moves": task["max_moves"],
        "num_workers": task["num_workers"],
        "training_stage": task["training_stage"],
    }), encoding="utf-8")
    holder._ensure_checkpoint_acceptance_worker = lambda: None

    assert holder._recover_pending_checkpoint_acceptance() == 0
    assert not pending_path.exists()
    assert holder._acceptance_queue.empty()


def test_success_removes_pending_only_after_durable_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)

    def successful_run(checkpoint_path: str, **kwargs) -> dict:
        report = {
            "checkpoint_path": checkpoint_path,
            "step": kwargs["step"],
            "passed": False,
            "task_id": task["task_id"],
            "checkpoint_sha256": task["checkpoint_sha256"],
            "frozen_suite_fingerprint": task.get("suite_fingerprint"),
            "opening_seed": task["opening_seed"],
            "opening_plies": task["opening_plies"],
            "inference_depth": task["inference_depth"],
            "max_moves": task["max_moves"],
            "num_workers": task["num_workers"],
            "training_stage": task["training_stage"],
        }
        report_path = (
            Path(kwargs["output_dir"])
            / f"acceptance_step_{kwargs['step']:06d}.json"
        )
        checkpoint_acceptance._write_json_atomic(report_path, report)
        return {**report, "report_path": str(report_path)}

    monkeypatch.setattr(
        checkpoint_acceptance, "run_checkpoint_acceptance", successful_run)

    holder._process_checkpoint_acceptance_task(task)

    assert not pending_path.exists()
    assert holder.stats.acceptance_history[-1]["step"] == task["step"]
    assert checkpoint_acceptance.successful_acceptance_report_path(
        holder.config.acceptance_dir, task) is not None


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


def test_checkpoint_hash_mismatch_writes_failure_without_running_evaluator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    holder = _holder(tmp_path)
    task = _make_task(tmp_path)
    pending_path = checkpoint_acceptance.persist_pending_acceptance_task(
        holder.config.acceptance_dir, task)
    Path(task["checkpoint_path"]).write_bytes(b"tampered checkpoint")
    evaluator_called = False

    def unexpected_run(*args, **kwargs):
        nonlocal evaluator_called
        evaluator_called = True
        raise AssertionError("evaluator must not run after a digest mismatch")

    monkeypatch.setattr(
        checkpoint_acceptance, "run_checkpoint_acceptance", unexpected_run)

    holder._process_checkpoint_acceptance_task(task)

    failure_path = checkpoint_acceptance.failure_acceptance_report_path(
        holder.config.acceptance_dir, task)
    assert evaluator_called is False
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
