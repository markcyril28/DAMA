import json
from pathlib import Path

import pytest

from dama.ai.ml import checkpoint_acceptance


_CHECKPOINT_SHA256 = "A" * 64
_SUITE_FINGERPRINT = "b" * 64


class _Stats:
    def __init__(self, opponent: str) -> None:
        if opponent == "random":
            self.payload = {
                "ml_wins": 90,
                "draws": 10,
                "opponent_wins": 0,
                "ml_as_p1_wins": 45,
                "ml_as_p1_draws": 5,
                "ml_as_p1_losses": 0,
                "ml_as_p2_wins": 45,
                "ml_as_p2_draws": 5,
                "ml_as_p2_losses": 0,
            }
        else:
            self.payload = {
                "ml_wins": 70,
                "draws": 0,
                "opponent_wins": 30,
                "ml_as_p1_wins": 35,
                "ml_as_p1_draws": 0,
                "ml_as_p1_losses": 15,
                "ml_as_p2_wins": 35,
                "ml_as_p2_draws": 0,
                "ml_as_p2_losses": 15,
            }
        self.payload["opening_suite_id"] = "fixed-suite"

    def to_dict(self) -> dict:
        return dict(self.payload)


class _Tester:
    calls = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def run_tests(self, num_games: int):
        assert num_games == 100
        return _Stats(self.kwargs["opponent_type"])


def test_promoted_checkpoint_runs_fixed_random_then_easy_protocol(
    monkeypatch, tmp_path: Path
) -> None:
    class _Executor:
        instances = []

        def __init__(self, *, max_workers, mp_context, initializer) -> None:
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.initializer = initializer
            self.entered = False
            self.exited = False
            self.instances.append(self)

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *_exc) -> None:
            self.exited = True

    _Tester.calls = []
    worker_context = object()
    monkeypatch.setattr(checkpoint_acceptance, "ModelVsAlgoTester", _Tester)
    monkeypatch.setattr(checkpoint_acceptance, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(
        checkpoint_acceptance, "_evaluation_worker_context",
        lambda: worker_context,
    )
    checkpoint_path = str(tmp_path / "model_step_136000.pt")
    task_id = checkpoint_acceptance.acceptance_task_id(
        checkpoint_path, 136000)
    report = checkpoint_acceptance.run_checkpoint_acceptance(
        checkpoint_path,
        step=136000,
        teacher_agreement=0.55,
        opening_plies=(2, 4, 6, 8),
        opening_seed=20260819,
        inference_depth=1,
        max_moves=200,
        num_workers=2,
        output_dir=str(tmp_path / "reports"),
        training_stage="policy_only",
        task_id=task_id,
        checkpoint_sha256=_CHECKPOINT_SHA256,
        suite_fingerprint=_SUITE_FINGERPRINT,
        teacher_correct_states=2750,
        teacher_total_states=5000,
    )

    assert [call["opponent_type"] for call in _Tester.calls] == ["random", "algorithm"]
    assert all(call["opening_seed"] == 20260819 for call in _Tester.calls)
    assert len(_Executor.instances) == 1
    executor = _Executor.instances[0]
    assert executor.max_workers == 2
    assert executor.mp_context is worker_context
    assert executor.initializer is checkpoint_acceptance._evaluation_worker_init
    assert executor.entered and executor.exited
    assert all(call["executor"] is executor for call in _Tester.calls)
    assert report["passed"] is True
    assert report["task_id"] == task_id
    assert report["opening_suite_id"] == "fixed-suite"
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["checks"]["random_exact_balanced_100_games"] is True
    assert saved["checks"]["easy_exact_balanced_100_games"] is True


@pytest.mark.parametrize(
    "opening_plies",
    (
        (0, 2),
        (True, 2),
        (2.5, 4),
        ("2", 4),
    ),
)
def test_acceptance_task_rejects_invalid_opening_plies(opening_plies) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        checkpoint_acceptance.make_pending_acceptance_task(
            "checkpoint.pt", step=1, teacher_agreement=0.55,
            opening_plies=opening_plies, opening_seed=1, inference_depth=1,
            max_moves=200, num_workers=1, training_stage="policy_only",
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT)


@pytest.mark.parametrize("checkpoint_path", (True, 1, ["checkpoint.pt"], ""))
def test_acceptance_task_rejects_invalid_checkpoint_path(checkpoint_path) -> None:
    with pytest.raises(ValueError, match="checkpoint_path must be"):
        checkpoint_acceptance.make_pending_acceptance_task(
            checkpoint_path, step=1, teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8), opening_seed=1,
            inference_depth=1, max_moves=200, num_workers=1,
            training_stage="policy_only", teacher_correct_states=2750,
            teacher_total_states=5000,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT)


@pytest.mark.parametrize("checkpoint_path", (True, 1, ["checkpoint.pt"], ""))
def test_direct_acceptance_rejects_invalid_checkpoint_before_evaluation(
    monkeypatch,
    tmp_path: Path,
    checkpoint_path,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid checkpoint path reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="checkpoint_path must be"):
        checkpoint_acceptance.run_checkpoint_acceptance(
            checkpoint_path,
            step=136000,
            teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8),
            opening_seed=20260819,
            inference_depth=1,
            max_moves=200,
            num_workers=2,
            output_dir=str(output_dir),
            training_stage="policy_only",
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT,
            teacher_correct_states=2750,
            teacher_total_states=5000,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("task_id", (True, 136000, "wrong-task-id"))
def test_direct_acceptance_rejects_invalid_task_id_before_evaluation(
    monkeypatch,
    tmp_path: Path,
    task_id,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid task id reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="task_id does not match"):
        checkpoint_acceptance.run_checkpoint_acceptance(
            str(tmp_path / "model_step_136000.pt"),
            step=136000,
            teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8),
            opening_seed=20260819,
            inference_depth=1,
            max_moves=200,
            num_workers=2,
            output_dir=str(output_dir),
            training_stage="policy_only",
            task_id=task_id,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT,
            teacher_correct_states=2750,
            teacher_total_states=5000,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "opening_plies",
    (
        (0, 2, 4, 6, 8),
        (True, 2, 4, 6, 8),
        (2.5, 4, 6, 8),
        ("2", 4, 6, 8),
    ),
)
def test_direct_acceptance_rejects_invalid_opening_before_evaluation(
    monkeypatch,
    tmp_path: Path,
    opening_plies,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid opening schedule reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"

    with pytest.raises(
        ValueError,
        match="opening plies must be non-empty and positive integers",
    ):
        checkpoint_acceptance.run_checkpoint_acceptance(
            str(tmp_path / "model_step_136000.pt"),
            step=136000,
            teacher_agreement=0.55,
            opening_plies=opening_plies,
            opening_seed=20260819,
            inference_depth=1,
            max_moves=200,
            num_workers=2,
            output_dir=str(output_dir),
            training_stage="policy_only",
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT,
            teacher_correct_states=2750,
            teacher_total_states=5000,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("opening_seed", (True, 20260819.5, "20260819"))
def test_acceptance_task_rejects_coerced_opening_seed(opening_seed) -> None:
    with pytest.raises(ValueError, match="opening_seed must be an integer"):
        checkpoint_acceptance.make_pending_acceptance_task(
            "checkpoint.pt", step=1, teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8), opening_seed=opening_seed,
            inference_depth=1, max_moves=200, num_workers=1,
            training_stage="policy_only", teacher_correct_states=2750,
            teacher_total_states=5000,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT)


@pytest.mark.parametrize("opening_seed", (True, 20260819.5, "20260819"))
def test_direct_acceptance_rejects_coerced_opening_seed_before_evaluation(
    monkeypatch,
    tmp_path: Path,
    opening_seed,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid opening seed reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="opening_seed must be an integer"):
        checkpoint_acceptance.run_checkpoint_acceptance(
            str(tmp_path / "model_step_136000.pt"),
            step=136000,
            teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8),
            opening_seed=opening_seed,
            inference_depth=1,
            max_moves=200,
            num_workers=2,
            output_dir=str(output_dir),
            training_stage="policy_only",
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT,
            teacher_correct_states=2750,
            teacher_total_states=5000,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("num_workers", (True, False, "2", 2.5, 0, -1))
def test_acceptance_task_rejects_invalid_num_workers(num_workers) -> None:
    with pytest.raises(ValueError, match="num_workers must be a positive integer"):
        checkpoint_acceptance.make_pending_acceptance_task(
            "checkpoint.pt", step=1, teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8), opening_seed=20260819,
            inference_depth=1, max_moves=200, num_workers=num_workers,
            training_stage="policy_only", teacher_correct_states=2750,
            teacher_total_states=5000,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT)


@pytest.mark.parametrize("num_workers", (True, False, "2", 2.5, 0, -1))
def test_direct_acceptance_rejects_invalid_num_workers_before_evaluation(
    monkeypatch,
    tmp_path: Path,
    num_workers,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid worker count reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="num_workers must be a positive integer"):
        checkpoint_acceptance.run_checkpoint_acceptance(
            str(tmp_path / "model_step_136000.pt"),
            step=136000,
            teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8),
            opening_seed=20260819,
            inference_depth=1,
            max_moves=200,
            num_workers=num_workers,
            output_dir=str(output_dir),
            training_stage="policy_only",
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT,
            teacher_correct_states=2750,
            teacher_total_states=5000,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "training_stage", (True, 1, ["policy_only"], "unknown", ""))
def test_acceptance_task_rejects_invalid_training_stage(training_stage) -> None:
    with pytest.raises(ValueError, match="training_stage must be"):
        checkpoint_acceptance.make_pending_acceptance_task(
            "checkpoint.pt", step=1, teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8), opening_seed=20260819,
            inference_depth=1, max_moves=200, num_workers=1,
            training_stage=training_stage, teacher_correct_states=2750,
            teacher_total_states=5000,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT)


@pytest.mark.parametrize(
    "training_stage", (True, 1, ["policy_only"], "unknown", ""))
def test_direct_acceptance_rejects_invalid_training_stage_before_evaluation(
    monkeypatch,
    tmp_path: Path,
    training_stage,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid training stage reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="training_stage must be"):
        checkpoint_acceptance.run_checkpoint_acceptance(
            str(tmp_path / "model_step_136000.pt"),
            step=136000,
            teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8),
            opening_seed=20260819,
            inference_depth=1,
            max_moves=200,
            num_workers=2,
            output_dir=str(output_dir),
            training_stage=training_stage,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT,
            teacher_correct_states=2750,
            teacher_total_states=5000,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("step", (True, "136000", 136000.5, -1))
def test_acceptance_task_rejects_invalid_step(step) -> None:
    with pytest.raises(ValueError, match="step must be a non-negative integer"):
        checkpoint_acceptance.make_pending_acceptance_task(
            "checkpoint.pt", step=step, teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8), opening_seed=20260819,
            inference_depth=1, max_moves=200, num_workers=1,
            training_stage="policy_only", teacher_correct_states=2750,
            teacher_total_states=5000,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT)


@pytest.mark.parametrize("schema_version", (True, 1.0, "1", 2))
def test_persisted_task_rejects_invalid_schema_version(
    tmp_path: Path,
    schema_version,
) -> None:
    task = checkpoint_acceptance.make_pending_acceptance_task(
        "checkpoint.pt", step=1, teacher_agreement=0.55,
        opening_plies=(2, 4, 6, 8), opening_seed=20260819,
        inference_depth=1, max_moves=200, num_workers=1,
        training_stage="policy_only", teacher_correct_states=2750,
        teacher_total_states=5000,
        checkpoint_sha256=_CHECKPOINT_SHA256,
        suite_fingerprint=_SUITE_FINGERPRINT)
    task["schema_version"] = schema_version

    with pytest.raises(
        ValueError,
        match="Unsupported pending acceptance task schema_version",
    ):
        checkpoint_acceptance.persist_pending_acceptance_task(tmp_path, task)


@pytest.mark.parametrize("step", (True, "136000", 136000.5, -1))
def test_direct_acceptance_rejects_invalid_step_before_evaluation(
    monkeypatch,
    tmp_path: Path,
    step,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid checkpoint step reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="step must be a non-negative integer"):
        checkpoint_acceptance.run_checkpoint_acceptance(
            str(tmp_path / "model_step_136000.pt"),
            step=step,
            teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8),
            opening_seed=20260819,
            inference_depth=1,
            max_moves=200,
            num_workers=2,
            output_dir=str(output_dir),
            training_stage="policy_only",
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT,
            teacher_correct_states=2750,
            teacher_total_states=5000,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("inference_depth", 0, "inference depth must be one of"),
        ("inference_depth", 4, "inference depth must be one of"),
        ("inference_depth", True, "inference depth must be one of"),
        ("max_moves", 0, "max_moves must be a positive integer"),
        ("max_moves", -1, "max_moves must be a positive integer"),
        ("max_moves", True, "max_moves must be a positive integer"),
    ),
)
def test_acceptance_task_rejects_invalid_game_bounds(
    field: str,
    value,
    message: str,
) -> None:
    arguments = {
        "checkpoint_path": "checkpoint.pt",
        "step": 1,
        "teacher_agreement": 0.55,
        "opening_plies": (2, 4, 6, 8),
        "opening_seed": 1,
        "inference_depth": 1,
        "max_moves": 200,
        "num_workers": 1,
        "training_stage": "policy_only",
        "checkpoint_sha256": _CHECKPOINT_SHA256,
        "suite_fingerprint": _SUITE_FINGERPRINT,
        "teacher_correct_states": 2750,
        "teacher_total_states": 5000,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        checkpoint_acceptance.make_pending_acceptance_task(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("inference_depth", 0, "inference depth must be one of"),
        ("max_moves", 0, "max_moves must be a positive integer"),
    ),
)
def test_direct_acceptance_rejects_invalid_game_bounds_before_evaluation(
    monkeypatch,
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid game bounds reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"
    arguments = {
        "checkpoint_path": str(tmp_path / "model_step_136000.pt"),
        "step": 136000,
        "teacher_agreement": 0.55,
        "opening_plies": (2, 4, 6, 8),
        "opening_seed": 20260819,
        "inference_depth": 1,
        "max_moves": 200,
        "num_workers": 2,
        "output_dir": str(output_dir),
        "training_stage": "policy_only",
        "checkpoint_sha256": _CHECKPOINT_SHA256,
        "suite_fingerprint": _SUITE_FINGERPRINT,
        "teacher_correct_states": 2750,
        "teacher_total_states": 5000,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        checkpoint_acceptance.run_checkpoint_acceptance(**arguments)

    assert not output_dir.exists()


@pytest.mark.parametrize(("teacher_agreement", "correct_states", "message"), (
    (float("nan"), 2750, "must be finite and within"),
    ("0.55", 2750, "must be a real number"),
    (True, 5000, "must be a real number"),
    (False, 0, "must be a real number"),
))
def test_invalid_teacher_agreement_fails_before_evaluation(
    monkeypatch,
    tmp_path: Path,
    teacher_agreement,
    correct_states: int,
    message: str,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid teacher evidence reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"

    with pytest.raises(
        ValueError,
        match=message,
    ):
        checkpoint_acceptance.run_checkpoint_acceptance(
            str(tmp_path / "model_step_136000.pt"),
            step=136000,
            teacher_agreement=teacher_agreement,
            opening_plies=(2, 4, 6, 8),
            opening_seed=20260819,
            inference_depth=1,
            max_moves=200,
            num_workers=2,
            output_dir=str(output_dir),
            training_stage="policy_only",
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT,
            teacher_correct_states=correct_states,
            teacher_total_states=5000,
        )

    assert not output_dir.exists()


def test_inconsistent_teacher_counts_fail_before_evaluation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def unexpected_worker_pool(**_kwargs):
        raise AssertionError("invalid teacher evidence reached the worker pool")

    monkeypatch.setattr(
        checkpoint_acceptance, "ProcessPoolExecutor", unexpected_worker_pool)
    output_dir = tmp_path / "reports"

    with pytest.raises(
        ValueError,
        match="quotient of its recorded counts",
    ):
        checkpoint_acceptance.run_checkpoint_acceptance(
            str(tmp_path / "model_step_136000.pt"),
            step=136000,
            teacher_agreement=0.55,
            opening_plies=(2, 4, 6, 8),
            opening_seed=20260819,
            inference_depth=1,
            max_moves=200,
            num_workers=2,
            output_dir=str(output_dir),
            training_stage="policy_only",
            checkpoint_sha256=_CHECKPOINT_SHA256,
            suite_fingerprint=_SUITE_FINGERPRINT,
            teacher_correct_states=2751,
            teacher_total_states=5000,
        )

    assert not output_dir.exists()
