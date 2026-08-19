import json
from pathlib import Path

import pytest

from dama.ai.ml import checkpoint_acceptance


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
    _Tester.calls = []
    monkeypatch.setattr(checkpoint_acceptance, "ModelVsAlgoTester", _Tester)
    report = checkpoint_acceptance.run_checkpoint_acceptance(
        str(tmp_path / "model_step_136000.pt"),
        step=136000,
        teacher_agreement=0.55,
        opening_plies=(0, 2, 4, 6, 8),
        opening_seed=20260819,
        inference_depth=1,
        max_moves=200,
        num_workers=2,
        output_dir=str(tmp_path / "reports"),
        training_stage="policy_only",
    )

    assert [call["opponent_type"] for call in _Tester.calls] == ["random", "algorithm"]
    assert all(call["opening_seed"] == 20260819 for call in _Tester.calls)
    assert report["passed"] is True
    assert report["opening_suite_id"] == "fixed-suite"
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["checks"]["random_exact_balanced_100_games"] is True
    assert saved["checks"]["easy_exact_balanced_100_games"] is True


def test_acceptance_task_rejects_zero_opening():
    with pytest.raises(ValueError, match="positive"):
        checkpoint_acceptance.make_pending_acceptance_task(
            "checkpoint.pt", step=1, teacher_agreement=0.55,
            opening_plies=(0, 2), opening_seed=1, inference_depth=1,
            max_moves=200, num_workers=1, training_stage="policy_only")
