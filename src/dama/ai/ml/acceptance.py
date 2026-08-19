"""Acceptance metrics and gates for policy-distillation checkpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple


ACCEPTANCE_GAMES_PER_OPPONENT = 100
ACCEPTANCE_GAMES_PER_SIDE = 50
WILSON_95_Z = 1.959963984540054
WILSON_95_METHOD = "wilson_score_95_draw_as_half_success"


def match_score(wins: int, draws: int, losses: int) -> float:
    """Return match points per game, using win=1, draw=0.5, loss=0."""
    total = _validate_wdl(wins, draws, losses)
    if total == 0:
        return 0.0
    return (wins + 0.5 * draws) / total


def wilson_score_interval_95(
    wins: int,
    draws: int,
    losses: int,
) -> Tuple[float, float]:
    """Return the predeclared 95 percent Wilson interval for match score.

    The half-point convention treats each draw as one half-success and keeps
    one trial per game. Thus the effective success count is ``wins + 0.5 *
    draws`` and the trial count is ``wins + draws + losses``. The standard
    Wilson score formula is then applied with z=1.959963984540054.

    This convention is deterministic, applies identically to every promoted
    checkpoint, and keeps the interval centered on the declared match score.
    """
    total = _validate_wdl(wins, draws, losses)
    if total == 0:
        return (0.0, 1.0)

    p_hat = (wins + 0.5 * draws) / total
    z2 = WILSON_95_Z * WILSON_95_Z
    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    radius = (
        WILSON_95_Z
        * math.sqrt(
            (p_hat * (1.0 - p_hat) / total)
            + (z2 / (4.0 * total * total))
        )
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def wdl_summary(wins: int, draws: int, losses: int) -> Dict[str, Any]:
    """Return raw counts, match score, and the declared confidence interval."""
    total = _validate_wdl(wins, draws, losses)
    lower, upper = wilson_score_interval_95(wins, draws, losses)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "total": total,
        "match_score": match_score(wins, draws, losses),
        "match_score_ci_95": {
            "lower": lower,
            "upper": upper,
            "method": WILSON_95_METHOD,
        },
    }


def evaluate_random_gate(random_result: Any) -> Dict[str, Any]:
    """Evaluate the predeclared random-opponent gate before easy games run."""
    random_data = _as_mapping(random_result)
    overall = _extract_wdl(random_data)
    p1 = _extract_wdl(random_data, "ml_as_p1_")
    p2 = _extract_wdl(random_data, "ml_as_p2_")
    summary = wdl_summary(*overall)
    protocol = (
        sum(overall) == ACCEPTANCE_GAMES_PER_OPPONENT
        and sum(p1) == ACCEPTANCE_GAMES_PER_SIDE
        and sum(p2) == ACCEPTANCE_GAMES_PER_SIDE
        and tuple(first + second for first, second in zip(p1, p2)) == overall
    )
    checks = {
        "random_exact_balanced_100_games": protocol,
        "random_match_score_at_least_0_80": summary["match_score"] >= 0.80,
        "random_lower_ci_above_0_70": (
            summary["match_score_ci_95"]["lower"] > 0.70
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": summary,
    }


@dataclass(frozen=True)
class AcceptanceDecision:
    """Deterministic result of the sequential policy-distillation gates."""

    passed: bool
    checks: Dict[str, bool]
    metrics: Dict[str, Any]
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "ci_method": WILSON_95_METHOD,
        }


def evaluate_acceptance_gates(
    teacher_agreement: float,
    random_result: Any,
    easy_result: Any,
) -> AcceptanceDecision:
    """Evaluate the predeclared sequential promotion and acceptance gates.

    ``random_result`` and ``easy_result`` may be mappings or objects exposing a
    ``to_dict()`` method, such as ``TestStatistics``. The helper also enforces
    the approved protocol size of 100 games per opponent and 50 games per side.
    """
    if not math.isfinite(teacher_agreement) or not 0.0 <= teacher_agreement <= 1.0:
        raise ValueError("teacher_agreement must be finite and within [0, 1]")

    random_data = _as_mapping(random_result)
    easy_data = _as_mapping(easy_result)
    easy_overall = _extract_wdl(easy_data)
    easy_p1 = _extract_wdl(easy_data, "ml_as_p1_")
    easy_p2 = _extract_wdl(easy_data, "ml_as_p2_")

    random_gate = evaluate_random_gate(random_data)
    random_summary = random_gate["metrics"]
    easy_summary = wdl_summary(*easy_overall)
    easy_p1_score = match_score(*easy_p1)
    easy_p2_score = match_score(*easy_p2)

    easy_protocol = (
        sum(easy_overall) == ACCEPTANCE_GAMES_PER_OPPONENT
        and sum(easy_p1) == ACCEPTANCE_GAMES_PER_SIDE
        and sum(easy_p2) == ACCEPTANCE_GAMES_PER_SIDE
        and tuple(p1 + p2 for p1, p2 in zip(easy_p1, easy_p2)) == easy_overall
    )

    checks = {
        "teacher_agreement_at_least_0_50": teacher_agreement >= 0.50,
        **random_gate["checks"],
        "easy_exact_balanced_100_games": easy_protocol,
        "easy_lower_ci_above_0_50": easy_summary["match_score_ci_95"]["lower"] > 0.50,
        "easy_p1_match_score_at_least_0_50": easy_p1_score >= 0.50,
        "easy_p2_match_score_at_least_0_50": easy_p2_score >= 0.50,
    }
    metrics = {
        "teacher_agreement": teacher_agreement,
        "random": random_summary,
        "easy": easy_summary,
        "easy_p1_match_score": easy_p1_score,
        "easy_p2_match_score": easy_p2_score,
    }
    thresholds = {
        "teacher_agreement_min": 0.50,
        "random_match_score_min": 0.80,
        "random_lower_ci_strict_min": 0.70,
        "easy_lower_ci_strict_min": 0.50,
        "easy_side_match_score_min": 0.50,
    }
    return AcceptanceDecision(
        passed=all(checks.values()),
        checks=checks,
        metrics=metrics,
        thresholds=thresholds,
    )


def _validate_wdl(wins: int, draws: int, losses: int) -> int:
    values = (wins, draws, losses)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("wins, draws, and losses must be integers")
    if any(value < 0 for value in values):
        raise ValueError("wins, draws, and losses must be non-negative")
    return sum(values)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return data
    raise TypeError("evaluation result must be a mapping or expose to_dict()")


def _extract_wdl(data: Mapping[str, Any], prefix: str = "") -> Tuple[int, int, int]:
    if prefix:
        wins = int(data.get(f"{prefix}wins", 0))
        draws = int(data.get(f"{prefix}draws", 0))
        losses = int(data.get(f"{prefix}losses", 0))
    else:
        wins = int(data.get("ml_wins", data.get("wins", 0)))
        draws = int(data.get("draws", 0))
        losses = int(data.get("opponent_wins", data.get("algo_wins", 0)))
    _validate_wdl(wins, draws, losses)
    return wins, draws, losses
