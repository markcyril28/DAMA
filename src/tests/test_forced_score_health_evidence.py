"""Invalid forced-move scores cannot authorize confidence or exploration advice."""

import pytest
import torch

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def _render(report):
    analyses = [fn(report) for fn in (
        analysis.analyze_loss, analysis.analyze_throughput,
        analysis.analyze_gradients, analysis.analyze_model_health,
        analysis.analyze_evaluations, analysis.analyze_selfplay,
        analysis.analyze_system,
    )]
    recommendations = analysis.generate_recommendations(report, *analyses)
    return analysis.format_markdown_report(report, *analyses, recommendations)


def _sample(collector, step, values, padded):
    scores = torch.tensor([values] if padded else values)
    counts = torch.tensor([len(values)])
    compute = (collector.compute_score_stats_padded if padded
               else collector.compute_score_stats)
    collector.record_training_step(
        step, loss=1.0, lr=1e-4, score_stats=compute(scores, counts),
    )


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("incremental", [False, True])
def test_forced_invalid_scores_override_earlier_confidence(
    tmp_path, padded, invalid, incremental,
):
    collector = StatsCollector(str(tmp_path), session_id="forced")
    _sample(collector, 200, [0.0, -1000.0], padded)
    _sample(collector, 400, [invalid], padded)
    # A forced move supplies score-health evidence but no entropy observation.
    assert collector.score_entropy.count == 1
    assert collector.score_entropy.summary()["recent_mean"] == 0.0
    hints = collector.generate_optimization_hints()
    assert any("nonfinite score distribution" in hint["hint"] for hint in hints)
    assert not any(hint["area"] == "model_behavior" for hint in hints)

    if incremental:
        collector.flush_incremental()
        report, warning = analysis.load_incremental_report(
            str(tmp_path / "incremental_forced.jsonl"))
        assert warning is None
    else:
        report = collector.generate_session_report()
    rendered = _render(report)
    assert "nonfinite score distribution" in rendered
    assert "mean_summary" in rendered
    assert "std_summary" in rendered
    assert "increasing noise_prob" not in rendered


@pytest.mark.parametrize("family", ["mean_summary", "std_summary"])
@pytest.mark.parametrize("field", ["recent_mean", "recent_stdev", "recent_min", "recent_max"])
def test_legacy_score_evidence_blocks_saved_confidence(family, field):
    report = {
        "score_distribution": {
            family: {"total_count": 1, "recent_mean": 0.0, field: float("nan")},
            "entropy_summary": {"total_count": 1, "recent_mean": 0.0},
        },
        "optimization_hints": [{
            "area": "model_behavior", "severity": "warning",
            "hint": "Average score entropy is very low (0.000). Consider increasing noise_prob.",
        }],
    }
    rendered = _render(report)
    assert "nonfinite score distribution" in rendered
    assert family in rendered
    assert "Average score entropy is very low" not in rendered
    assert "increasing noise_prob" not in rendered
    assert "score distribution" in rendered


@pytest.mark.parametrize("family", ["mean", "std"])
def test_terminal_score_window_overrides_shorter_saved_hint(tmp_path, family):
    collector = StatsCollector(str(tmp_path), session_id="window")
    collector.record_training_step(1, 1.0, 1e-4, score_stats={family: float("nan")})
    for step in range(2, 102):
        collector.record_training_step(
            step, 1.0, 1e-4, score_stats={family: 0.0, "entropy": 0.0})
    report = collector.generate_session_report()
    assert any(hint["area"] == "model_behavior" for hint in report["optimization_hints"])
    rendered = _render(report)
    assert "nonfinite score distribution" in rendered
    assert "Average score entropy is very low" not in rendered
    assert "increasing noise_prob" not in rendered


@pytest.mark.parametrize("older_invalid", [False, True])
def test_healthy_recent_score_window_keeps_confidence_hint(tmp_path, older_invalid):
    collector = StatsCollector(str(tmp_path), session_id="healthy")
    if older_invalid:
        _sample(collector, 0, [float("nan")], padded=True)
    for step in range(1, 1001):
        collector.record_training_step(
            step, 1.0, 1e-4,
            score_stats={"mean": 0.0, "std": 0.0, "entropy": 0.0})
    rendered = _render(collector.generate_session_report())
    assert "nonfinite score distribution" not in rendered
    assert "Average score entropy is very low (0.000)" in rendered
