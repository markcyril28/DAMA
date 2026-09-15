"""Saved entropy hints must agree with the report's supplied evidence window."""

import pytest

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


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_terminal_entropy_window_overrides_shorter_collector_hint(tmp_path, invalid):
    collector = StatsCollector(str(tmp_path), session_id="entropy_window")
    # The saved summary spans 1000 observations; the collector hint spans 100.
    for step, entropy in enumerate([invalid] + [0.0] * 100, start=1):
        collector.record_training_step(step, 1.0, 1e-4, score_stats={"entropy": entropy})
    report = collector.generate_session_report()
    assert report["score_distribution"]["entropy_summary"]["recent_nonfinite_count"] == 1
    assert any(hint["area"] == "model_behavior" for hint in report["optimization_hints"])
    rendered = _render(report)
    assert "1 recorded nonfinite observations" in rendered
    assert "Average score entropy is very low" not in rendered
    assert "increasing noise_prob" not in rendered
    assert "entropy tuning hint withheld" in rendered


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_legacy_invalid_entropy_blocks_saved_confidence_hint(invalid):
    report = {
        "score_distribution": {"entropy_summary": {
            "total_count": 1, "recent_mean": 0.0, "recent_max": invalid,
        }},
        "optimization_hints": [{
            "area": "model_behavior", "severity": "warning",
            "hint": "Average score entropy is very low (0.000). Consider increasing noise_prob.",
        }],
    }
    rendered = _render(report)
    assert "nonfinite summary statistics" in rendered
    assert "Average score entropy is very low" not in rendered
    assert "increasing noise_prob" not in rendered
    assert "entropy tuning hint withheld" in rendered


@pytest.mark.parametrize("older_invalid", [False, True])
def test_finite_recent_entropy_keeps_collector_hint(tmp_path, older_invalid):
    collector = StatsCollector(str(tmp_path), session_id="finite_entropy")
    values = ([float("nan")] if older_invalid else []) + [0.0] * 1000
    for step, entropy in enumerate(values, start=1):
        collector.record_training_step(step, 1.0, 1e-4, score_stats={"entropy": entropy})
    report = collector.generate_session_report()
    assert report["score_distribution"]["entropy_summary"]["recent_nonfinite_count"] == 0
    rendered = _render(report)
    assert "Average score entropy is very low (0.000)" in rendered
    assert "entropy tuning hint withheld" not in rendered
