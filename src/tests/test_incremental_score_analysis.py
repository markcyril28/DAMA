"""Score diagnostics retain sampled evidence when terminal reports are missing."""

import json

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def _recommendations(report):
    return analysis.generate_recommendations(
        report, analysis.analyze_loss(report), analysis.analyze_throughput(report),
        analysis.analyze_gradients(report), analysis.analyze_model_health(report),
        analysis.analyze_evaluations(report), analysis.analyze_selfplay(report),
        analysis.analyze_system(report),
    )


def _entropy_recommendations(report):
    return [rec for rec in _recommendations(report)
            if "entropy" in rec["recommendation"].lower()]


def test_incremental_scores_reach_analyzer_without_terminal_report(tmp_path):
    collector = StatsCollector(str(tmp_path), session_id="scores")
    collector.record_training_step(
        200, loss=1.0, lr=1e-4,
        score_stats={"mean": 0.1, "std": 0.2, "entropy": 0.0, "top1_margin": 0.4},
    )
    collector.flush_incremental()
    stream = tmp_path / "incremental_scores.jsonl"
    first, warning = analysis.load_incremental_report(str(stream))
    assert warning is None
    assert first["score_distribution"]["entropy_summary"]["latest"] == {
        "step": 200, "value": 0.0,
    }
    assert _entropy_recommendations(first)[0]["category"] == "Model Behavior"

    collector.record_training_step(
        400, loss=0.9, lr=1e-4,
        score_stats={"mean": 0.3, "std": 0.4, "entropy": 1.0, "top1_margin": 0.6},
    )
    collector.flush_incremental()
    report, warning = analysis.load_incremental_report(str(stream))
    assert warning is None
    for name, expected_mean, latest_value in (
        ("mean_summary", 0.2, 0.3),
        ("std_summary", 0.3, 0.4),
        ("entropy_summary", 0.5, 1.0),
        ("top1_margin_summary", 0.5, 0.6),
    ):
        summary = report["score_distribution"][name]
        assert summary["recent_mean"] == pytest.approx(expected_mean)
        assert summary["recent_finite_count"] == 2
        assert summary["latest"] == {"step": 400, "value": latest_value}
    assert _entropy_recommendations(report) == []
    assert not list(tmp_path.glob("session_report_*.json"))


def test_legacy_incremental_scores_remain_unavailable(tmp_path):
    stream = tmp_path / "incremental_legacy.jsonl"
    stream.write_text(json.dumps({"timestamp": "2026-09-14T12:00:00"}) + "\n")
    report, warning = analysis.load_incremental_report(str(stream))
    assert warning is None
    assert report["score_distribution"] == {}
    assert _entropy_recommendations(report) == []


@pytest.mark.parametrize("values,expected_category,invalid_count", [
    ([], None, 0),
    ([0.0], "Model Behavior", 0),
    ([0.1, 0.2], "Model Behavior", 0),
    ([0.3], None, 0),
    ([float("nan")], "Stability", 1),
    ([float("inf")], "Stability", 1),
    ([-float("inf")], "Stability", 1),
    ([0.0, float("nan")], "Stability", 1),
    ([0.1, float("inf"), float("nan")], "Stability", 2),
])
def test_recorded_entropy_preserves_zero_and_invalid_evidence(
        tmp_path, values, expected_category, invalid_count):
    collector = StatsCollector(str(tmp_path), session_id="entropy")
    for step, value in enumerate(values, start=1):
        collector.record_training_step(
            step * 200, loss=1.0, lr=1e-4, score_stats={"entropy": value},
        )
    report = collector.generate_session_report()
    recs = _entropy_recommendations(report)
    assert [rec["category"] for rec in recs] == (
        [expected_category] if expected_category else [])
    if invalid_count:
        assert f"{invalid_count} recorded nonfinite observations" in recs[0]["recommendation"]
        assert "config_change" not in recs[0]
    elif values and expected_category:
        assert f"{sum(values) / len(values):.3f}" in recs[0]["recommendation"]


@pytest.mark.parametrize("summary,expected_category", [
    ({"total_count": 2, "recent_finite_count": 0, "recent_mean": 0.0}, None),
    ({"total_count": 2, "recent_finite_count": 1, "recent_nonfinite_count": 0,
      "nonfinite_count": 1, "recent_mean": 0.0}, "Model Behavior"),
    ({"total_count": 1, "recent_mean": 0.0}, "Model Behavior"),
    ({"total_count": 0, "recent_mean": 0.0}, None),
    ({"total_count": 1}, None),
])
def test_entropy_uses_recent_evidence_and_preserves_legacy_fallback(
        summary, expected_category):
    report = {"score_distribution": {"entropy_summary": summary}}
    assert [rec["category"] for rec in _entropy_recommendations(report)] == (
        [expected_category] if expected_category else [])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["recent_mean", "recent_stdev", "recent_min", "recent_max"])
def test_legacy_invalid_entropy_statistics_report_stability_without_counts(invalid, field):
    summary = {"total_count": 10, "recent_mean": 0.0, field: invalid}
    report = {"score_distribution": {"entropy_summary": summary}}
    recs = _entropy_recommendations(report)
    assert len(recs) == 1
    assert recs[0]["category"] == "Stability"
    assert "nonfinite" in recs[0]["recommendation"].lower()
    assert "recorded nonfinite observations" not in recs[0]["recommendation"]
    assert "config_change" not in recs[0]
