"""Loss summaries must distinguish finite zero from invalid or missing evidence."""

import math

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def _collector(tmp_path, values):
    collector = StatsCollector(str(tmp_path), session_id="loss")
    for step, value in enumerate(values, start=1):
        collector.record_training_step(step * 200, value, 1e-4)
    return collector


def _render(report):
    analyses = [fn(report) for fn in (
        analysis.analyze_loss, analysis.analyze_throughput,
        analysis.analyze_gradients, analysis.analyze_model_health,
        analysis.analyze_evaluations, analysis.analyze_selfplay,
        analysis.analyze_system,
    )]
    recommendations = analysis.generate_recommendations(report, *analyses)
    markdown = analysis.format_markdown_report(report, *analyses, recommendations)
    return analyses[0], recommendations, markdown


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("incremental", [False, True])
def test_invalid_only_loss_never_reports_measured_zero_or_clean_training(
    tmp_path, invalid, incremental,
):
    collector = _collector(tmp_path, [invalid])
    if incremental:
        collector.flush_incremental()
        report, warning = analysis.load_incremental_report(
            str(tmp_path / "incremental_loss.jsonl"))
        assert warning is None
    else:
        report = collector.generate_session_report()
    result, recs, markdown = _render(report)
    assert result["running_mean"] is None
    assert result["running_min"] is None
    assert result["recent_mean"] is None
    assert result["recent_stdev"] is None
    assert result["has_data"] is True
    assert result["finite_count"] == 0
    assert result["recent_nonfinite_count"] == 1
    assert any("nonfinite loss" in rec["recommendation"].lower() for rec in recs)
    assert "UNAVAILABLE" in markdown.split("## Loss Analysis")[1].split("## ")[0]
    assert "No issues detected" not in markdown
    assert "0 recorded loss samples" not in markdown
    assert "| Running Mean | 0.000000 |" not in markdown


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_mixed_loss_preserves_finite_statistics_without_convergence_advice(
    tmp_path, invalid,
):
    report = _collector(tmp_path, [invalid, 0.0, 2.0]).generate_session_report()
    report["convergence"].update({
        "loss_is_plateauing": True, "loss_plateau_steps": 1000,
        "loss_improvement_1000": 0.0, "loss_improvement_pct_1000": 0.0,
    })
    report["optimization_hints"] = [{
        "area": "convergence", "severity": "warning",
        "hint": "Sampled loss EMA is plateauing. Consider reducing learning rate.",
    }]
    result, recs, markdown = _render(report)
    assert result["running_mean"] == 1.0
    assert result["running_min"] == 0.0
    assert result["recent_mean"] == 1.0
    assert result["recent_stdev"] == pytest.approx(math.sqrt(2))
    assert result["recent_finite_count"] == 2
    assert result["recent_nonfinite_count"] == 1
    assert result["is_plateauing"] is False
    assert result["loss_improvement_1000"] is None
    assert not any(rec["category"] == "Convergence" for rec in recs)
    assert "PARTIAL" in markdown
    assert "Consider reducing learning rate" not in markdown
    assert "No issues detected" not in markdown


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["recent_mean", "recent_max", "running_mean"])
def test_legacy_invalid_loss_does_not_establish_finite_safe_zero(invalid, field):
    summary = {
        "total_count": 1, "running_mean": 0.0, "running_min": 0.0,
        "recent_mean": 0.0, "recent_stdev": 0.0, field: invalid,
    }
    result, recs, markdown = _render({"loss": {"summary": summary}})
    assert result["has_nonfinite"] is True
    assert result["nonfinite_count"] is None
    assert result["recent_nonfinite_count"] is None
    assert result["running_mean"] is None
    assert result["running_min"] is None
    assert result["recent_mean"] is None
    assert any("nonfinite loss" in rec["recommendation"].lower() for rec in recs)
    assert "recorded nonfinite observations" not in markdown
    assert "No issues detected" not in markdown


@pytest.mark.parametrize("legacy", [False, True])
def test_measured_zero_loss_remains_valid(tmp_path, legacy):
    report = _collector(tmp_path, [0.0] * 5).generate_session_report()
    if legacy:
        summary = report["loss"]["summary"]
        for key in ("finite_count", "nonfinite_count", "recent_finite_count",
                    "recent_nonfinite_count"):
            summary.pop(key)
    result, _, markdown = _render(report)
    assert result["running_mean"] == 0.0
    assert result["running_min"] == 0.0
    assert result["recent_mean"] == 0.0
    assert result.get("has_nonfinite", False) is False
    assert result["is_plateauing"] is True
    assert "| Running Mean | 0.000000 |" in markdown


def test_recent_invalid_loss_does_not_erase_all_time_finite_statistics(tmp_path):
    collector = _collector(tmp_path, [2.0] + [float("nan")] * 1000)
    result, _, markdown = _render(collector.generate_session_report())
    assert result["running_mean"] == 2.0
    assert result["running_min"] == 2.0
    assert result["recent_mean"] is None
    assert result["finite_count"] == 1
    assert result["recent_finite_count"] == 0
    assert result["recent_nonfinite_count"] == 1000
    assert "| Running Mean | 2.000000 |" in markdown
    assert "| Recent Mean | Unavailable |" in markdown


def test_recent_finite_loss_retains_old_invalid_evidence_without_hiding_values(tmp_path):
    collector = _collector(tmp_path, [float("nan")] + [2.0] * 1000)
    result, recs, _ = _render(collector.generate_session_report())
    assert result["running_mean"] == 2.0
    assert result["recent_mean"] == 2.0
    assert result["nonfinite_count"] == 1
    assert result["recent_nonfinite_count"] == 0
    assert result["is_plateauing"] is False
    assert any("nonfinite loss" in rec["recommendation"].lower() for rec in recs)


def test_missing_loss_stays_missing():
    result, recs, markdown = _render({})
    assert result["has_data"] is False
    assert result["running_mean"] is None
    assert result["recent_mean"] is None
    assert "MISSING (0 recorded loss samples)" in markdown
    assert any(rec["category"] == "Data Quality" for rec in recs)


def test_empty_recent_window_does_not_reuse_all_time_counts():
    result, _, _ = _render({"loss": {"summary": {
        "total_count": 4, "finite_count": 4, "nonfinite_count": 0,
        "recent_finite_count": 0, "recent_nonfinite_count": 0,
        "running_mean": 2.0, "running_min": 1.0,
        "recent_mean": 0.0, "recent_stdev": 0.0,
    }}})
    assert result["running_mean"] == 2.0
    assert result["running_min"] == 1.0
    assert result["recent_mean"] is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_console_invalid_only_loss_discloses_missing_finite_evidence(
    tmp_path, capsys, invalid,
):
    collector = _collector(tmp_path, [invalid])
    collector.print_session_summary()
    console = capsys.readouterr().out
    loss_block = console.split("  Loss:")[1].split("\n\n")[0]
    assert "Unavailable (no finite observations)" in loss_block
    assert "Nonfinite (last 100): 1" in loss_block
    assert "Nonfinite (all time): 1" in loss_block
    assert "0.000000" not in loss_block


def test_console_missing_loss_is_not_measured_zero(tmp_path, capsys):
    collector = _collector(tmp_path, [])
    collector.print_session_summary()
    console = capsys.readouterr().out
    loss_block = console.split("  Loss:")[1].split("\n\n")[0]
    assert "Unavailable (no finite observations)" in loss_block
    assert "0.000000" not in loss_block


def test_console_finite_zero_and_mixed_values_remain_visible(tmp_path, capsys):
    collector = _collector(tmp_path, [float("nan"), 0.0])
    collector.print_session_summary()
    console = capsys.readouterr().out
    assert "Latest (avg 100):   0.000000" in console
    assert "Best:               0.000000" in console
    assert "Nonfinite (last 100): 1" in console


def test_console_recent_window_does_not_erase_finite_all_time_best(tmp_path, capsys):
    collector = _collector(tmp_path, [2.0] + [float("nan")] * 100)
    collector.print_session_summary()
    console = capsys.readouterr().out
    assert "Latest (avg 100):   Unavailable" in console
    assert "Best:               2.000000" in console
    assert "Nonfinite (last 100): 100" in console


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_ema_never_crashes_or_supports_convergence(tmp_path, invalid):
    report = _collector(tmp_path, [0.0] * 5).generate_session_report()
    # Legacy summaries can establish zero without counts when their sampled
    # statistics are finite; an invalid EMA must not erase that evidence.
    for key in ("finite_count", "nonfinite_count", "recent_finite_count",
                "recent_nonfinite_count"):
        report["loss"]["summary"].pop(key)
    report["convergence"]["loss_ema"] = invalid
    result, recs, markdown = _render(report)
    assert result["running_mean"] == 0.0
    assert result["recent_mean"] == 0.0
    assert result["loss_ema"] is None
    assert result["is_plateauing"] is False
    assert result["convergence_available"] is False
    assert any("nonfinite" in rec["recommendation"].lower() for rec in recs)
    assert not any(rec["category"] == "Convergence" for rec in recs)
    assert "No issues detected" not in markdown


def test_missing_ema_cannot_support_plateau_claim(tmp_path):
    report = _collector(tmp_path, [1.0] * 5).generate_session_report()
    report["convergence"].pop("loss_ema")
    result, recs, _ = _render(report)
    assert result["is_plateauing"] is False
    assert not any("plateau" in rec["recommendation"].lower() for rec in recs)


def test_collector_hints_and_console_withhold_plateau_after_invalid_loss(
    tmp_path, capsys,
):
    collector = _collector(tmp_path, [float("nan"), 1.0, 1.0, 1.0, 1.0, 1.0])
    assert collector.get_convergence_metrics()["loss_is_plateauing"] is True
    hints = collector.generate_optimization_hints()
    assert not any(hint["area"] == "convergence" for hint in hints)
    assert any("nonfinite loss" in hint["hint"].lower() for hint in hints)
    collector.print_session_summary()
    console = capsys.readouterr().out
    assert "reducing learning rate" not in console
    assert "Nonfinite (all time): 1" in console


def test_collector_finite_plateau_hint_remains_available(tmp_path):
    collector = _collector(tmp_path, [1.0] * 5)
    hints = collector.generate_optimization_hints()
    assert any(hint["area"] == "convergence" for hint in hints)
