"""Invalid health samples must remain visible without becoming zero updates."""

import math

import pytest
import torch

from dama.ai.ml.stats_collector import MetricBuffer, StatsCollector
from scripts import analyze_training_stats as analysis


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("incremental", [False, True])
def test_nonfinite_model_updates_do_not_suggest_increasing_lr(
    tmp_path, invalid, incremental,
):
    collector = StatsCollector(output_dir=str(tmp_path), session_id="invalid")
    model = torch.nn.Linear(1, 1, bias=False)
    collector.record_model_health(model, step=5000)
    with torch.no_grad():
        model.weight.fill_(invalid)
    collector.record_model_health(model, step=10000)

    if incremental:
        collector.flush_incremental()
        report, warning = analysis.load_incremental_report(
            str(tmp_path / "incremental_invalid.jsonl"))
        assert warning is None
    else:
        report = collector.generate_session_report()
    result = analysis.analyze_model_health(report)
    assert "mean_update_ratio" not in result
    assert "lr_possibly_too_low" not in result
    assert result["recent_nonfinite_update_observations"] == 1

    analyses = {
        "loss_analysis": analysis.analyze_loss(report),
        "throughput_analysis": analysis.analyze_throughput(report),
        "grad_analysis": analysis.analyze_gradients(report),
        "model_analysis": result,
        "eval_analysis": analysis.analyze_evaluations(report),
        "selfplay_analysis": analysis.analyze_selfplay(report),
        "system_analysis": analysis.analyze_system(report),
    }
    recommendations = analysis.generate_recommendations(report, **analyses)
    assert any("nonfinite weight-update" in item["recommendation"]
               for item in recommendations)
    assert not any(item["category"] == "Learning Rate"
                   for item in recommendations)


def test_metric_summary_counts_recent_invalid_samples_after_eviction():
    buffer = MetricBuffer(maxlen=3)
    for step, value in enumerate([0.5, 0.004, float("nan"), float("inf")]):
        buffer.append(value, step)
    summary = buffer.summary(2)
    assert summary["finite_count"] == 2
    assert summary["nonfinite_count"] == 2
    assert summary["recent_finite_count"] == 0
    assert summary["recent_nonfinite_count"] == 2
    result = analysis.analyze_model_health({
        "model_health": {"weight_update_ratio_summaries": {
            "expired": summary,
            "observed": {"total_count": 1, "recent_mean": 0.004},
        }},
    })
    assert result["mean_update_ratio"] == pytest.approx(0.004)
    assert result["recent_nonfinite_update_observations"] == 2


def test_mixed_finite_and_invalid_samples_keep_measured_mean():
    buffer = MetricBuffer()
    for step, value in enumerate([float("nan"), 0.004, float("inf"), 0.0]):
        buffer.append(value, step)
    summary = buffer.summary()
    assert summary["recent_finite_count"] == 2
    assert summary["recent_nonfinite_count"] == 2
    result = analysis.analyze_model_health({
        "model_health": {"weight_update_ratio_summaries": {"weight": summary}},
    })
    assert result["mean_update_ratio"] == pytest.approx(0.002)
    assert result["recent_nonfinite_update_observations"] == 2
    assert "lr_possibly_too_high" not in result
    assert "lr_possibly_too_low" not in result


def test_invalid_weight_does_not_turn_unchanged_bias_into_low_lr_advice(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path))
    model = torch.nn.Linear(1, 1)
    collector.record_model_health(model, step=5000)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    collector.record_model_health(model, step=10000)

    report = collector.generate_session_report()
    result = analysis.analyze_model_health(report)
    assert result["mean_update_ratio"] == 0.0
    assert result["recent_nonfinite_update_observations"] == 1
    assert "lr_possibly_too_low" not in result
    analyses = {
        "loss_analysis": analysis.analyze_loss(report),
        "throughput_analysis": analysis.analyze_throughput(report),
        "grad_analysis": analysis.analyze_gradients(report),
        "model_analysis": result,
        "eval_analysis": analysis.analyze_evaluations(report),
        "selfplay_analysis": analysis.analyze_selfplay(report),
        "system_analysis": analysis.analyze_system(report),
    }
    recommendations = analysis.generate_recommendations(report, **analyses)
    assert not any(item["category"] == "Learning Rate" for item in recommendations)
    markdown = analysis.format_markdown_report(
        report, **analyses, recommendations=recommendations)
    assert "Unavailable (nonfinite updates)" in markdown


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_legacy_invalid_only_update_is_not_low_lr_evidence(invalid):
    result = analysis.analyze_model_health({
        "model_health": {"weight_update_ratio_summaries": {
            "weight": {"total_count": 1, "recent_mean": 0.0,
                       "recent_stdev": 0.0, "recent_min": invalid,
                       "recent_max": invalid},
        }},
    })
    assert "mean_update_ratio" not in result
    assert "lr_possibly_too_low" not in result
    assert result["recent_nonfinite_update_layers"] == 1
    assert "recent_nonfinite_update_observations" not in result


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), None])
def test_legacy_nonfinite_or_missing_means_are_not_measured_updates(invalid):
    result = analysis.analyze_model_health({
        "model_health": {"weight_update_ratio_summaries": {
            "invalid": {"total_count": 1, "recent_mean": invalid},
            "zero": {"total_count": 1, "recent_mean": 0.0},
        }},
    })
    assert math.isfinite(result["mean_update_ratio"])
    assert result["mean_update_ratio"] == 0.0
    if invalid is None:
        assert result["lr_possibly_too_low"] is True
    else:
        assert "lr_possibly_too_low" not in result
        assert result["recent_nonfinite_update_layers"] == 1
        assert "recent_nonfinite_update_observations" not in result


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("has_finite", [False, True])
def test_legacy_invalid_extrema_preserve_update_evidence(invalid, has_finite):
    buffer = MetricBuffer()
    # Put NaN first so the old raw extrema expose it even in a mixed window.
    buffer.append(invalid, 5000)
    if has_finite:
        buffer.append(0.004, 10000)
    legacy = buffer.summary()
    for key in ("finite_count", "nonfinite_count", "recent_finite_count",
                "recent_nonfinite_count", "recent_finite_max"):
        legacy.pop(key)
    report = {"model_health": {"weight_update_ratio_summaries": {
        "invalid": legacy,
        "observed": {"total_count": 1, "recent_mean": 0.004},
    }}}
    result = analysis.analyze_model_health(report)
    assert result["mean_update_ratio"] == pytest.approx(0.004)
    assert result["recent_nonfinite_update_layers"] == 1
    assert "recent_nonfinite_update_observations" not in result
    assert "lr_possibly_too_high" not in result
    assert "lr_possibly_too_low" not in result
    assert ("invalid" in dict(result["top_update_layers"])) is has_finite

    analyses = {
        "loss_analysis": analysis.analyze_loss(report),
        "throughput_analysis": analysis.analyze_throughput(report),
        "grad_analysis": analysis.analyze_gradients(report),
        "model_analysis": result,
        "eval_analysis": analysis.analyze_evaluations(report),
        "selfplay_analysis": analysis.analyze_selfplay(report),
        "system_analysis": analysis.analyze_system(report),
    }
    recommendations = analysis.generate_recommendations(report, **analyses)
    assert any("nonfinite weight-update" in item["recommendation"]
               for item in recommendations)
    assert not any(item["category"] == "Learning Rate"
                   for item in recommendations)
    markdown = analysis.format_markdown_report(
        report, **analyses, recommendations=recommendations)
    assert "Unavailable (nonfinite updates)" in markdown
