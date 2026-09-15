"""Nonfinite gradient telemetry must not become zero-gradient tuning advice."""

import math

import pytest
import torch

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def _collector(tmp_path, values):
    collector = StatsCollector(str(tmp_path), session_id="gradients")
    for step, value in enumerate(values, start=1):
        collector.record_training_step(step=step * 200, loss=1.0,
                                       lr=1e-4, grad_norm=value)
    return collector


def _render(report):
    analyses = {
        "loss_analysis": analysis.analyze_loss(report),
        "throughput_analysis": analysis.analyze_throughput(report),
        "grad_analysis": analysis.analyze_gradients(report),
        "model_analysis": analysis.analyze_model_health(report),
        "eval_analysis": analysis.analyze_evaluations(report),
        "selfplay_analysis": analysis.analyze_selfplay(report),
        "system_analysis": analysis.analyze_system(report),
    }
    recommendations = analysis.generate_recommendations(report, **analyses)
    markdown = analysis.format_markdown_report(
        report, **analyses, recommendations=recommendations)
    return analyses["grad_analysis"], recommendations, markdown


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("incremental", [False, True])
def test_invalid_gradients_are_unavailable_not_vanishing(
    tmp_path, capsys, invalid, incremental,
):
    collector = _collector(tmp_path, [invalid])
    if incremental:
        collector.flush_incremental()
        report, warning = analysis.load_incremental_report(
            str(tmp_path / "incremental_gradients.jsonl"))
        assert warning is None
    else:
        report = collector.generate_session_report()

    convergence = report["convergence"]
    assert convergence["grad_vanishing"] is False
    assert convergence["grad_exploding"] is False
    assert convergence["grad_norm_mean"] is None
    assert convergence["grad_norm_max"] is None
    assert convergence["grad_norm_recent_finite_count"] == 0
    assert convergence["grad_norm_recent_nonfinite_count"] == 1
    hints = collector.generate_optimization_hints()
    assert any("nonfinite gradient" in item["hint"].lower() for item in hints)
    assert not any("increasing learning rate" in item["hint"] for item in hints)
    assert not any("reducing grad_clip_norm" in item["hint"] for item in hints)

    result, recommendations, markdown = _render(report)
    assert result["has_data"] is True
    assert result["has_finite_data"] is False
    assert result["global_mean"] is None
    assert result["vanishing"] is False
    assert result["recent_nonfinite_count"] == 1
    assert any("nonfinite gradient" in item["recommendation"].lower()
               for item in recommendations)
    assert not any("grad_clip_norm" in item.get("config_change", "")
                   for item in recommendations)
    gradient_markdown = markdown.split("## Gradient Health")[1].split("## ")[0]
    assert "UNAVAILABLE" in gradient_markdown
    assert "0 recorded gradient samples" not in gradient_markdown
    assert "No ✓" not in gradient_markdown
    assert "| Mean Norm | 0.0000 |" not in gradient_markdown
    collector.print_session_summary()
    console = capsys.readouterr().out
    assert "Unavailable (no finite observations)" in console
    assert "Nonfinite (last 100): 1" in console


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_mixed_gradients_keep_finite_statistics_and_flag_invalid(tmp_path, invalid):
    collector = _collector(tmp_path, [invalid, 2.0, 4.0])
    report = collector.generate_session_report()
    result, _, markdown = _render(report)
    assert result["global_mean"] == pytest.approx(3.0)
    assert result["global_stdev"] == pytest.approx(math.sqrt(2))
    assert result["global_max"] == pytest.approx(4.0)
    assert result["recent_finite_count"] == 2
    assert result["recent_nonfinite_count"] == 1
    assert result["vanishing"] is False
    gradient_markdown = markdown.split("## Gradient Health")[1].split("## ")[0]
    assert "nonfinite" in gradient_markdown
    assert "| Mean Norm | 3.0000 |" in gradient_markdown
    assert "No ✓" not in gradient_markdown


@pytest.mark.parametrize("value,vanishing,exploding", [
    (0.0, True, False), (2.0, False, False), (150.0, False, True),
])
def test_finite_gradients_preserve_zero_and_explosion_evidence(
    tmp_path, value, vanishing, exploding,
):
    collector = _collector(tmp_path, [value])
    result, recommendations, _ = _render(collector.generate_session_report())
    assert result["global_mean"] == value
    assert result["global_max"] == value
    assert result["vanishing"] is vanishing
    assert result["exploding"] is exploding
    assert result["recent_finite_count"] == 1
    assert result["recent_nonfinite_count"] == 0
    assert any("grad_clip_norm" in item.get("config_change", "")
               for item in recommendations) is exploding


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["recent_mean", "recent_max"])
def test_explicit_legacy_nonfinite_values_override_unsafe_flags(invalid, field):
    summary = {"total_count": 1, "recent_mean": 0.0,
               "recent_stdev": 0.0, "recent_max": 0.0}
    summary[field] = invalid
    result, recommendations, markdown = _render({
        "gradient_norms": {"global_summary": summary},
        "convergence": {"grad_vanishing": True, "grad_exploding": True},
    })
    assert result["vanishing"] is False
    assert result["exploding"] is False
    assert result["has_nonfinite"] is True
    # The old summary identifies invalid evidence but not a sample count.
    assert result["recent_nonfinite_count"] is None
    assert any("nonfinite gradient" in item["recommendation"].lower()
               for item in recommendations)
    assert "Unavailable (nonfinite gradients)" in markdown


def test_recent_invalid_window_does_not_reuse_old_finite_evidence(tmp_path):
    collector = _collector(tmp_path, [2.0] + [float("nan")] * 1000)
    result, _, _ = _render(collector.generate_session_report())
    assert result["has_finite_data"] is False
    assert result["global_mean"] is None
    assert result["vanishing"] is False
    assert result["recent_finite_count"] == 0
    assert result["recent_nonfinite_count"] == 1000


def test_terminal_summary_and_convergence_keep_their_distinct_windows(tmp_path):
    collector = _collector(tmp_path, [150.0, float("nan")] + [2.0] * 100)
    report = collector.generate_session_report()
    assert report["convergence"]["grad_norm_max"] == 2.0
    assert report["convergence"]["grad_exploding"] is False
    result, _, _ = _render(report)
    assert result["global_mean"] == pytest.approx(350.0 / 101)
    assert result["global_max"] == 150.0
    assert result["recent_finite_count"] == 101
    assert result["recent_nonfinite_count"] == 1
    assert result["exploding"] is False


@pytest.mark.parametrize("multiply_parameter", [False, True])
def test_actual_preclip_invalid_norm_keeps_finite_explosion_without_tuning(
    tmp_path, multiply_parameter,
):
    parameter = torch.nn.Parameter(torch.zeros(1))
    # This is the same producer used by the trainer; clipping returns the
    # norm before clipping. A finite forward loss can have invalid gradients.
    term = parameter.sqrt()
    loss = 1.0 + (parameter * term if multiply_parameter else term)
    assert loss.item() == 1.0
    loss.backward()
    invalid_norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0).item()
    assert not math.isfinite(invalid_norm)
    collector = _collector(tmp_path, [150.0, invalid_norm])
    report = collector.generate_session_report()
    assert report["convergence"]["grad_exploding"] is True
    result, recommendations, _ = _render(report)
    assert result["exploding"] is True
    assert result["vanishing"] is False
    assert result["global_max"] == 150.0
    assert not any("grad_clip_norm" in item.get("config_change", "")
                   for item in recommendations)
    assert not any("reducing grad_clip_norm" in item["hint"]
                   for item in collector.generate_optimization_hints())


def test_nonfinite_gradients_withhold_model_low_lr_advice(tmp_path):
    report = _collector(tmp_path, [float("nan"), 0.0]).generate_session_report()
    report["model_health"] = {"weight_update_ratio_summaries": {
        "weight": {"total_count": 1, "recent_mean": 0.0},
    }}
    result, recommendations, markdown = _render(report)
    assert result["global_mean"] == 0.0
    assert result["vanishing"] is False
    assert not any(item["category"] == "Learning Rate" for item in recommendations)
    assert "| LR assessment | Unavailable (nonfinite gradients) |" in markdown


def test_missing_gradients_remain_missing():
    result, _, markdown = _render({})
    assert result["has_data"] is False
    assert result["has_finite_data"] is False
    assert result["vanishing"] is False
    assert "MISSING (0 recorded gradient samples)" in markdown


def test_legacy_saved_gradient_tuning_hints_are_withheld(tmp_path):
    report = _collector(tmp_path, [float("inf")]).generate_session_report()
    report["optimization_hints"] = [
        {"area": "stability", "severity": "warning", "hint": (
            "Very small gradient norms detected (< 1e-6). "
            "Consider: (1) increasing learning rate.")},
        {"area": "stability", "severity": "critical", "hint": (
            "Gradient norms exceeding 100.0 detected. "
            "Consider: (1) reducing learning rate, (2) reducing grad_clip_norm.")},
        {"area": "stability", "severity": "critical", "hint": "Inspect corrupt weights."},
    ]
    _, _, markdown = _render(report)
    assert "increasing learning rate" not in markdown
    assert "reducing grad_clip_norm" not in markdown
    assert "Historical gradient tuning hint withheld" in markdown
    assert "Inspect corrupt weights." in markdown
