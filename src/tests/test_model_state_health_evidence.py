"""Parameter and BatchNorm state evidence must survive diagnostic analysis."""

import pytest
import torch

from dama.ai.ml.stats_collector import MetricBuffer, StatsCollector
from scripts import analyze_training_stats as analysis


FAMILIES = (
    "param_norm_summaries",
    "bn_running_mean_norms",
    "bn_running_var_means",
)


def _analyses(report):
    return {
        "loss_analysis": analysis.analyze_loss(report),
        "throughput_analysis": analysis.analyze_throughput(report),
        "grad_analysis": analysis.analyze_gradients(report),
        "model_analysis": analysis.analyze_model_health(report),
        "eval_analysis": analysis.analyze_evaluations(report),
        "selfplay_analysis": analysis.analyze_selfplay(report),
        "system_analysis": analysis.analyze_system(report),
    }


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("incremental", [False, True])
def test_nonfinite_state_reaches_terminal_and_incremental_analysis(
    tmp_path, family, invalid, incremental,
):
    collector = StatsCollector(str(tmp_path), session_id="state")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    collector.record_training_step(200, loss=1.0, lr=1e-4)
    if family != "param_norm_summaries":
        collector.record_model_health(model, 5000)
    with torch.no_grad():
        if family == "param_norm_summaries":
            model[0].weight.fill_(invalid)
        elif family == "bn_running_mean_norms":
            model[1].running_mean.fill_(invalid)
        else:
            model[1].running_var.fill_(invalid)
    collector.record_model_health(model, 10000)
    if incremental:
        collector.flush_incremental()
        report, warning = analysis.load_incremental_report(
            str(tmp_path / "incremental_state.jsonl"))
        assert warning is None
    else:
        report = collector.generate_session_report()

    analyses = _analyses(report)
    health = analyses["model_analysis"]
    layer = "0.weight" if family == "param_norm_summaries" else "bn_1"
    assert health["nonfinite_state_layers"] == {family: [layer]}
    assert "lr_possibly_too_low" not in health
    assert "lr_possibly_too_high" not in health
    if family == "param_norm_summaries":
        # A first snapshot cannot measure an update, even for an invalid weight.
        assert "mean_update_ratio" not in health
    else:
        assert health["mean_update_ratio"] == 0.0
    recs = analysis.generate_recommendations(report, **analyses)
    assert any("nonfinite model-state" in rec["recommendation"] for rec in recs)
    assert not any(rec["category"] == "Learning Rate" for rec in recs)
    markdown = analysis.format_markdown_report(report, recommendations=recs, **analyses)
    assert "nonfinite model-state" in markdown
    assert "| LR Too High?" not in markdown
    assert "| LR Too Low?" not in markdown
    if family != "param_norm_summaries":
        assert "| LR assessment | Unavailable (nonfinite model state) |" in markdown


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("field", ["recent_mean", "recent_stdev", "recent_min", "recent_max"])
def test_legacy_nonfinite_state_statistics_remain_visible(family, field):
    report = {"model_health": {family: {
        "layer": {"total_count": 10, "recent_mean": 0.0, field: float("nan")},
    }}}
    analyses = _analyses(report)
    assert analyses["model_analysis"]["nonfinite_state_layers"] == {family: ["layer"]}
    recs = analysis.generate_recommendations(report, **analyses)
    state_rec = next(rec for rec in recs if "nonfinite model-state" in rec["recommendation"])
    assert "observations" not in state_rec["recommendation"]


@pytest.mark.parametrize("family", FAMILIES)
def test_old_invalid_state_does_not_override_finite_recent_window(family):
    state = MetricBuffer()
    state.append(float("nan"), 1)
    state.append(0.0, 2)
    update = MetricBuffer()
    update.append(0.004, 2)
    report = {"model_health": {
        family: {"layer": state.summary(1)},
        "weight_update_ratio_summaries": {"weight": update.summary()},
    }}
    result = analysis.analyze_model_health(report)
    assert "nonfinite_state_layers" not in result
    assert result["mean_update_ratio"] == 0.004
    assert result["lr_possibly_too_low"] is False


def test_healthy_zero_state_keeps_measured_zero_update_advice():
    value = MetricBuffer()
    value.append(0.0, 1)
    report = {"model_health": {
        family: {"layer": value.summary()} for family in
        (*FAMILIES, "weight_update_ratio_summaries")
    }}
    result = analysis.analyze_model_health(report)
    assert "nonfinite_state_layers" not in result
    assert result["mean_update_ratio"] == 0.0
    assert result["lr_possibly_too_low"] is True


@pytest.mark.parametrize("grad_norm", [0.0, 101.0])
@pytest.mark.parametrize("state_evidence", ["healthy", "recent_invalid", "older_invalid"])
def test_model_state_evidence_controls_generated_and_saved_tuning_hints(
    tmp_path, grad_norm, state_evidence,
):
    collector = StatsCollector(str(tmp_path), session_id="state_hints")
    model = torch.nn.BatchNorm1d(2)
    collector.record_model_health(model, 1)
    if state_evidence != "healthy":
        model.running_mean.fill_(float("nan"))
        collector.record_model_health(model, 2)
        if state_evidence == "older_invalid":
            model.running_mean.zero_()
            # Only the supplied 50-observation window controls the assessment.
            for step in range(3, 53):
                collector.record_model_health(model, step)
    for step in (200, 400, 600, 800):
        collector.record_training_step(step, 1.0, 1e-4, grad_norm=grad_norm)
    report = collector.generate_session_report()
    analyses = _analyses(report)
    assert analyses["loss_analysis"]["is_plateauing"] is True
    assert analyses["grad_analysis"]["global_mean"] == grad_norm
    assert any(hint["area"] == "convergence" for hint in report["optimization_hints"])
    gradient_hint = ("Very small gradient norms detected" if grad_norm == 0.0
                     else "Gradient norms exceeding 100.0 detected")
    assert any(hint["hint"].startswith(gradient_hint)
               for hint in report["optimization_hints"])
    recs = analysis.generate_recommendations(report, **analyses)
    markdown = analysis.format_markdown_report(report, recommendations=recs, **analyses)
    if state_evidence == "recent_invalid":
        assert not any(rec["category"] == "Convergence" for rec in recs)
        assert not any("Gradient explosion detected" in rec["recommendation"] for rec in recs)
        assert gradient_hint not in markdown
        assert "Consider: (1) reducing learning rate" not in markdown
        assert "nonfinite model-state" in markdown
        assert "Model state tuning hint withheld" in markdown
    else:
        assert any(rec["category"] == "Convergence" for rec in recs)
        assert any("Gradient explosion detected" in rec["recommendation"]
                   for rec in recs) is (grad_norm > 100.0)
        assert gradient_hint in markdown
        assert "Consider: (1) reducing learning rate" in markdown
        assert "Model state tuning hint withheld" not in markdown
