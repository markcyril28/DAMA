"""Keep measured model health available when only incremental logs survive."""

import json

import pytest
import torch

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def test_incremental_model_health_reaches_analyzer_without_terminal_report(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path))
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    collector.record_model_health(model, 5000)
    collector.flush_incremental()

    stream = tmp_path / f"incremental_{collector.session_id}.jsonl"
    first, warning = analysis.load_incremental_report(str(stream))
    assert warning is None
    assert analysis.analyze_model_health(first) == {"num_layers": 1}

    with torch.no_grad():
        model.weight.fill_(2.0)
    collector.record_model_health(model, 10000)
    collector.flush_incremental()

    report, warning = analysis.load_incremental_report(str(stream))
    assert warning is None
    result = analysis.analyze_model_health(report)
    assert result["num_layers"] == 1
    assert result["mean_update_ratio"] == pytest.approx(0.5)
    assert result["lr_possibly_too_high"] is True
    assert report["model_health"]["param_norm_summaries"]["weight"]["latest"]["step"] == 10000
    assert not list(tmp_path.glob("session_report_*.json"))


@pytest.mark.parametrize("ratios,expected", [
    ({"unobserved": {"total_count": 0, "recent_mean": 0.0}}, None),
    ({"unobserved": {"total_count": 0, "recent_mean": 0.0},
      "observed": {"total_count": 1, "recent_mean": 0.004}}, 0.004),
    ({"observed": {"total_count": 1, "recent_mean": 0.0}}, 0.0),
])
def test_model_health_does_not_treat_unobserved_updates_as_zero(ratios, expected):
    result = analysis.analyze_model_health({
        "model_health": {"weight_update_ratio_summaries": ratios},
    })
    if expected is None:
        assert "mean_update_ratio" not in result
        assert "lr_possibly_too_low" not in result
    else:
        assert result["mean_update_ratio"] == pytest.approx(expected)
        assert result["lr_possibly_too_low"] is (expected == 0.0)


def test_legacy_incremental_model_health_remains_unavailable(tmp_path):
    stream = tmp_path / "incremental_legacy.jsonl"
    stream.write_text(json.dumps({"timestamp": "2026-09-10T14:30:57"}) + "\n")
    report, warning = analysis.load_incremental_report(str(stream))
    assert warning is None
    assert report["model_health"] == {}
    assert analysis.analyze_model_health(report) == {"num_layers": 0}
