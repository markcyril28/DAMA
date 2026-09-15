"""Interrupted sessions retain sampled model health without terminal export."""

import json

import pytest
import torch

from dama.ai.ml.stats_collector import StatsCollector


def _last_snapshot(collector):
    collector.flush_incremental()
    path = collector.output_dir / f"incremental_{collector.session_id}.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


def test_incremental_model_health_distinguishes_unobserved_parameters(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path), session_id="unobserved")

    assert _last_snapshot(collector)["model_health"] == {
        "param_norm_summaries": {},
        "weight_update_ratio_summaries": {},
        "bn_running_mean_norms": {},
        "bn_running_var_means": {},
    }
    assert not list(tmp_path.glob("session_report_*.json"))


def test_incremental_model_health_retains_updates_without_terminal_export(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path), session_id="interrupted")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[1].running_mean.copy_(torch.tensor([3.0, 4.0]))
        model[1].running_var.copy_(torch.tensor([2.0, 4.0]))
    collector.record_model_health(model, step=5000)
    first = _last_snapshot(collector)["model_health"]

    with torch.no_grad():
        model[0].weight.fill_(2.0)
        model[1].running_mean.copy_(torch.tensor([0.0, 2.0]))
        model[1].running_var.copy_(torch.tensor([4.0, 8.0]))
    collector.record_model_health(model, step=10000)
    second = _last_snapshot(collector)["model_health"]

    assert first["param_norm_summaries"]["0.weight"]["latest"] == {
        "step": 5000, "value": 2.0,
    }
    assert first["weight_update_ratio_summaries"]["0.weight"]["latest"] is None
    assert first["weight_update_ratio_summaries"]["0.weight"]["total_count"] == 0
    assert second["param_norm_summaries"]["0.weight"]["latest"] == {
        "step": 10000, "value": 4.0,
    }
    assert second["weight_update_ratio_summaries"]["0.weight"]["latest"] == {
        "step": 10000, "value": 0.5,
    }
    assert second["bn_running_mean_norms"]["bn_1"]["latest"] == {
        "step": 10000, "value": 2.0,
    }
    assert second["bn_running_var_means"]["bn_1"]["latest"] == {
        "step": 10000, "value": 6.0,
    }

    terminal = collector.generate_session_report()["model_health"]
    for family, summaries in terminal.items():
        for name, summary in summaries.items():
            assert second[family][name] == {
                **summary, "latest": second[family][name]["latest"],
            }
    assert not list(tmp_path.glob("session_report_*.json"))


def test_incremental_model_health_bounds_recent_summary_window(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path), session_id="window")
    model = torch.nn.Linear(1, 1, bias=False)
    for step in range(1, 61):
        with torch.no_grad():
            model.weight.fill_(float(step))
        collector.record_model_health(model, step=step)

    summary = _last_snapshot(collector)["model_health"]["param_norm_summaries"]["weight"]
    assert summary["total_count"] == 60
    assert summary["recent_min"] == 11.0
    assert summary["recent_max"] == 60.0
    assert summary["recent_mean"] == pytest.approx(35.5)
    assert summary["latest"] == {"step": 60, "value": 60.0}
