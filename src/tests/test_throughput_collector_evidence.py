"""Collector summaries and hints must retain invalid performance evidence."""

import json
import math

import pytest

from dama.ai.ml.stats_collector import StatsCollector


def _collector(tmp_path):
    collector = StatsCollector(str(tmp_path), session_id="performance_evidence")
    collector._flush_max_seconds = float("inf")
    return collector


def _persisted(collector, source):
    if source == "terminal":
        return collector.generate_session_report()
    collector.flush_incremental()
    return json.loads((collector.output_dir / "incremental_performance_evidence.jsonl").read_text())


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("series,prefix,mean_key,std_key", [
    ("throughput_samples_sec", "throughput", "throughput_mean", "throughput_stdev"),
    ("step_time_sec", "step_time", "step_time_mean_sec", "step_time_stdev_sec"),
])
def test_invalid_only_performance_is_unavailable_in_collector_outputs(
    tmp_path, capsys, source, invalid, series, prefix, mean_key, std_key,
):
    collector = _collector(tmp_path)
    getattr(collector, series).append(invalid, 200)
    report = _persisted(collector, source)
    conv = report["convergence"]
    assert conv[mean_key] is None
    assert conv[std_key] is None
    assert conv[f"{prefix}_recent_finite_count"] == 0
    assert conv[f"{prefix}_recent_nonfinite_count"] == 1
    assert conv.get("throughput_cv") is None
    json.dumps(conv, allow_nan=False)
    hints = collector.generate_optimization_hints()
    assert any("nonfinite" in hint["hint"].lower() for hint in hints)
    assert not any("Throughput is highly variable" in hint["hint"] for hint in hints)
    collector.print_session_summary()
    output = capsys.readouterr().out.split("  Throughput:", 1)[1]
    assert "Unavailable" in output
    assert "Nonfinite" in output
    assert "Samples/sec:        0.0" not in output


@pytest.mark.parametrize("invalid_series", ["throughput_samples_sec", "step_time_sec"])
def test_mixed_performance_retains_finite_values_without_tuning(tmp_path, invalid_series):
    collector = _collector(tmp_path)
    for step, value in enumerate([2.0, 8.0], 1):
        collector.throughput_samples_sec.append(value, step)
        collector.step_time_sec.append(value / 10, step)
    getattr(collector, invalid_series).append(float("nan"), 3)
    conv = collector.get_convergence_metrics()
    assert conv["throughput_mean"] == 5.0
    assert conv["throughput_stdev"] == pytest.approx(math.sqrt(18))
    assert conv["step_time_mean_sec"] == 0.5
    assert conv["throughput_cv"] is None
    hints = collector.generate_optimization_hints()
    assert any("nonfinite" in hint["hint"].lower() for hint in hints)
    assert not any("Throughput is highly variable" in hint["hint"] for hint in hints)


def test_measured_zero_and_empty_performance_remain_distinct(tmp_path, capsys):
    collector = _collector(tmp_path)
    assert collector.get_convergence_metrics().get("throughput_cv") is None
    collector.print_session_summary()
    assert "  Throughput:" not in capsys.readouterr().out
    collector.throughput_samples_sec.append(0.0, 1)
    collector.step_time_sec.append(0.0, 1)
    conv = collector.get_convergence_metrics()
    assert conv["throughput_cv"] == 0.0
    assert conv["throughput_mean"] == 0.0
    assert conv["step_time_mean_sec"] == 0.0
    collector.print_session_summary()
    assert "Samples/sec:        0.0" in capsys.readouterr().out


def test_healthy_recent_performance_survives_older_invalid_observation(tmp_path):
    collector = _collector(tmp_path)
    collector.throughput_samples_sec.append(float("nan"), 0)
    collector.step_time_sec.append(float("inf"), 0)
    for step, value in enumerate([2.0, 8.0] * 50, 1):
        collector.throughput_samples_sec.append(value, step)
        collector.step_time_sec.append(value / 10, step)
    conv = collector.get_convergence_metrics()
    assert conv["throughput_cv"] > 0.5
    assert conv["throughput_recent_nonfinite_count"] == 0
    assert conv["step_time_recent_nonfinite_count"] == 0
    hints = collector.generate_optimization_hints()
    assert any("Throughput is highly variable" in hint["hint"] for hint in hints)
    assert not any("nonfinite" in hint["hint"].lower() for hint in hints)


def test_nonfinite_timing_from_public_recording_remains_visible(tmp_path):
    collector = _collector(tmp_path)
    collector.record_training_step(1, loss=1.0, lr=1e-4, batch_size=32,
                                   step_time=float("inf"))
    conv = collector.get_convergence_metrics()
    assert conv["step_time_mean_sec"] is None
    assert conv["step_time_recent_nonfinite_count"] == 1
    assert conv["throughput_cv"] is None
