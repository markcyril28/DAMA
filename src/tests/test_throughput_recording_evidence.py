"""Public training telemetry retains invalid timings without inventing rates."""

import json
import math

import pytest

from dama.ai.ml.stats_collector import StatsCollector


def _collector(tmp_path):
    collector = StatsCollector(str(tmp_path), session_id="recording_evidence")
    collector._flush_max_seconds = float("inf")
    return collector


def _record(collector, step, step_time, batch_size=8):
    collector.record_training_step(
        step, loss=1.0, lr=1e-4, batch_size=batch_size, step_time=step_time,
    )


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("has_history", [False, True])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_recorded_invalid_timing_survives_export_without_usable_sample_rate(
    tmp_path, capsys, source, has_history, invalid,
):
    collector = _collector(tmp_path)
    if has_history:
        _record(collector, 1, 4.0)
        _record(collector, 2, 1.0)
        assert collector.get_convergence_metrics()["throughput_cv"] > 0.5
    _record(collector, 3, invalid)

    if source == "terminal":
        report = collector.generate_session_report()
        rate_summary = report["throughput"]["samples_per_sec"]
        timing_summary = report["throughput"]["step_time_sec"]
    else:
        collector.flush_incremental()
        report = json.loads(
            (tmp_path / "incremental_recording_evidence.jsonl").read_text()
        )
        rate_summary = report["throughput_summary"]
        timing_summary = report["step_time_summary"]

    for summary in (rate_summary, timing_summary):
        assert summary["recent_nonfinite_count"] == 1
        assert summary["recent_finite_count"] == (2 if has_history else 0)
    conv = report["convergence"]
    assert conv["throughput_mean"] == (5.0 if has_history else None)
    assert conv["step_time_mean_sec"] == (2.5 if has_history else None)
    assert conv["throughput_cv"] is None
    hints = collector.generate_optimization_hints()
    assert any("nonfinite" in hint["hint"].lower() for hint in hints)
    assert not any("Throughput is highly variable" in hint["hint"] for hint in hints)
    collector.print_session_summary()
    output = capsys.readouterr().out
    assert "Nonfinite Step time" in output
    if not has_history:
        assert "Unavailable (no finite observations)" in output
        assert "Samples/sec:        0.0" not in output


@pytest.mark.parametrize("step_time", [None, 0.0, -1.0])
def test_missing_or_nonpositive_finite_timing_keeps_existing_omission(tmp_path, step_time):
    collector = _collector(tmp_path)
    _record(collector, 1, step_time)
    assert collector.step_time_sec.count == 0
    assert collector.throughput_samples_sec.count == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_invalid_timing_without_batch_count_still_warns(tmp_path, invalid):
    collector = _collector(tmp_path)
    _record(collector, 1, invalid, batch_size=0)
    assert collector.throughput_samples_sec.count == 0
    assert collector.step_time_sec.count == 1
    assert not math.isfinite(collector.step_time_sec.last_n_values(1)[0])
    hints = collector.generate_optimization_hints()
    assert any("nonfinite" in hint["hint"].lower() for hint in hints)


def test_recorded_recent_window_recovers_after_invalid_timing(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, 0, float("nan"))
    for step in range(1, 101):
        _record(collector, step, 4.0 if step % 2 else 1.0)
    report = collector.generate_session_report()
    assert report["throughput"]["samples_per_sec"]["nonfinite_count"] == 1
    conv = report["convergence"]
    assert conv["throughput_recent_nonfinite_count"] == 0
    assert conv["step_time_recent_nonfinite_count"] == 0
    assert conv["throughput_mean"] == 5.0
    assert conv["throughput_cv"] > 0.5
    hints = collector.generate_optimization_hints()
    assert any("Throughput is highly variable" in hint["hint"] for hint in hints)
    assert not any("nonfinite" in hint["hint"].lower() for hint in hints)
