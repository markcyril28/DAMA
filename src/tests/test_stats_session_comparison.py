"""Session comparisons preserve measured zero without inventing missing data."""

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def _report(tmp_path, session_id, loss, win_rate):
    collector = StatsCollector(str(tmp_path), session_id=session_id)
    collector.record_training_step(step=200, loss=loss, lr=1e-4)
    collector.record_evaluation(200, 1, {"ml_win_rate": win_rate})
    return collector.generate_session_report()


def _row(markdown, metric):
    return next(line for line in markdown.splitlines()
                if line.startswith(f"| {metric} |"))


@pytest.mark.parametrize("reverse", [False, True])
def test_comparison_retains_measured_zero_loss_and_win_rate(tmp_path, reverse):
    first = _report(tmp_path, "first", loss=0.0, win_rate=0.0)
    second = _report(tmp_path, "second", loss=1.0, win_rate=0.5)
    reports = (second, first) if reverse else (first, second)
    markdown = analysis.compare_sessions(*reports)

    loss_row = _row(markdown, "Loss (recent mean)")
    assert "0.000000" in loss_row
    assert ("-1.000000" if reverse else "+1.000000") in loss_row
    assert ("(-100.0%)" if reverse else "(N/A)") in loss_row
    assert "0.000000" in _row(markdown, "Loss (best)")
    assert "0.000" in _row(markdown, "Win Rate")
    # A 50% result produces a measured Elo difference of zero.
    assert "N/A |" not in _row(markdown, "ELO Diff")


def test_equal_zero_measurements_are_unchanged(tmp_path):
    report = _report(tmp_path, "zero", loss=0.0, win_rate=0.5)
    row = _row(analysis.compare_sessions(report, report), "Loss (recent mean)")
    assert "| 0.000000 | 0.000000 | +0.000000 (N/A) = |" in row


def test_empty_collector_stays_unavailable(tmp_path):
    empty = StatsCollector(str(tmp_path), session_id="empty").generate_session_report()
    measured = _report(tmp_path, "measured", loss=0.0, win_rate=0.0)
    markdown = analysis.compare_sessions(empty, measured)
    assert "| N/A | 0.000000 | - |" in _row(markdown, "Loss (recent mean)")
    assert "| N/A | 0.000000 | - |" in _row(markdown, "Loss (best)")
    assert "| N/A | N/A | - |" in _row(markdown, "Throughput (samples/s)")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_comparison_values_remain_unavailable(tmp_path, invalid):
    report = _report(tmp_path, "measured", loss=1.0, win_rate=0.5)
    invalid_report = _report(tmp_path, "invalid", loss=invalid, win_rate=0.0)
    invalid_report["evaluations"][-1]["ml_win_rate"] = invalid
    markdown = analysis.compare_sessions(invalid_report, report)
    assert "| N/A | 1.000000 | - |" in _row(markdown, "Loss (recent mean)")
    assert "| N/A | 1.000000 | - |" in _row(markdown, "Loss (best)")
    assert "| N/A | 0.500 | - |" in _row(markdown, "Win Rate")


def test_missing_evaluation_fields_do_not_become_zero(tmp_path):
    report = _report(tmp_path, "measured", loss=0.0, win_rate=0.0)
    markdown = analysis.compare_sessions({"evaluations": [{}]}, report)
    assert "| N/A | 0.000 | - |" in _row(markdown, "Win Rate")
    assert "| N/A | -800 | - |" in _row(markdown, "ELO Diff")


def test_legacy_numeric_summaries_without_counts_remain_comparable():
    report = {"loss": {"summary": {"recent_mean": 1.0, "running_min": 1.0}}}
    zero = {"loss": {"summary": {"recent_mean": 0.0, "running_min": 0.0}}}
    markdown = analysis.compare_sessions(report, zero)
    assert "| 1.000000 | 0.000000 | -1.000000 (-100.0%) ✓ |" in _row(
        markdown, "Loss (recent mean)")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_legacy_invalid_recent_extrema_do_not_establish_measured_zero(invalid):
    legacy = {"loss": {"summary": {
        "total_count": 1, "recent_mean": 0.0,
        "recent_min": invalid, "recent_max": invalid,
    }}}
    measured = {"loss": {"summary": {"recent_mean": 1.0}}}
    markdown = analysis.compare_sessions(legacy, measured)
    assert "| N/A | 1.000000 | - |" in _row(markdown, "Loss (recent mean)")
