"""Session comparisons disclose invalid evidence before judging improvements."""

import json

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def _report(tmp_path, session_id, source, *, losses=(), throughput=(), timing=()):
    collector = StatsCollector(str(tmp_path), session_id=session_id)
    for step, value in enumerate(losses, start=1):
        collector.record_training_step(step, value, 1e-4)
    # Exercise actual summaries, including values a broken timer can produce.
    for step, value in enumerate(throughput, start=1):
        collector.throughput_samples_sec.append(value, step)
    for step, value in enumerate(timing, start=1):
        collector.step_time_sec.append(value, step)
    if source == "terminal":
        return json.loads(json.dumps(collector.generate_session_report()))
    collector.flush_incremental()
    report, warning = analysis.load_incremental_report(
        str(tmp_path / f"incremental_{session_id}.jsonl"))
    assert warning is None
    return report


def _row(markdown, metric):
    return next(line for line in markdown.splitlines()
                if line.startswith(f"| {metric} |"))


def _assert_no_verdict(row):
    assert "✓" not in row
    assert "✗" not in row
    assert " = |" not in row


def _assert_diagnostic(markdown, session, metric):
    assert any(session.lower() in line.lower()
               and "nonfinite" in line.lower()
               and metric.lower() in line.lower()
               for line in markdown.splitlines())


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_loss_keeps_finite_values_without_improvement_verdict(
    tmp_path, source, reverse,
):
    healthy = _report(tmp_path, "healthy", source, losses=[2.0])
    invalid = _report(tmp_path, "invalid", source, losses=[float("nan"), 1.0])
    reports = (invalid, healthy) if reverse else (healthy, invalid)
    markdown = analysis.compare_sessions(*reports)
    for label in ("Loss (recent mean)", "Loss (best)"):
        row = _row(markdown, label)
        assert "1.000000" in row and "2.000000" in row
        assert ("+1.000000" if reverse else "-1.000000") in row
        _assert_no_verdict(row)
    _assert_diagnostic(markdown, "Session A" if reverse else "Session B", "loss")


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_throughput_keeps_finite_values_without_improvement_verdict(
    tmp_path, source, reverse,
):
    healthy = _report(tmp_path, "healthy", source, throughput=[4.0])
    invalid = _report(
        tmp_path, "invalid", source, throughput=[float("nan"), 2.0, 8.0])
    reports = (invalid, healthy) if reverse else (healthy, invalid)
    markdown = analysis.compare_sessions(*reports)
    row = _row(markdown, "Throughput (samples/s)")
    assert "4.0" in row and "5.0" in row
    assert ("-1.0" if reverse else "+1.0") in row
    _assert_no_verdict(row)
    _assert_diagnostic(markdown, "Session A" if reverse else "Session B", "throughput")


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_invalid_only_throughput_is_unavailable_and_explained(tmp_path, source, invalid):
    report = _report(tmp_path, "invalid", source, throughput=[invalid])
    healthy = _report(tmp_path, "healthy", source, throughput=[4.0])
    markdown = analysis.compare_sessions(report, healthy)
    row = _row(markdown, "Throughput (samples/s)")
    assert "| N/A | 4.0 | - |" in row
    _assert_no_verdict(row)
    _assert_diagnostic(markdown, "Session A", "throughput")


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_nonfinite_timing_prevents_throughput_improvement_verdict(tmp_path, source):
    healthy = _report(tmp_path, "healthy", source, throughput=[4.0], timing=[1.0])
    invalid = _report(
        tmp_path, "invalid", source, throughput=[8.0], timing=[float("inf"), 0.5])
    markdown = analysis.compare_sessions(healthy, invalid)
    row = _row(markdown, "Throughput (samples/s)")
    assert "| 4.0 | 8.0 | +4.0" in row
    _assert_no_verdict(row)
    _assert_diagnostic(markdown, "Session B", "throughput")
    assert any("session b" in line.lower()
               and "nonfinite" in line.lower()
               and ("timing" in line.lower() or "step time" in line.lower())
               for line in markdown.splitlines())


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_healthy_recent_samples_remain_comparable_after_older_invalid(tmp_path, source):
    window = 1000 if source == "terminal" else 100
    healthy = _report(tmp_path, "healthy", source, losses=[2.0], throughput=[4.0])
    recovered = _report(
        tmp_path, "recovered", source,
        losses=[float("nan")] + [1.0] * window,
        throughput=[float("inf")] + [8.0] * window,
    )
    markdown = analysis.compare_sessions(healthy, recovered)
    assert "✓" in _row(markdown, "Loss (recent mean)")
    assert "✓" in _row(markdown, "Throughput (samples/s)")
    _assert_no_verdict(_row(markdown, "Loss (best)"))
    _assert_diagnostic(markdown, "Session B", "loss")


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_healthy_measured_zero_stays_comparable(tmp_path, source):
    report = _report(tmp_path, "zero", source, losses=[0.0], throughput=[0.0])
    markdown = analysis.compare_sessions(report, report)
    for label in ("Loss (recent mean)", "Loss (best)", "Throughput (samples/s)"):
        assert "(N/A) = |" in _row(markdown, label)
    assert "nonfinite" not in markdown.lower()


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_measured_zero_with_invalid_samples_does_not_imply_unchanged_health(tmp_path, source):
    report = _report(
        tmp_path, "mixed_zero", source, losses=[float("nan"), 0.0],
        throughput=[float("inf"), 0.0],
    )
    markdown = analysis.compare_sessions(report, report)
    for label in ("Loss (recent mean)", "Loss (best)", "Throughput (samples/s)"):
        row = _row(markdown, label)
        assert "0.0" in row and "| N/A |" not in row
        _assert_no_verdict(row)
    for session in ("Session A", "Session B"):
        _assert_diagnostic(markdown, session, "loss")
        _assert_diagnostic(markdown, session, "throughput")


@pytest.mark.parametrize("empty_evidence", ["buffered_count", "finite_count"])
@pytest.mark.parametrize("section,subsection,label", [
    ("loss", "summary", "Loss (recent mean)"),
    ("throughput", "samples_per_sec", "Throughput (samples/s)"),
])
def test_legacy_empty_recent_evidence_cannot_borrow_lifetime_count(
    empty_evidence, section, subsection, label,
):
    report = {section: {subsection: {
        "total_count": 4, empty_evidence: 0, "recent_mean": 0.0,
        "recent_stdev": 0.0,
    }}}
    markdown = analysis.compare_sessions(report, report)
    assert "| N/A | N/A | - |" in _row(markdown, label)


@pytest.mark.parametrize("section,subsection,key,label,invalid", [
    ("throughput", "samples_per_sec", "recent_mean", "Throughput (samples/s)", -1.0),
    ("throughput", "samples_per_sec", "recent_mean", "Throughput (samples/s)", True),
    ("loss", "summary", "recent_mean", "Loss (recent mean)", True),
    ("loss", "summary", "running_min", "Loss (best)", False),
])
def test_invalid_summary_scalars_do_not_become_comparable_measurements(
    section, subsection, key, label, invalid,
):
    report = {section: {subsection: {
        "total_count": 1, "finite_count": 1, "recent_finite_count": 1, key: invalid,
    }}}
    markdown = analysis.compare_sessions(report, report)
    assert "| N/A | N/A | - |" in _row(markdown, label)
