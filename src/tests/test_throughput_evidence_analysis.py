"""Performance reports preserve measured samples without inventing safe zero."""

import json
import math

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


COUNT_KEYS = (
    "total_count", "finite_count", "nonfinite_count",
    "recent_finite_count", "recent_nonfinite_count",
)
SAMPLE_FIELDS = (
    "samples_per_sec_mean", "samples_per_sec_stdev",
    "step_time_mean", "step_time_stdev",
)
SAVED_HINT = "Throughput is highly variable (CV=0.80). Check dataloader workers."


def _outputs(report):
    analyses = [fn(report) for fn in (
        analysis.analyze_loss, analysis.analyze_throughput,
        analysis.analyze_gradients, analysis.analyze_model_health,
        analysis.analyze_evaluations, analysis.analyze_selfplay,
        analysis.analyze_system,
    )]
    recommendations = analysis.generate_recommendations(report, *analyses)
    markdown = analysis.format_markdown_report(report, *analyses, recommendations)
    throughput = analyses[1]
    json.dumps(throughput, allow_nan=False)
    return throughput, recommendations, markdown


def _saved_hint(report):
    report["optimization_hints"] = [{
        "area": "performance", "severity": "warning", "hint": SAVED_HINT,
    }]


def _section(markdown):
    return markdown.split("## Throughput\n", 1)[1].split("\n## ", 1)[0]


def _row(markdown, label):
    return next(line for line in _section(markdown).splitlines()
                if line.startswith(f"| {label} |"))


def _assert_invalid_diagnostic(recommendations, markdown):
    diagnostics = [rec["recommendation"].lower() for rec in recommendations]
    assert any("nonfinite" in text and (
        "throughput" in text or "timing" in text or "performance" in text
    ) for text in diagnostics)
    assert not any("dataloader bottleneck" in text for text in diagnostics)
    assert SAVED_HINT not in markdown
    assert "No issues detected" not in markdown


def _collector_report(tmp_path, source, throughput=(), timings=()):
    collector = StatsCollector(str(tmp_path), session_id="throughput_evidence")
    # Append through the real collector's buffers: record_training_step omits
    # zero/nonpositive timing, but saved/legacy summaries may contain it.
    for step, value in enumerate(throughput, start=1):
        collector.throughput_samples_sec.append(value, step)
    for step, value in enumerate(timings, start=1):
        collector.step_time_sec.append(value, step)
    if source == "terminal":
        return json.loads(json.dumps(collector.generate_session_report()))
    collector.flush_incremental()
    report, warning = analysis.load_incremental_report(
        str(tmp_path / "incremental_throughput_evidence.jsonl"))
    assert warning is None
    return report


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_empty_collector_does_not_fabricate_performance_measurements(tmp_path, source):
    report = _collector_report(tmp_path, source)
    result, _, markdown = _outputs(report)
    assert result["has_data"] is False
    assert {key: result[key] for key in SAMPLE_FIELDS} == dict.fromkeys(SAMPLE_FIELDS)
    assert result["throughput_cv"] is None
    assert result["steps_per_hour"] is None
    for series in ("samples_per_sec", "step_time_sec"):
        assert result["sample_evidence"][series] == dict.fromkeys(COUNT_KEYS, 0)
    assert "MISSING (0 recorded throughput samples)" in markdown


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_invalid_only_samples_retain_evidence_without_measured_zero(
    tmp_path, source, invalid,
):
    report = _collector_report(tmp_path, source, [invalid], [invalid])
    _saved_hint(report)
    result, recs, markdown = _outputs(report)
    assert result["has_data"] is True
    assert result["has_nonfinite"] is True
    assert {key: result[key] for key in SAMPLE_FIELDS} == dict.fromkeys(SAMPLE_FIELDS)
    assert result["throughput_cv"] is None
    for series in ("samples_per_sec", "step_time_sec"):
        assert result["sample_evidence"][series] == {
            "total_count": 1, "finite_count": 0, "nonfinite_count": 1,
            "recent_finite_count": 0, "recent_nonfinite_count": 1,
        }
    assert "Unavailable" in _row(markdown, "Samples/sec")
    assert "Unavailable" in _row(markdown, "Step Time")
    assert "MISSING (0 recorded throughput samples)" not in markdown
    _assert_invalid_diagnostic(recs, markdown)


@pytest.mark.parametrize("series,present_label,missing_label,prefix", [
    ("samples_per_sec", "Samples/sec", "Step Time", "samples_per_sec"),
    ("step_time_sec", "Step Time", "Samples/sec", "step_time"),
])
def test_one_sampled_series_does_not_invent_the_other(
    series, present_label, missing_label, prefix,
):
    report = {"throughput": {series: {
        "total_count": 2, "recent_mean": 2.0, "recent_stdev": 1.0,
    }}}
    result, _, markdown = _outputs(report)
    assert result["has_data"] is True
    assert result[f"{prefix}_mean"] == 2.0
    assert result[f"{prefix}_stdev"] == 1.0
    assert "Unavailable" not in _row(markdown, present_label)
    assert "Unavailable" in _row(markdown, missing_label)
    assert "Unavailable" in _row(markdown, "Steps/Hour")
    assert "Unavailable" in _row(markdown, "Throughput CV")
    evidence = result["sample_evidence"][series]
    assert evidence["total_count"] == 2
    assert all(evidence[key] is None for key in COUNT_KEYS[1:])


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_mixed_recent_samples_preserve_finite_statistics_and_suppress_tuning(
    tmp_path, source,
):
    report = _collector_report(
        tmp_path, source, [float("nan"), 2.0, 4.0], [0.25, float("inf"), 0.75])
    report["convergence"]["throughput_cv"] = 0.8
    _saved_hint(report)
    result, recs, markdown = _outputs(report)
    assert result["samples_per_sec_mean"] == 3.0
    assert result["samples_per_sec_stdev"] == pytest.approx(math.sqrt(2))
    assert result["step_time_mean"] == 0.5
    assert result["step_time_stdev"] == pytest.approx(math.sqrt(0.125))
    assert result["throughput_cv"] is None
    assert result["has_nonfinite"] is True
    for series in ("samples_per_sec", "step_time_sec"):
        assert result["sample_evidence"][series]["recent_finite_count"] == 2
        assert result["sample_evidence"][series]["recent_nonfinite_count"] == 1
    assert "3.0" in _row(markdown, "Samples/sec")
    assert "500.0" in _row(markdown, "Step Time")
    _assert_invalid_diagnostic(recs, markdown)


@pytest.mark.parametrize("legacy", [False, True])
def test_measured_finite_zero_remains_valid(tmp_path, legacy):
    report = _collector_report(tmp_path, "terminal", [0.0], [0.0])
    report["convergence"]["overall_steps_per_hour"] = 0.0
    if legacy:
        for summary in report["throughput"].values():
            for key in COUNT_KEYS[1:]:
                summary.pop(key, None)
    result, _, markdown = _outputs(report)
    assert {key: result[key] for key in SAMPLE_FIELDS} == dict.fromkeys(SAMPLE_FIELDS, 0)
    assert result["has_nonfinite"] is False
    assert result["throughput_cv"] == 0.0
    assert result["steps_per_hour"] == 0.0
    assert "0.0 ± 0.0" in _row(markdown, "Samples/sec")
    assert "0.0 ms ± 0.0 ms" in _row(markdown, "Step Time")
    assert "0.000" in _row(markdown, "Throughput CV")


@pytest.mark.parametrize("known_finite", [False, True])
def test_legacy_invalid_extrema_are_visible_without_inventing_counts(known_finite):
    report = {"throughput": {"samples_per_sec": {
        "total_count": 3, "recent_mean": 0.0, "recent_stdev": 0.0,
        "recent_min": float("inf"), "recent_max": float("inf"),
    }}, "convergence": {"throughput_cv": 0.8}}
    if known_finite:
        report["throughput"]["samples_per_sec"]["recent_finite_count"] = 1
    _saved_hint(report)
    result, recs, markdown = _outputs(report)
    assert result["has_nonfinite"] is True
    assert result["samples_per_sec_mean"] == (0.0 if known_finite else None)
    assert result["samples_per_sec_stdev"] == (0.0 if known_finite else None)
    assert result["throughput_cv"] is None
    evidence = result["sample_evidence"]["samples_per_sec"]
    assert evidence["total_count"] == 3
    assert evidence["recent_finite_count"] == (1 if known_finite else None)
    assert all(evidence[key] is None for key in (
        "finite_count", "nonfinite_count", "recent_nonfinite_count"))
    _assert_invalid_diagnostic(recs, markdown)


def test_bad_recent_scalars_are_unavailable_independently_and_json_safe():
    for invalid in (None, "malformed", True, -1.0, float("nan"), float("inf")):
        for field, output in (("recent_mean", "samples_per_sec_mean"),
                              ("recent_stdev", "samples_per_sec_stdev")):
            report = {"throughput": {"samples_per_sec": {
                "total_count": 2, "recent_finite_count": 2,
                "recent_nonfinite_count": 0, "recent_mean": 4.0,
                "recent_stdev": 1.0, field: invalid,
            }}, "convergence": {"throughput_cv": 0.75}}
            _saved_hint(report)
            result, _, markdown = _outputs(report)
            assert result[output] is None, (field, invalid)
            other = "samples_per_sec_stdev" if field == "recent_mean" else "samples_per_sec_mean"
            assert result[other] == (1.0 if field == "recent_mean" else 4.0)
            assert result["throughput_cv"] is None
            assert SAVED_HINT not in markdown
            assert "Unavailable" in _row(markdown, "Samples/sec")


def test_bad_session_scalars_never_crash_markdown_or_escape_into_json():
    for invalid in (None, "malformed", True, -1.0, float("nan"), float("inf")):
        report = {"throughput": {"samples_per_sec": {
            "total_count": 1, "recent_mean": 4.0, "recent_stdev": 0.0,
        }}, "convergence": {
            "throughput_cv": invalid, "overall_steps_per_hour": invalid,
        }}
        result, recs, markdown = _outputs(report)
        assert result["throughput_cv"] is None, invalid
        assert result["steps_per_hour"] is None, invalid
        assert "Unavailable" in _row(markdown, "Steps/Hour")
        assert "Unavailable" in _row(markdown, "Throughput CV")
        assert not any("dataloader bottleneck" in rec["recommendation"].lower()
                       for rec in recs)


@pytest.mark.parametrize("empty_evidence", ["recent_finite_count", "buffered_count"])
def test_explicit_empty_recent_window_cannot_borrow_lifetime_samples(empty_evidence):
    report = {"throughput": {"samples_per_sec": {
        "total_count": 4, "finite_count": 4, "nonfinite_count": 0,
        empty_evidence: 0, "recent_nonfinite_count": 0,
        "recent_mean": 0.0, "recent_stdev": 0.0,
    }}, "convergence": {"throughput_cv": 0.8, "overall_steps_per_hour": 12.0}}
    result, _, markdown = _outputs(report)
    assert result["has_data"] is True
    assert result["samples_per_sec_mean"] is None
    assert result["samples_per_sec_stdev"] is None
    assert result["throughput_cv"] is None
    assert result["steps_per_hour"] == 12.0
    assert "12" in _row(markdown, "Steps/Hour")


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_healthy_recent_window_remains_usable_after_lifetime_invalid(tmp_path, source):
    recent_count = 1000 if source == "terminal" else 100
    values = [float("nan")] + [2.0, 8.0] * (recent_count // 2)
    report = _collector_report(tmp_path, source, values)
    _saved_hint(report)
    result, recs, markdown = _outputs(report)
    assert result["samples_per_sec_mean"] == 5.0
    assert result["has_nonfinite"] is False
    assert result["throughput_cv"] > 0.3
    evidence = result["sample_evidence"]["samples_per_sec"]
    assert evidence["nonfinite_count"] == 1
    assert evidence["recent_nonfinite_count"] == 0
    assert evidence["recent_finite_count"] == recent_count
    assert any("dataloader bottleneck" in rec["recommendation"].lower()
               for rec in recs)
    assert SAVED_HINT in markdown


def test_terminal_invalid_evidence_is_not_overruled_by_shorter_cv_window(tmp_path):
    report = _collector_report(
        tmp_path, "terminal", [float("nan")] + [2.0, 8.0] * 50)
    assert report["convergence"]["throughput_cv"] > 0.3
    _saved_hint(report)
    result, recs, markdown = _outputs(report)
    assert result["samples_per_sec_mean"] == 5.0
    assert result["sample_evidence"]["samples_per_sec"]["recent_nonfinite_count"] == 1
    assert result["throughput_cv"] is None
    _assert_invalid_diagnostic(recs, markdown)


def test_legacy_convergence_only_values_are_independent_session_evidence():
    result, recs, markdown = _outputs({"convergence": {
        "throughput_cv": 0.8, "overall_steps_per_hour": 12.0,
    }})
    assert result["has_data"] is False
    assert result["throughput_cv"] == 0.8
    assert result["steps_per_hour"] == 12.0
    assert result["sample_evidence"]["samples_per_sec"] == dict.fromkeys(COUNT_KEYS)
    assert "MISSING (0 recorded throughput samples)" in markdown
    assert "12" in _row(markdown, "Steps/Hour")
    assert any("dataloader bottleneck" in rec["recommendation"].lower()
               for rec in recs)


def test_explicit_missing_throughput_samples_cannot_support_saved_cv():
    report = {"throughput": {"samples_per_sec": {
        "total_count": 0, "recent_mean": 0.0, "recent_stdev": 0.0,
    }}, "convergence": {"throughput_cv": 0.8}}
    _saved_hint(report)
    result, recs, markdown = _outputs(report)
    assert result["throughput_cv"] is None
    assert result["samples_per_sec_mean"] is None
    assert SAVED_HINT not in markdown
    assert not any("dataloader bottleneck" in rec["recommendation"].lower()
                   for rec in recs)
