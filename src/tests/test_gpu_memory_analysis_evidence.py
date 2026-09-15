"""GPU allocation advice follows the evidence in the persisted recent window.

Mixed allocation windows keep finite descriptive statistics, but a nonfinite
observation means the window cannot support batch-size tuning, either as an
analyzer recommendation or as a retained collector hint.
"""

import json

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


NONFINITE_RECOMMENDATION = "Nonfinite GPU allocated-memory evidence in the recent summary."
EVIDENCE_PREFIX = "GPU allocated-memory evidence: "
WITHHELD_HINT = "- GPU memory tuning hint withheld:"
LOW_HINT = {
    "area": "performance", "severity": "info",
    "hint": ("GPU VRAM utilization is only 12%. "
             "You could increase batch_size to better utilize the GPU."),
}
HIGH_HINT = {
    "area": "performance", "severity": "warning",
    "hint": "GPU VRAM utilization is 97%, near capacity. Consider reducing batch_size.",
}
HEALTHY_SUMMARY = {
    "total_count": 2, "recent_finite_count": 2, "recent_nonfinite_count": 0,
    "recent_mean": 1000.0, "recent_min": 900.0, "recent_max": 1100.0,
}
NAN = float("nan")
INF = float("inf")


def _outputs(report):
    analyses = [fn(report) for fn in (
        analysis.analyze_loss, analysis.analyze_throughput,
        analysis.analyze_gradients, analysis.analyze_model_health,
        analysis.analyze_evaluations, analysis.analyze_selfplay,
        analysis.analyze_system,
    )]
    recommendations = analysis.generate_recommendations(report, *analyses)
    markdown = analysis.format_markdown_report(report, *analyses, recommendations)
    system = analyses[-1]
    assert json.loads(json.dumps(system, allow_nan=False)) == system
    return system, recommendations, markdown


def _persisted_report(tmp_path, source, samples, vram_gb=8.0):
    collector = StatsCollector(str(tmp_path), session_id=f"gpu_allocation_{source}")
    for step, value in enumerate(samples):
        collector.gpu_mem_allocated_mb.append(value, step)
    if source == "terminal":
        report = analysis.load_session_report(collector.export_session_report())
    else:
        collector.flush_incremental()
        report, warning = analysis.load_incremental_report(
            str(tmp_path / f"incremental_{collector.session_id}.jsonl"))
        assert warning is None
    report.setdefault("meta", {})["gpu_vram_gb"] = vram_gb
    # Terminal hints depend on whether this host has CUDA; tests supply their own.
    report["optimization_hints"] = []
    return report


def _summary_report(summary, hints=(), vram_gb=8.0):
    return {
        "meta": {"gpu_vram_gb": vram_gb},
        "gpu_memory": {"allocated_mb": dict(summary)},
        "optimization_hints": [dict(hint) for hint in hints],
    }


def _batch_size_advice(recommendations):
    return [rec for rec in recommendations if rec.get("category") == "Batch Size"]


def _memory_evidence(recommendations):
    return [rec for rec in recommendations
            if rec.get("recommendation", "").startswith(NONFINITE_RECOMMENDATION)]


def _assert_evidence_instead_of_advice(recommendations):
    assert _batch_size_advice(recommendations) == []
    (evidence,) = _memory_evidence(recommendations)
    assert evidence["category"] == "Performance"
    assert evidence["priority"] == "MEDIUM"
    assert "config_change" not in evidence


# Python min/max keep a NaN only when it is the oldest sample, so both orders
# must be detected through the recorded counts rather than the extrema.
MIXED_WINDOWS = {
    "nan_first": [NAN, 512.0],
    "nan_last": [512.0, NAN],
    "positive_infinity": [512.0, INF],
    "negative_infinity": [-INF, 512.0],
}


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("samples", MIXED_WINDOWS.values(), ids=MIXED_WINDOWS.keys())
def test_mixed_allocation_window_withholds_batch_size_advice(tmp_path, source, samples):
    system, recommendations, markdown = _outputs(
        _persisted_report(tmp_path, source, samples))

    _assert_evidence_instead_of_advice(recommendations)
    assert ("GPU allocated-memory evidence: 1 recorded nonfinite observations in the "
            "recent window. Allocation-based batch-size advice is unavailable.") in markdown
    assert "GPU VRAM utilization is only" not in markdown
    # Finite descriptive values stay visible but do not authorize tuning.
    assert system["gpu_mem_mean_mb"] == 512.0
    assert system["gpu_mem_utilization"] == pytest.approx(0.064)


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_invalid_only_allocation_window_reports_its_evidence(tmp_path, source):
    system, recommendations, markdown = _outputs(
        _persisted_report(tmp_path, source, [NAN, INF]))
    _assert_evidence_instead_of_advice(recommendations)
    assert f"{EVIDENCE_PREFIX}2 recorded nonfinite observations" in markdown
    assert system.get("gpu_mem_mean_mb") is None
    assert system.get("gpu_mem_utilization") is None


def test_invalid_sample_at_recent_window_edge_withholds_advice(tmp_path):
    # Terminal memory summaries span the last 100 samples.
    _, recommendations, _ = _outputs(
        _persisted_report(tmp_path, "terminal", [NAN] + [512.0] * 99))
    _assert_evidence_instead_of_advice(recommendations)


@pytest.mark.parametrize("invalid", [NAN, INF])
def test_legacy_nonfinite_extrema_without_counts_withhold_advice(invalid):
    system, recommendations, markdown = _outputs(_summary_report({
        "total_count": 2, "recent_mean": 1000.0,
        "recent_min": 1000.0, "recent_max": invalid,
    }))
    _assert_evidence_instead_of_advice(recommendations)
    assert (f"{EVIDENCE_PREFIX}nonfinite summary statistics "
            "(observation count unavailable) in the recent window.") in markdown
    assert system["gpu_mem_utilization"] == pytest.approx(0.125)
    assert system["gpu_mem_recent_finite_count"] is None
    assert system["gpu_mem_recent_nonfinite_count"] is None


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("samples, finite, nonfinite, flagged", [
    ([500.0, 524.0], 2, 0, False),
    ([512.0, NAN], 1, 1, True),
    ([NAN, INF], 0, 2, True),
    ([NAN] + [512.0] * 100, 100, 0, False),
], ids=["healthy", "mixed", "invalid_only", "invalid_left_window"])
def test_recent_allocation_counts_reach_system_analysis(
        tmp_path, source, samples, finite, nonfinite, flagged):
    system, _, _ = _outputs(_persisted_report(tmp_path, source, samples))
    assert system["gpu_mem_recent_finite_count"] == finite
    assert system["gpu_mem_recent_nonfinite_count"] == nonfinite
    assert system["gpu_mem_has_nonfinite"] is flagged


@pytest.mark.parametrize("count", [True, -1, 1.5, "1"])
def test_malformed_recent_counts_are_not_reported_as_counts(count):
    system, _, _ = _outputs(_summary_report({
        **HEALTHY_SUMMARY, "recent_finite_count": count, "recent_nonfinite_count": count,
    }))
    assert system["gpu_mem_recent_finite_count"] is None
    assert system["gpu_mem_recent_nonfinite_count"] is None


@pytest.mark.parametrize("hint", [LOW_HINT, HIGH_HINT], ids=["low", "high"])
@pytest.mark.parametrize("summary, vram_gb", [
    ({**HEALTHY_SUMMARY, "recent_finite_count": 1, "recent_nonfinite_count": 1}, 8.0),
    ({"total_count": 2, "recent_mean": 1000.0, "recent_min": NAN, "recent_max": 1000.0}, 8.0),
    ({"total_count": 2, "recent_mean": -1.0, "recent_min": -1.0, "recent_max": -1.0}, 8.0),
    (HEALTHY_SUMMARY, None),
], ids=["counted_nonfinite", "legacy_nonfinite_extrema", "negative_mean", "unknown_capacity"])
def test_collector_memory_hint_is_withheld_without_valid_evidence(summary, vram_gb, hint):
    _, _, markdown = _outputs(_summary_report(summary, hints=[hint], vram_gb=vram_gb))
    hints_section = markdown.split("### Additional Hints from Collector", 1)[1]
    assert WITHHELD_HINT in hints_section
    assert hint["hint"] not in markdown


# Guards against over-suppression: valid evidence keeps existing advice.

@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_healthy_low_allocation_keeps_batch_size_advice(tmp_path, source):
    _, recommendations, markdown = _outputs(
        _persisted_report(tmp_path, source, [500.0, 524.0]))
    assert _memory_evidence(recommendations) == []
    (advice,) = _batch_size_advice(recommendations)
    assert advice["recommendation"].startswith("GPU VRAM utilization is only 6%.")
    assert EVIDENCE_PREFIX not in markdown


def test_invalid_sample_leaving_recent_window_restores_advice(tmp_path):
    _, recommendations, markdown = _outputs(
        _persisted_report(tmp_path, "terminal", [NAN] + [512.0] * 100))
    assert _memory_evidence(recommendations) == []
    assert len(_batch_size_advice(recommendations)) == 1
    assert EVIDENCE_PREFIX not in markdown


@pytest.mark.parametrize("hint", [LOW_HINT, HIGH_HINT], ids=["low", "high"])
def test_collector_memory_hint_is_kept_with_valid_evidence(hint):
    _, _, markdown = _outputs(_summary_report(HEALTHY_SUMMARY, hints=[hint]))
    assert hint["hint"] in markdown
    assert WITHHELD_HINT not in markdown
