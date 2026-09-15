"""Persisted GPU cap diagnostics remain usable in machine and human reports."""

import json

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


SCALAR_KEYS = ("gpu_power_w", "gpu_sm_clock_mhz", "gpu_temp_c")
DIAGNOSTIC_KEYS = (*SCALAR_KEYS, "gpu_throttle_reasons")
FIELDS = ("recent_mean", "recent_min", "recent_max", "latest_value", "latest_step")
ROW_LABELS = ("gpu power", "gpu sm clock", "gpu temperature", "gpu throttle")


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
    # The JSON mode must not emit NaN or Infinity for unavailable diagnostics.
    assert json.loads(json.dumps(system, allow_nan=False)) == system
    return system["gpu_diagnostics"], recommendations, markdown


def _rows(markdown, label):
    section = markdown.split("## System Resources\n", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if label in line.lower()]
    assert rows, f"Missing {label} evidence from System Resources"
    return "\n".join(rows).lower()


def _persisted_report(collector, tmp_path, source):
    if source == "terminal":
        return analysis.load_session_report(collector.export_session_report())
    collector.flush_incremental()
    report, warning = analysis.load_incremental_report(
        str(tmp_path / f"incremental_{collector.session_id}.jsonl"))
    assert warning is None
    return report


def _summary(value=12.5, step=37):
    return {
        "total_count": 2,
        "recent_mean": value,
        "recent_min": value,
        "recent_max": value,
        "latest": {"value": value, "step": step},
    }


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_collector_gpu_evidence_reaches_json_and_markdown(tmp_path, source):
    collector = StatsCollector(str(tmp_path), session_id="gpu_evidence")
    samples = {
        "gpu_power_w": (30.0, 50.0),
        "gpu_sm_clock_mhz": (900.0, 1300.0),
        "gpu_temp_c": (60.0, 70.0),
        # A bitmask's numerical maximum and mean do not describe its latest state.
        "gpu_throttle_reasons": (64.0, 4.0),
    }
    for key, (first, last) in samples.items():
        getattr(collector, key).append(first, 100)
        getattr(collector, key).append(last, 120)
    report = _persisted_report(collector, tmp_path, source)
    diagnostics, recommendations, markdown = _outputs(report)

    for key, label in zip(SCALAR_KEYS, ROW_LABELS):
        first, last = samples[key]
        expected = {
            "recent_mean": (first + last) / 2,
            "recent_min": first,
            "recent_max": last,
            "latest_value": last,
            "latest_step": 120,
        }
        assert diagnostics[key] == expected
        rows = _rows(markdown, label)
        for value in (first, (first + last) / 2, last, 120):
            assert str(int(value)) in rows
        assert "unavailable" not in rows
    assert diagnostics["gpu_throttle_reasons"] == {
        **dict.fromkeys(FIELDS), "latest_value": 4, "latest_step": 120,
    }
    throttle_row = _rows(markdown, "gpu throttle")
    assert "0x4" in throttle_row and "120" in throttle_row
    assert "0x40" not in throttle_row
    assert "34" not in throttle_row

    # Telemetry exposure alone must not create new tuning recommendations.
    without_diagnostics = {**report, "system": {
        key: value for key, value in report["system"].items()
        if key not in DIAGNOSTIC_KEYS
    }}
    assert recommendations == _outputs(without_diagnostics)[1]


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_measured_gpu_zero_is_preserved_including_zero_mask(tmp_path, source):
    collector = StatsCollector(str(tmp_path), session_id="gpu_zero")
    for key in DIAGNOSTIC_KEYS:
        getattr(collector, key).append(0.0, 0)
    diagnostics, _, markdown = _outputs(_persisted_report(collector, tmp_path, source))
    for key in SCALAR_KEYS:
        assert diagnostics[key] == dict.fromkeys(FIELDS, 0)
    assert diagnostics["gpu_throttle_reasons"] == {
        **dict.fromkeys(FIELDS), "latest_value": 0, "latest_step": 0,
    }
    assert "0x0" in _rows(markdown, "gpu throttle")
    for label in ROW_LABELS[:3]:
        assert "unavailable" not in _rows(markdown, label)


def test_missing_gpu_evidence_is_visible_as_unavailable():
    diagnostics, _, markdown = _outputs({})
    assert diagnostics == {key: dict.fromkeys(FIELDS) for key in DIAGNOSTIC_KEYS}
    for label in ROW_LABELS:
        assert "unavailable" in _rows(markdown, label)


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_empty_collector_does_not_report_zero_gpu_resources(tmp_path, source):
    collector = StatsCollector(str(tmp_path), session_id="gpu_unobserved")
    diagnostics, _, markdown = _outputs(_persisted_report(collector, tmp_path, source))
    assert diagnostics == {key: dict.fromkeys(FIELDS) for key in DIAGNOSTIC_KEYS}
    for label in ROW_LABELS:
        assert "unavailable" in _rows(markdown, label)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -1, True, "12"])
def test_invalid_scalar_gpu_values_are_unavailable(invalid):
    diagnostics, _, markdown = _outputs({"system": {
        key: _summary(invalid) for key in SCALAR_KEYS
    }})
    for key, label in zip(SCALAR_KEYS, ROW_LABELS):
        assert diagnostics[key] == dict.fromkeys(FIELDS)
        assert "unavailable" in _rows(markdown, label)


@pytest.mark.parametrize("invalid", [float("nan"), -1, 1.5, False, "4"])
def test_invalid_throttle_mask_does_not_fall_back_to_mean_or_max(invalid):
    summary = _summary(8.0)
    summary["latest"]["value"] = invalid
    diagnostics, _, markdown = _outputs({"system": {"gpu_throttle_reasons": summary}})
    assert diagnostics["gpu_throttle_reasons"] == dict.fromkeys(FIELDS)
    assert "unavailable" in _rows(markdown, "gpu throttle")


@pytest.mark.parametrize("invalid", [-1, 1.5, True, "37"])
def test_invalid_latest_step_does_not_hide_valid_gpu_value(invalid):
    diagnostics, _, markdown = _outputs({"system": {
        **{key: _summary(12.5, invalid) for key in SCALAR_KEYS},
        "gpu_throttle_reasons": _summary(4.0, invalid),
    }})
    for key in DIAGNOSTIC_KEYS:
        assert diagnostics[key]["latest_step"] is None
        assert diagnostics[key]["latest_value"] == (4 if key == "gpu_throttle_reasons" else 12.5)
    assert "0x4" in _rows(markdown, "gpu throttle")


@pytest.mark.parametrize("counter", ["total_count", "recent_finite_count"])
def test_zero_count_summary_does_not_turn_missing_samples_into_zero(counter):
    summaries = {key: {**_summary(0), counter: 0} for key in DIAGNOSTIC_KEYS}
    for summary in summaries.values():
        summary["latest"] = None
    diagnostics, _, _ = _outputs({"system": summaries})
    assert diagnostics == {key: dict.fromkeys(FIELDS) for key in DIAGNOSTIC_KEYS}


def test_legacy_summaries_preserve_independent_scalar_values_without_inventing_latest():
    diagnostics, _, markdown = _outputs({"system": {
        "gpu_power_w": {"total_count": 2, "recent_mean": 40.0,
                        "recent_min": float("nan"), "recent_max": 50.0},
        "gpu_throttle_reasons": {"total_count": 2, "recent_mean": 34.0,
                                 "recent_min": 4.0, "recent_max": 64.0},
    }})
    assert diagnostics["gpu_power_w"] == {
        "recent_mean": 40.0, "recent_min": None, "recent_max": 50.0,
        "latest_value": None, "latest_step": None,
    }
    assert diagnostics["gpu_throttle_reasons"] == dict.fromkeys(FIELDS)
    assert "unavailable" in _rows(markdown, "gpu power")
    assert "unavailable" in _rows(markdown, "gpu throttle")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_legacy_safe_mean_zero_does_not_hide_nonfinite_gpu_samples(invalid):
    diagnostics, _, markdown = _outputs({"system": {
        key: {"total_count": 1, "recent_mean": 0.0,
              "recent_min": invalid, "recent_max": invalid}
        for key in SCALAR_KEYS
    }})
    for key, label in zip(SCALAR_KEYS, ROW_LABELS):
        assert diagnostics[key] == dict.fromkeys(FIELDS)
        assert "unavailable" in _rows(markdown, label)


def test_empty_buffer_does_not_reuse_lifetime_gpu_count():
    diagnostics, _, _ = _outputs({"system": {
        key: {**_summary(4.0), "buffered_count": 0}
        for key in DIAGNOSTIC_KEYS
    }})
    assert diagnostics == {key: dict.fromkeys(FIELDS) for key in DIAGNOSTIC_KEYS}
