"""Persisted allocator and trainer memory remain visible without GPU metadata."""

import json

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


MEMORY_KEYS = ("gpu_mem_allocated_mb", "gpu_mem_reserved_mb", "process_rss_gb")
FIELDS = ("recent_mean", "recent_min", "recent_max", "latest_value", "latest_step")
LABELS = (
    "GPU allocated memory (MB)",
    "GPU reserved memory (MB)",
    "Trainer process RSS (GB)",
)


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
    return system, markdown


def _memory_cells(markdown, label):
    section = markdown.split("## System Resources\n", 1)[1].split("\n## ", 1)[0]
    assert "| Recent mean | Recent min | Recent max | Latest | Latest sample step |" in section
    matches = [line for line in section.splitlines() if line.startswith(f"| {label} |")]
    assert len(matches) == 1, f"Expected one {label} row in System Resources"
    cells = [cell.strip() for cell in matches[0].strip("|").split("|")]
    assert len(cells) == 6
    return cells[1:]


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


def _report(summaries, **meta):
    return {
        "meta": meta,
        "gpu_memory": {
            "allocated_mb": summaries.get("gpu_mem_allocated_mb", {}),
            "reserved_mb": summaries.get("gpu_mem_reserved_mb", {}),
        },
        "system": {"process_rss_gb": summaries.get("process_rss_gb", {})},
    }


def _assert_unavailable(system, markdown):
    assert system["memory_diagnostics"] == {
        key: dict.fromkeys(FIELDS) for key in MEMORY_KEYS
    }
    for alias in ("gpu_mem_mean_mb", "gpu_mem_max_mb", "gpu_mem_utilization"):
        assert system.get(alias) is None
    for label in LABELS:
        assert _memory_cells(markdown, label) == ["Unavailable"] * len(FIELDS)


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_persisted_memory_reaches_json_and_markdown_without_vram(tmp_path, source):
    collector = StatsCollector(str(tmp_path), session_id="memory_evidence")
    samples = {
        "gpu_mem_allocated_mb": (1536.0, 1024.0),
        "gpu_mem_reserved_mb": (2560.0, 2048.0),
        "process_rss_gb": (2.5, 3.5),
    }
    for key, (first, last) in samples.items():
        getattr(collector, key).append(first, 100)
        getattr(collector, key).append(last, 120)
    report = _persisted_report(collector, tmp_path, source)
    report.setdefault("meta", {}).pop("gpu_vram_gb", None)
    system, markdown = _outputs(report)

    for key, label in zip(MEMORY_KEYS, LABELS):
        first, last = samples[key]
        expected = {
            "recent_mean": (first + last) / 2,
            "recent_min": min(first, last),
            "recent_max": max(first, last),
            "latest_value": last,
            "latest_step": 120,
        }
        assert system["memory_diagnostics"][key] == expected
        assert [float(cell) for cell in _memory_cells(markdown, label)] == pytest.approx(
            [expected[field] for field in FIELDS])
    assert system["gpu_mem_mean_mb"] == 1280
    assert system["gpu_mem_max_mb"] == 1536
    assert system.get("gpu_mem_utilization") is None


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_persisted_measured_zero_memory_is_preserved(tmp_path, source):
    collector = StatsCollector(str(tmp_path), session_id="memory_zero")
    for key in MEMORY_KEYS:
        getattr(collector, key).append(0.0, 0)
    report = _persisted_report(collector, tmp_path, source)
    report.setdefault("meta", {})["gpu_vram_gb"] = 8.0
    system, markdown = _outputs(report)
    for key, label in zip(MEMORY_KEYS, LABELS):
        assert system["memory_diagnostics"][key] == dict.fromkeys(FIELDS, 0)
        assert [float(cell) for cell in _memory_cells(markdown, label)] == [0] * len(FIELDS)
    for alias in ("gpu_mem_mean_mb", "gpu_mem_max_mb", "gpu_mem_utilization"):
        assert system[alias] == 0


def test_missing_memory_evidence_is_visible_as_unavailable():
    _assert_unavailable(*_outputs({}))


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_empty_collector_does_not_claim_zero_memory(tmp_path, source):
    collector = StatsCollector(str(tmp_path), session_id="memory_empty")
    _assert_unavailable(*_outputs(_persisted_report(collector, tmp_path, source)))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -1, True, "12"])
def test_invalid_memory_cannot_reach_reports_or_legacy_aliases(invalid):
    report = _report({key: _summary(invalid) for key in MEMORY_KEYS}, gpu_vram_gb=8.0)
    _assert_unavailable(*_outputs(report))


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_persisted_nonfinite_only_memory_does_not_become_safe_zero(tmp_path, source):
    collector = StatsCollector(str(tmp_path), session_id="memory_nonfinite")
    for key in MEMORY_KEYS:
        getattr(collector, key).append(float("nan"), 100)
        getattr(collector, key).append(float("inf"), 120)
    report = _persisted_report(collector, tmp_path, source)
    report.setdefault("meta", {})["gpu_vram_gb"] = 8.0
    _assert_unavailable(*_outputs(report))


@pytest.mark.parametrize("counter", ["total_count", "recent_finite_count", "buffered_count"])
def test_empty_memory_window_does_not_reuse_lifetime_count(counter):
    summaries = {
        key: {**_summary(0), counter: 0, "latest": None} for key in MEMORY_KEYS
    }
    _assert_unavailable(*_outputs(_report(summaries, gpu_vram_gb=8.0)))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_legacy_safe_zero_memory_mean_is_unavailable_with_nonfinite_extrema(invalid):
    summaries = {
        key: {"total_count": 1, "recent_mean": 0.0,
              "recent_min": invalid, "recent_max": invalid}
        for key in MEMORY_KEYS
    }
    _assert_unavailable(*_outputs(_report(summaries, gpu_vram_gb=8.0)))


def test_legacy_memory_summaries_preserve_independent_values_without_inventing_latest():
    summaries = {
        key: {"total_count": 2, "recent_mean": 4000.0,
              "recent_min": float("nan"), "recent_max": 5000.0}
        for key in MEMORY_KEYS
    }
    system, markdown = _outputs(_report(summaries, gpu_vram_gb=8.0))
    for key, label in zip(MEMORY_KEYS, LABELS):
        assert system["memory_diagnostics"][key] == {
            "recent_mean": 4000.0, "recent_min": None, "recent_max": 5000.0,
            "latest_value": None, "latest_step": None,
        }
        mean, minimum, maximum, latest, step = _memory_cells(markdown, label)
        assert float(mean) == 4000.0
        assert float(maximum) == 5000.0
        assert minimum == latest == step == "Unavailable"
    assert system["gpu_mem_mean_mb"] == 4000.0
    assert system["gpu_mem_max_mb"] == 5000.0
    # Collector units are decimal MB and GB; 4000 MB / 8 GB is exactly 50%.
    assert system["gpu_mem_utilization"] == pytest.approx(0.5)


@pytest.mark.parametrize("invalid_step", [-1, 1.5, True, "37"])
def test_invalid_sample_step_does_not_hide_valid_memory(invalid_step):
    system, markdown = _outputs(_report({
        key: _summary(12.5, invalid_step) for key in MEMORY_KEYS
    }))
    for key, label in zip(MEMORY_KEYS, LABELS):
        assert system["memory_diagnostics"][key]["latest_value"] == 12.5
        assert system["memory_diagnostics"][key]["latest_step"] is None
        assert _memory_cells(markdown, label)[-1] == "Unavailable"


RESOURCE_SERIES = (
    ("gpu_utilization_pct", "gpu_utilization", "gpu_compute_utilization", "GPU Compute Util"),
    ("cpu_percent", "cpu_percent", "cpu_percent_mean", "CPU Usage"),
    ("ram_used_gb", "ram_used_gb", "ram_used_gb_mean", "RAM Usage"),
)


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("samples, expected", [
    ([float("nan"), float("inf")], None),
    ([0.0], 0.0),
    ([float("nan"), 12.0], 12.0),
])
def test_persisted_resource_means_require_finite_samples(tmp_path, source, samples, expected):
    collector = StatsCollector(str(tmp_path), session_id="system_resource_evidence")
    for buffer_name, _, _, _ in RESOURCE_SERIES:
        for step, value in enumerate(samples):
            getattr(collector, buffer_name).append(value, step)
    system, markdown = _outputs(_persisted_report(collector, tmp_path, source))
    for _, _, alias, label in RESOURCE_SERIES:
        assert system.get(alias) == expected
        if expected is None:
            assert f"| {label} | 0" not in markdown
        else:
            assert f"| {label} | {expected:.0f}" in markdown


@pytest.mark.parametrize("summary", [
    {"total_count": 1},
    {"total_count": 1, "recent_mean": -1.0},
    {"total_count": 1, "recent_mean": True},
    {"total_count": 1, "recent_mean": "12"},
    {"total_count": 1, "recent_mean": float("nan")},
    {"total_count": 1, "recent_mean": 0.0, "recent_max": float("inf")},
    {"total_count": 1, "recent_mean": 0.0, "buffered_count": 0},
])
def test_invalid_or_missing_legacy_resource_means_remain_unavailable(summary):
    system, markdown = _outputs({
        "system": {key: dict(summary) for _, key, _, _ in RESOURCE_SERIES}
    })
    for _, _, alias, label in RESOURCE_SERIES:
        assert system.get(alias) is None
        assert f"| {label} | 0" not in markdown


@pytest.mark.parametrize("capacity", [None, 0, -1, True, "8", float("nan"), float("inf")])
def test_unavailable_gpu_capacity_is_consistent_in_json_and_both_markdown_sections(capacity):
    report = _report({"gpu_mem_allocated_mb": _summary(4000.0)}, gpu_vram_gb=capacity)
    system, markdown = _outputs(report)
    assert system["gpu_vram_gb"] is None
    assert system.get("gpu_mem_utilization") is None
    assert system["gpu_mem_mean_mb"] == 4000.0
    assert "**GPU:** N/A (unknown)" in markdown
    assert "| VRAM | Unknown |" in markdown


def test_missing_gpu_capacity_is_not_reported_as_zero_gb():
    system, markdown = _outputs({})
    assert system["gpu_vram_gb"] is None
    assert "| VRAM | Unknown |" in markdown
