"""Saved self-play waiting remains visible in terminal and partial analyses."""

from datetime import datetime, timedelta
import json

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


WAIT_KEYS = (
    "gpu_idle_wait_count",
    "gpu_idle_wait_seconds",
    "gpu_idle_wait_max_seconds",
    "gpu_idle_wait_pct",
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
    return analyses[1], recommendations, markdown


def _wait_rows(markdown):
    section = markdown.split("## Throughput\n", 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines() if "gpu idle wait" in line.lower()]


def _wait_advice(recommendations):
    return [r for r in recommendations if r["category"] == "Self-Play"
            and "wait" in r["recommendation"].lower()]


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("training_samples", [False, True])
def test_collector_waits_reach_analysis_and_human_report(
    tmp_path, source, training_samples,
):
    collector = StatsCollector(str(tmp_path), session_id="wait_evidence")
    collector.session_start = datetime.now() - timedelta(seconds=200)
    collector.record_gpu_idle_wait(25.0)
    collector.record_gpu_idle_wait(75.0)
    if training_samples:
        collector.record_training_step(
            1, 0.75, 2e-4, batch_size=2048, step_time=0.25,
        )
    if source == "terminal":
        report = collector.generate_session_report()
    else:
        collector.flush_incremental()
        report, _ = analysis.load_incremental_report(
            str(tmp_path / "incremental_wait_evidence.jsonl"))

    throughput, recommendations, markdown = _outputs(report)
    assert throughput["has_data"] is training_samples
    assert throughput["gpu_idle_wait_count"] == 2
    assert throughput["gpu_idle_wait_seconds"] == 100.0
    assert throughput["gpu_idle_wait_max_seconds"] == 75.0
    assert throughput["gpu_idle_wait_pct"] == report["summary"]["gpu_idle_wait_pct"]
    assert 49 < throughput["gpu_idle_wait_pct"] <= 50
    # The JSON analysis must carry these fields independently of terminal data.
    encoded = json.loads(json.dumps(throughput, allow_nan=False))
    assert {key: encoded[key] for key in WAIT_KEYS} == {
        key: report["summary"][key] for key in WAIT_KEYS}
    rows = _wait_rows(markdown)
    assert len(rows) == 4
    assert any("100.0" in row for row in rows)
    assert any("75.0" in row for row in rows)
    advice = _wait_advice(recommendations)
    assert len(advice) == 1
    text = advice[0]["recommendation"].lower()
    assert "100.0" in text
    assert "generation" in text and "admission" in text
    assert "config_change" not in advice[0]
    assert "increase cpu_workers" not in text
    assert "raise cpu_workers" not in text
    if not training_samples:
        assert "MISSING (0 recorded throughput samples)" in markdown


@pytest.mark.parametrize("source", ["terminal", "incremental"])
def test_collector_zero_wait_is_measured_and_does_not_request_tuning(tmp_path, source):
    collector = StatsCollector(str(tmp_path), session_id="no_wait")
    if source == "terminal":
        report = collector.generate_session_report()
    else:
        collector.flush_incremental()
        report, _ = analysis.load_incremental_report(
            str(tmp_path / "incremental_no_wait.jsonl"))
    throughput, recommendations, markdown = _outputs(report)
    assert {key: throughput[key] for key in WAIT_KEYS} == dict.fromkeys(WAIT_KEYS, 0)
    rows = _wait_rows(markdown)
    assert len(rows) == 4
    assert all("unavailable" not in row.lower() for row in rows)
    assert not _wait_advice(recommendations)


def test_legacy_report_does_not_invent_zero_wait():
    throughput, recommendations, markdown = _outputs({})
    assert {key: throughput[key] for key in WAIT_KEYS} == dict.fromkeys(WAIT_KEYS)
    rows = _wait_rows(markdown)
    assert len(rows) == 4
    assert all("unavailable" in row.lower() for row in rows)
    assert not _wait_advice(recommendations)


@pytest.mark.parametrize("field", WAIT_KEYS)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -1])
def test_invalid_wait_measurements_remain_unavailable(field, invalid):
    throughput, recommendations, markdown = _outputs({"summary": {field: invalid}})
    assert {key: throughput[key] for key in WAIT_KEYS} == dict.fromkeys(WAIT_KEYS)
    json.dumps(throughput, allow_nan=False)
    assert len(_wait_rows(markdown)) == 4
    assert all("unavailable" in row.lower() for row in _wait_rows(markdown))
    assert not _wait_advice(recommendations)


@pytest.mark.parametrize("field,invalid", [
    ("gpu_idle_wait_count", 1.5),
    ("gpu_idle_wait_count", True),
    ("gpu_idle_wait_count", False),
    ("gpu_idle_wait_pct", 100.01),
])
def test_impossible_wait_count_or_percentage_is_not_reported(field, invalid):
    throughput, recommendations, markdown = _outputs({"summary": {field: invalid}})
    assert throughput[field] is None
    assert all("unavailable" in row.lower() for row in _wait_rows(markdown))
    assert not _wait_advice(recommendations)


def test_percentage_only_legacy_evidence_still_explains_waiting():
    throughput, recommendations, markdown = _outputs({
        "summary": {"gpu_idle_wait_pct": 52.65757335909824},
        "convergence": {"throughput_cv": 0.408},
    })
    assert throughput["gpu_idle_wait_seconds"] is None
    assert throughput["gpu_idle_wait_pct"] == 52.65757335909824
    advice = _wait_advice(recommendations)
    assert len(advice) == 1
    text = advice[0]["recommendation"].lower()
    assert "52.66%" in text
    assert "generation" in text and "admission" in text
    assert "config_change" not in advice[0]
    assert any("52.66%" in row for row in _wait_rows(markdown))


def test_longest_wait_and_event_count_alone_do_not_infer_session_wait_share():
    throughput, recommendations, markdown = _outputs({"summary": {
        "gpu_idle_wait_count": 4, "gpu_idle_wait_max_seconds": 12.0,
    }})
    assert throughput["gpu_idle_wait_count"] == 4
    assert throughput["gpu_idle_wait_max_seconds"] == 12.0
    assert throughput["gpu_idle_wait_seconds"] is None
    assert throughput["gpu_idle_wait_pct"] is None
    assert len(_wait_rows(markdown)) == 4
    assert not _wait_advice(recommendations)


def test_recorded_zero_seconds_does_not_take_positive_percentage_as_extra_wait():
    throughput, recommendations, _ = _outputs({"summary": {
        "gpu_idle_wait_seconds": 0.0, "gpu_idle_wait_pct": 25.0,
    }})
    assert throughput["gpu_idle_wait_seconds"] == 0.0
    assert throughput["gpu_idle_wait_pct"] == 25.0
    assert not _wait_advice(recommendations)


def test_known_wait_seconds_survive_invalid_count_and_percentage():
    throughput, recommendations, markdown = _outputs({"summary": {
        "gpu_idle_wait_count": 0.5,
        "gpu_idle_wait_seconds": 7.5,
        "gpu_idle_wait_pct": float("nan"),
    }})
    assert throughput["gpu_idle_wait_count"] is None
    assert throughput["gpu_idle_wait_seconds"] == 7.5
    assert throughput["gpu_idle_wait_pct"] is None
    advice = _wait_advice(recommendations)
    assert len(advice) == 1
    assert "7.5" in advice[0]["recommendation"]
    assert "%" not in advice[0]["recommendation"]
    assert any("7.5" in row for row in _wait_rows(markdown))


def test_full_session_wait_and_integral_float_count_are_reported():
    throughput, recommendations, markdown = _outputs({"summary": {
        "gpu_idle_wait_count": 2.0, "gpu_idle_wait_pct": 100.0,
    }})
    assert throughput["gpu_idle_wait_count"] == 2
    assert throughput["gpu_idle_wait_pct"] == 100.0
    assert len(_wait_advice(recommendations)) == 1
    assert any("100.00%" in row for row in _wait_rows(markdown))
