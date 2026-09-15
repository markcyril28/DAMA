"""Interrupted runs retain score measurements and honest entropy diagnostics."""

import json

import pytest

from dama.ai.ml.stats_collector import StatsCollector


def _snapshot(collector):
    collector.flush_incremental()
    stream = collector.output_dir / f"incremental_{collector.session_id}.jsonl"
    return json.loads(stream.read_text().splitlines()[-1])


def test_unobserved_score_metrics_are_explicitly_empty(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path))

    scores = _snapshot(collector)["score_distribution"]

    assert set(scores) == {
        "mean_summary", "std_summary", "entropy_summary", "top1_margin_summary",
    }
    for summary in scores.values():
        assert summary["total_count"] == 0
        assert summary["recent_finite_count"] == 0
        assert summary["latest"] is None
    assert collector.generate_optimization_hints() == []


def test_sampled_score_metrics_survive_without_terminal_export(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path), flush_every=10000)
    for step, score_stats in [
        (1000, {"mean": 1.0, "std": 2.0, "entropy": 0.5, "top1_margin": 3.0}),
        (2000, {"mean": 0.0, "std": 0.0, "entropy": 0.0, "top1_margin": 0.0}),
        # An ordinary loss observation must not advance the score sample step.
        (2200, None),
    ]:
        collector.record_training_step(step, loss=1.0, lr=0.001, score_stats=score_stats)

    scores = _snapshot(collector)["score_distribution"]
    terminal = collector.generate_session_report()["score_distribution"]

    for name, summary in terminal.items():
        assert scores[name] == {**summary, "latest": {"step": 2000, "value": 0.0}}
        assert summary["total_count"] == 2
    assert not list(tmp_path.glob("session_report_*.json"))


def test_incremental_scores_use_terminal_thousand_sample_window(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path), flush_every=10000)
    for step in range(1, 1002):
        collector.record_training_step(
            step, loss=1.0, lr=0.001, score_stats={"entropy": float(step)},
        )

    summary = _snapshot(collector)["score_distribution"]["entropy_summary"]

    assert summary["total_count"] == 1001
    assert summary["recent_finite_count"] == 1000
    assert summary["recent_min"] == 2.0
    assert summary["recent_mean"] == pytest.approx(501.5)
    assert summary["latest"] == {"step": 1001, "value": 1001.0}


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("finite_zero", [False, True])
def test_invalid_entropy_is_stability_evidence_not_confidence(tmp_path, invalid, finite_zero):
    collector = StatsCollector(output_dir=str(tmp_path))
    if finite_zero:
        collector.record_training_step(1000, loss=1.0, lr=0.001, score_stats={"entropy": 0.0})
    collector.record_training_step(2000, loss=1.0, lr=0.001, score_stats={"entropy": invalid})

    hints = collector.generate_optimization_hints()

    assert not any(hint["area"] == "model_behavior" for hint in hints)
    stability = [hint for hint in hints if hint["area"] == "stability"]
    assert len(stability) == 1
    assert "nonfinite score entropy" in stability[0]["hint"]
    summary = _snapshot(collector)["score_distribution"]["entropy_summary"]
    assert summary["recent_finite_count"] == int(finite_zero)
    assert summary["recent_nonfinite_count"] == 1


def test_measured_zero_entropy_retains_confidence_hint(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path))
    collector.record_training_step(1000, loss=1.0, lr=0.001, score_stats={"entropy": 0.0})

    hints = collector.generate_optimization_hints()

    assert len(hints) == 1
    assert hints[0]["area"] == "model_behavior"
    assert "0.000" in hints[0]["hint"]


def test_old_invalid_entropy_does_not_poison_recent_hint_window(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path), flush_every=10000)
    collector.record_training_step(1, loss=1.0, lr=0.001, score_stats={"entropy": float("nan")})
    for step in range(2, 102):
        collector.record_training_step(step, loss=1.0, lr=0.001, score_stats={"entropy": 0.0})

    hints = collector.generate_optimization_hints()

    assert len(hints) == 1
    assert hints[0]["area"] == "model_behavior"
