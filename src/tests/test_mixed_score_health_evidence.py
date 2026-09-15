"""Finite rows cannot conceal invalid model scores in the same sample."""

import math

import pytest
import torch

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def _score_stats(rows, padded):
    counts = torch.tensor([len(row) for row in rows])
    if not padded:
        return StatsCollector.compute_score_stats(
            torch.tensor([value for row in rows for value in row]), counts)
    scores = torch.full((len(rows), max(map(len, rows))), -float("inf"))
    for index, row in enumerate(rows):
        scores[index, :len(row)] = torch.tensor(row)
    return StatsCollector.compute_score_stats_padded(scores, counts)


def _render(report):
    analyses = [fn(report) for fn in (
        analysis.analyze_loss, analysis.analyze_throughput,
        analysis.analyze_gradients, analysis.analyze_model_health,
        analysis.analyze_evaluations, analysis.analyze_selfplay,
        analysis.analyze_system,
    )]
    recommendations = analysis.generate_recommendations(report, *analyses)
    return analysis.format_markdown_report(report, *analyses, recommendations)


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("finite_multi", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_mixed_invalid_scores_reach_reports_without_confidence_advice(
    tmp_path, padded, invalid, finite_multi, incremental,
):
    collector = StatsCollector(str(tmp_path), session_id="mixed")
    collector.record_training_step(
        100, 1.0, 1e-4, score_stats=_score_stats([[0.0, -1000.0]], padded))
    rows = [[invalid], [0.0, -1000.0] if finite_multi else [0.0]]
    sampled = _score_stats(rows, padded)
    assert math.isnan(sampled["mean"])
    assert math.isnan(sampled["std"])
    collector.record_training_step(200, 1.0, 1e-4, score_stats=sampled)
    # Invalid forced rows supply no entropy of their own. Other measured rows
    # may still supply valid entropy, without authorizing healthy-model advice.
    assert collector.score_entropy.count == (2 if finite_multi else 1)
    assert collector.score_entropy.summary()["recent_nonfinite_count"] == 0
    hints = collector.generate_optimization_hints()
    assert any("nonfinite score distribution" in hint["hint"] for hint in hints)
    assert not any(hint["area"] == "model_behavior" for hint in hints)
    if incremental:
        collector.flush_incremental()
        report, warning = analysis.load_incremental_report(
            str(tmp_path / "incremental_mixed.jsonl"))
        assert warning is None
    else:
        report = collector.generate_session_report()
    for key in ("mean_summary", "std_summary"):
        assert report["score_distribution"][key]["recent_nonfinite_count"] == 1
        assert report["score_distribution"][key]["recent_finite_count"] == 1
    rendered = _render(report)
    assert "nonfinite score distribution" in rendered
    assert "increasing noise_prob" not in rendered


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("rows", [[[0.0], [0.0]], [[2.0], [0.0, -1000.0]]])
def test_finite_scores_and_negative_infinity_padding_remain_measured(padded, rows):
    sampled = _score_stats(rows, padded)
    flattened = torch.tensor([value for row in rows for value in row])
    assert sampled["mean"] == pytest.approx(flattened.mean().item())
    assert sampled["std"] == pytest.approx(flattened.std().item())
