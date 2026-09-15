"""Score sampling preserves invalid observations and excludes padded moves."""

import json
import math

import pytest
import torch

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def _score_stats(rows, padded):
    counts = torch.tensor([len(row) for row in rows])
    if not padded:
        scores = torch.tensor([value for row in rows for value in row])
        return StatsCollector.compute_score_stats(scores, counts)
    scores = torch.full((len(rows), max(map(len, rows), default=0)), -float("inf"))
    for index, row in enumerate(rows):
        scores[index, :len(row)] = torch.tensor(row)
    return StatsCollector.compute_score_stats_padded(scores, counts)


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_all_invalid_score_sample_replaces_prior_healthy_evidence(tmp_path, padded, invalid):
    collector = StatsCollector(str(tmp_path), session_id="invalid_scores")
    collector.record_training_step(
        200, loss=1.0, lr=1e-4,
        score_stats=_score_stats([[0.0, 0.0]], padded),
    )
    sampled = _score_stats([[invalid, invalid]], padded)
    assert set(sampled) == {"mean", "std", "entropy", "top1_margin"}
    assert not math.isfinite(sampled["mean"])
    assert not math.isfinite(sampled["std"])
    assert not math.isfinite(sampled["entropy"])
    collector.record_training_step(400, loss=1.0, lr=1e-4, score_stats=sampled)
    collector.flush_incremental()

    stream = tmp_path / "incremental_invalid_scores.jsonl"
    report, warning = analysis.load_incremental_report(str(stream))
    assert warning is None
    entropy = report["score_distribution"]["entropy_summary"]
    assert entropy["latest"]["step"] == 400
    assert entropy["recent_finite_count"] == 1
    assert entropy["recent_nonfinite_count"] == 1
    assert any("nonfinite score entropy" in hint["hint"]
               for hint in collector.generate_optimization_hints())
    assert json.loads(stream.read_text())["score_distribution"]["mean_summary"][
        "recent_nonfinite_count"] == 1


@pytest.mark.parametrize("padded", [False, True])
def test_invalid_single_move_records_scores_without_inventing_entropy(padded):
    sampled = _score_stats([[float("nan")]], padded)
    assert set(sampled) == {"mean", "std"}
    assert math.isnan(sampled["mean"])
    assert math.isnan(sampled["std"])


@pytest.mark.parametrize("padded", [False, True])
def test_empty_score_batch_remains_unobserved(padded):
    assert _score_stats([], padded) == {}


@pytest.mark.parametrize("padded", [False, True])
def test_padding_cannot_outrank_finite_negative_scores(padded):
    sampled = _score_stats([[-2e9, -4e9], [3.0, 2.0, 0.0]], padded)
    assert sampled["top1_margin"] == pytest.approx((2e9 + 1.0) / 2)
