"""Sparse loss observations must not masquerade as optimizer steps."""

import json

import pytest

from dama.ai.ml.stats_collector import StatsCollector


def _collector(tmp_path):
    collector = StatsCollector(str(tmp_path), session_id="windows", flush_every=10000)
    collector._flush_max_seconds = float("inf")
    return collector


def _record(collector, observations):
    for step, loss in observations:
        collector.record_training_step(step=step, loss=loss, lr=2e-4)


def test_sparse_windows_use_optimizer_steps_and_publish_evidence(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, [(408000 + i * 200, 2.0 if i <= 12 else 1.0)
                        for i in range(1, 26)])
    collector.flush_incremental()
    conv = json.loads((tmp_path / "incremental_windows.jsonl").read_text())["convergence"]
    assert conv["loss_window_basis"] == "optimizer_steps"
    assert conv["loss_window_sample_counts"] == {"100": 1, "1000": 5, "5000": 25}
    assert conv["loss_improvement_100"] is None
    assert conv["loss_improvement_pct_100"] is None
    assert conv["loss_improvement_1000"] is None
    assert conv["loss_improvement_pct_1000"] is None
    assert conv["loss_improvement_5000"] == pytest.approx(1.0)
    assert conv["loss_improvement_pct_5000"] == pytest.approx(50.0)


def test_dense_window_preserves_boundaries_and_half_means(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, [(100, 1000.0)] + [(step, 2.0 if step <= 150 else 1.0)
                                       for step in range(101, 201)])
    conv = collector.get_convergence_metrics()
    assert conv["loss_window_sample_counts"]["100"] == 100
    assert conv["loss_improvement_100"] == pytest.approx(1.0)
    assert conv["loss_improvement_pct_100"] == pytest.approx(50.0)


def test_irregular_windows_compare_temporal_halves(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, [(step, 2.0) for step in range(1, 10)] + [(100, 1.0)])
    assert collector.get_convergence_metrics()["loss_improvement_100"] == pytest.approx(1.0)


def test_window_requires_finite_samples_in_both_halves(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, [(step, 1.0) for step in range(90, 101)])
    assert collector.get_convergence_metrics()["loss_improvement_100"] is None
    _record(collector, [(150, float("nan")), (160, float("inf"))])
    conv = collector.get_convergence_metrics()
    assert conv["loss_window_sample_counts"]["100"] == 11
    assert conv["loss_improvement_100"] is None


def test_plateau_duration_uses_observed_step_span(tmp_path, capsys):
    collector = _collector(tmp_path)
    _record(collector, [(408200 + i * 200, 1.0) for i in range(502)])
    conv = collector.get_convergence_metrics()
    assert conv["loss_plateau_steps"] == 100200
    assert conv["loss_plateau_observations"] == 502
    assert conv["loss_is_plateauing"] is True
    hint = next(h["hint"] for h in collector.generate_optimization_hints()
                if h["area"] == "convergence")
    assert "sampled" in hint.lower() and "100200" in hint and "502" in hint
    collector.print_session_summary()
    assert "Improvement (1K)" not in capsys.readouterr().out


@pytest.mark.parametrize("interruption", [(600, float("nan")), (600, float("inf")),
                                          (400, 1.0), (100, 1.0)])
def test_plateau_cannot_bridge_nonfinite_or_nonincreasing_steps(tmp_path, interruption):
    collector = _collector(tmp_path)
    _record(collector, [(200, 1.0), (400, 1.0), interruption])
    assert collector.get_convergence_metrics()["loss_plateau_steps"] == 0
    _record(collector, [(800, 1.0)])
    expected = 0 if interruption[0] == 600 else 800 - interruption[0]
    assert collector.get_convergence_metrics()["loss_plateau_steps"] == expected


def test_loss_window_does_not_mix_repeated_steps_from_a_rollback(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, [(step, 5.0) for step in range(1, 101)])
    _record(collector, [(step, 1.0) for step in range(90, 101)])
    conv = collector.get_convergence_metrics()
    assert conv["loss_window_sample_counts"]["100"] == 11
    assert conv["loss_improvement_100"] is None


def test_plateau_hint_threshold_counts_steps(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, [(100, 1.0), (300, 1.0), (600, 1.0)])
    assert collector.get_convergence_metrics()["loss_is_plateauing"] is False
    _record(collector, [(700, 1.0)])
    conv = collector.get_convergence_metrics()
    assert conv["loss_plateau_steps"] == 600
    assert conv["loss_plateau_observations"] == 4
    assert conv["loss_is_plateauing"] is True


@pytest.mark.parametrize("basis,unit", [(None, "sampled records"),
                                        ("optimizer_steps", "optimizer steps")])
@pytest.mark.parametrize("plateau", [False, True])
def test_analysis_labels_legacy_and_corrected_convergence(basis, unit, plateau):
    from scripts import analyze_training_stats as analysis

    report = {
        "loss": {"summary": {"total_count": 20}},
        "convergence": {
            "loss_ema": 1.0,
            "loss_plateau_steps": 600,
            "loss_is_plateauing": plateau,
            "loss_improvement_1000": 0.0,
            "loss_improvement_pct_1000": 0.0,
        },
    }
    if basis is not None:
        report["convergence"]["loss_window_basis"] = basis
    analyses = {
        "loss_analysis": analysis.analyze_loss(report),
        "throughput_analysis": analysis.analyze_throughput(report),
        "grad_analysis": analysis.analyze_gradients(report),
        "model_analysis": analysis.analyze_model_health(report),
        "eval_analysis": analysis.analyze_evaluations(report),
        "selfplay_analysis": analysis.analyze_selfplay(report),
        "system_analysis": analysis.analyze_system(report),
    }
    recommendations = analysis.generate_recommendations(report, **analyses)
    convergence_hint = next(r["recommendation"] for r in recommendations
                            if r["category"] == "Convergence")
    assert (f"600 {unit}" if plateau else f"1K {unit}") in convergence_hint
    markdown = analysis.format_markdown_report(report, **analyses,
                                               recommendations=recommendations)
    assert f"Improvement (1000 {unit})" in markdown
