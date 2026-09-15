"""Self-play outcomes survive analysis of terminal and interrupted sessions."""

import copy
import json

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


def _outputs(report):
    analyses = [fn(report) for fn in (
        analysis.analyze_loss, analysis.analyze_throughput,
        analysis.analyze_gradients, analysis.analyze_model_health,
        analysis.analyze_evaluations, analysis.analyze_selfplay,
        analysis.analyze_system,
    )]
    recommendations = analysis.generate_recommendations(report, *analyses)
    markdown = analysis.format_markdown_report(report, *analyses, recommendations)
    return analyses[5], recommendations, markdown


def _record(**changes):
    record = {
        "step": 42, "epoch": 3, "num_games": 4, "num_entries": 80,
        "elapsed_sec": 2.0, "games_per_sec": 2.0, "entries_per_sec": 40.0,
        "result_distribution": {"p1_win": 0, "p2_win": 0, "draw": 4, "unknown": 0},
        "game_length_stats": {"min": 10, "mean": 20, "max": 30},
        "game_length_basis": "recorded_post_opening_plies",
    }
    record.update(changes)
    return record


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("outcome", ["draw", "unknown", "balanced"])
def test_real_collector_outcomes_and_lengths_reach_json_and_markdown(tmp_path, source, outcome):
    collector = StatsCollector(str(tmp_path), session_id="outcomes", flush_every=1000)
    distributions = {
        "draw": {"p1_win": 0, "p2_win": 0, "draw": 4, "unknown": 0},
        "unknown": {"p1_win": 0, "p2_win": 0, "draw": 0, "unknown": 4},
        "balanced": {"p1_win": 2, "p2_win": 2, "draw": 0, "unknown": 0},
    }
    lengths = [0, 0, 0, 0] if outcome == "unknown" else [10, 20, 20, 30]
    collector.record_selfplay_epoch(
        42, 3, 4, sum(lengths), 2.0, result_distribution=distributions[outcome],
        game_lengths=lengths, game_length_basis="recorded_post_opening_plies",
    )
    if source == "terminal":
        report = collector.generate_session_report()
    else:
        collector.flush_incremental()
        report, _ = analysis.load_incremental_report(str(tmp_path / "incremental_outcomes.jsonl"))
    sp, recommendations, markdown = _outputs(report)
    assert sp["latest_result_distribution"] == distributions[outcome]
    assert sp["latest_draw_fraction"] == (1.0 if outcome == "draw" else 0.0)
    assert sp["latest_unknown_fraction"] == (1.0 if outcome == "unknown" else 0.0)
    assert sp["latest_game_length_stats"]["min"] == min(lengths)
    assert sp["latest_game_length_basis"] == "recorded_post_opening_plies"
    assert sp["latest_step"] == 42
    assert json.loads(json.dumps(sp, allow_nan=False))["latest_result_distribution"] == distributions[outcome]
    section = markdown.split("## Self-Play\n", 1)[1].split("\n## ", 1)[0]
    assert "post-opening" in section
    assert "P1 wins" in section and "Unknown outcomes" in section
    messages = "\n".join(r["recommendation"] for r in recommendations)
    if outcome == "draw":
        assert "100.0%" in messages and "draw" in messages.lower()
    elif outcome == "unknown":
        assert "unknown outcomes" in messages.lower()
        assert "zero recorded post-opening plies" in messages
        assert "zero-length games" not in messages
    else:
        assert not any(r["category"] == "Self-Play" for r in recommendations)


def test_newest_cycle_does_not_inherit_old_outcomes_or_length():
    report = {"selfplay": [_record(), _record(result_distribution=None, game_length_stats=None,
                                               game_length_basis=None, step=100)]}
    before = copy.deepcopy(report)
    sp, recommendations, markdown = _outputs(report)
    assert report == before
    assert sp["latest_step"] == 100
    assert sp["latest_result_distribution"] is None
    assert sp["latest_game_length_stats"] is None
    assert sp["latest_game_length_basis"] is None
    assert not any("draw rate" in r["recommendation"].lower() for r in recommendations)
    assert "Unavailable" in markdown.split("## Self-Play\n", 1)[1]


@pytest.mark.parametrize("distribution", [
    None, {}, {"draw": 4}, {"p1_win": 0, "p2_win": 0, "draw": 3, "unknown": 0},
    {"p1_win": 0, "p2_win": 0, "draw": 4, "unknown": -1},
    {"p1_win": 0, "p2_win": 0, "draw": 4, "unknown": False},
    {"p1_win": 0, "p2_win": 0, "draw": 3.5, "unknown": 0.5},
    {"p1_win": 0, "p2_win": 0, "draw": float("nan"), "unknown": 0},
])
def test_missing_or_invalid_outcome_counts_are_unavailable(distribution):
    sp, recommendations, markdown = _outputs({"selfplay": [_record(result_distribution=distribution)]})
    assert sp["latest_result_distribution"] is None
    assert sp["latest_draw_fraction"] is None
    assert sp["latest_unknown_fraction"] is None
    assert not any("draw rate" in r["recommendation"].lower() for r in recommendations)
    assert "| P1 wins | Unavailable |" in markdown
    json.dumps(sp, allow_nan=False)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, False])
def test_invalid_lengths_do_not_become_measured_zero(invalid):
    sp, recommendations, _ = _outputs({"selfplay": [_record(game_length_stats={"min": invalid, "mean": 20, "max": 30})]})
    assert sp["latest_game_length_stats"]["min"] is None
    assert sp["latest_game_length_stats"]["mean"] == 20
    assert not any("zero recorded post-opening" in r["recommendation"] for r in recommendations)
    json.dumps(sp, allow_nan=False)


def test_no_games_does_not_invent_draw_rate():
    sp, recommendations, _ = _outputs({"selfplay": [_record(
        num_games=0, num_entries=0, result_distribution=dict.fromkeys(("p1_win", "p2_win", "draw", "unknown"), 0),
        game_length_stats=None, game_length_basis=None,
    )]})
    assert sp["latest_result_distribution"] == {"p1_win": 0, "p2_win": 0, "draw": 0, "unknown": 0}
    assert sp["latest_draw_fraction"] is None
    assert not any("draw rate" in r["recommendation"].lower() for r in recommendations)


def test_incremental_cycle_count_is_not_presented_as_complete_history():
    sp, _, markdown = _outputs({"meta": {"data_source": "incremental"},
                               "summary": {"total_selfplay_epochs": 339}, "selfplay": [_record()]})
    assert sp["record_scope"] == "latest_snapshot"
    assert sp["num_selfplay_epochs"] == 1
    section = markdown.split("## Self-Play\n", 1)[1].split("\n## ", 1)[0]
    assert "latest cycle snapshot" in section.lower()
    assert "| Games in available records | 4 |" in section
    assert "339" not in section


def test_legacy_length_basis_stays_unknown():
    sp, recommendations, markdown = _outputs({"selfplay": [_record(
        game_length_stats={"min": 0, "mean": 10, "max": 20}, game_length_basis=None)]})
    assert sp["latest_game_length_basis"] is None
    assert "| Length basis | Unavailable |" in markdown
    assert any("basis is unavailable" in r["recommendation"] for r in recommendations)


def test_draw_fraction_counts_unknown_games_in_denominator():
    sp, recommendations, _ = _outputs({"selfplay": [_record(result_distribution={
        "p1_win": 0, "p2_win": 0, "draw": 2, "unknown": 2})]})
    assert sp["latest_draw_fraction"] == 0.5
    assert sp["latest_unknown_fraction"] == 0.5
    assert not any("high draw rate" in r["recommendation"].lower() for r in recommendations)


def test_empty_report_does_not_claim_selfplay_health():
    sp, recommendations, markdown = _outputs({})
    assert sp["num_selfplay_epochs"] == 0
    assert not any(r["category"] == "Self-Play" for r in recommendations)
    assert "| Status | Unavailable (no self-play records) |" in markdown
