"""Latest incremental evaluation snapshots must not become session history."""

import json
from pathlib import Path

import pytest

from dama.ai.ml.stats_collector import StatsCollector
from scripts import analyze_training_stats as analysis


HISTORICAL_FIELDS = (
    "first_win_rate", "best_win_rate", "win_rate_delta", "first_elo", "best_elo",
    "elo_delta", "steps_per_elo", "avg_p1_win_rate", "avg_p2_win_rate",
    "side_imbalance",
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
    return analyses[4], recommendations, markdown


def _section(markdown):
    return markdown.split("## Evaluation Progress\n", 1)[1].split("\n## ", 1)[0]


@pytest.fixture
def persisted_evaluations(tmp_path):
    collector = StatsCollector(str(tmp_path), session_id="evaluation_scope", flush_every=1000)
    for epoch, win_rate in enumerate((0.8, 0.9, 0.2), start=1):
        collector.record_evaluation(epoch * 100, epoch, {
            "ml_win_rate": win_rate, "draw_rate": 0.0,
            "ml_as_p1_win_rate": win_rate, "ml_as_p2_win_rate": win_rate,
        })
        collector.flush_incremental()
    terminal_path = Path(collector.export_session_report())
    incremental_path = tmp_path / "incremental_evaluation_scope.jsonl"
    return json.loads(terminal_path.read_text()), incremental_path


def test_terminal_report_retains_evaluation_history(persisted_evaluations):
    terminal, _ = persisted_evaluations
    evaluation, recommendations, markdown = _outputs(terminal)

    assert evaluation["record_scope"] == "available_history"
    assert evaluation["num_evaluations"] == evaluation["total_evaluations"] == 3
    assert evaluation["latest_step"] == 300
    assert evaluation["first_win_rate"] == 0.8
    assert evaluation["best_win_rate"] == 0.9
    assert evaluation["latest_win_rate"] == 0.2
    assert evaluation["win_rate_delta"] == pytest.approx(-0.6)
    assert evaluation["first_elo"] == terminal["evaluations"][0]["estimated_elo_diff"]
    assert evaluation["best_elo"] == terminal["evaluations"][1]["estimated_elo_diff"]
    assert evaluation["elo_delta"] == pytest.approx(
        terminal["evaluations"][-1]["estimated_elo_diff"]
        - terminal["evaluations"][0]["estimated_elo_diff"])
    assert evaluation["avg_p1_win_rate"] == pytest.approx(1.9 / 3)
    assert evaluation["avg_p2_win_rate"] == pytest.approx(1.9 / 3)
    assert evaluation["side_imbalance"] == 0.0
    assert any("DECREASED" in item["recommendation"] for item in recommendations)
    section = _section(markdown)
    assert "Available evaluation history" in section
    assert "| First Win Rate | 80.0% |" in section
    assert "| Best Win Rate | 90.0% |" in section
    assert "| Win Rate Delta | -60.0% |" in section


def test_incremental_snapshot_preserves_latest_without_inventing_history(persisted_evaluations):
    terminal, incremental_path = persisted_evaluations
    report, warning = analysis.load_incremental_report(str(incremental_path))
    assert warning is None
    assert report["summary"]["total_evaluations"] == 3
    assert report["evaluations"] == [terminal["evaluations"][-1]]

    evaluation, recommendations, markdown = _outputs(report)
    assert evaluation["record_scope"] == "latest_snapshot"
    assert evaluation["num_evaluations"] == 1
    assert evaluation["total_evaluations"] == 3
    assert evaluation["latest_step"] == 300
    assert evaluation["latest_win_rate"] == 0.2
    assert evaluation["latest_elo"] == terminal["evaluations"][-1]["estimated_elo_diff"]
    for field in HISTORICAL_FIELDS:
        assert evaluation.get(field) is None, field
    assert not any("DECREASED" in item["recommendation"] for item in recommendations)
    section = _section(markdown)
    assert "Latest evaluation snapshot" in section
    assert "| Available evaluation records | 1 |" in section
    assert "| Recorded session evaluations | 3 |" in section
    assert "| Latest evaluation step | 300 |" in section
    assert "| Latest Win Rate | 20.0% |" in section
    for label in ("First Win Rate", "Best Win Rate", "Win Rate Delta", "First ELO Diff", "Best ELO Diff"):
        assert f"| {label} | Unavailable |" in section
    json.dumps(evaluation, allow_nan=False)


def test_legacy_incremental_count_is_unknown_even_with_previous_cumulative_count(persisted_evaluations):
    _, incremental_path = persisted_evaluations
    rows = [json.loads(line) for line in incremental_path.read_text().splitlines()]
    rows[-1]["session_summary"].pop("evaluations_recorded")
    incremental_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    report, _ = analysis.load_incremental_report(str(incremental_path))
    evaluation, _, markdown = _outputs(report)
    assert report["summary"]["total_evaluations"] is None
    assert evaluation["num_evaluations"] == 1
    assert evaluation["total_evaluations"] is None
    assert evaluation["latest_step"] == 300
    assert "| Recorded session evaluations | Unavailable |" in _section(markdown)


def test_partial_latest_evaluation_does_not_borrow_prior_snapshot_values(persisted_evaluations):
    _, incremental_path = persisted_evaluations
    rows = [json.loads(line) for line in incremental_path.read_text().splitlines()]
    rows[-1]["latest_records"]["evaluation"] = {"step": 300}
    incremental_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    report, _ = analysis.load_incremental_report(str(incremental_path))
    evaluation, recommendations, markdown = _outputs(report)
    assert report["evaluations"] == [{"step": 300}]
    assert evaluation["num_evaluations"] == 1
    assert evaluation["total_evaluations"] == 3
    assert evaluation["latest_step"] == 300
    assert evaluation["latest_win_rate"] is None
    assert evaluation["latest_elo"] is None
    for field in HISTORICAL_FIELDS:
        assert evaluation.get(field) is None, field
    assert not any("DECREASED" in item["recommendation"] for item in recommendations)
    section = _section(markdown)
    assert "| Latest Win Rate | Unavailable |" in section
    assert "| Latest ELO Diff | Unavailable |" in section


@pytest.mark.parametrize("source", ["terminal", "incremental"])
@pytest.mark.parametrize("win_rate, draw_rate", [(0.0, 1.0), (0.5, 0.0)])
def test_measured_zero_win_rate_and_elo_stay_available(tmp_path, source, win_rate, draw_rate):
    collector = StatsCollector(str(tmp_path), session_id="zero")
    collector.record_evaluation(0, 0, {"ml_win_rate": win_rate, "draw_rate": draw_rate})
    if source == "terminal":
        report = json.loads(Path(collector.export_session_report()).read_text())
    else:
        collector.flush_incremental()
        report, _ = analysis.load_incremental_report(str(tmp_path / "incremental_zero.jsonl"))
    evaluation, _, markdown = _outputs(report)
    assert evaluation["latest_step"] == 0
    assert evaluation["latest_win_rate"] == win_rate
    assert evaluation["latest_elo"] == 0.0
    assert evaluation["total_evaluations"] == 1
    section = _section(markdown)
    assert f"| Latest Win Rate | {win_rate * 100:.1f}% |" in section
    elo_row = next(line for line in section.splitlines()
                   if line.startswith("| Latest ELO Diff |"))
    assert float(elo_row.split("|")[2]) == 0.0
    if source == "terminal":
        assert evaluation["first_win_rate"] == win_rate
        assert evaluation["best_win_rate"] == win_rate
        assert evaluation["win_rate_delta"] == evaluation["elo_delta"] == 0.0


def test_missing_latest_evaluation_does_not_reuse_earlier_snapshots(persisted_evaluations):
    _, incremental_path = persisted_evaluations
    rows = [json.loads(line) for line in incremental_path.read_text().splitlines()]
    rows[-1]["latest_records"].pop("evaluation")
    incremental_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    report, _ = analysis.load_incremental_report(str(incremental_path))
    evaluation, recommendations, _ = _outputs(report)
    assert report["summary"]["total_evaluations"] == 3
    assert report["evaluations"] == []
    assert evaluation["num_evaluations"] == 0
    assert not any("DECREASED" in item["recommendation"] for item in recommendations)


def test_sparse_flush_with_truncated_tail_preserves_complete_snapshot_scope(tmp_path):
    collector = StatsCollector(str(tmp_path), session_id="sparse")
    for epoch, win_rate in enumerate((0.8, 0.9, 0.2), start=1):
        collector.record_evaluation(epoch * 100, epoch, {"ml_win_rate": win_rate})
    collector.flush_incremental()
    path = tmp_path / "incremental_sparse.jsonl"
    assert len(path.read_text().splitlines()) == 1
    with path.open("a") as stream:
        stream.write('{"session_summary":{"evaluations_recorded":4},'
                     '"latest_records":{"evaluation":{"step":400,')

    report, warning = analysis.load_incremental_report(str(path))
    evaluation, recommendations, markdown = _outputs(report)
    assert "truncated final row" in warning
    assert report["evaluations"] == [collector.eval_records[-1]]
    assert evaluation["total_evaluations"] == 3
    assert evaluation["num_evaluations"] == 1
    assert evaluation["latest_step"] == 300
    assert evaluation["latest_win_rate"] == 0.2
    assert evaluation["latest_elo"] == collector.eval_records[-1]["estimated_elo_diff"]
    for field in HISTORICAL_FIELDS:
        assert evaluation.get(field) is None, field
    assert not any("DECREASED" in item["recommendation"] for item in recommendations)
    assert "Latest evaluation snapshot" in _section(markdown)
