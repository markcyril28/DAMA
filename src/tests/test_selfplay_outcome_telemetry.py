"""Completed self-play cycles expose game outcomes without retaining worker rows."""

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
import concurrent.futures
import copy
import json
from types import SimpleNamespace

import pytest

from dama.ai.ml import trainer as trainer_module
from dama.ai.ml.stats_collector import StatsCollector


@pytest.mark.parametrize("pool_mode", ["healthy", "broken", "startup_failure"])
@pytest.mark.parametrize("zero_moves", [False, True])
def test_completed_cycle_persists_outcomes_and_lengths(
    tmp_path, monkeypatch, pool_mode, zero_moves,
):
    delivered = []
    original_rows = []

    def play_batch(batch):
        rows = []
        for task in batch:
            if zero_moves:
                continue
            game_id = task[8]
            index = int(game_id.rsplit("-", 1)[1])
            winner = [1, 2, None][index]
            # Different lengths ensure outcomes are counted per game, and
            # alternating perspectives cannot turn losses into extra wins.
            for ply in range(index + 1):
                turn = 1 + ply % 2
                rows.append({
                    "game_id": game_id,
                    "trajectory_source": "algorithm",
                    "state": {"turn": turn},
                    "result": 0 if winner is None else (1 if turn == winner else -1),
                    "opening_plies": 8,
                })
        original_rows.extend(copy.deepcopy(rows))
        delivered.append(rows)
        return rows

    class Pool:
        def __init__(self, *args, **kwargs):
            if pool_mode == "startup_failure":
                raise OSError("controlled pool startup failure")
            self._processes = {}
            self._executor_manager_thread = None
            self.submissions = 0

        def submit(self, function, batch):
            future = Future()
            self.submissions += 1
            if pool_mode == "broken" and self.submissions > 1:
                future.set_exception(BrokenProcessPool("controlled worker death"))
            else:
                future.set_result(function(batch))
            return future

        def shutdown(self, wait=False, cancel_futures=False):
            pass

    trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
    trainer.config = SimpleNamespace(
        selfplay_opening_plies=(8,), selfplay_opening_seed=20260819,
        selfplay_opponent_focus="algorithm", selfplay_focus_side="both",
        max_moves_per_sample=32, algo_vs_algo_games=3,
        algo_vs_algo_enabled=True, selfplay_difficulties=("easy",),
        selfplay_max_moves=0 if zero_moves else 3, selfplay_noise_prob=0.1,
        cpu_workers=3, recovery_enforced=False,
        algo_vs_algo_difficulties=("easy",), teacher_difficulty="hard",
        inference_depth=1,
    )
    trainer.step, trainer.epoch = 10, 2
    trainer.stats = SimpleNamespace(generation_cycles_completed=0)
    trainer.stats_collector = StatsCollector(
        output_dir=str(tmp_path), session_id="outcomes", flush_every=1000,
    )
    trainer._stopped = False
    trainer._allocate_generation_cycle_id = lambda: 0
    trainer._runtime_model_path = lambda _name: tmp_path / "unused.pt"
    trainer._service_control_queue = lambda: None
    trainer._annotate_selfplay_entries = lambda *args, **kwargs: None
    trainer._balance_side_sample_weights = lambda entries: None
    trainer._cleanup_runtime_model_file = lambda path: None
    trainer._cleanup_runtime_models_dir = lambda: None
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(trainer_module, "_play_games_batch_worker_algo", play_batch)

    count = trainer.run_selfplay(0, skip_replay=True, collect_dicts=True)
    assert count == (0 if zero_moves else 6)
    assert trainer.stats.generation_cycles_completed == 1
    assert trainer._last_selfplay_dicts == original_rows
    assert all(rows == [] for rows in delivered)
    trainer.stats_collector.flush_incremental()
    row = json.loads((tmp_path / "incremental_outcomes.jsonl").read_text().splitlines()[-1])
    record = row["latest_records"]["selfplay"]
    assert record["num_games"] == 3
    assert record["result_distribution"] == (
        {"p1_win": 0, "p2_win": 0, "draw": 0, "unknown": 3}
        if zero_moves else {"p1_win": 1, "p2_win": 1, "draw": 1, "unknown": 0}
    )
    assert record["game_length_basis"] == "recorded_post_opening_plies"
    assert record["game_length_stats"]["min"] == (0 if zero_moves else 1)
    assert record["game_length_stats"]["max"] == (0 if zero_moves else 3)
    assert record["game_length_stats"]["mean"] == (0 if zero_moves else 2)


def test_invalid_or_conflicting_results_are_unknown_without_mutating_rows():
    rows = [
        {"game_id": "conflict", "state": {"turn": 1}, "result": 1},
        {"game_id": "conflict", "state": {"turn": 2}, "result": 1},
        {"game_id": "missing", "state": {"turn": 1}},
        {"game_id": "invalid", "state": {"turn": 3}, "result": -1},
        {"game_id": "boolean_turn", "state": {"turn": True}, "result": 1},
        {"game_id": "boolean_result", "state": {"turn": 1}, "result": False},
    ]
    before = copy.deepcopy(rows)
    results, lengths = trainer_module._summarize_selfplay_batch(rows, 5)
    assert results == {"p1_win": 0, "p2_win": 0, "draw": 0, "unknown": 5}
    assert sorted(lengths) == [1, 1, 1, 1, 2]
    assert rows == before


def test_draw_cycle_enables_existing_data_quality_hint(tmp_path):
    rows = [
        {"game_id": f"draw-{index}", "state": {"turn": 1}, "result": 0}
        for index in range(3)
    ]
    results, lengths = trainer_module._summarize_selfplay_batch(rows, 3)
    collector = StatsCollector(output_dir=str(tmp_path), flush_every=1000)
    collector.record_selfplay_epoch(
        step=0, epoch=0, num_games=3, num_entries=3, elapsed_sec=1.0,
        result_distribution=results, game_lengths=lengths,
    )
    hints = collector.generate_optimization_hints()
    assert any(hint["area"] == "data_quality" and "100%" in hint["hint"] for hint in hints)


def test_interleaved_rows_count_games_instead_of_player_results():
    rows = [
        {"game_id": "p1", "state": {"turn": 2}, "result": -1},
        {"game_id": "p2", "state": {"turn": 1}, "result": -1},
        {"game_id": "draw", "state": {"turn": 1}, "result": 0},
        {"game_id": "p1", "state": {"turn": 1}, "result": 1},
        {"game_id": "draw", "state": {"turn": 2}, "result": 0},
    ]
    results, lengths = trainer_module._summarize_selfplay_batch(rows, 3)
    assert results == {"p1_win": 1, "p2_win": 1, "draw": 1, "unknown": 0}
    assert sorted(lengths) == [1, 2, 2]
