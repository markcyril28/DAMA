"""Focused tests for policy-distillation evaluation and inference controls."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Future

import pytest
import torch

from dama.ai.ml.acceptance import (
    WILSON_95_METHOD,
    evaluate_acceptance_gates,
    match_score,
    wilson_score_interval_95,
)
from dama.ai.ml import inference
from dama.ai.ml import model_vs_algo
from dama.ai.ml.model_vs_algo import (
    TestStatistics as EvaluationStatistics,
    _build_balanced_game_specs,
    _choose_opponent_move,
)


def _result(
    p1_wins: int,
    p1_draws: int,
    p1_losses: int,
    p2_wins: int,
    p2_draws: int,
    p2_losses: int,
) -> dict:
    return {
        "ml_wins": p1_wins + p2_wins,
        "draws": p1_draws + p2_draws,
        "algo_wins": p1_losses + p2_losses,
        "ml_as_p1_wins": p1_wins,
        "ml_as_p1_draws": p1_draws,
        "ml_as_p1_losses": p1_losses,
        "ml_as_p2_wins": p2_wins,
        "ml_as_p2_draws": p2_draws,
        "ml_as_p2_losses": p2_losses,
    }


def test_wilson_interval_uses_draw_as_half_success() -> None:
    assert match_score(20, 60, 20) == pytest.approx(0.5)
    lower, upper = wilson_score_interval_95(20, 60, 20)
    assert lower == pytest.approx(0.4038315, abs=1e-6)
    assert upper == pytest.approx(0.5961685, abs=1e-6)


def test_statistics_records_raw_wdl_match_score_and_ci_method() -> None:
    stats = EvaluationStatistics(
        total_games=100,
        ml_wins=80,
        draws=10,
        algo_wins=10,
        ml_as_p1_wins=40,
        ml_as_p1_draws=5,
        ml_as_p1_losses=5,
        ml_as_p2_wins=40,
        ml_as_p2_draws=5,
        ml_as_p2_losses=5,
    )
    data = stats.to_dict()
    assert data["overall_wdl"]["wins"] == 80
    assert data["overall_wdl"]["draws"] == 10
    assert data["overall_wdl"]["losses"] == 10
    assert data["match_score"] == pytest.approx(0.85)
    assert data["ci_method"] == WILSON_95_METHOD
    assert data["ml_as_p1_wdl"]["total"] == 50
    assert data["ml_as_p2_wdl"]["total"] == 50


def test_acceptance_gates_pass_only_with_exact_protocol_and_thresholds() -> None:
    random_result = _result(45, 0, 5, 45, 0, 5)
    easy_result = _result(32, 5, 13, 33, 5, 12)
    decision = evaluate_acceptance_gates(0.50, random_result, easy_result)
    assert decision.passed
    assert all(decision.checks.values())

    unbalanced = dict(easy_result)
    unbalanced["ml_as_p1_wins"] -= 1
    unbalanced["ml_as_p2_wins"] += 1
    failed = evaluate_acceptance_gates(0.50, random_result, unbalanced)
    assert not failed.passed
    assert not failed.checks["easy_exact_balanced_100_games"]

    inconsistent = dict(easy_result)
    inconsistent["ml_wins"] -= 1
    inconsistent["draws"] += 1
    failed = evaluate_acceptance_gates(0.50, random_result, inconsistent)
    assert not failed.passed
    assert not failed.checks["easy_exact_balanced_100_games"]


def test_balanced_specs_pair_the_same_fixed_suite_across_sides() -> None:
    first, seed, suite_id = _build_balanced_game_specs(
        model_path="checkpoint.pt",
        difficulty="easy",
        opponent_type="random",
        num_games=100,
        max_moves=200,
        opening_plies=(0, 2, 4, 6, 8),
        opening_seed=7719,
        ml_inference_depth=1,
    )
    second, second_seed, second_suite_id = _build_balanced_game_specs(
        model_path="checkpoint.pt",
        difficulty="easy",
        opponent_type="random",
        num_games=100,
        max_moves=200,
        opening_plies=(0, 2, 4, 6, 8),
        opening_seed=7719,
        ml_inference_depth=1,
    )

    assert first == second
    assert seed == second_seed == 7719
    assert suite_id == second_suite_id
    assert suite_id.startswith("sha256:")
    assert Counter(spec[3] for spec in first) == {1: 50, 2: 50}
    assert set(Counter(spec[5] for spec in first).values()) == {2}

    with pytest.raises(ValueError, match="even"):
        _build_balanced_game_specs(
            "checkpoint.pt", "easy", "random", 99, 200, (0, 2), 7719, 1
        )


def test_injected_evaluation_executor_is_reused_without_internal_pool(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def _fake_batch(batch):
        (_model, _difficulty, opponent, ml_player_values,
         _max_moves, openings, depth) = batch
        return [
            {
                "result": "draw",
                "ml_player": player,
                "winner": None,
                "num_moves": 12,
                "ml_moves": 6,
                "algo_moves": 6,
                "game_time_ms": 1.0,
                "opponent_type": opponent,
                "opening_plies": opening[0],
                "opening_seed": opening[1],
                "ml_inference_depth": depth,
            }
            for player, opening in zip(ml_player_values, openings)
        ]

    class _Executor:
        def submit(self, function, batch):
            calls.append((function, batch))
            future = Future()
            future.set_result(_fake_batch(batch))
            return future

    class _UnexpectedPool:
        def __init__(self, *args, **kwargs):
            raise AssertionError("injected executor must bypass pool creation")

    monkeypatch.setattr(model_vs_algo, "ProcessPoolExecutor", _UnexpectedPool)
    executor = _Executor()
    tester = model_vs_algo.ModelVsAlgoTester(
        model_path="unused.pt",
        algo_difficulty="easy",
        num_workers=1,
        stats_dir=str(tmp_path),
        opening_plies=(2, 4),
        opening_seed=7719,
        opponent_type="random",
        executor=executor,
    )

    stats = tester.run_tests(num_games=4)

    assert stats.total_games == 4
    assert stats.draws == 4
    assert len(calls) == 1
    assert calls[0][0] is model_vs_algo._play_test_games_batch


def test_evaluation_worker_limits_pytorch_thread_pools(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(("intra", value)))
    monkeypatch.setattr(
        torch, "set_num_interop_threads", lambda value: calls.append(("interop", value))
    )

    model_vs_algo._evaluation_worker_init()

    assert calls == [("intra", 1), ("interop", 1)]


def test_linux_evaluation_context_preloads_cpu_inference_stack(monkeypatch) -> None:
    calls = []
    sentinel = object()
    monkeypatch.setattr(model_vs_algo.sys, "platform", "linux")
    monkeypatch.setattr(
        model_vs_algo.mp,
        "get_all_start_methods",
        lambda: ["spawn", "forkserver"],
    )
    monkeypatch.setattr(
        model_vs_algo.mp, "set_forkserver_preload",
        lambda modules: calls.append(("preload", tuple(modules))),
    )
    monkeypatch.setattr(
        model_vs_algo.mp, "get_context",
        lambda method: calls.append(("context", method)) or sentinel,
    )

    assert model_vs_algo._evaluation_worker_context() is sentinel
    assert calls == [
        ("preload", model_vs_algo._EVALUATION_FORKSERVER_PRELOAD),
        ("context", "forkserver"),
    ]


def test_evaluation_context_falls_back_to_spawn(monkeypatch) -> None:
    calls = []
    sentinel = object()
    monkeypatch.setattr(model_vs_algo.sys, "platform", "win32")
    monkeypatch.setattr(
        model_vs_algo.mp, "set_forkserver_preload",
        lambda _modules: calls.append("unexpected preload"),
    )
    monkeypatch.setattr(
        model_vs_algo.mp, "get_context",
        lambda method: calls.append(method) or sentinel,
    )

    assert model_vs_algo._evaluation_worker_context() is sentinel
    assert calls == ["spawn"]


def test_standalone_evaluation_pool_installs_thread_limiter(
    monkeypatch, tmp_path
) -> None:
    instances = []
    worker_context = object()

    class _Executor:
        def __init__(self, *, max_workers, mp_context, initializer):
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.initializer = initializer
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def submit(self, _function, batch):
            (_model, _difficulty, opponent, ml_player_values,
             _max_moves, openings, depth) = batch
            future = Future()
            future.set_result([
                {
                    "result": "draw",
                    "ml_player": player,
                    "winner": None,
                    "num_moves": 12,
                    "ml_moves": 6,
                    "algo_moves": 6,
                    "game_time_ms": 1.0,
                    "opponent_type": opponent,
                    "opening_plies": opening[0],
                    "opening_seed": opening[1],
                    "ml_inference_depth": depth,
                }
                for player, opening in zip(ml_player_values, openings)
            ])
            return future

    monkeypatch.setattr(model_vs_algo, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(
        model_vs_algo,
        "_evaluation_worker_context",
        lambda: worker_context,
    )
    stats = model_vs_algo.ModelVsAlgoTester(
        model_path="unused.pt",
        algo_difficulty="easy",
        num_workers=2,
        stats_dir=str(tmp_path),
        opening_plies=(2, 4),
        opening_seed=7719,
        opponent_type="random",
    ).run_tests(num_games=4)

    assert stats.total_games == 4
    assert len(instances) == 1
    assert instances[0].max_workers == 2
    assert instances[0].mp_context is worker_context
    assert instances[0].initializer is model_vs_algo._evaluation_worker_init


def test_random_opponent_uses_uniform_choice_path() -> None:
    legal_moves = ("a", "b", "c")
    rng = __import__("random").Random(9321)
    counts = Counter(
        _choose_opponent_move(object(), legal_moves, "easy", "random", rng)
        for _ in range(12000)
    )
    assert set(counts) == set(legal_moves)
    assert all(abs(count - 4000) < 200 for count in counts.values())


def test_batch_game_times_partition_the_batch_wall(monkeypatch) -> None:
    """Interleaved batch play attributes the shared clock across games.

    Before the attribution fix every game in a batch reported the whole
    batch's elapsed wall as its own duration, so the per-game times summed
    to roughly ``batch_size`` times the true wall.  Attributed shares are a
    partition of the round walls, so their sum can never exceed the wall of
    the whole call.
    """
    import time as _time

    class _FirstMoveModel:
        def forward_padded(self, boards, move_features, move_counts):
            scores = torch.zeros(
                (boards.shape[0], move_features.shape[1]), dtype=torch.float32)
            scores[:, 0] = 1.0
            return scores

    monkeypatch.setattr(
        inference, "get_model", lambda *_args, **_kwargs: _FirstMoveModel())

    args = (
        "unused.pt", "easy", "random", [1, 1, 2, 2], 60,
        [(2, 11), (2, 12), (4, 13), (4, 14)], 1,
    )
    wall_start = _time.perf_counter()
    records = model_vs_algo._play_test_games_batch(args)
    wall_ms = (_time.perf_counter() - wall_start) * 1000.0

    assert len(records) == 4
    times = [record["game_time_ms"] for record in records]
    assert all(value > 0.0 for value in times)
    assert sum(times) <= wall_ms


def test_algorithm_opponent_failure_never_falls_back_to_random(monkeypatch) -> None:
    from dama.ai.algorithmic import search

    monkeypatch.setattr(search, "get_best_move", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="failed to return a legal"):
        _choose_opponent_move(
            object(), ("a", "b"), "easy", "algorithm",
            __import__("random").Random(1),
        )


class _TreeState:
    def __init__(self, name: str, children: dict[str, "_TreeState"] | None = None):
        self.name = name
        self.children = children or {}

    def legal_moves(self):
        return list(self.children)

    def apply_move(self, move):
        return self.children[move]


class _FakeModel:
    value_head = object()


def test_depth_one_preserves_argmax_and_depth_two_uses_negamax(monkeypatch) -> None:
    leaf_bad_high = _TreeState("leaf_bad_high", {"unused": _TreeState("unused1")})
    leaf_bad_low = _TreeState("leaf_bad_low", {"unused": _TreeState("unused2")})
    leaf_good_low = _TreeState("leaf_good_low", {"unused": _TreeState("unused3")})
    leaf_good_high = _TreeState("leaf_good_high", {"unused": _TreeState("unused4")})
    child_bad = _TreeState("child_bad", {"reply1": leaf_bad_high, "reply2": leaf_bad_low})
    child_good = _TreeState("child_good", {"reply1": leaf_good_low, "reply2": leaf_good_high})
    root = _TreeState("root", {"bad": child_bad, "good": child_good})

    monkeypatch.setattr(
        inference,
        "_score_policy_moves",
        lambda state, moves, model, device: torch.tensor([10.0, 0.0]),
    )
    values = {
        "leaf_bad_high": 0.9,
        "leaf_bad_low": -0.8,
        "leaf_good_low": 0.2,
        "leaf_good_high": 0.3,
    }
    monkeypatch.setattr(
        inference,
        "_value_for_state",
        lambda state, model, device: values[state.name],
    )

    model = _FakeModel()
    moves = root.legal_moves()
    assert inference._select_move_with_model(
        root, moves, model, device="cpu", depth=1
    ) == "bad"
    assert inference._select_move_with_model(
        root, moves, model, device="cpu", depth=2
    ) == "good"


def test_search_depth_validation_and_value_head_requirement() -> None:
    root = _TreeState("root", {"a": _TreeState("leaf", {"unused": _TreeState("x")})})
    with pytest.raises(ValueError, match="one of 1, 2, or 3"):
        inference._select_move_with_model(root, root.legal_moves(), _FakeModel(), "cpu", 4)

    class _PolicyOnlyModel:
        value_head = None

    two_move_root = _TreeState(
        "root",
        {
            "a": _TreeState("leaf_a", {"unused": _TreeState("x")}),
            "b": _TreeState("leaf_b", {"unused": _TreeState("y")}),
        },
    )
    with pytest.raises(ValueError, match="requires a model value head"):
        inference._select_move_with_model(
            two_move_root,
            two_move_root.legal_moves(),
            _PolicyOnlyModel(),
            "cpu",
            2,
        )
