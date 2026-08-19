"""Focused tests for policy-distillation trajectory generation."""

import random

import pytest

from dama.ai.ml import selfplay
from dama.ai.ml.replay import ReplayEntry
from dama.game_state import GameState
from dama.types import Move


def _last_legal_move(state, difficulty, use_parallel=False):
    assert difficulty == selfplay.HARD_TEACHER_DIFFICULTY
    return state.legal_moves()[-1]


def _opening_seed_for_first_exploration_index(index: int) -> int:
    for opening_seed in range(10000):
        rng = random.Random(
            (opening_seed ^ 0x6A09E667F3BCC909) & ((1 << 64) - 1)
        )
        rng.random()
        if rng.randrange(7) == index:
            return opening_seed
    raise AssertionError("could not find deterministic exploration seed")


def test_exact_policy_distillation_allocation():
    assert selfplay.allocate_policy_distillation_games(0) == (0, 0)
    assert selfplay.allocate_policy_distillation_games(10) == (7, 3)
    assert selfplay.allocate_policy_distillation_games(240) == (168, 72)

    with pytest.raises(ValueError, match="divisible by 10"):
        selfplay.allocate_policy_distillation_games(9)
    with pytest.raises(ValueError, match="non-negative"):
        selfplay.allocate_policy_distillation_games(-10)
    with pytest.raises(TypeError, match="integer"):
        selfplay.allocate_policy_distillation_games(True)


def test_random_training_opening_is_seeded_and_legal():
    initial = GameState.initial()
    opened_a, applied_a = selfplay.apply_random_training_opening(
        initial, opening_plies=6, opening_seed=2718)
    opened_b, applied_b = selfplay.apply_random_training_opening(
        initial, opening_plies=6, opening_seed=2718)

    rng = random.Random(2718)
    expected = initial
    expected_applied = 0
    for _ in range(6):
        moves = expected.legal_moves()
        if not moves:
            break
        expected = expected.apply_move(rng.choice(moves))
        expected_applied += 1

    assert applied_a == applied_b == expected_applied
    assert opened_a.to_compact() == opened_b.to_compact() == expected.to_compact()
    assert initial.to_compact() == GameState.initial().to_compact()


def test_exploration_keeps_hard_label_and_applies_played_action(monkeypatch):
    monkeypatch.setattr(selfplay, "get_best_move", _last_legal_move)
    opening_seed = _opening_seed_for_first_exploration_index(0)

    entries = selfplay.play_single_game(
        max_moves=2,
        noise_prob=1.0,
        p1_policy="algorithmic",
        p2_policy="algorithmic",
        return_dicts=True,
        trajectory_source="algorithm",
        game_id="noise-game",
        opening_seed=opening_seed,
    )

    first = entries[0]
    assert first["chosen_index"] == len(first["legal_moves"]) - 1
    assert first["played_index"] == 0
    assert first["chosen_index"] != first["played_index"]
    assert first["was_exploration"] is True
    assert first["teacher_difficulty"] == "hard"
    assert first["trajectory_source"] == "algorithm"
    assert first["game_id"] == "noise-game"

    first_state = GameState.from_compact(first["state"])
    played_move = Move.from_dict(first["legal_moves"][first["played_index"]])
    assert entries[1]["state"] == first_state.apply_move(played_move).to_compact()


def test_model_trajectory_uses_model_action_but_hard_teacher_label(monkeypatch):
    monkeypatch.setattr(selfplay, "get_best_move", _last_legal_move)
    monkeypatch.setattr(selfplay, "get_ml_move_idx", lambda *args, **kwargs: 0)

    entries = selfplay.play_single_game(
        max_moves=1,
        noise_prob=0.0,
        p1_policy="ml",
        p2_policy="ml",
        model_path="unused.pt",
        return_dicts=True,
        game_id="model-game",
    )

    entry = entries[0]
    assert entry["chosen_index"] == len(entry["legal_moves"]) - 1
    assert entry["played_index"] == 0
    assert entry["chosen_index"] != entry["played_index"]
    assert entry["was_exploration"] is False
    assert entry["trajectory_source"] == "current_model"
    assert entry["teacher_difficulty"] == "hard"


def test_model_trajectory_fails_closed_when_inference_is_unavailable(monkeypatch):
    monkeypatch.setattr(selfplay, "get_best_move", _last_legal_move)
    monkeypatch.setattr(selfplay, "get_ml_move_idx", None)

    with pytest.raises(RuntimeError, match="inference is unavailable"):
        selfplay.play_single_game(
            max_moves=1,
            noise_prob=0.0,
            p1_policy="ml",
            p2_policy="ml",
            model_path="missing.pt",
            return_dicts=True,
        )


def test_enhanced_model_trajectory_uses_configured_search_depth(monkeypatch):
    observed = []
    monkeypatch.setattr(selfplay, "get_best_move", _last_legal_move)

    def _model_choice(*args, **kwargs):
        observed.append(kwargs["depth"])
        return 0

    monkeypatch.setattr(selfplay, "get_ml_move_idx", _model_choice)
    selfplay.play_single_game(
        max_moves=1,
        noise_prob=0.0,
        p1_policy="ml",
        p2_policy="ml",
        model_path="unused.pt",
        return_dicts=True,
        inference_depth=3,
    )

    assert observed == [3]


def test_interleaved_model_trajectory_keeps_hard_label(monkeypatch):
    torch = pytest.importorskip("torch")

    class FirstMoveModel:
        def forward_padded(self, boards, move_features, move_counts):
            scores = torch.zeros(
                (boards.shape[0], move_features.shape[1]), dtype=torch.float32)
            scores[:, 0] = 1.0
            return scores

    monkeypatch.setattr(selfplay, "_FORK_MODEL", FirstMoveModel())
    monkeypatch.setattr(selfplay, "_HAS_COMPACT_STATE", False)
    monkeypatch.setattr(selfplay, "_HAS_FAST_MOVEGEN", False)
    monkeypatch.setattr(selfplay, "_HAS_FAST_ENCODE", False)
    monkeypatch.setattr(selfplay, "get_best_move", _last_legal_move)

    tasks = [
        (
            "medium", 1, 0.0, 1, "ml", "ml", "unused.pt", "cpu",
            0, seed, "current_model", f"interleaved-{seed}", "hard",
        )
        for seed in (11, 12)
    ]
    entries = selfplay.play_games_interleaved(tasks)

    assert len(entries) == 2
    for entry in entries:
        assert entry["chosen_index"] == len(entry["legal_moves"]) - 1
        assert entry["played_index"] == 0
        assert entry["chosen_index"] != entry["played_index"]
        assert entry["was_exploration"] is False
        assert entry["trajectory_source"] == "current_model"
        assert entry["teacher_difficulty"] == "hard"


def test_replay_metadata_round_trip_and_old_entry_compatibility():
    state = GameState.initial()
    moves = [move.to_dict() for move in state.legal_moves()]
    old_data = {
        "state": state.to_compact(),
        "legal_moves": moves,
        "chosen_index": 0,
        "result": 0,
    }
    restored_old = ReplayEntry.from_dict(old_data)
    assert restored_old.played_index is None
    assert restored_old.trajectory_source is None
    assert restored_old.to_dict() == old_data

    new_entry = ReplayEntry(
        state=state.to_compact(),
        legal_moves=moves,
        chosen_index=1,
        played_index=0,
        result=0,
        trajectory_source="current_model",
        was_exploration=True,
        teacher_difficulty="hard",
        opening_plies=4,
        game_id="round-trip",
    )
    restored_new = ReplayEntry.from_dict(new_entry.to_dict())
    assert restored_new == new_entry


def test_non_hard_teacher_is_rejected():
    with pytest.raises(ValueError, match="teacher_difficulty"):
        selfplay.play_single_game(max_moves=0, teacher_difficulty="medium")


@pytest.mark.skipif(
    not selfplay._HAS_FAST_GAME,
    reason="compiled full-game extension is not available in this interpreter",
)
def test_compiled_generator_emits_opening_and_label_audit_metadata():
    from dama.ai.algorithmic._fast_search import play_full_game_cy

    random.seed(314159)
    result = play_full_game_cy(
        p1_difficulty="easy",
        p2_difficulty="easy",
        max_moves=1,
        noise_prob=1.0,
        opening_plies=2,
        opening_seed=99,
        trajectory_source="algorithm",
        game_id="compiled-game",
    )

    entry = result["entries"][0]
    assert entry["state"]["move_count"] == 2
    assert entry["opening_plies"] == result["opening_plies"] == 2
    assert 0 <= entry["chosen_index"] < len(entry["legal_moves"])
    assert 0 <= entry["played_index"] < len(entry["legal_moves"])
    assert entry["was_exploration"] is True
    assert entry["teacher_difficulty"] == "hard"
    assert entry["trajectory_source"] == "algorithm"
    assert entry["game_id"] == "compiled-game"
