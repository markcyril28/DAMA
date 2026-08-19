import numpy as np

from dama.ai.ml.dataset import (
    BOARD_PLANES,
    MOVE_FEATURE_SIZE,
    _encode_board_fast,
    _encode_moves_fast,
)
from dama.game_state import GameState


def _rotate(position):
    return [7 - position[0], 7 - position[1]]


def _swap_rotate_state(state: dict) -> dict:
    return {
        "p1_men": [_rotate(value) for value in state["p2_men"]],
        "p1_kings": [_rotate(value) for value in state["p2_kings"]],
        "p2_men": [_rotate(value) for value in state["p1_men"]],
        "p2_kings": [_rotate(value) for value in state["p1_kings"]],
        "turn": 2 if state["turn"] == 1 else 1,
        "move_count": state.get("move_count", 0),
    }


def _rotate_move(move: dict) -> dict:
    return {
        "path": [_rotate(value) for value in move["path"]],
        "captures": [_rotate(value) for value in move.get("captures", [])],
        "promotion": bool(move.get("promotion", False)),
    }


def _path(move: dict) -> tuple:
    return tuple(tuple(value) for value in move["path"])


def test_only_rules_valid_rotation_is_already_encoding_identical() -> None:
    state = GameState.initial()
    state = state.apply_move(state.legal_moves()[1])
    state = state.apply_move(state.legal_moves()[-1])
    compact = state.to_compact()
    legal_moves = [move.to_dict() for move in state.legal_moves()]
    transformed_state = _swap_rotate_state(compact)
    transformed_moves = [_rotate_move(move) for move in legal_moves]

    transformed_legal = {
        _path(move.to_dict())
        for move in GameState.from_compact(transformed_state).legal_moves()
    }
    assert {_path(move) for move in transformed_moves} == transformed_legal

    original_board = np.zeros((BOARD_PLANES, 8, 8), dtype=np.float32)
    transformed_board = np.zeros_like(original_board)
    original_move_features = np.zeros(
        (len(legal_moves), MOVE_FEATURE_SIZE), dtype=np.float32)
    transformed_move_features = np.zeros_like(original_move_features)
    _encode_board_fast(compact, original_board)
    _encode_board_fast(transformed_state, transformed_board)
    _encode_moves_fast(compact, legal_moves, original_move_features)
    _encode_moves_fast(
        transformed_state, transformed_moves, transformed_move_features)

    np.testing.assert_array_equal(original_board, transformed_board)
    np.testing.assert_array_equal(
        original_move_features, transformed_move_features)


def test_diagonal_and_quarter_turn_transforms_are_not_valid_augmentations() -> None:
    state = {
        "p1_men": [[2, 1]],
        "p1_kings": [],
        "p2_men": [[7, 0]],
        "p2_kings": [],
        "turn": 1,
        "move_count": 0,
    }
    legal = [move.to_dict() for move in GameState.from_compact(state).legal_moves()]

    diagonal_state = {
        **state,
        "p1_men": [[1, 2]],
        "p2_men": [[0, 7]],
    }
    diagonal_paths = {
        tuple((col, row) for row, col in move["path"])
        for move in legal
    }
    actual_diagonal_paths = {
        _path(move.to_dict())
        for move in GameState.from_compact(diagonal_state).legal_moves()
    }
    assert diagonal_paths != actual_diagonal_paths

    playable = (2, 1)
    quarter_turn = (playable[1], 7 - playable[0])
    horizontal_reflection = (playable[0], 7 - playable[1])
    assert sum(playable) % 2 == 1
    assert sum(quarter_turn) % 2 == 0
    assert sum(horizontal_reflection) % 2 == 0
