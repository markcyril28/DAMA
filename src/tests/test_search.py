"""Behavioral tests for the algorithmic alpha-beta search.

These guard the Cython `_fast_search` extension, which had **no** test coverage
before: `test_game_logic.py` exercises board/movegen/rules but never the search,
so a stale or broken `_fast_search.so` could ship undetected (this happened once
- the committed binary lagged its `.pyx` by 2 days and silently ran outdated
search code; see CLAUDE.md "Cython Extensions" and the Journal stale-.so
landmine). The launcher mtime-staleness guard checks *freshness*, not *behavior*;
these tests check behavior.

Designed to pass whether or not the compiled extension is present: every
difficulty must return a legal move (covers both the Cython fast path and the
pure-Python fallback), and when the `.so` is loaded we additionally validate the
raw binary call directly.
"""
import pytest

from dama.types import Move
from dama.ai.algorithmic import search as search_mod
from dama.ai.algorithmic.search import get_best_move


NON_CUSTOM_DIFFICULTIES = ("easy", "medium", "hard")


@pytest.mark.parametrize("difficulty", NON_CUSTOM_DIFFICULTIES)
def test_get_best_move_returns_legal_move(initial_game_state, difficulty):
    """Search must return a legal move on a non-terminal position.

    Exercises whichever path is live (Cython fast path when built, else the
    pure-Python fallback), so it is the cross-host behavioral guard.
    """
    legal = set(initial_game_state.legal_moves())
    assert legal, "start position must have legal moves"

    move = get_best_move(initial_game_state, difficulty=difficulty)

    assert move is not None, f"{difficulty}: search returned None on a non-terminal position"
    assert move in legal, f"{difficulty}: search returned an illegal move {move}"
    # Move is a frozen dataclass using tuples (hashability contract).
    assert isinstance(move.path, tuple) and isinstance(move.captures, tuple)


def test_fast_search_binary_smoke(initial_game_state):
    """Smoke-test the compiled extension directly: it must return a legal move.

    This is a GROSS-breakage guard, not a Cython-vs-Python *parity* test. It does
    not compare the binary's chosen move against the pure-Python search, because
    alpha-beta may pick different equally-optimal moves on ties, which would make
    a strict `==` comparison flaky.

    NOTE (uncovered): it also does not exercise the two regressions behind the
    stale-`.so` landmine (CLAUDE.md "Cython Extensions"). The opening position
    has zero kings, so it never stresses MAX_MOVES (the 64->128 multi-king fix),
    and it finishes far under a 0.2s budget, so it never hits the CLOCK_MONOTONIC
    deadline path. Those paths remain untested (see Journal Pass 96).

    Skips on hosts without the built `.so` (pure-Python fallback) so a fresh
    clone is not forced to compile before tests pass; runs wherever the binary
    is present (local/server training hosts).
    """
    if not getattr(search_mod, "_HAS_FAST_SEARCH", False):
        pytest.skip("Cython _fast_search not built — pure-Python fallback in use")

    legal = set(initial_game_state.legal_moves())
    result = search_mod._fast_search(
        initial_game_state, "medium",
        time_budget_override=0.2, max_depth_override=4,
    )

    assert isinstance(result, dict), f"_fast_search must return a dict, got {type(result)}"
    move_dict = result.get("move")
    assert move_dict is not None, "_fast_search returned no move on a non-terminal position"
    move = Move.from_dict(move_dict)
    assert move in legal, f"_fast_search returned an illegal move {move}"


def test_fast_move_generation_matches_python_on_reachable_positions(
    initial_game_state,
):
    """The compiled generator must preserve the Python oracle's legal moves.

    Capture coordinates are compared as sets because their serialized order is
    not part of move application, while path order is.  The deterministic walk
    exercises both quiet and forced-capture positions without making search
    tie-breaking part of the contract.
    """
    if not getattr(search_mod, "_HAS_FAST_SEARCH", False):
        pytest.skip("Cython _fast_search not built - pure-Python fallback in use")

    from dama.ai.algorithmic._fast_search import fast_generate_moves

    def move_key(move):
        return (
            tuple(move.path),
            frozenset(move.captures),
            bool(move.promotion),
        )

    state = initial_game_state
    saw_capture = False
    saw_quiet = False
    for step in range(96):
        python_moves = state.legal_moves()
        compiled_moves = [
            Move.from_dict(move) for move in fast_generate_moves(state)
        ]
        assert len(compiled_moves) == len(python_moves)
        assert {move_key(move) for move in compiled_moves} == {
            move_key(move) for move in python_moves
        }

        saw_capture |= any(move.is_capture for move in python_moves)
        saw_quiet |= any(not move.is_capture for move in python_moves)
        if not python_moves:
            state = initial_game_state
            continue
        state = state.apply_move(python_moves[(step * 7 + 3) % len(python_moves)])

    assert saw_capture
    assert saw_quiet


def test_fast_capture_generation_matches_python_on_seeded_sparse_boards():
    """Exercise multi-jump and flying captures beyond one reachable walk."""
    if not getattr(search_mod, "_HAS_FAST_SEARCH", False):
        pytest.skip("Cython _fast_search not built - pure-Python fallback in use")

    import random

    from dama.ai.algorithmic._fast_search import fast_generate_moves
    from dama.board import Board
    from dama.game_state import GameState
    from dama.types import Piece, PieceType, Player

    rng = random.Random(20260828)
    playable = [
        (row, col)
        for row in range(8)
        for col in range(8)
        if Board.is_playable(row, col)
    ]
    saw_capture = False
    saw_multi_capture = False
    saw_flying_capture = False

    def move_key(move):
        return (
            tuple(move.path),
            frozenset(move.captures),
            bool(move.promotion),
        )

    for case in range(256):
        current = Player.ONE if case % 2 == 0 else Player.TWO
        opponent = current.opponent()
        occupied = rng.sample(playable, rng.randint(4, 10))
        own_count = rng.randint(1, min(4, len(occupied) - 1))
        board = Board()
        for position in occupied[:own_count]:
            piece_type = (
                PieceType.KING if rng.random() < 0.45 else PieceType.MAN
            )
            board.set_piece(position, Piece(current, piece_type))
        for position in occupied[own_count:]:
            piece_type = (
                PieceType.KING if rng.random() < 0.35 else PieceType.MAN
            )
            board.set_piece(position, Piece(opponent, piece_type))

        state = GameState(board, current, case)
        python_moves = state.legal_moves()
        compiled_moves = [
            Move.from_dict(move) for move in fast_generate_moves(state)
        ]
        assert len(compiled_moves) == len(python_moves), case
        assert {move_key(move) for move in compiled_moves} == {
            move_key(move) for move in python_moves
        }, case

        saw_capture |= any(move.is_capture for move in python_moves)
        saw_multi_capture |= any(
            move.num_captures > 1 for move in python_moves
        )
        saw_flying_capture |= any(
            move.is_capture and board.get_piece(move.start).is_king
            for move in python_moves
        )

    assert saw_capture
    assert saw_multi_capture
    assert saw_flying_capture


def test_fast_static_evaluation_matches_python_on_reachable_positions(
    initial_game_state,
):
    """The table-driven evaluator must preserve every reference score term."""
    if not getattr(search_mod, "_HAS_FAST_SEARCH", False):
        pytest.skip("Cython _fast_search not built - pure-Python fallback in use")

    from dama.ai.algorithmic._fast_search import _fast_evaluate_static
    from dama.ai.algorithmic.eval import _evaluate_material, _evaluate_position

    state = initial_game_state
    seen_players = set()
    for step in range(96):
        current = state.current_player
        opponent = current.opponent()
        expected = _evaluate_material(
            state.board, current, opponent
        ) + _evaluate_position(state.board, current, opponent)
        assert _fast_evaluate_static(state) == expected
        seen_players.add(current)

        moves = state.legal_moves()
        if not moves:
            state = initial_game_state
            continue
        state = state.apply_move(moves[(step * 7 + 3) % len(moves)])

    assert len(seen_players) == 2
