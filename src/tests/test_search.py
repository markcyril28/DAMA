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
