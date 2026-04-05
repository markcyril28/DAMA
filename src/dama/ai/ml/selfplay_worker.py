#!/usr/bin/env python3
"""
Standalone self-play worker for use with GNU Parallel.

Usage:
    python -m dama.ai.ml.selfplay_worker --games 10 --output games_001.jsonl

Or with GNU Parallel:
    seq 1 32 | parallel -j 32 python -m dama.ai.ml.selfplay_worker --games 16 --output games_{}.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dama.types import Player
from dama.ai.ml.scoring import score_game_dicts

# Try Cython full game loop first (2-7x faster than Python game loop).
try:
    from dama.ai.algorithmic._fast_search import play_full_game_cy
    _HAS_FAST_GAME = True
except ImportError:
    _HAS_FAST_GAME = False

from dama.game_state import GameState
from dama.board import Board
from dama.ai.algorithmic.search import get_best_move


def play_single_game(
    difficulty: str = 'medium',
    max_moves: int = 200,
    noise_prob: float = 0.1,
    start_player: int = 1,
    p1_difficulty: str = None,
    p2_difficulty: str = None,
) -> list:
    """Play a single game and return entries as dicts.

    Uses Cython full game loop when available (2-7x faster).
    Falls back to Python game loop otherwise.

    Args:
        difficulty: Default difficulty (used when p1/p2_difficulty not set)
        p1_difficulty: Difficulty for Player 1 (overrides difficulty if set)
        p2_difficulty: Difficulty for Player 2 (overrides difficulty if set)
    """
    _p1_diff = p1_difficulty or difficulty
    _p2_diff = p2_difficulty or difficulty

    # Fast path: Cython full game loop runs entirely in C.
    # Avoids all GameState/Board/Move/Piece Python object creation.
    if _HAS_FAST_GAME:
        result = play_full_game_cy(
            p1_difficulty=_p1_diff, p2_difficulty=_p2_diff,
            max_moves=max_moves, noise_prob=noise_prob,
            start_player=start_player,
        )
        entries = result['entries']
        score_game_dicts(
            entry_dicts=entries, winner_int=result['winner'],
            total_moves=result['num_moves'], max_moves=max_moves,
            final_state_dict=result['final_state'],
            p1_captures=result['p1_captures'], p2_captures=result['p2_captures'],
        )
        return entries

    # Python fallback
    start_player = Player(start_player)
    state = GameState(
        board=Board.initial(),
        current_player=start_player,
        move_count=0,
    )
    entries = []
    move_count = 0
    player_captures = {Player.ONE: 0, Player.TWO: 0}

    while move_count < max_moves:
        legal_moves = state.legal_moves()
        if not legal_moves:
            break

        cur_diff = _p1_diff if state.current_player == Player.ONE else _p2_diff

        # Select move with optional noise for exploration
        if random.random() < noise_prob and len(legal_moves) > 1:
            chosen_move = random.choice(legal_moves)
        else:
            chosen_move = get_best_move(state, cur_diff, use_parallel=False)
            if chosen_move is None:
                chosen_move = random.choice(legal_moves)

        # Find the index of chosen move
        try:
            chosen_index = legal_moves.index(chosen_move)
        except ValueError:
            for i, m in enumerate(legal_moves):
                if m.path == chosen_move.path:
                    chosen_index = i
                    break
            else:
                chosen_index = 0
                chosen_move = legal_moves[0]

        # Track captures
        if chosen_move.is_capture:
            player_captures[state.current_player] += chosen_move.num_captures

        # Record the position
        entry = {
            'state': state.to_compact(),
            'legal_moves': [m.to_dict() for m in legal_moves],
            'chosen_index': chosen_index,
            'result': 0,
            'score': 0.0,
        }
        entries.append(entry)

        # Apply the move
        state = state.apply_move(chosen_move)
        move_count += 1

    # Determine winner and score entries
    winner = state.winner()
    winner_int = int(winner) if winner is not None else None

    for entry in entries:
        turn = entry['state']['turn']
        if winner is None:
            entry['result'] = 0
        elif turn == winner_int:
            entry['result'] = 1
        else:
            entry['result'] = -1

    score_game_dicts(
        entry_dicts=entries,
        winner_int=winner_int,
        total_moves=move_count,
        max_moves=max_moves,
        final_state_dict=state.to_compact(),
        p1_captures=player_captures[Player.ONE],
        p2_captures=player_captures[Player.TWO],
    )

    return entries


def main():
    parser = argparse.ArgumentParser(description='Self-play worker for GNU Parallel')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard'], help='AI difficulty')
    parser.add_argument('--p1-difficulty', type=str, default=None,
                        choices=['easy', 'medium', 'hard'],
                        help='Player 1 difficulty (overrides --difficulty)')
    parser.add_argument('--p2-difficulty', type=str, default=None,
                        choices=['easy', 'medium', 'hard'],
                        help='Player 2 difficulty (overrides --difficulty)')
    parser.add_argument('--max-moves', type=int, default=200, help='Max moves per game')
    parser.add_argument('--noise-prob', type=float, default=0.1, help='Random move probability')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_entries = 0

    with open(output_path, 'w') as f:
        for game_idx in range(args.games):
            # Alternate starting player
            start_player = 1 if game_idx % 2 == 0 else 2

            entries = play_single_game(
                difficulty=args.difficulty,
                max_moves=args.max_moves,
                noise_prob=args.noise_prob,
                start_player=start_player,
                p1_difficulty=args.p1_difficulty,
                p2_difficulty=args.p2_difficulty,
            )

            for entry in entries:
                f.write(json.dumps(entry) + '\n')

            total_entries += len(entries)

    mode = "algo-vs-algo" if args.p1_difficulty or args.p2_difficulty else args.difficulty
    cy_label = " (Cython)" if _HAS_FAST_GAME else ""
    print(f"Generated {total_entries} entries from {args.games} games ({mode}{cy_label}) -> {output_path}")


if __name__ == '__main__':
    main()
