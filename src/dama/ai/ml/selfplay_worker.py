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

from dama.ai.ml.scoring import score_game_dicts

# Try Cython full game loop first (2-7x faster than Python game loop).
try:
    from dama.ai.algorithmic._fast_search import play_full_game_cy
    _HAS_FAST_GAME = True
except ImportError:
    _HAS_FAST_GAME = False

from dama.ai.ml.selfplay import (
    HARD_TEACHER_DIFFICULTY,
    play_single_game as _play_single_game_python,
)


def play_single_game(
    difficulty: str = 'medium',
    max_moves: int = 200,
    noise_prob: float = 0.1,
    start_player: int = 1,
    p1_difficulty: str = None,
    p2_difficulty: str = None,
    teacher_difficulty: str = HARD_TEACHER_DIFFICULTY,
    opening_plies: int = 0,
    opening_seed: int = 0,
    trajectory_source: str = 'algorithm',
    game_id: str = None,
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
            teacher_difficulty=teacher_difficulty,
            opening_plies=opening_plies,
            opening_seed=opening_seed,
            trajectory_source=trajectory_source,
            game_id=game_id,
        )
        entries = result['entries']
        score_game_dicts(
            entry_dicts=entries, winner_int=result['winner'],
            total_moves=result['num_moves'], max_moves=max_moves,
            final_state_dict=result['final_state'],
            p1_captures=result['p1_captures'], p2_captures=result['p2_captures'],
        )
        return entries

    # Use the shared Python implementation so label/action separation and
    # replay metadata stay identical to the in-process generator.
    return _play_single_game_python(
        difficulty=difficulty,
        max_moves=max_moves,
        noise_prob=noise_prob,
        start_player=start_player,
        p1_policy='algorithmic',
        p2_policy='algorithmic',
        p1_difficulty=_p1_diff,
        p2_difficulty=_p2_diff,
        return_dicts=True,
        teacher_difficulty=teacher_difficulty,
        opening_plies=opening_plies,
        opening_seed=opening_seed,
        trajectory_source=trajectory_source,
        game_id=game_id,
    )


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
    parser.add_argument('--opening-plies', type=int, default=0,
                        help='Seeded random legal plies before replay recording')
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
                opening_plies=args.opening_plies,
                opening_seed=(args.seed + game_idx if args.seed is not None else game_idx),
                trajectory_source='algorithm',
                game_id=f"worker-game-{game_idx}",
            )

            for entry in entries:
                f.write(json.dumps(entry) + '\n')

            total_entries += len(entries)

    mode = "algo-vs-algo" if args.p1_difficulty or args.p2_difficulty else args.difficulty
    cy_label = " (Cython)" if _HAS_FAST_GAME else ""
    print(f"Generated {total_entries} entries from {args.games} games ({mode}{cy_label}) -> {output_path}")


if __name__ == '__main__':
    main()
