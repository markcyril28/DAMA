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
from dama.game_state import GameState
from dama.board import Board
from dama.ai.algorithmic.search import get_best_move


def play_single_game(
    difficulty: str = 'medium',
    max_moves: int = 200,
    noise_prob: float = 0.1,
    start_player: int = 1,
) -> list:
    """Play a single game and return entries as dicts."""
    start_player = Player(start_player)
    state = GameState(
        board=Board.initial(),
        current_player=start_player,
        move_count=0,
    )
    entries = []
    move_count = 0

    while not state.is_terminal() and move_count < max_moves:
        legal_moves = state.legal_moves()
        if not legal_moves:
            break

        # Select move with optional noise for exploration
        if random.random() < noise_prob and len(legal_moves) > 1:
            chosen_move = random.choice(legal_moves)
        else:
            chosen_move = get_best_move(state, difficulty)
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

        # Record the position
        entry = {
            'state': state.to_compact(),
            'legal_moves': [m.to_dict() for m in legal_moves],
            'chosen_index': chosen_index,
            'result': 0,  # Will be filled after game ends
        }
        entries.append(entry)

        # Apply the move
        state = state.apply_move(chosen_move)
        move_count += 1

    # Determine winner
    winner = state.winner()

    # Update results from each player's perspective
    for entry in entries:
        turn = entry['state']['turn']
        player = Player(turn)

        if winner is None:
            entry['result'] = 0  # Draw
        elif winner == player:
            entry['result'] = 1  # Win
        else:
            entry['result'] = -1  # Loss

    return entries


def main():
    parser = argparse.ArgumentParser(description='Self-play worker for GNU Parallel')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--difficulty', type=str, default='medium', 
                        choices=['easy', 'medium', 'hard'], help='AI difficulty')
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
            )
            
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
            
            total_entries += len(entries)
    
    print(f"Generated {total_entries} entries from {args.games} games -> {output_path}")


if __name__ == '__main__':
    main()
