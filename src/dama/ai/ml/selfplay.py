"""Self-play for generating training data."""

import random
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from ...types import Move, Player
from ...game_state import GameState
from ...board import Board
from ...ai.algorithmic.search import get_best_move
from .scoring import score_game_entries, score_game_dicts

try:
    from .inference import get_ml_move
except ImportError:
    get_ml_move = None
from .replay import ReplayEntry

# Cython-accelerated full game loop — runs entire algo-vs-algo games in C.
# Eliminates all Python object creation (GameState, Board, Move, Piece) during
# gameplay. Falls back to Python play_single_game if extension not built.
try:
    from ...ai.algorithmic._fast_search import play_full_game_cy
    _HAS_FAST_GAME = True
except ImportError:
    _HAS_FAST_GAME = False


@dataclass
class GameRecord:
    """Record of a single game."""
    entries: List[ReplayEntry]
    winner: Optional[Player]
    num_moves: int


def play_single_game(
    difficulty: str = 'medium',
    max_moves: int = 200,
    noise_prob: float = 0.1,
    start_player: Player = Player.ONE,
    p1_policy: str = 'algorithmic',
    p2_policy: str = 'algorithmic',
    model_path: str = 'models/latest.pt',
    device=None,
    p1_difficulty: str = None,
    p2_difficulty: str = None,
    return_dicts: bool = False,
) -> Union[GameRecord, List[dict]]:
    """
    Play a single self-play game using the algorithmic AI as teacher.

    Args:
        difficulty: Default AI difficulty level (used when p1/p2_difficulty not set)
        max_moves: Maximum moves before declaring draw
        noise_prob: Probability of playing a random move (for exploration)
        p1_difficulty: Difficulty for Player 1 (overrides difficulty if set)
        p2_difficulty: Difficulty for Player 2 (overrides difficulty if set)
        return_dicts: If True, return List[dict] directly instead of GameRecord

    Returns:
        GameRecord (default) or List[dict] when return_dicts=True
    """
    start_player = Player(start_player)
    # Per-player difficulties default to the shared difficulty
    _p1_diff = p1_difficulty or difficulty
    _p2_diff = p2_difficulty or difficulty

    state = GameState(
        board=Board.initial(),
        current_player=start_player,
        move_count=0,
    )
    entries = []
    move_count = 0
    player_captures = {Player.ONE: 0, Player.TWO: 0}

    def _select_move(policy: str, legal_moves: List[Move], player_diff: str) -> Move:
        if random.random() < noise_prob and len(legal_moves) > 1:
            return random.choice(legal_moves)
        if policy == 'ml' and get_ml_move is not None:
            try:
                chosen = get_ml_move(state, model_path=model_path, device=device)
                if chosen is not None:
                    return chosen
            except Exception as e:
                print(f"  Warning: ML inference failed, using random move: {e}")
            return random.choice(legal_moves)
        # use_parallel=False: self-play already parallelizes at the game level
        # via ProcessPoolExecutor. Creating threads per move per worker causes
        # massive oversubscription (N_workers × CPU_COUNT threads).
        # With Cython fast search, this is moot (runs in C, single-threaded),
        # but the flag prevents thread storms if falling back to Python search.
        chosen = get_best_move(state, player_diff, use_parallel=False)
        if chosen is None:
            return random.choice(legal_moves)
        return chosen

    while not state.is_terminal() and move_count < max_moves:
        legal_moves = state.legal_moves()
        if not legal_moves:
            break

        policy = p1_policy if state.current_player == Player.ONE else p2_policy
        cur_diff = _p1_diff if state.current_player == Player.ONE else _p2_diff
        chosen_move = _select_move(policy, legal_moves, cur_diff)

        # Find the index of chosen move
        try:
            chosen_index = legal_moves.index(chosen_move)
        except ValueError:
            # If exact match not found, find by path
            for i, m in enumerate(legal_moves):
                if m.path == chosen_move.path:
                    chosen_index = i
                    break
            else:
                chosen_index = 0
                chosen_move = legal_moves[0]

        # Track captures per player
        if chosen_move.is_capture:
            player_captures[state.current_player] += chosen_move.num_captures

        # Record the position — build dicts directly when return_dicts=True
        # to skip ReplayEntry construction + to_dict() roundtrip
        if return_dicts:
            entry = {
                'state': state.to_compact(),
                'legal_moves': [m.to_dict() for m in legal_moves],
                'chosen_index': chosen_index,
                'result': 0,
                'score': 0.0,
            }
        else:
            entry = ReplayEntry(
                state=state.to_compact(),
                legal_moves=[m.to_dict() for m in legal_moves],
                chosen_index=chosen_index,
                result=0,  # Will be filled after game ends
                score=0.0,  # Will be filled by scoring system
            )
        entries.append(entry)

        # Apply the move
        state = state.apply_move(chosen_move)
        move_count += 1

    # Determine winner
    winner = state.winner()

    if return_dicts:
        # Dict path — update results and score via dict-native scoring
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
        return entries  # Return list of dicts directly
    else:
        # ReplayEntry path — original behavior
        for i, entry in enumerate(entries):
            turn = entry.state['turn']
            player = Player(turn)
            if winner is None:
                entry.result = 0
            elif winner == player:
                entry.result = 1
            else:
                entry.result = -1

        score_game_entries(
            entries=entries,
            winner=winner,
            total_moves=move_count,
            max_moves=max_moves,
            final_state=state,
            player_captures=player_captures,
        )
        return GameRecord(entries=entries, winner=winner, num_moves=move_count)


def _play_games_batch_worker_full(batch_args: list) -> List[dict]:
    """Batched worker for ML-policy games.

    Uses return_dicts=True to skip ReplayEntry construction + to_dict()
    roundtrip — builds dicts directly in the game loop.
    """
    all_entries = []
    for (difficulty, max_moves, noise_prob, start_player,
         p1_policy, p2_policy, model_path, _device) in batch_args:
        entry_dicts = play_single_game(
            difficulty=difficulty, max_moves=max_moves,
            noise_prob=noise_prob, start_player=start_player,
            p1_policy=p1_policy, p2_policy=p2_policy,
            model_path=model_path, device='cpu',
            return_dicts=True,
        )
        all_entries.extend(entry_dicts)
    return all_entries


def _play_game_worker_algo_vs_algo(
    args: Tuple[str, str, int, float, int]
) -> List[dict]:
    """Worker function for algo-vs-algo self-play with per-player difficulties.

    Uses Cython full game loop when available for 2-7x speedup.
    Falls back to Python play_single_game with return_dicts=True.
    """
    p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player = args
    if _HAS_FAST_GAME:
        result = play_full_game_cy(
            p1_difficulty=p1_difficulty, p2_difficulty=p2_difficulty,
            max_moves=max_moves, noise_prob=noise_prob,
            start_player=start_player,
        )
        score_game_dicts(
            entry_dicts=result['entries'], winner_int=result['winner'],
            total_moves=result['num_moves'], max_moves=max_moves,
            final_state_dict=result['final_state'],
            p1_captures=result['p1_captures'], p2_captures=result['p2_captures'],
        )
        return result['entries']
    return play_single_game(
        difficulty=p1_difficulty,
        max_moves=max_moves,
        noise_prob=noise_prob,
        start_player=start_player,
        p1_policy='algorithmic',
        p2_policy='algorithmic',
        p1_difficulty=p1_difficulty,
        p2_difficulty=p2_difficulty,
        return_dicts=True,
    )


def _play_games_batch_worker_algo(batch_args: list) -> List[dict]:
    """Batched worker for algo-vs-algo games with per-player difficulties.

    Uses Cython full game loop when available (per-player difficulties + TT).
    """
    all_entries = []
    if _HAS_FAST_GAME:
        for p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player in batch_args:
            result = play_full_game_cy(
                p1_difficulty=p1_difficulty, p2_difficulty=p2_difficulty,
                max_moves=max_moves, noise_prob=noise_prob,
                start_player=start_player,
            )
            score_game_dicts(
                entry_dicts=result['entries'], winner_int=result['winner'],
                total_moves=result['num_moves'], max_moves=max_moves,
                final_state_dict=result['final_state'],
                p1_captures=result['p1_captures'], p2_captures=result['p2_captures'],
            )
            all_entries.extend(result['entries'])
    else:
        for p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player in batch_args:
            entry_dicts = play_single_game(
                difficulty=p1_difficulty,
                max_moves=max_moves, noise_prob=noise_prob,
                start_player=start_player,
                p1_policy='algorithmic', p2_policy='algorithmic',
                p1_difficulty=p1_difficulty, p2_difficulty=p2_difficulty,
                return_dicts=True,
            )
            all_entries.extend(entry_dicts)
    return all_entries


