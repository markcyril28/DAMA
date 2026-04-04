"""Self-play for generating training data."""

import random
import platform
import sys
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
from .replay import ReplayEntry, ReplayBuffer

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
) -> GameRecord:
    """
    Play a single self-play game using the algorithmic AI as teacher.

    Args:
        difficulty: Default AI difficulty level (used when p1/p2_difficulty not set)
        max_moves: Maximum moves before declaring draw
        noise_prob: Probability of playing a random move (for exploration)
        p1_difficulty: Difficulty for Player 1 (overrides difficulty if set)
        p2_difficulty: Difficulty for Player 2 (overrides difficulty if set)

    Returns:
        GameRecord with all positions and the chosen moves
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

        # Record the position
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

    # Update results from each player's perspective
    for i, entry in enumerate(entries):
        turn = entry.state['turn']
        player = Player(turn)

        if winner is None:
            entry.result = 0  # Draw
        elif winner == player:
            entry.result = 1  # Win
        else:
            entry.result = -1  # Loss

    # Compute detailed scores using the scoring system
    score_game_entries(
        entries=entries,
        winner=winner,
        total_moves=move_count,
        max_moves=max_moves,
        final_state=state,
        player_captures=player_captures,
    )

    return GameRecord(entries=entries, winner=winner, num_moves=move_count)


def _play_game_worker_cy(args: Tuple[str, int, float, int]) -> List[dict]:
    """Cython-accelerated worker — runs entire game in C (algo-vs-algo, same difficulty)."""
    difficulty, max_moves, noise_prob, start_player = args
    from .scoring import score_game_dicts as _score
    result = play_full_game_cy(
        p1_difficulty=difficulty,
        p2_difficulty=difficulty,
        max_moves=max_moves,
        noise_prob=noise_prob,
        start_player=start_player,
    )
    _score(
        entry_dicts=result['entries'],
        winner_int=result['winner'],
        total_moves=result['num_moves'],
        max_moves=max_moves,
        final_state_dict=result['final_state'],
        p1_captures=result['p1_captures'],
        p2_captures=result['p2_captures'],
    )
    return result['entries']


def _play_game_worker(args: Tuple[str, int, float, int]) -> List[dict]:
    """Worker function for parallel self-play."""
    difficulty, max_moves, noise_prob, start_player = args
    record = play_single_game(
        difficulty,
        max_moves,
        noise_prob,
        start_player,
        p1_policy='algorithmic',
        p2_policy='algorithmic',
    )
    return [e.to_dict() for e in record.entries]


def _play_games_batch_worker(batch_args: list) -> List[dict]:
    """Run multiple algo-only games in one worker call to reduce IPC overhead.

    Instead of 1 game per task (high per-task pickle + scheduling overhead),
    each worker processes a batch of games and returns all entries at once.
    Reduces IPC from N_games submissions to N_workers submissions.
    """
    all_entries = []
    for difficulty, max_moves, noise_prob, start_player in batch_args:
        record = play_single_game(
            difficulty, max_moves, noise_prob, start_player,
            p1_policy='algorithmic', p2_policy='algorithmic',
        )
        all_entries.extend(e.to_dict() for e in record.entries)
    return all_entries


def _play_games_batch_worker_full(batch_args: list) -> List[dict]:
    """Batched worker for ML-policy games."""
    all_entries = []
    for (difficulty, max_moves, noise_prob, start_player,
         p1_policy, p2_policy, model_path, _device) in batch_args:
        record = play_single_game(
            difficulty=difficulty, max_moves=max_moves,
            noise_prob=noise_prob, start_player=start_player,
            p1_policy=p1_policy, p2_policy=p2_policy,
            model_path=model_path, device='cpu',
        )
        all_entries.extend(e.to_dict() for e in record.entries)
    return all_entries


def _play_game_worker_full(
    args: Tuple[str, int, float, int, str, str, str, object]
) -> List[dict]:
    """Worker function for parallel self-play with optional ML policy.

    Always uses CPU for ML inference to avoid CUDA multi-process issues.
    """
    (difficulty, max_moves, noise_prob, start_player, p1_policy,
     p2_policy, model_path, _device) = args
    record = play_single_game(
        difficulty=difficulty,
        max_moves=max_moves,
        noise_prob=noise_prob,
        start_player=start_player,
        p1_policy=p1_policy,
        p2_policy=p2_policy,
        model_path=model_path,
        device='cpu',  # Force CPU in worker processes
    )
    return [e.to_dict() for e in record.entries]


def _play_game_worker_algo_vs_algo_cy(
    args: Tuple[str, str, int, float, int]
) -> List[dict]:
    """Cython-accelerated algo-vs-algo worker with per-player difficulties."""
    p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player = args
    from .scoring import score_game_dicts as _score
    result = play_full_game_cy(
        p1_difficulty=p1_difficulty,
        p2_difficulty=p2_difficulty,
        max_moves=max_moves,
        noise_prob=noise_prob,
        start_player=start_player,
    )
    _score(
        entry_dicts=result['entries'],
        winner_int=result['winner'],
        total_moves=result['num_moves'],
        max_moves=max_moves,
        final_state_dict=result['final_state'],
        p1_captures=result['p1_captures'],
        p2_captures=result['p2_captures'],
    )
    return result['entries']


def _play_game_worker_algo_vs_algo(
    args: Tuple[str, str, int, float, int]
) -> List[dict]:
    """Worker function for algo-vs-algo self-play with per-player difficulties."""
    p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player = args
    record = play_single_game(
        difficulty=p1_difficulty,
        max_moves=max_moves,
        noise_prob=noise_prob,
        start_player=start_player,
        p1_policy='algorithmic',
        p2_policy='algorithmic',
        p1_difficulty=p1_difficulty,
        p2_difficulty=p2_difficulty,
    )
    return [e.to_dict() for e in record.entries]


def _play_games_batch_worker_algo(batch_args: list) -> List[dict]:
    """Batched worker for algo-vs-algo games with per-player difficulties."""
    all_entries = []
    for p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player in batch_args:
        record = play_single_game(
            difficulty=p1_difficulty,
            max_moves=max_moves, noise_prob=noise_prob,
            start_player=start_player,
            p1_policy='algorithmic', p2_policy='algorithmic',
            p1_difficulty=p1_difficulty, p2_difficulty=p2_difficulty,
        )
        all_entries.extend(e.to_dict() for e in record.entries)
    return all_entries


class SelfPlayRunner:
    """
    Runs self-play games in parallel to generate training data.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        num_workers: int = 10,
        difficulty: str = 'medium',
        max_moves: int = 200,
        noise_prob: float = 0.1,
        p1_policy: str = 'algorithmic',
        p2_policy: str = 'algorithmic',
        model_path: str = 'models/latest.pt',
        device=None,
    ):
        self.replay_buffer = replay_buffer
        self.num_workers = num_workers
        self.difficulty = difficulty
        self.max_moves = max_moves
        self.noise_prob = noise_prob
        self.p1_policy = p1_policy
        self.p2_policy = p2_policy
        self.model_path = model_path
        self.device = device

        self._running = False
        self._games_completed = 0

    def run_games(self, num_games: int, callback=None) -> int:
        """
        Run self-play games and store in replay buffer.

        Args:
            num_games: Number of games to play
            callback: Optional callback(games_completed, total_games)

        Returns:
            Total number of training entries generated
        """
        self._running = True
        self._games_completed = 0
        total_entries = 0

        # Start new replay file
        self.replay_buffer.start_new_file()

        # Prepare arguments for workers (balance starting player across games)
        num_p1 = num_games // 2
        num_p2 = num_games - num_p1
        start_players = ([Player.ONE] * num_p1) + ([Player.TWO] * num_p2)
        random.shuffle(start_players)
        args = [
            (self.difficulty, self.max_moves, self.noise_prob, int(start))
            for start in start_players
        ]

        can_parallel = self.num_workers > 1

        # On Windows, ProcessPoolExecutor can be problematic - limit workers and add timeout
        is_windows = platform.system() == 'Windows'
        effective_workers = min(self.num_workers, 8) if is_windows else self.num_workers

        # Always use ProcessPoolExecutor for true parallelism (avoids GIL).
        # ML inference uses CPU in worker processes to avoid CUDA multi-process issues.
        # Batch multiple games per worker to reduce IPC overhead (pickle + scheduling).
        if can_parallel:
            uses_ml = self.p1_policy == 'ml' or self.p2_policy == 'ml'
            if uses_ml:
                task_args = [
                    (
                        self.difficulty,
                        self.max_moves,
                        self.noise_prob,
                        int(start),
                        self.p1_policy,
                        self.p2_policy,
                        self.model_path,
                        self.device,
                    )
                    for start in start_players
                ]
                batch_worker_fn = _play_games_batch_worker_full
            else:
                task_args = args
                batch_worker_fn = _play_games_batch_worker

            # Split games into batches — one batch per worker
            batch_size = max(1, (num_games + effective_workers - 1) // effective_workers)
            batches = [
                task_args[i:i + batch_size]
                for i in range(0, len(task_args), batch_size)
            ]

            print(f"  Starting parallel self-play with {effective_workers} workers, "
                  f"{len(batches)} batches of ~{batch_size} games...")
            sys.stdout.flush()
            try:
                with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                    futures = [executor.submit(batch_worker_fn, b) for b in batches]
                    # O(1) future→batch lookup (was O(n) via list.index)
                    future_to_batch = {f: i for i, f in enumerate(futures)}
                    print(f"  Submitted {len(futures)} batch tasks...")
                    sys.stdout.flush()

                    for future in as_completed(futures):
                        if not self._running:
                            break

                        try:
                            entries_data = future.result(timeout=300 * batch_size)
                            # Write dicts directly — skip dict→ReplayEntry→dict round-trip
                            self.replay_buffer.add_entry_dicts(entries_data)
                            total_entries += len(entries_data)

                            batch_idx = future_to_batch[future]
                            games_in_batch = len(batches[batch_idx])
                            self._games_completed += games_in_batch
                            if callback:
                                callback(self._games_completed, num_games)

                        except Exception as e:
                            print(f"Self-play error: {e}")

            except Exception as e:
                # Fallback to sequential if multiprocessing fails
                print(f"Parallel self-play failed ({e}), falling back to sequential")
                can_parallel = False

        if not can_parallel:
            for i, start_player in enumerate(start_players):
                if not self._running:
                    break

                record = play_single_game(
                    self.difficulty,
                    self.max_moves,
                    self.noise_prob,
                    start_player,
                    p1_policy=self.p1_policy,
                    p2_policy=self.p2_policy,
                    model_path=self.model_path,
                    device=self.device,
                )
                self.replay_buffer.add_entries(record.entries)
                total_entries += len(record.entries)

                self._games_completed += 1
                if callback:
                    callback(self._games_completed, num_games)

        self.replay_buffer.close()
        return total_entries

    def stop(self) -> None:
        """Stop running games."""
        self._running = False

    @property
    def games_completed(self) -> int:
        return self._games_completed
