"""Model vs Algorithm self-play testing."""

import json
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
import random

from ...types import Move, Player
from ...game_state import GameState


class TestResult(Enum):
    """Result of a single test game."""
    ML_WIN = "ml_win"
    ALGO_WIN = "algo_win"
    DRAW = "draw"


@dataclass
class GameTestRecord:
    """Record of a single test game."""
    result: TestResult
    ml_player: Player
    winner: Optional[Player]
    num_moves: int
    ml_moves: int
    algo_moves: int
    game_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'result': self.result.value,
            'ml_player': self.ml_player.value,
            'winner': self.winner.value if self.winner else None,
            'num_moves': self.num_moves,
            'ml_moves': self.ml_moves,
            'algo_moves': self.algo_moves,
            'game_time_ms': self.game_time_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameTestRecord':
        return cls(
            result=TestResult(data['result']),
            ml_player=Player(data['ml_player']),
            winner=Player(data['winner']) if data['winner'] else None,
            num_moves=data['num_moves'],
            ml_moves=data['ml_moves'],
            algo_moves=data['algo_moves'],
            game_time_ms=data['game_time_ms'],
        )


@dataclass
class TestStatistics:
    """Statistics from model vs algorithm testing."""
    total_games: int = 0
    ml_wins: int = 0
    algo_wins: int = 0
    draws: int = 0
    
    ml_as_p1_wins: int = 0
    ml_as_p1_losses: int = 0
    ml_as_p1_draws: int = 0
    
    ml_as_p2_wins: int = 0
    ml_as_p2_losses: int = 0
    ml_as_p2_draws: int = 0
    
    avg_game_length: float = 0.0
    avg_game_time_ms: float = 0.0
    
    # Model info
    model_path: str = ""
    algo_difficulty: str = "medium"
    
    # Timestamp
    start_time: str = ""
    end_time: str = ""
    
    # Game records
    games: List[Dict] = field(default_factory=list)
    
    @property
    def ml_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.ml_wins / self.total_games
    
    @property
    def algo_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.algo_wins / self.total_games
    
    @property
    def draw_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.draws / self.total_games
    
    @property
    def ml_as_p1_games(self) -> int:
        return self.ml_as_p1_wins + self.ml_as_p1_losses + self.ml_as_p1_draws
    
    @property
    def ml_as_p2_games(self) -> int:
        return self.ml_as_p2_wins + self.ml_as_p2_losses + self.ml_as_p2_draws
    
    @property
    def ml_as_p1_win_rate(self) -> float:
        games = self.ml_as_p1_games
        if games == 0:
            return 0.0
        return self.ml_as_p1_wins / games
    
    @property
    def ml_as_p2_win_rate(self) -> float:
        games = self.ml_as_p2_games
        if games == 0:
            return 0.0
        return self.ml_as_p2_wins / games
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_games': self.total_games,
            'ml_wins': self.ml_wins,
            'algo_wins': self.algo_wins,
            'draws': self.draws,
            'ml_win_rate': self.ml_win_rate,
            'algo_win_rate': self.algo_win_rate,
            'draw_rate': self.draw_rate,
            'ml_as_p1_wins': self.ml_as_p1_wins,
            'ml_as_p1_losses': self.ml_as_p1_losses,
            'ml_as_p1_draws': self.ml_as_p1_draws,
            'ml_as_p1_win_rate': self.ml_as_p1_win_rate,
            'ml_as_p2_wins': self.ml_as_p2_wins,
            'ml_as_p2_losses': self.ml_as_p2_losses,
            'ml_as_p2_draws': self.ml_as_p2_draws,
            'ml_as_p2_win_rate': self.ml_as_p2_win_rate,
            'avg_game_length': self.avg_game_length,
            'avg_game_time_ms': self.avg_game_time_ms,
            'model_path': self.model_path,
            'algo_difficulty': self.algo_difficulty,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'games': self.games,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestStatistics':
        stats = cls(
            total_games=data.get('total_games', 0),
            ml_wins=data.get('ml_wins', 0),
            algo_wins=data.get('algo_wins', 0),
            draws=data.get('draws', 0),
            ml_as_p1_wins=data.get('ml_as_p1_wins', 0),
            ml_as_p1_losses=data.get('ml_as_p1_losses', 0),
            ml_as_p1_draws=data.get('ml_as_p1_draws', 0),
            ml_as_p2_wins=data.get('ml_as_p2_wins', 0),
            ml_as_p2_losses=data.get('ml_as_p2_losses', 0),
            ml_as_p2_draws=data.get('ml_as_p2_draws', 0),
            avg_game_length=data.get('avg_game_length', 0.0),
            avg_game_time_ms=data.get('avg_game_time_ms', 0.0),
            model_path=data.get('model_path', ''),
            algo_difficulty=data.get('algo_difficulty', 'medium'),
            start_time=data.get('start_time', ''),
            end_time=data.get('end_time', ''),
            games=data.get('games', []),
        )
        return stats


def _play_single_test_game(args: Tuple[str, str, int, int]) -> Dict[str, Any]:
    """
    Play a single test game between ML model and algorithm.

    Args:
        args: (model_path, difficulty, ml_player_value, max_moves)

    Returns:
        Game record as dict
    """
    model_path, difficulty, ml_player_value, max_moves = args

    # Import here to avoid loading in main process
    from .inference import get_ml_move
    from ..algorithmic.search import get_best_move

    ml_player = Player(ml_player_value)

    state = GameState.initial()
    move_count = 0
    ml_moves = 0
    algo_moves = 0

    start_time = time.time()

    while move_count < max_moves:
        legal_moves = state.legal_moves()
        if not legal_moves:
            break

        current_player = state.current_player

        if current_player == ml_player:
            # ML model's turn
            try:
                chosen_move = get_ml_move(state, model_path)
                if chosen_move is None:
                    chosen_move = random.choice(legal_moves)
            except Exception as e:
                print(f"  Warning: ML inference failed in test game, using random move: {e}")
                chosen_move = random.choice(legal_moves)
            ml_moves += 1
        else:
            # Algorithm's turn
            chosen_move = get_best_move(state, difficulty)
            if chosen_move is None:
                chosen_move = random.choice(legal_moves)
            algo_moves += 1

        # Validate move
        if chosen_move not in legal_moves:
            # Try to find matching move by path
            matching = [m for m in legal_moves if m.path == chosen_move.path]
            if matching:
                chosen_move = matching[0]
            else:
                chosen_move = random.choice(legal_moves)

        state = state.apply_move(chosen_move)
        move_count += 1

    game_time_ms = (time.time() - start_time) * 1000

    # Determine result
    winner = state.winner()

    if winner is None:
        result = TestResult.DRAW
    elif winner == ml_player:
        result = TestResult.ML_WIN
    else:
        result = TestResult.ALGO_WIN

    record = GameTestRecord(
        result=result,
        ml_player=ml_player,
        winner=winner,
        num_moves=move_count,
        ml_moves=ml_moves,
        algo_moves=algo_moves,
        game_time_ms=game_time_ms,
    )

    return record.to_dict()


def _play_test_games_batch(args: Tuple[str, str, List[int], int]) -> List[Dict[str, Any]]:
    """Play multiple test games interleaved with batched ML inference.

    Same semantics as calling _play_single_test_game N times, but batches
    ML forward passes across all active games.  ~2-3x faster for ML moves.
    Algo moves are still sequential (alpha-beta can't be batched).

    Args:
        args: (model_path, difficulty, ml_player_values, max_moves)
              ml_player_values is a list of Player int values, one per game.

    Returns:
        List of game record dicts.
    """
    import torch
    import numpy as np
    from .inference import get_model
    from ..algorithmic.search import get_best_move
    # [Pass 67] Fast encoding: Cython (~6-7x faster) or Python dict-based (~2x)
    try:
        from ._fast_encode import (
            encode_board_fast_cy as _cy_board,
            encode_moves_fast_cy as _cy_moves,
        )
    except ImportError:
        _cy_board = None
        _cy_moves = None
    from .dataset import _encode_board_fast, _encode_moves_fast

    model_path, difficulty, ml_player_values, max_moves = args
    n = len(ml_player_values)

    # Load model once for the entire batch
    try:
        model = get_model(model_path, 'cpu')
    except Exception:
        # Fall back to sequential
        results = []
        for mpv in ml_player_values:
            results.append(_play_single_test_game(
                (model_path, difficulty, mpv, max_moves)))
        return results

    # Initialize all games
    games = []
    for mpv in ml_player_values:
        games.append({
            'state': GameState.initial(),
            'ml_player': Player(mpv),
            'move_count': 0,
            'ml_moves': 0,
            'algo_moves': 0,
            'start_time': time.time(),
        })

    active = list(range(n))

    while active:
        ml_requests = []      # (game_idx, legal_moves)
        algo_requests = []    # (game_idx, legal_moves)
        immediate = []        # (game_idx, legal_moves, chosen_idx)

        new_active = []
        for i in active:
            g = games[i]
            if g['move_count'] >= max_moves:
                continue
            legal_moves = g['state'].legal_moves()
            if not legal_moves:
                continue
            new_active.append(i)

            if len(legal_moves) == 1:
                immediate.append((i, legal_moves, 0))
                continue

            if g['state'].current_player == g['ml_player']:
                ml_requests.append((i, legal_moves))
            else:
                algo_requests.append((i, legal_moves))

        active = new_active
        if not active:
            break

        # Immediate moves (forced single moves)
        for game_idx, legal_moves, idx in immediate:
            g = games[game_idx]
            if g['state'].current_player == g['ml_player']:
                g['ml_moves'] += 1
            else:
                g['algo_moves'] += 1
            g['state'] = g['state'].apply_move(legal_moves[idx])
            g['move_count'] += 1

        # Batched ML inference
        if ml_requests:
            batch_sz = len(ml_requests)
            max_m = max(len(lm) for _, lm in ml_requests)

            boards = np.zeros((batch_sz, 5, 8, 8), dtype=np.float32)
            all_mf = np.zeros((batch_sz, max_m, 8), dtype=np.float32)
            counts = np.zeros(batch_sz, dtype=np.int32)

            for j, (game_idx, legal_moves) in enumerate(ml_requests):
                sd = games[game_idx]['state'].to_compact()
                md = [m.to_dict() for m in legal_moves]
                if _cy_board is not None:
                    _cy_board(sd, boards[j])
                    counts[j] = _cy_moves(sd, md, all_mf[j])
                else:
                    _encode_board_fast(sd, boards[j])
                    counts[j] = _encode_moves_fast(sd, md, all_mf[j])

            with torch.inference_mode():
                scores = model.forward_padded(
                    torch.from_numpy(boards),
                    torch.from_numpy(all_mf),
                    torch.from_numpy(counts),
                )

            for j, (game_idx, legal_moves) in enumerate(ml_requests):
                best_idx = scores[j, :len(legal_moves)].argmax().item()
                g = games[game_idx]
                g['ml_moves'] += 1
                g['state'] = g['state'].apply_move(legal_moves[best_idx])
                g['move_count'] += 1

        # Sequential algo moves
        for game_idx, legal_moves in algo_requests:
            g = games[game_idx]
            move = get_best_move(g['state'], difficulty, use_parallel=False)
            if move is None:
                move = random.choice(legal_moves)
            elif move not in legal_moves:
                matching = [m for m in legal_moves if m.path == move.path]
                move = matching[0] if matching else random.choice(legal_moves)
            g['algo_moves'] += 1
            g['state'] = g['state'].apply_move(move)
            g['move_count'] += 1

    # Build results
    results = []
    for g in games:
        game_time_ms = (time.time() - g['start_time']) * 1000
        winner = g['state'].winner()
        if winner is None:
            result = TestResult.DRAW
        elif winner == g['ml_player']:
            result = TestResult.ML_WIN
        else:
            result = TestResult.ALGO_WIN

        results.append(GameTestRecord(
            result=result,
            ml_player=g['ml_player'],
            winner=winner,
            num_moves=g['move_count'],
            ml_moves=g['ml_moves'],
            algo_moves=g['algo_moves'],
            game_time_ms=game_time_ms,
        ).to_dict())

    return results


class ModelVsAlgoTester:
    """
    Run test games between ML model and algorithmic AI.
    
    Features:
    - Run multiple games with ML as both Player 1 and Player 2
    - Collect statistics on win rates
    - Save results to file
    """
    
    def __init__(
        self,
        model_path: str = "models/latest.pt",
        algo_difficulty: str = "medium",
        num_workers: int = 4,
        max_moves: int = 200,
        stats_dir: str = "models/test_stats",
    ):
        self.model_path = model_path
        self.algo_difficulty = algo_difficulty
        self.num_workers = num_workers
        self.max_moves = max_moves
        self.stats_dir = Path(stats_dir)
        
        self._running = False
        self._games_completed = 0
    
    def run_tests(
        self,
        num_games: int = 100,
        callback: Optional[Callable[[int, int, TestStatistics], None]] = None,
    ) -> TestStatistics:
        """
        Run test games.
        
        Args:
            num_games: Total number of games to play (half as P1, half as P2)
            callback: Optional callback(games_completed, total_games, current_stats)
        
        Returns:
            TestStatistics with all results
        """
        self._running = True
        self._games_completed = 0
        
        stats = TestStatistics(
            model_path=self.model_path,
            algo_difficulty=self.algo_difficulty,
            start_time=datetime.now().isoformat(),
        )
        
        # Split games between ML as P1 and ML as P2
        games_as_p1 = num_games // 2
        games_as_p2 = num_games - games_as_p1
        
        # Prepare arguments
        args_list = []
        for _ in range(games_as_p1):
            args_list.append((self.model_path, self.algo_difficulty, Player.ONE.value, self.max_moves))
        for _ in range(games_as_p2):
            args_list.append((self.model_path, self.algo_difficulty, Player.TWO.value, self.max_moves))
        
        # Shuffle to mix up the order
        random.shuffle(args_list)
        
        total_moves = 0
        total_time_ms = 0.0

        def _ingest_result(record_data: Dict[str, Any]) -> None:
            """Update stats from a single completed game."""
            nonlocal total_moves, total_time_ms
            record = GameTestRecord.from_dict(record_data)

            stats.total_games += 1
            total_moves += record.num_moves
            total_time_ms += record.game_time_ms

            if record.result == TestResult.ML_WIN:
                stats.ml_wins += 1
                if record.ml_player == Player.ONE:
                    stats.ml_as_p1_wins += 1
                else:
                    stats.ml_as_p2_wins += 1
            elif record.result == TestResult.ALGO_WIN:
                stats.algo_wins += 1
                if record.ml_player == Player.ONE:
                    stats.ml_as_p1_losses += 1
                else:
                    stats.ml_as_p2_losses += 1
            else:
                stats.draws += 1
                if record.ml_player == Player.ONE:
                    stats.ml_as_p1_draws += 1
                else:
                    stats.ml_as_p2_draws += 1

            stats.games.append(record_data)
            self._games_completed += 1

            stats.avg_game_length = total_moves / stats.total_games
            stats.avg_game_time_ms = total_time_ms / stats.total_games

            if callback:
                callback(self._games_completed, num_games, stats)

        # Batch games per worker for interleaved ML inference (~2-3x faster).
        # Each worker plays multiple games with batched forward passes.
        _GAMES_PER_BATCH = max(2, num_games // self.num_workers)
        batches = []
        for i in range(0, len(args_list), _GAMES_PER_BATCH):
            chunk = args_list[i:i + _GAMES_PER_BATCH]
            ml_player_vals = [a[2] for a in chunk]
            batches.append((self.model_path, self.algo_difficulty,
                            ml_player_vals, self.max_moves))

        # Use 'spawn' context so child processes don't inherit
        # the parent's CUDA state (fork + CUDA = "Cannot re-initialize CUDA")
        spawn_ctx = mp.get_context('spawn')
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=spawn_ctx) as executor:
                futures = [executor.submit(_play_test_games_batch, batch)
                           for batch in batches]

                for future in as_completed(futures):
                    if not self._running:
                        break

                    try:
                        for record_data in future.result():
                            _ingest_result(record_data)
                    except Exception as e:
                        print(f"Test batch error: {e}")

        except Exception as e:
            print(f"Parallel testing failed ({e}), falling back to sequential")
            # Sequential fallback — use batched function directly
            for batch in batches:
                if not self._running:
                    break

                try:
                    for record_data in _play_test_games_batch(batch):
                        _ingest_result(record_data)
                except Exception as e:
                    print(f"Test batch error: {e}")
        
        stats.end_time = datetime.now().isoformat()
        
        # Save stats
        self._save_stats(stats)
        
        return stats
    
    def _save_stats(self, stats: TestStatistics) -> str:
        """Save test statistics to file."""
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_{timestamp}.json"
        filepath = self.stats_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        
        # Also save as latest
        latest_path = self.stats_dir / "latest_test.json"
        with open(latest_path, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        
        print(f"Test stats saved: {filepath}")
        return str(filepath)
    
    def stop(self) -> None:
        """Stop running tests."""
        self._running = False
    
    @property
    def games_completed(self) -> int:
        return self._games_completed


def load_test_stats(stats_file: str = "models/test_stats/latest_test.json") -> Optional[TestStatistics]:
    """Load test statistics from file."""
    path = Path(stats_file)
    if not path.exists():
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return TestStatistics.from_dict(data)
    except Exception:
        return None


def list_test_results(stats_dir: str = "models/test_stats") -> List[Dict[str, Any]]:
    """List all available test result files."""
    path = Path(stats_dir)
    if not path.exists():
        return []
    
    results = []
    for f in path.glob("test_*.json"):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                results.append({
                    'path': str(f),
                    'name': f.name,
                    'total_games': data.get('total_games', 0),
                    'ml_win_rate': data.get('ml_win_rate', 0),
                    'start_time': data.get('start_time', ''),
                })
        except Exception:
            continue
    
    # Sort by time
    results.sort(key=lambda x: x['start_time'], reverse=True)
    return results
