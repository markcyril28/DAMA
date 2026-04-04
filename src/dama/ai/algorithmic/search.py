"""Alpha-beta search for algorithmic AI with multithreading support."""

import time
import os
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
import multiprocessing

from ...types import Move, Player
from ...game_state import GameState
from ...config import get_config
from .eval import evaluate, quick_evaluate, set_custom_weights


# Time budgets for difficulty levels (seconds)
TIME_BUDGETS = {
    'easy': 0.2,
    'medium': 0.8,
    'hard': 2.5,
}

# Max depth limits as fallback
MAX_DEPTHS = {
    'easy': 3,
    'medium': 5,
    'hard': 8,
}

# Number of worker threads (default to CPU count)
NUM_THREADS = max(1, multiprocessing.cpu_count())


@dataclass
class SearchResult:
    """Result of a search."""
    move: Optional[Move]
    score: float
    depth: int
    nodes: int


@dataclass
class SharedSearchState:
    """Shared state between parallel search threads."""
    best_score: float = float('-inf')
    best_move: Optional[Move] = None
    best_depth: int = 0
    total_nodes: int = 0
    lock: Lock = field(default_factory=Lock)
    stop_event: Event = field(default_factory=Event)
    
    def update_best(self, move: Move, score: float, depth: int, nodes: int) -> None:
        """Thread-safe update of best move."""
        with self.lock:
            self.total_nodes += nodes
            if score > self.best_score or (score == self.best_score and depth > self.best_depth):
                self.best_score = score
                self.best_move = move
                self.best_depth = depth
                # If winning move found, signal other threads to stop
                if score >= 9000:
                    self.stop_event.set()
    
    def add_nodes(self, nodes: int) -> None:
        """Thread-safe addition of node count."""
        with self.lock:
            self.total_nodes += nodes


class AlphaBetaSearch:
    """Alpha-beta search with iterative deepening."""

    def __init__(self, time_budget: float = 1.0, max_depth: int = 10):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.nodes_searched = 0
        self.start_time = 0.0
        self.timeout = False

    def search(self, state: GameState) -> SearchResult:
        """
        Find the best move using iterative deepening alpha-beta search.

        Returns the best move found within the time budget.
        """
        self.nodes_searched = 0
        self.start_time = time.time()
        self.timeout = False

        best_move: Optional[Move] = None
        best_score = float('-inf')
        best_depth = 0

        moves = state.legal_moves()
        if not moves:
            return SearchResult(None, -10000, 0, 0)

        if len(moves) == 1:
            # Only one legal move, no need to search
            return SearchResult(moves[0], 0, 0, 1)

        # Order moves for better pruning
        moves = self._order_moves(moves, state)

        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            if self._is_timeout():
                break

            try:
                move, score = self._search_root(state, moves, depth)
                if not self.timeout and move is not None:
                    best_move = move
                    best_score = score
                    best_depth = depth

                    # If we found a winning move, stop searching
                    if score >= 9000:
                        break

            except TimeoutError:
                break

        return SearchResult(best_move, best_score, best_depth, self.nodes_searched)

    def _search_root(self, state: GameState, moves: List[Move], depth: int) -> Tuple[Optional[Move], float]:
        """Search at the root level."""
        best_move = moves[0] if moves else None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in moves:
            if self._is_timeout():
                raise TimeoutError()

            new_state = state.apply_move(move)
            score = -self._alphabeta(new_state, depth - 1, -beta, -alpha)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

        return best_move, best_score

    def _alphabeta(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        """
        Alpha-beta search.

        Returns score from the perspective of the current player.
        """
        self.nodes_searched += 1

        if self._is_timeout():
            raise TimeoutError()

        # Terminal node or max depth reached
        if depth == 0 or state.is_terminal():
            return evaluate(state)

        moves = state.legal_moves()
        if not moves:
            # No legal moves means this player loses
            return -10000

        # Order moves for better pruning
        moves = self._order_moves(moves, state)

        best_score = float('-inf')

        for move in moves:
            new_state = state.apply_move(move)
            score = -self._alphabeta(new_state, depth - 1, -beta, -alpha)

            best_score = max(best_score, score)
            alpha = max(alpha, score)

            if alpha >= beta:
                break  # Beta cutoff

        return best_score

    def _order_moves(self, moves: List[Move], state: GameState) -> List[Move]:
        """
        Order moves for better pruning.

        Priority: captures -> promotions -> center moves -> others
        """
        def move_priority(move: Move) -> Tuple[int, int, int]:
            # Higher priority = searched first
            capture_priority = move.num_captures * 10
            promotion_priority = 5 if move.promotion else 0

            # Prefer moves toward center
            end_row, end_col = move.end
            center_dist = abs(3.5 - end_row) + abs(3.5 - end_col)
            center_priority = int(7 - center_dist)

            return (capture_priority, promotion_priority, center_priority)

        return sorted(moves, key=move_priority, reverse=True)

    def _is_timeout(self) -> bool:
        """Check if time budget is exceeded."""
        if self.timeout:
            return True

        elapsed = time.time() - self.start_time
        if elapsed >= self.time_budget:
            self.timeout = True
            return True

        return False


class ParallelAlphaBetaSearch:
    """
    Parallel alpha-beta search using multiple threads.
    
    Uses two parallelization strategies:
    1. Root parallelization: Different threads search different root moves
    2. Lazy SMP: Multiple threads search the same tree with different parameters
    """
    
    def __init__(self, time_budget: float = 1.0, max_depth: int = 10, 
                 num_threads: Optional[int] = None):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.num_threads = num_threads or NUM_THREADS
        self.start_time = 0.0
        self.shared_state: Optional[SharedSearchState] = None
    
    def search(self, state: GameState) -> SearchResult:
        """
        Find the best move using parallel alpha-beta search.
        
        Returns the best move found within the time budget.
        """
        self.start_time = time.time()
        self.shared_state = SharedSearchState()
        
        moves = state.legal_moves()
        if not moves:
            return SearchResult(None, -10000, 0, 0)
        
        if len(moves) == 1:
            return SearchResult(moves[0], 0, 0, 1)
        
        # Order moves for better distribution
        ordered_moves = self._order_moves(moves, state)
        
        # Use root parallelization for many moves, lazy SMP for fewer
        if len(ordered_moves) >= self.num_threads:
            self._parallel_root_search(state, ordered_moves)
        else:
            self._lazy_smp_search(state, ordered_moves)
        
        return SearchResult(
            self.shared_state.best_move,
            self.shared_state.best_score,
            self.shared_state.best_depth,
            self.shared_state.total_nodes
        )
    
    def _parallel_root_search(self, state: GameState, moves: List[Move]) -> None:
        """
        Parallel search at root level - each thread searches different moves.
        """
        # Distribute moves among threads
        move_chunks = self._distribute_moves(moves)
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for chunk in move_chunks:
                if chunk:  # Only submit if chunk has moves
                    future = executor.submit(
                        self._search_move_chunk, 
                        state, 
                        chunk
                    )
                    futures.append(future)
            
            # Wait for all threads to complete or timeout
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass  # Ignore errors from individual threads
    
    def _lazy_smp_search(self, state: GameState, moves: List[Move]) -> None:
        """
        Lazy SMP search - multiple threads search the same tree
        with slightly different parameters for better coverage.
        """
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for thread_id in range(self.num_threads):
                # Each thread uses slightly different depth/aspiration
                depth_offset = thread_id % 2  # Alternate depths
                future = executor.submit(
                    self._smp_worker,
                    state,
                    moves,
                    thread_id,
                    depth_offset
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass
    
    def _search_move_chunk(self, state: GameState, moves: List[Move]) -> None:
        """Search a chunk of root moves."""
        searcher = AlphaBetaSearch(
            time_budget=self.time_budget,
            max_depth=self.max_depth
        )
        searcher.start_time = self.start_time
        
        for move in moves:
            if self.shared_state.stop_event.is_set():
                break
            if self._is_timeout():
                break
            
            try:
                new_state = state.apply_move(move)
                
                # Iterative deepening for this move
                best_score = float('-inf')
                best_depth = 0
                
                for depth in range(1, self.max_depth + 1):
                    if self._is_timeout() or self.shared_state.stop_event.is_set():
                        break
                    
                    try:
                        searcher.nodes_searched = 0
                        searcher.timeout = False
                        score = -searcher._alphabeta(new_state, depth - 1, float('-inf'), float('inf'))
                        
                        if not searcher.timeout:
                            best_score = score
                            best_depth = depth
                            
                            # Update shared state
                            self.shared_state.update_best(move, score, depth, searcher.nodes_searched)
                            
                            if score >= 9000:
                                break
                    except TimeoutError:
                        break
                        
            except Exception:
                continue
    
    def _smp_worker(self, state: GameState, moves: List[Move], 
                    thread_id: int, depth_offset: int) -> None:
        """
        SMP worker thread - searches the full tree with varied parameters.
        """
        searcher = AlphaBetaSearch(
            time_budget=self.time_budget,
            max_depth=self.max_depth
        )
        searcher.start_time = self.start_time
        
        # Vary move ordering slightly based on thread ID
        thread_moves = moves.copy()
        if thread_id > 0 and len(thread_moves) > 1:
            # Rotate moves to start from different positions
            rotation = thread_id % len(thread_moves)
            thread_moves = thread_moves[rotation:] + thread_moves[:rotation]
        
        # Iterative deepening
        for depth in range(1 + depth_offset, self.max_depth + 1, 1):
            if self._is_timeout() or self.shared_state.stop_event.is_set():
                break
            
            try:
                for move in thread_moves:
                    if self._is_timeout() or self.shared_state.stop_event.is_set():
                        break
                    
                    searcher.nodes_searched = 0
                    searcher.timeout = False
                    
                    new_state = state.apply_move(move)
                    score = -searcher._alphabeta(new_state, depth - 1, float('-inf'), float('inf'))
                    
                    if not searcher.timeout:
                        self.shared_state.update_best(move, score, depth, searcher.nodes_searched)
                        
                        if score >= 9000:
                            return
                            
            except TimeoutError:
                break
            except Exception:
                continue
    
    def _distribute_moves(self, moves: List[Move]) -> List[List[Move]]:
        """Distribute moves among threads."""
        chunks: List[List[Move]] = [[] for _ in range(self.num_threads)]
        for i, move in enumerate(moves):
            chunks[i % self.num_threads].append(move)
        return chunks
    
    def _order_moves(self, moves: List[Move], state: GameState) -> List[Move]:
        """Order moves for better parallel distribution."""
        def move_priority(move: Move) -> Tuple[int, int, int]:
            capture_priority = move.num_captures * 10
            promotion_priority = 5 if move.promotion else 0
            end_row, end_col = move.end
            center_dist = abs(3.5 - end_row) + abs(3.5 - end_col)
            center_priority = int(7 - center_dist)
            return (capture_priority, promotion_priority, center_priority)
        
        return sorted(moves, key=move_priority, reverse=True)
    
    def _is_timeout(self) -> bool:
        """Check if time budget is exceeded."""
        elapsed = time.time() - self.start_time
        return elapsed >= self.time_budget


def get_best_move(state: GameState, difficulty: str = 'medium', 
                  custom_params: Optional[Dict[str, Any]] = None,
                  use_parallel: bool = True,
                  num_threads: Optional[int] = None) -> Optional[Move]:
    """
    Get the best move for the current player.

    Args:
        state: Current game state
        difficulty: 'easy', 'medium', 'hard', or 'custom'
        custom_params: Custom parameters dict when difficulty is 'custom'
        use_parallel: Whether to use multithreaded parallel search (default True)
        num_threads: Number of threads to use (default: CPU count)

    Returns:
        The best move, or None if no legal moves exist.
    """
    if difficulty == 'custom':
        # Use custom parameters from config or provided params
        if custom_params:
            time_budget = custom_params.get('time_budget', 1.0)
            max_depth = custom_params.get('max_depth', 6)
            # Set custom evaluation weights
            weights = {
                'man': custom_params.get('weight_man', 100),
                'king': custom_params.get('weight_king', 200),
                'mobility': custom_params.get('weight_mobility', 5),
                'advancement': custom_params.get('weight_advancement', 2),
                'center_control': custom_params.get('weight_center_control', 3),
                'back_rank': custom_params.get('weight_back_rank', 10),
            }
            set_custom_weights(weights)
        else:
            # Load from config
            config = get_config()
            algo_config = config.ai.algorithmic
            time_budget = algo_config.time_budget
            max_depth = algo_config.max_depth
            weights = {
                'man': algo_config.weight_man,
                'king': algo_config.weight_king,
                'mobility': algo_config.weight_mobility,
                'advancement': algo_config.weight_advancement,
                'center_control': algo_config.weight_center_control,
                'back_rank': algo_config.weight_back_rank,
            }
            set_custom_weights(weights)
    else:
        time_budget = TIME_BUDGETS.get(difficulty, TIME_BUDGETS['medium'])
        max_depth = MAX_DEPTHS.get(difficulty, MAX_DEPTHS['medium'])
        # Reset to default weights for non-custom difficulty
        set_custom_weights(None)

    # Use parallel search for better performance on multi-core systems
    if use_parallel and NUM_THREADS > 1:
        search = ParallelAlphaBetaSearch(
            time_budget=time_budget, 
            max_depth=max_depth,
            num_threads=num_threads or NUM_THREADS
        )
    else:
        search = AlphaBetaSearch(time_budget=time_budget, max_depth=max_depth)
    
    result = search.search(state)

    return result.move


def get_best_move_async(state: GameState, difficulty: str = 'medium',
                        custom_params: Optional[Dict[str, Any]] = None,
                        num_threads: Optional[int] = None) -> SearchResult:
    """
    Get the best move with full search result information.
    
    This is useful when you need detailed search statistics.
    
    Args:
        state: Current game state
        difficulty: 'easy', 'medium', 'hard', or 'custom'
        custom_params: Custom parameters dict when difficulty is 'custom'
        num_threads: Number of threads to use (default: CPU count)
    
    Returns:
        SearchResult with move, score, depth searched, and nodes searched.
    """
    if difficulty == 'custom':
        if custom_params:
            time_budget = custom_params.get('time_budget', 1.0)
            max_depth = custom_params.get('max_depth', 6)
            weights = {
                'man': custom_params.get('weight_man', 100),
                'king': custom_params.get('weight_king', 200),
                'mobility': custom_params.get('weight_mobility', 5),
                'advancement': custom_params.get('weight_advancement', 2),
                'center_control': custom_params.get('weight_center_control', 3),
                'back_rank': custom_params.get('weight_back_rank', 10),
            }
            set_custom_weights(weights)
        else:
            config = get_config()
            algo_config = config.ai.algorithmic
            time_budget = algo_config.time_budget
            max_depth = algo_config.max_depth
            weights = {
                'man': algo_config.weight_man,
                'king': algo_config.weight_king,
                'mobility': algo_config.weight_mobility,
                'advancement': algo_config.weight_advancement,
                'center_control': algo_config.weight_center_control,
                'back_rank': algo_config.weight_back_rank,
            }
            set_custom_weights(weights)
    else:
        time_budget = TIME_BUDGETS.get(difficulty, TIME_BUDGETS['medium'])
        max_depth = MAX_DEPTHS.get(difficulty, MAX_DEPTHS['medium'])
        set_custom_weights(None)

    if NUM_THREADS > 1:
        search = ParallelAlphaBetaSearch(
            time_budget=time_budget,
            max_depth=max_depth,
            num_threads=num_threads or NUM_THREADS
        )
    else:
        search = AlphaBetaSearch(time_budget=time_budget, max_depth=max_depth)
    
    return search.search(state)
