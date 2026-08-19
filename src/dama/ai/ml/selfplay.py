"""Self-play for generating training data."""

import gc
import random
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from ...types import Move, Player
from ...game_state import GameState
from ...board import Board
from ...ai.algorithmic.search import get_best_move
from .scoring import score_game_entries, score_game_dicts

try:
    from .inference import get_ml_move, get_ml_move_idx
except ImportError:
    get_ml_move = None
    get_ml_move_idx = None
from .replay import ReplayEntry

# Cython-accelerated full game loop — runs entire algo-vs-algo games in C.
# Eliminates all Python object creation (GameState, Board, Move, Piece) during
# gameplay. Falls back to Python play_single_game if extension not built.
try:
    from ...ai.algorithmic._fast_search import play_full_game_cy
    _HAS_FAST_GAME = True
except ImportError:
    _HAS_FAST_GAME = False

# [Pass 70] Cython-accelerated legal move generation for interleaved self-play.
# C movegen + dict creation is ~50-100x faster than Python legal_moves() + to_dict().
# Eliminates ~12 Move object creations per position (~3s savings per 200-game cycle).
try:
    from ...ai.algorithmic._fast_search import fast_generate_moves as _fast_gen_moves
    _HAS_FAST_MOVEGEN = True
except ImportError:
    _HAS_FAST_MOVEGEN = False

# [Pass 75] Compact-state API: operate on raw board bytes (64-byte int8 array)
# instead of GameState/Board/Move objects. Eliminates ~100-200μs of Python object
# creation per position (Move.from_dict + Board.__init__ + GameState.__init__).
try:
    from ...ai.algorithmic._fast_search import (
        init_board_bytes as _init_board_bytes,
        gen_moves_from_board as _gen_moves_board,
        apply_move_board as _apply_move_board,
        board_bytes_to_compact as _board_to_compact,
    )
    _HAS_COMPACT_STATE = True
except ImportError:
    _HAS_COMPACT_STATE = False

# [Pass 67] Fast encoding for interleaved self-play inference.
# Cython versions are 6-7x faster than Python move_encoder.encode_board/moves.
# Python _encode_*_fast are ~2x faster (avoid GameState/Move object traversal).
# Both write directly into pre-allocated numpy arrays from compact dicts.
try:
    from ._fast_encode import (
        encode_board_fast_cy as _cy_encode_board,
        encode_moves_fast_cy as _cy_encode_moves,
    )
    _HAS_FAST_ENCODE = True
except ImportError:
    _HAS_FAST_ENCODE = False

from .dataset import _encode_board_fast, _encode_moves_fast


def _selfplay_worker_init(seed: int = 0):
    """Per-worker initializer for self-play ProcessPoolExecutor pools.

    The seed is experiment-derived, never PID-derived or time-derived.
    Trajectory exploration itself uses per-game Random instances so worker
    scheduling cannot change generated actions.

    [Pass 101] gc.freeze() moves every object inherited from the parent via
    fork (the trainer's live CUDA tensors: training model, in-GPU replay
    buffer, optimizer state) into a permanent generation the cyclic GC never
    scans.  Without this, an automatic GC pass in the worker (triggered by
    object allocation deep inside _fast_search's PyTuple_New) sweeps one of
    those inherited CUDA tensors; its destructor calls ExchangeDevice
    (cudaSetDevice), which is invalid in a fork that never initialized CUDA,
    raising cudaErrorInitializationError from a C++ destructor -> std::terminate
    -> the worker dies (BrokenProcessPool, then the cycle finishes sequentially).
    freeze() leaves normal GC active for the worker's OWN allocations, so it
    cannot leak; it also avoids GC traversal of inherited pages (fewer COW
    faults, aligning with the Pass 70 copy-on-write goal).
    """
    random.seed(int(seed) & 0xFFFFFFFF)
    gc.freeze()

# [Pass 70] Fork-inherited model for self-play workers.
# On Linux (fork start method), parent sets this global before creating the
# ProcessPoolExecutor. Workers inherit the model via copy-on-write — zero
# torch.save/load disk I/O.  model.eval() + inference_mode means no COW
# faults on tensor data pages (read-only access).
# Set to None when not in a self-play cycle (cleanup prevents memory leaks).
_FORK_MODEL = None


HARD_TEACHER_DIFFICULTY = 'hard'


def allocate_policy_distillation_games(total_games: int) -> Tuple[int, int]:
    """Return an exact 70 percent algorithm, 30 percent model allocation.

    Exact integer allocation requires a cycle size divisible by ten. Keeping
    that constraint explicit prevents silent rounding drift across cycles.

    Returns:
        ``(algorithm_games, current_model_games)``
    """
    if isinstance(total_games, bool) or not isinstance(total_games, int):
        raise TypeError("total_games must be an integer")
    if total_games < 0:
        raise ValueError("total_games must be non-negative")
    if total_games % 10 != 0:
        raise ValueError("total_games must be divisible by 10 for an exact 70/30 split")
    return total_games * 7 // 10, total_games * 3 // 10


def apply_random_training_opening(
    state: GameState,
    opening_plies: int = 0,
    opening_seed: Optional[int] = 0,
) -> Tuple[GameState, int]:
    """Apply a deterministic sequence of uniformly random legal opening plies.

    Opening moves are trajectory setup only and are not emitted as replay
    labels. A local RNG keeps opening selection independent from the 0.10
    played-action exploration stream.
    """
    if isinstance(opening_plies, bool) or not isinstance(opening_plies, int):
        raise TypeError("opening_plies must be an integer")
    if opening_plies < 0:
        raise ValueError("opening_plies must be non-negative")

    rng = random.Random(opening_seed)
    applied = 0
    for _ in range(opening_plies):
        legal_moves = state.legal_moves()
        if not legal_moves:
            break
        state = state.apply_move(rng.choice(legal_moves))
        applied += 1
    return state, applied


def _validate_teacher_difficulty(teacher_difficulty: str) -> None:
    if teacher_difficulty != HARD_TEACHER_DIFFICULTY:
        raise ValueError(
            f"Policy-distillation labels require teacher_difficulty="
            f"{HARD_TEACHER_DIFFICULTY!r}, got {teacher_difficulty!r}"
        )


def _move_index(move, legal_moves, default: Optional[int] = None) -> int:
    """Match a Move or move dict to a legal-move sequence without reordering it."""
    if move is None:
        if default is None:
            raise RuntimeError("Teacher or behavior policy returned no move")
        return default
    move_path = move.get('path') if isinstance(move, dict) else move.path
    move_path = tuple(tuple(p) for p in move_path)
    for idx, candidate in enumerate(legal_moves):
        path = candidate.get('path') if isinstance(candidate, dict) else candidate.path
        if tuple(tuple(p) for p in path) == move_path:
            return idx
    if default is None:
        raise RuntimeError("Teacher or behavior move is outside the legal move list")
    return default


def _trajectory_source(p1_policy: str, p2_policy: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return 'current_model' if 'ml' in (p1_policy, p2_policy) else 'algorithm'


def _entry_dict(
    state: dict,
    legal_moves: list,
    teacher_index: int,
    played_index: int,
    trajectory_source: str,
    was_exploration: bool,
    opening_plies: int,
    game_id: Optional[str],
) -> dict:
    entry = {
        'state': state,
        'legal_moves': legal_moves,
        'chosen_index': teacher_index,
        'played_index': played_index,
        'trajectory_source': trajectory_source,
        'was_exploration': bool(was_exploration),
        'teacher_difficulty': HARD_TEACHER_DIFFICULTY,
        'opening_plies': opening_plies,
        'result': 0,
        'score': 0.0,
    }
    if game_id is not None:
        entry['game_id'] = str(game_id)
    return entry


def _normalize_ml_task(task: tuple) -> dict:
    """Normalize legacy or extended ML-trajectory task tuples."""
    if len(task) < 8:
        raise ValueError("ML self-play task requires at least 8 fields")
    difficulty, max_moves, noise_prob, start_player = task[:4]
    p1_policy, p2_policy, model_path, device = task[4:8]
    opening_plies = task[8] if len(task) > 8 else 0
    opening_seed = task[9] if len(task) > 9 else 0
    explicit_source = task[10] if len(task) > 10 else None
    game_id = task[11] if len(task) > 11 else None
    teacher_difficulty = task[12] if len(task) > 12 else HARD_TEACHER_DIFFICULTY
    inference_depth = task[13] if len(task) > 13 else 1
    _validate_teacher_difficulty(teacher_difficulty)
    if isinstance(inference_depth, bool) or int(inference_depth) not in (1, 2, 3):
        raise ValueError("ML trajectory inference depth must be 1, 2, or 3")
    return {
        'difficulty': difficulty,
        'max_moves': int(max_moves),
        'noise_prob': float(noise_prob),
        'start_player': int(start_player),
        'p1_policy': p1_policy,
        'p2_policy': p2_policy,
        'model_path': model_path,
        'device': device,
        'opening_plies': int(opening_plies),
        'opening_seed': opening_seed,
        'trajectory_source': _trajectory_source(
            p1_policy, p2_policy, explicit_source),
        'game_id': str(game_id) if game_id is not None else None,
        'teacher_difficulty': teacher_difficulty,
        'inference_depth': int(inference_depth),
    }


def _normalize_algo_task(task: tuple) -> dict:
    """Normalize legacy or extended algorithm-trajectory task tuples."""
    if len(task) < 5:
        raise ValueError("Algorithm self-play task requires at least 5 fields")
    p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player = task[:5]
    opening_plies = task[5] if len(task) > 5 else 0
    opening_seed = task[6] if len(task) > 6 else 0
    trajectory_source = task[7] if len(task) > 7 else 'algorithm'
    game_id = task[8] if len(task) > 8 else None
    teacher_difficulty = task[9] if len(task) > 9 else HARD_TEACHER_DIFFICULTY
    _validate_teacher_difficulty(teacher_difficulty)
    return {
        'p1_difficulty': p1_difficulty,
        'p2_difficulty': p2_difficulty,
        'max_moves': int(max_moves),
        'noise_prob': float(noise_prob),
        'start_player': int(start_player),
        'opening_plies': int(opening_plies),
        'opening_seed': opening_seed,
        'trajectory_source': trajectory_source or 'algorithm',
        'game_id': str(game_id) if game_id is not None else None,
        'teacher_difficulty': teacher_difficulty,
    }


def _apply_compact_random_opening(
    board_bytes,
    player: int,
    opening_plies: int,
    opening_seed: Optional[int],
):
    """Apply a seeded legal opening through the compact Cython board API."""
    if opening_plies < 0:
        raise ValueError("opening_plies must be non-negative")
    rng = random.Random(opening_seed)
    captures = {1: 0, 2: 0}
    applied = 0
    for _ in range(opening_plies):
        moves = _gen_moves_board(board_bytes, player)
        if not moves:
            break
        move = moves[rng.randrange(len(moves))]
        previous_player = player
        board_bytes, player, num_captures = _apply_move_board(
            board_bytes, player, move)
        captures[previous_player] += num_captures
        applied += 1
    return board_bytes, player, captures, applied


# Symmetry augmentation is intentionally omitted. The only global board
# symmetry that preserves dark squares and men's forward direction is a
# 180-degree rotation combined with a player swap. The encoder already applies
# exactly that side-to-move normalization, so augmentation would duplicate the
# encoded board and move tensors rather than add information.



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
    teacher_difficulty: str = HARD_TEACHER_DIFFICULTY,
    opening_plies: int = 0,
    opening_seed: Optional[int] = 0,
    trajectory_source: Optional[str] = None,
    game_id: Optional[str] = None,
    inference_depth: int = 1,
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
        teacher_difficulty: Must be ``hard`` for policy distillation.
        opening_plies: Random legal setup plies played before replay recording.
        opening_seed: Seed used only for deterministic opening selection.
        trajectory_source: Audit label such as ``algorithm`` or ``current_model``.
        game_id: Optional stable game identifier retained in replay metadata.
        inference_depth: Current-model behavior depth. Values 2 and 3 require
            an enhanced-stage value head.

    Returns:
        GameRecord (default) or List[dict] when return_dicts=True
    """
    _validate_teacher_difficulty(teacher_difficulty)
    if isinstance(inference_depth, bool) or inference_depth not in (1, 2, 3):
        raise ValueError("inference_depth must be one of 1, 2, or 3")
    start_player = Player(start_player)
    # Per-player difficulties default to the shared difficulty
    _p1_diff = p1_difficulty or difficulty
    _p2_diff = p2_difficulty or difficulty

    state = GameState(
        board=Board.initial(),
        current_player=start_player,
        move_count=0,
    )
    state, applied_opening_plies = apply_random_training_opening(
        state, opening_plies=opening_plies, opening_seed=opening_seed)
    behavior_rng = random.Random(
        (int(opening_seed or 0) ^ 0x6A09E667F3BCC909) & ((1 << 64) - 1)
    )
    source = _trajectory_source(p1_policy, p2_policy, trajectory_source)
    entries = []
    move_count = 0
    player_captures = {1: 0, 2: 0}

    def _select_behavior_index(
        policy: str,
        legal_moves: List[Move],
        player_diff: str,
        teacher_index: int,
    ) -> int:
        """Select the non-exploration behavior action, independent of its label."""
        if policy == 'ml':
            # A model trajectory must never silently become an algorithmic
            # trajectory.  Doing so leaves the audit label saying
            # ``current_model`` while the action was selected by the teacher.
            # Fail closed so the caller can quarantine the incomplete cycle.
            if get_ml_move_idx is None:
                raise RuntimeError(
                    "Current-model trajectory inference is unavailable"
                )
            try:
                # get_ml_move_idx accepts pre-computed legal_moves, returns
                # index directly — avoids redundant legal_moves() generation
                # inside get_ml_move and the O(n) index search afterward.
                idx = get_ml_move_idx(
                    state, legal_moves, model_path=model_path, device=device,
                    depth=inference_depth,
                )
                if idx is not None and 0 <= int(idx) < len(legal_moves):
                    return int(idx)
                if idx is not None:
                    raise RuntimeError(
                        f"Current-model trajectory returned illegal index {idx}"
                    )
            except Exception as e:
                raise RuntimeError(
                    "Current-model trajectory inference failed"
                ) from e
            raise RuntimeError(
                "Current-model trajectory returned no move for a legal state"
            )
        if policy == 'algorithmic' and player_diff == HARD_TEACHER_DIFFICULTY:
            return teacher_index
        # use_parallel=False: self-play already parallelizes at the game level
        # via ProcessPoolExecutor. Creating threads per move per worker causes
        # massive oversubscription (N_workers × CPU_COUNT threads).
        # With Cython fast search, this is moot (runs in C, single-threaded),
        # but the flag prevents thread storms if falling back to Python search.
        chosen = get_best_move(state, player_diff, use_parallel=False)
        return _move_index(chosen, legal_moves)

    while move_count < max_moves:
        legal_moves = state.legal_moves()
        if not legal_moves:
            break

        policy = p1_policy if state.current_player == Player.ONE else p2_policy
        cur_diff = _p1_diff if state.current_player == Player.ONE else _p2_diff
        if len(legal_moves) == 1:
            teacher_index = 0
            played_index = 0
            was_exploration = False
        else:
            teacher_move = get_best_move(
                state, HARD_TEACHER_DIFFICULTY, use_parallel=False)
            teacher_index = _move_index(teacher_move, legal_moves)
            was_exploration = behavior_rng.random() < noise_prob
            if was_exploration:
                played_index = behavior_rng.randrange(len(legal_moves))
            else:
                played_index = _select_behavior_index(
                    policy, legal_moves, cur_diff, teacher_index)
        played_move = legal_moves[played_index]

        # Track captures per player
        if played_move.is_capture:
            player_captures[int(state.current_player)] += played_move.num_captures

        # Record the position — build dicts directly when return_dicts=True
        # to skip ReplayEntry construction + to_dict() roundtrip
        if return_dicts:
            entry = _entry_dict(
                state.to_compact(), [m.to_dict() for m in legal_moves],
                teacher_index, played_index, source, was_exploration,
                applied_opening_plies, game_id)
        else:
            entry = ReplayEntry(
                state=state.to_compact(),
                legal_moves=[m.to_dict() for m in legal_moves],
                chosen_index=teacher_index,
                result=0,  # Will be filled after game ends
                score=0.0,  # Will be filled by scoring system
                played_index=played_index,
                trajectory_source=source,
                was_exploration=was_exploration,
                teacher_difficulty=HARD_TEACHER_DIFFICULTY,
                opening_plies=applied_opening_plies,
                game_id=str(game_id) if game_id is not None else None,
            )
        entries.append(entry)

        # Apply the move
        state = state.apply_move(played_move)
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
            p1_captures=player_captures[1],
            p2_captures=player_captures[2],
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



def play_games_interleaved(batch_args: list) -> List[dict]:
    """Play multiple games interleaved with batched ML inference.

    Instead of playing games sequentially (one forward pass per ML move),
    this advances all active games one step at a time and batches ML
    inference requests into a single forward_padded() call.  CPU BLAS
    amortizes per-layer setup overhead across positions, yielding ~30-40%
    faster ML inference vs single-position calls.

    Algo moves are processed sequentially (alpha-beta can't be batched).

    Falls back to sequential play_single_game when the model can't be
    loaded or no games use ML policy.

    Args:
        batch_args: List of legacy or extended tuples:
            (difficulty, max_moves, noise_prob, start_player,
             p1_policy, p2_policy, model_path, device,
             opening_plies=0, opening_seed=0, trajectory_source=None,
             game_id=None, teacher_difficulty='hard')

    Returns:
        List of entry dicts (same format as play_single_game return_dicts=True)
    """
    import torch
    import numpy as np
    from .inference import get_model

    if not batch_args:
        return []

    specs = [_normalize_ml_task(task) for task in batch_args]

    # Extract model_path from first arg (same for all games in batch)
    _model_path = specs[0]['model_path']

    # Load model once for the entire batch.
    # [Pass 70] Try fork-inherited model first (zero disk I/O on Linux fork).
    # Falls back to get_model() which loads from disk via cache.
    model = None
    has_ml = any(
        spec['p1_policy'] == 'ml' or spec['p2_policy'] == 'ml'
        for spec in specs)
    if has_ml:
        if _FORK_MODEL is not None:
            model = _FORK_MODEL
        else:
            try:
                model = get_model(_model_path, 'cpu')
            except Exception:
                model = None

    # A missing model is acceptable only for an algorithm-only batch.  A
    # current-model trajectory must fail closed rather than silently becoming
    # algorithmic data with an incorrect provenance label.
    if model is None:
        if has_ml:
            raise RuntimeError(
                "Current-model self-play requires a loadable ML model"
            )
        return _play_games_batch_sequential(batch_args)

    n = len(batch_args)

    # [Pass 75] Compact-state path: maintain board as raw bytes (int8[64])
    # instead of GameState objects. Eliminates Move.from_dict + Board.__init__
    # + GameState.__init__ per position (~100-200μs saved per move).
    _use_compact = _HAS_COMPACT_STATE

    # Initialize all games
    # [Pass 80] Use int keys {1: 0, 2: 0} for captures dict instead of
    # {Player.ONE: 0, Player.TWO: 0}.  Eliminates ~10K Player() IntEnum
    # constructor calls per self-play cycle (~2μs each = ~20ms per cycle).
    # score_game_dicts expects int keys (p1_captures, p2_captures).
    games = []
    if _use_compact:
        _init_bb = _init_board_bytes()
        for spec in specs:
            bb, player, opening_captures, applied_opening = (
                _apply_compact_random_opening(
                    _init_bb, spec['start_player'], spec['opening_plies'],
                    spec['opening_seed']))
            games.append({
                'bb': bb,  # raw board bytes (64 int8)
                'pl': player,  # current player (1 or 2)
                'entries': [],
                'p1_policy': spec['p1_policy'],
                'p2_policy': spec['p2_policy'],
                'difficulty': spec['difficulty'],
                'noise_prob': spec['noise_prob'],
                'max_moves': spec['max_moves'],
                'move_count': 0,
                'state_move_count': applied_opening,
                'captures': opening_captures,
                'opening_plies': applied_opening,
                'trajectory_source': spec['trajectory_source'],
                'game_id': spec['game_id'],
                'rng': random.Random(
                    (int(spec['opening_seed'] or 0) ^ 0x6A09E667F3BCC909)
                    & ((1 << 64) - 1)
                ),
            })
    else:
        for spec in specs:
            state, applied_opening = apply_random_training_opening(
                GameState(
                    board=Board.initial(),
                    current_player=Player(spec['start_player']),
                    move_count=0,
                ),
                opening_plies=spec['opening_plies'],
                opening_seed=spec['opening_seed'],
            )
            games.append({
                'state': state,
                'entries': [],
                'p1_policy': spec['p1_policy'],
                'p2_policy': spec['p2_policy'],
                'difficulty': spec['difficulty'],
                'noise_prob': spec['noise_prob'],
                'max_moves': spec['max_moves'],
                'move_count': 0,
                'captures': {1: 0, 2: 0},
                'opening_plies': applied_opening,
                'trajectory_source': spec['trajectory_source'],
                'game_id': spec['game_id'],
                'rng': random.Random(
                    (int(spec['opening_seed'] or 0) ^ 0x6A09E667F3BCC909)
                    & ((1 << 64) - 1)
                ),
            })

    active = list(range(n))

    # [Pass 71] Pre-allocate numpy arrays for ML inference.
    # Avoids ~75-150 np.zeros() allocation+zeroing cycles per game batch.
    # Fixed max_moves=32 (config default) ensures contiguous memory layout
    # and enables JIT-traced model to use fixed tensor shapes.
    _MAX_M = 32  # matches config max_moves_per_sample
    _boards_buf = np.zeros((n, 5, 8, 8), dtype=np.float32)
    _mf_buf = np.zeros((n, _MAX_M, 8), dtype=np.float32)
    _counts_buf = np.zeros(n, dtype=np.int32)

    # [Pass 84] Pre-create torch tensor views of the numpy buffers.
    # torch.from_numpy shares memory (zero copy), so encoding writes to
    # numpy are visible through these tensors.  Avoids ~3 × torch.from_numpy
    # Python object creations per ML inference round (~150 rounds/batch).
    _boards_t = torch.from_numpy(_boards_buf)
    _mf_t = torch.from_numpy(_mf_buf)
    _counts_t = torch.from_numpy(_counts_buf)

    # [Pass 70] Use Cython movegen when available — generates move dicts
    # directly in C, ~50-100x faster than Python legal_moves() + to_dict().
    # Only create a Move object for the single chosen move (for apply_move).
    _use_fast_movegen = _HAS_FAST_MOVEGEN

    while active:
        # Partition active games by action type.
        # [Pass 69] Pre-compute compact dicts once per position in the
        # partition step, shared by ALL paths (immediate, ML, algo).
        # [Pass 70] With fast movegen, md comes directly from Cython C
        # movegen (no Move objects created at all). Move objects are only
        # created for the chosen move via Move.from_dict().
        ml_requests = []     # (game_idx, teacher_idx, sd, md)
        algo_requests = []   # (game_idx, teacher_idx, sd, md)
        immediate = []       # (game_idx, teacher_idx, played_idx, explored, sd, md)

        new_active = []
        for i in active:
            g = games[i]
            if g['move_count'] >= g['max_moves']:
                g['_ended'] = 'max_moves'  # draw
                continue

            # [Pass 75] Compact path: movegen + to_compact from raw board bytes.
            # Avoids _load_board (Python dict iteration) and to_compact overhead.
            if _use_compact:
                md = _gen_moves_board(g['bb'], g['pl'])
            elif _use_fast_movegen:
                md = _fast_gen_moves(g['state'])
            else:
                legal_moves = g['state'].legal_moves()
                md = [m.to_dict() for m in legal_moves]

            if not md:
                g['_ended'] = 'no_moves'  # current player loses
                continue
            new_active.append(i)

            if _use_compact:
                sd = _board_to_compact(
                    g['bb'], g['pl'], g['state_move_count'])
            else:
                sd = g['state'].to_compact()

            # Single legal move — no inference needed
            if len(md) == 1:
                immediate.append((i, 0, 0, False, sd, md))
                continue

            if _use_compact:
                policy = g['p1_policy'] if g['pl'] == 1 else g['p2_policy']
            else:
                policy = g['p1_policy'] if g['state'].current_player == Player.ONE else g['p2_policy']

            teacher_state = (
                GameState.from_compact(sd) if _use_compact else g['state'])
            teacher_move = get_best_move(
                teacher_state, HARD_TEACHER_DIFFICULTY, use_parallel=False)
            teacher_idx = _move_index(teacher_move, md)

            # Exploration changes the played action but never the hard label.
            if g['rng'].random() < g['noise_prob']:
                immediate.append((
                    i, teacher_idx, g['rng'].randrange(len(md)), True, sd, md))
            elif policy == 'ml':
                ml_requests.append((i, teacher_idx, sd, md))
            elif g['difficulty'] == HARD_TEACHER_DIFFICULTY:
                immediate.append((
                    i, teacher_idx, teacher_idx, False, sd, md))
            else:
                algo_requests.append((i, teacher_idx, sd, md))

        active = new_active
        if not active:
            break

        # Immediate moves — apply chosen move from dict
        for game_idx, teacher_idx, played_idx, explored, sd, md in immediate:
            game = games[game_idx]
            played_md = md[played_idx]
            game['entries'].append(_entry_dict(
                sd, md, teacher_idx, played_idx,
                game['trajectory_source'], explored,
                game['opening_plies'], game['game_id']))
            if _use_compact:
                prev_pl = game['pl']
                game['bb'], game['pl'], ncaps = _apply_move_board(
                    game['bb'], game['pl'], played_md)
                if ncaps > 0:
                    game['captures'][prev_pl] += ncaps
                game['state_move_count'] += 1
            else:
                captures = played_md.get('captures', ())
                if captures:
                    game['captures'][int(game['state'].current_player)] += len(captures)
                game['state'] = game['state'].apply_move(Move.from_dict(played_md))
            game['move_count'] += 1

        # Batched ML inference — encode using partition-step dicts
        if ml_requests:
            batch_sz = len(ml_requests)

            # [Pass 71] Reuse pre-allocated buffers instead of np.zeros() per round.
            boards = _boards_buf[:batch_sz]
            all_mf = _mf_buf[:batch_sz]
            counts = _counts_buf[:batch_sz]

            # Dicts already pre-computed in partition step — just encode
            for j, (game_idx, _teacher_idx, sd, md) in enumerate(ml_requests):
                if _HAS_FAST_ENCODE:
                    _cy_encode_board(sd, boards[j])
                    counts[j] = _cy_encode_moves(sd, md, all_mf[j])
                else:
                    _encode_board_fast(sd, boards[j])
                    counts[j] = _encode_moves_fast(sd, md, all_mf[j])

            with torch.inference_mode():
                scores = model.forward_padded(
                    _boards_t[:batch_sz],
                    _mf_t[:batch_sz],
                    _counts_t[:batch_sz],
                )

            # [Pass 71] Batch argmax: single torch op for all positions.
            best_indices_list = scores.argmax(dim=1).tolist()

            for j, (game_idx, teacher_idx, sd, md) in enumerate(ml_requests):
                best_idx = best_indices_list[j]
                if not 0 <= best_idx < len(md):
                    raise RuntimeError(
                        "Current-model trajectory returned an illegal move index"
                    )
                game = games[game_idx]
                played_md = md[best_idx]
                game['entries'].append(_entry_dict(
                    sd, md, teacher_idx, best_idx,
                    game['trajectory_source'], False,
                    game['opening_plies'], game['game_id']))
                if _use_compact:
                    prev_pl = game['pl']
                    game['bb'], game['pl'], ncaps = _apply_move_board(
                        game['bb'], game['pl'], played_md)
                    if ncaps > 0:
                        game['captures'][prev_pl] += ncaps
                    game['state_move_count'] += 1
                else:
                    captures = played_md.get('captures', ())
                    if captures:
                        game['captures'][int(game['state'].current_player)] += len(captures)
                    game['state'] = game['state'].apply_move(Move.from_dict(played_md))
                game['move_count'] += 1

        # Sequential algo moves — match by path start/end positions
        for game_idx, teacher_idx, sd, md in algo_requests:
            # Compact path: reconstruct GameState from compact dict for algo search.
            # Cost is ~50μs — negligible vs 200ms-2.5s search.
            if _use_compact:
                _algo_state = GameState.from_compact(sd)
            else:
                _algo_state = games[game_idx]['state']
            move = get_best_move(_algo_state, games[game_idx]['difficulty'],
                                 use_parallel=False)
            if move is None:
                raise RuntimeError(
                    "Algorithm trajectory behavior returned no legal move"
                )
            else:
                # Match algo Move to dict by path start+end positions.
                # Cython fast_search and fast_generate_moves use the same C
                # move generator, so paths are identical. Start+end match is
                # sufficient; full-path fallback handles rare multi-capture
                # ambiguities.
                # [Pass 81] Tuple comparison (dp[0] == mp_start) is faster than
                # 4 element-by-element int comparisons — CPython tuple __eq__
                # short-circuits on first mismatch with C-level comparison.
                mp = move.path
                mp_start = mp[0]
                mp_end = mp[-1]
                idx = None
                for k, m_dict in enumerate(md):
                    dp = m_dict['path']
                    if dp[0] == mp_start and dp[-1] == mp_end:
                        idx = k
                        break
                if idx is None:
                    raise RuntimeError(
                        "Algorithm trajectory behavior move is outside the legal set"
                    )
            game = games[game_idx]
            played_md = md[idx]
            game['entries'].append(_entry_dict(
                sd, md, teacher_idx, idx,
                game['trajectory_source'], False,
                game['opening_plies'], game['game_id']))
            if _use_compact:
                prev_pl = game['pl']
                game['bb'], game['pl'], ncaps = _apply_move_board(
                    game['bb'], game['pl'], played_md)
                if ncaps > 0:
                    game['captures'][prev_pl] += ncaps
                game['state_move_count'] += 1
            else:
                captures = played_md.get('captures', ())
                if captures:
                    game['captures'][int(game['state'].current_player)] += len(captures)
                game['state'] = game['state'].apply_move(Move.from_dict(played_md))
            game['move_count'] += 1

    # Score all games' entries
    all_entries = []
    for g in games:
        entries = g['entries']
        if not entries:
            continue
        # Use cached end reason to avoid redundant legal_moves() call in
        # winner() → is_terminal() → has_legal_moves().  The game loop
        # already determined why each game ended.
        end_reason = g.get('_ended')
        if _use_compact:
            # Compact path: use raw player int for winner detection
            if end_reason == 'no_moves':
                # g['pl'] is current player who has no moves → opponent wins
                winner_int = 2 if g['pl'] == 1 else 1
            elif end_reason == 'max_moves':
                winner_int = None
            else:
                winner_int = None  # safety fallback (should not reach here)
            final_state_dict = _board_to_compact(
                g['bb'], g['pl'], g['state_move_count'])
        else:
            state = g['state']
            if end_reason == 'no_moves':
                winner = state.current_player.opponent()
            elif end_reason == 'max_moves':
                winner = None
            else:
                winner = state.winner()  # fallback for games still active
            winner_int = int(winner) if winner is not None else None
            final_state_dict = state.to_compact()

        # Entries are initialized with result=0 (draw) during the game loop.
        # Only update for decisive games — skips the loop entirely for draws.
        if winner_int is not None:
            for entry in entries:
                entry['result'] = 1 if entry['state']['turn'] == winner_int else -1

        score_game_dicts(
            entry_dicts=entries,
            winner_int=winner_int,
            total_moves=g['move_count'],
            max_moves=g['max_moves'],
            final_state_dict=final_state_dict,
            p1_captures=g['captures'][1],
            p2_captures=g['captures'][2],
        )
        all_entries.extend(entries)

    return all_entries


def _play_games_batch_sequential(batch_args: list) -> List[dict]:
    """Sequential fallback for _play_games_batch_worker_full."""
    all_entries = []
    for task in batch_args:
        spec = _normalize_ml_task(task)
        entry_dicts = play_single_game(
            difficulty=spec['difficulty'], max_moves=spec['max_moves'],
            noise_prob=spec['noise_prob'], start_player=spec['start_player'],
            p1_policy=spec['p1_policy'], p2_policy=spec['p2_policy'],
            model_path=spec['model_path'], device='cpu',
            return_dicts=True,
            teacher_difficulty=spec['teacher_difficulty'],
            opening_plies=spec['opening_plies'],
            opening_seed=spec['opening_seed'],
            trajectory_source=spec['trajectory_source'],
            game_id=spec['game_id'],
            inference_depth=spec['inference_depth'],
        )
        all_entries.extend(entry_dicts)
    return all_entries


def _play_games_batch_worker_full(batch_args: list) -> List[dict]:
    """Batched worker for ML-policy games.

    Uses interleaved game play with batched ML inference when multiple
    ML-policy games are in the batch.  This amortizes CPU forward pass
    overhead across positions (~30-40% faster ML inference per position).

    Falls back to sequential play for single-game batches or when no
    games use ML policy.
    """
    if not batch_args:
        return []

    # Interleaved play benefits from batching ≥2 games
    specs = [_normalize_ml_task(task) for task in batch_args]
    has_ml = any(
        spec['p1_policy'] == 'ml' or spec['p2_policy'] == 'ml'
        for spec in specs)
    if has_ml and any(spec['inference_depth'] > 1 for spec in specs):
        return _play_games_batch_sequential(batch_args)
    if has_ml and len(batch_args) >= 2:
        return play_games_interleaved(batch_args)

    return _play_games_batch_sequential(batch_args)


def _play_game_worker_algo_vs_algo(
    args: tuple
) -> List[dict]:
    """Worker function for algo-vs-algo self-play with per-player difficulties.

    Uses Cython full game loop when available for 2-7x speedup.
    Falls back to Python play_single_game with return_dicts=True.
    """
    spec = _normalize_algo_task(args)
    if _HAS_FAST_GAME:
        result = play_full_game_cy(
            p1_difficulty=spec['p1_difficulty'],
            p2_difficulty=spec['p2_difficulty'],
            max_moves=spec['max_moves'], noise_prob=spec['noise_prob'],
            start_player=spec['start_player'],
            teacher_difficulty=spec['teacher_difficulty'],
            opening_plies=spec['opening_plies'],
            opening_seed=spec['opening_seed'],
            trajectory_source=spec['trajectory_source'],
            game_id=spec['game_id'],
        )
        score_game_dicts(
            entry_dicts=result['entries'], winner_int=result['winner'],
            total_moves=result['num_moves'], max_moves=spec['max_moves'],
            final_state_dict=result['final_state'],
            p1_captures=result['p1_captures'], p2_captures=result['p2_captures'],
        )
        return result['entries']
    return play_single_game(
        difficulty=spec['p1_difficulty'],
        max_moves=spec['max_moves'],
        noise_prob=spec['noise_prob'],
        start_player=spec['start_player'],
        p1_policy='algorithmic',
        p2_policy='algorithmic',
        p1_difficulty=spec['p1_difficulty'],
        p2_difficulty=spec['p2_difficulty'],
        return_dicts=True,
        teacher_difficulty=spec['teacher_difficulty'],
        opening_plies=spec['opening_plies'],
        opening_seed=spec['opening_seed'],
        trajectory_source=spec['trajectory_source'],
        game_id=spec['game_id'],
    )


def _play_games_batch_worker_algo(batch_args: list) -> List[dict]:
    """Batched worker for algo-vs-algo games with per-player difficulties.

    Uses Cython full game loop when available (per-player difficulties + TT).
    """
    all_entries = []
    if _HAS_FAST_GAME:
        for task in batch_args:
            spec = _normalize_algo_task(task)
            result = play_full_game_cy(
                p1_difficulty=spec['p1_difficulty'],
                p2_difficulty=spec['p2_difficulty'],
                max_moves=spec['max_moves'], noise_prob=spec['noise_prob'],
                start_player=spec['start_player'],
                teacher_difficulty=spec['teacher_difficulty'],
                opening_plies=spec['opening_plies'],
                opening_seed=spec['opening_seed'],
                trajectory_source=spec['trajectory_source'],
                game_id=spec['game_id'],
            )
            score_game_dicts(
                entry_dicts=result['entries'], winner_int=result['winner'],
                total_moves=result['num_moves'], max_moves=spec['max_moves'],
                final_state_dict=result['final_state'],
                p1_captures=result['p1_captures'], p2_captures=result['p2_captures'],
            )
            all_entries.extend(result['entries'])
    else:
        for task in batch_args:
            spec = _normalize_algo_task(task)
            entry_dicts = play_single_game(
                difficulty=spec['p1_difficulty'],
                max_moves=spec['max_moves'], noise_prob=spec['noise_prob'],
                start_player=spec['start_player'],
                p1_policy='algorithmic', p2_policy='algorithmic',
                p1_difficulty=spec['p1_difficulty'],
                p2_difficulty=spec['p2_difficulty'],
                return_dicts=True,
                teacher_difficulty=spec['teacher_difficulty'],
                opening_plies=spec['opening_plies'],
                opening_seed=spec['opening_seed'],
                trajectory_source=spec['trajectory_source'],
                game_id=spec['game_id'],
            )
            all_entries.extend(entry_dicts)
    return all_entries
