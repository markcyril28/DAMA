"""Dataset for training the move scorer model."""

import hashlib
import json
import gc
import time
import os
import random
import psutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ...types import Move, Player
from ...game_state import GameState
from ...board import Board
from .replay import ReplayBuffer, ReplayEntry
from .move_encoder import encode_board, encode_moves, MOVE_FEATURE_SIZE, BOARD_PLANES
from .scoring import compute_reward_weight, compute_reward_weights_batch

# Try to import Cython-accelerated encoding functions (~6-7x faster).
# Falls back to pure Python if the extension isn't built.
try:
    from ._fast_encode import (
        encode_board_fast_cy as _cy_encode_board,
        encode_moves_fast_cy as _cy_encode_moves,
        preprocess_chunk_cy as _cy_preprocess_chunk,
    )
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False

try:
    from ._fast_encode import preprocess_dicts_chunk_cy as _cy_preprocess_dicts_chunk
    _HAS_CYTHON_DICTS = True
except ImportError:
    _HAS_CYTHON_DICTS = False


def get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except Exception:
        return 0.0


def get_total_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        mem = psutil.virtual_memory()
        return mem.total / (1024 ** 3)
    except Exception:
        return 0.0


class DamaDataset(Dataset):
    """
    PyTorch dataset for training data.

    Each item returns:
    - board: (BOARD_PLANES, 8, 8) tensor
    - move_features: (num_moves, MOVE_FEATURE_SIZE) tensor
    - target: int (index of chosen move)
    - reward_weight: float (reward-based weight for loss)
    """

    def __init__(self, entries: List[ReplayEntry]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, float, float]:
        entry = self.entries[idx]

        # Reconstruct game state
        state = GameState.from_compact(entry.state)

        # Encode board
        board = encode_board(state)

        # Reconstruct moves and encode them
        moves = [Move.from_dict(m) for m in entry.legal_moves]
        move_features = encode_moves(state, moves)

        # Compute reward weight from score
        reward_weight = compute_reward_weight(entry.score)

        return (
            torch.from_numpy(board),
            torch.from_numpy(move_features),
            entry.chosen_index,
            reward_weight,
            float(entry.result),  # value target
        )


def _encode_board_fast(state_dict: dict, planes: np.ndarray) -> None:
    """Encode board state directly from compact dict into pre-allocated planes.

    Avoids creating Board, GameState, Piece objects — writes directly to numpy.
    ~2x faster than encode_board(GameState.from_compact(d)) for preprocessing.
    """
    turn = state_dict['turn']  # 1 or 2 (Player.ONE or Player.TWO)
    # Map piece lists to plane indices based on whose turn it is.
    # Current player's pieces go to planes 0 (men) and 1 (kings).
    # Opponent's pieces go to planes 2 (men) and 3 (kings).
    if turn == 1:
        mapping = (('p1_men', 0), ('p1_kings', 1), ('p2_men', 2), ('p2_kings', 3))
    else:
        mapping = (('p2_men', 0), ('p2_kings', 1), ('p1_men', 2), ('p1_kings', 3))

    flip = turn == 2

    planes[:] = 0.0
    for key, plane_idx in mapping:
        for pos in state_dict.get(key, ()):
            row = 7 - pos[0] if flip else pos[0]
            planes[plane_idx, row, pos[1]] = 1.0
    planes[4, :, :] = 1.0


def _encode_moves_fast(
    state_dict: dict,
    legal_moves: list,
    out: np.ndarray,
) -> int:
    """Encode moves directly from dicts into pre-allocated array.

    Avoids creating Move/Piece objects. Returns the number of valid moves encoded.
    """
    # Build a set of king positions for the current player to check piece type.
    turn = state_dict['turn']
    king_key = 'p1_kings' if turn == 1 else 'p2_kings'
    king_set = {(pos[0], pos[1]) for pos in state_dict.get(king_key, ())}
    flip = turn == 2

    n = min(len(legal_moves), out.shape[0])
    for i in range(n):
        m = legal_moves[i]
        path = m['path']
        captures = m.get('captures', ())
        promotion = m.get('promotion', False)
        start = path[0]
        end = path[-1]
        is_king = (start[0], start[1]) in king_set

        start_r = 7 - start[0] if flip else start[0]
        end_r = 7 - end[0] if flip else end[0]

        out[i, 0] = start_r / 7.0
        out[i, 1] = start[1] / 7.0
        out[i, 2] = end_r / 7.0
        out[i, 3] = end[1] / 7.0
        out[i, 4] = 1.0 if captures else 0.0
        num_captures = len(captures)
        out[i, 5] = min(num_captures / 4.0, 1.0)
        out[i, 6] = 1.0 if promotion else 0.0
        out[i, 7] = 1.0 if is_king else 0.0

    return n


def _entry_signature(entries: List[Any], max_samples: int = 64) -> str:
    """Build a compact fingerprint for a set of replay entries.

    Uses a sampled subset so cache validation is quick while still stable for
    ordered entry order changes across runs (replay append/sampling).
    """
    if not entries:
        return "empty"

    n = len(entries)
    if n <= max_samples:
        sample_indices = range(n)
    else:
        sample_indices = {
            int(round(i * (n - 1) / (max_samples - 1)))
            for i in range(max_samples)
        }
    h = hashlib.blake2b(digest_size=8)
    h.update(str(n).encode('utf-8'))

    for idx in sorted(sample_indices):
        entry = entries[idx]
        if isinstance(entry, dict):
            payload = entry
        else:
            payload = {
                'state': getattr(entry, 'state', None),
                'legal_moves_len': len(getattr(entry, 'legal_moves', [])),
                'chosen_index': getattr(entry, 'chosen_index', -1),
                'result': getattr(entry, 'result', 0),
                'score': float(getattr(entry, 'score', 0.0)),
            }
        # Sort keys for deterministic ordering; compact separators to reduce CPU cost.
        h.update(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'))

    return h.hexdigest()


def _preprocess_pool_init():
    """Per-worker initializer for the forked preprocessing pools below.

    [Pass 101] These pools fork from the trainer parent, which holds live CUDA
    tensors (training model, in-GPU replay buffer, optimizer state).  gc.freeze()
    moves every inherited object into a permanent generation the cyclic GC never
    scans, so a worker's automatic GC pass (tripped by object allocation in the
    encoders) can never sweep an inherited CUDA tensor whose destructor would
    call cudaSetDevice in a fork that never initialized CUDA
    (cudaErrorInitializationError -> std::terminate -> dead worker).  Mirrors the
    self-play fix in selfplay._selfplay_worker_init; GC stays active for the
    worker's OWN allocations (no leak).  Harmless no-op on the spawn path (a
    spawned worker inherits no CUDA state).
    """
    gc.freeze()


def _preprocess_chunk(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Worker function for parallel preprocessing of a chunk of replay entries.

    Accepts serialized (dict) entries to avoid pickling ReplayEntry objects.
    Uses Cython-accelerated encoding when available (~6-7x faster).
    Returns numpy arrays for the chunk.
    """
    entry_dicts, max_moves_per_sample = args
    n = len(entry_dicts)

    boards = np.zeros((n, BOARD_PLANES, 8, 8), dtype=np.float32)
    all_move_features = np.zeros((n, max_moves_per_sample, MOVE_FEATURE_SIZE), dtype=np.float32)
    move_counts = np.zeros(n, dtype=np.int32)
    targets = np.zeros(n, dtype=np.int32)
    reward_weights = np.zeros(n, dtype=np.float32)
    value_targets = np.zeros(n, dtype=np.float32)
    scores_arr = np.zeros(n, dtype=np.float32)

    if _HAS_CYTHON_DICTS:
        # Single Cython call for entire chunk — eliminates Python per-entry loop.
        _cy_preprocess_dicts_chunk(
            entry_dicts, 0, n, max_moves_per_sample,
            boards, all_move_features, move_counts, targets, scores_arr, value_targets,
        )
    else:
        _cy_board = _cy_encode_board if _HAS_CYTHON else None
        _cy_moves = _cy_encode_moves if _HAS_CYTHON else None

        for i, ed in enumerate(entry_dicts):
            state_dict = ed['state']
            if _cy_board is not None:
                _cy_board(state_dict, boards[i])
            else:
                _encode_board_fast(state_dict, boards[i])

            legal_moves = ed['legal_moves']
            if _cy_moves is not None:
                num_moves = _cy_moves(state_dict, legal_moves, all_move_features[i])
            else:
                num_moves = _encode_moves_fast(state_dict, legal_moves, all_move_features[i])

            move_counts[i] = num_moves
            chosen_idx = ed['chosen_index']
            if num_moves > 0:
                targets[i] = min(chosen_idx, num_moves - 1)
            else:
                targets[i] = 0
            scores_arr[i] = ed.get('score', 0.0)
            value_targets[i] = float(ed.get('result', 0))

    # Vectorized reward weight computation — single numpy call for the whole chunk.
    reward_weights[:] = compute_reward_weights_batch(scores_arr)

    return boards, all_move_features, move_counts, targets, reward_weights, value_targets


# ---------------------------------------------------------------------------
# Fork-optimized preprocessing: zero-serialization I/O
# ---------------------------------------------------------------------------
# On Linux with fork start method, child processes inherit the parent's
# memory via copy-on-write.  We store the entries list in a module global
# and send only (start, end) index ranges to workers — no to_dict()
# conversion, no pickle of entry data.  Workers read entries directly from
# the inherited list.
#
# Output uses multiprocessing.shared_memory: the parent pre-allocates
# SharedMemory blocks for all output arrays, workers write directly to
# their slices, and the parent wraps the result as numpy arrays with zero
# copy.  This eliminates:
#   - pickle serialization of output arrays (~120MB per chunk)
#   - np.concatenate across chunks (~2GB memcpy for 600K entries)
#   - peak 2× memory (worker arrays + concatenated arrays → 1× shared)
# ---------------------------------------------------------------------------
_fork_entries: Optional[list] = None
_fork_max_moves: int = 32

# Shared-memory output globals (set by parent before forking)
_fork_shm_names: Optional[dict] = None  # {'boards': name, 'move_features': name, ...}
_fork_total_n: int = 0  # total dataset size


def _preprocess_chunk_fork_shm(args: Tuple[int, int]) -> None:
    """Worker that reads from fork-inherited global and writes to shared memory.

    Zero input serialization (fork-inherited entries) AND zero output
    serialization (writes directly to SharedMemory).  Returns nothing —
    parent reads from the same shared memory after workers finish.
    """
    from multiprocessing.shared_memory import SharedMemory as _SHM

    start_idx, end_idx = args
    entries = _fork_entries
    max_moves_per_sample = _fork_max_moves
    n = end_idx - start_idx
    total_n = _fork_total_n
    names = _fork_shm_names

    # Attach to parent's shared memory blocks and create numpy views
    shm_boards = _SHM(name=names['boards'], create=False)
    shm_mf = _SHM(name=names['move_features'], create=False)
    shm_mc = _SHM(name=names['move_counts'], create=False)
    shm_tgt = _SHM(name=names['targets'], create=False)
    shm_rw = _SHM(name=names['reward_weights'], create=False)
    shm_vt = _SHM(name=names['value_targets'], create=False)

    try:
        boards = np.ndarray((total_n, BOARD_PLANES, 8, 8), dtype=np.float32, buffer=shm_boards.buf)
        all_mf = np.ndarray((total_n, max_moves_per_sample, MOVE_FEATURE_SIZE), dtype=np.float32, buffer=shm_mf.buf)
        move_counts = np.ndarray(total_n, dtype=np.int32, buffer=shm_mc.buf)
        targets = np.ndarray(total_n, dtype=np.int32, buffer=shm_tgt.buf)
        reward_weights = np.ndarray(total_n, dtype=np.float32, buffer=shm_rw.buf)
        value_targets = np.ndarray(total_n, dtype=np.float32, buffer=shm_vt.buf)

        # Slice views for this worker's chunk (writes go directly to shared memory)
        b_slice = boards[start_idx:end_idx]
        mf_slice = all_mf[start_idx:end_idx]
        mc_slice = move_counts[start_idx:end_idx]
        tgt_slice = targets[start_idx:end_idx]
        vt_slice = value_targets[start_idx:end_idx]

        scores_arr = np.zeros(n, dtype=np.float32)

        if _HAS_CYTHON:
            _cy_preprocess_chunk(
                entries, start_idx, end_idx, max_moves_per_sample,
                b_slice, mf_slice, mc_slice, tgt_slice, scores_arr, vt_slice,
            )
        else:
            for i in range(n):
                entry = entries[start_idx + i]
                state_dict = entry.state
                _encode_board_fast(state_dict, b_slice[i])

                num_moves = _encode_moves_fast(state_dict, entry.legal_moves, mf_slice[i])

                mc_slice[i] = num_moves
                chosen_idx = entry.chosen_index
                tgt_slice[i] = min(chosen_idx, num_moves - 1) if num_moves > 0 else 0
                scores_arr[i] = entry.score
                vt_slice[i] = float(entry.result)

        reward_weights[start_idx:end_idx] = compute_reward_weights_batch(scores_arr)
    finally:
        # Close (detach) shared memory handles — parent still owns them
        shm_boards.close()
        shm_mf.close()
        shm_mc.close()
        shm_tgt.close()
        shm_rw.close()
        shm_vt.close()


def _preprocess_chunk_fork(args: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy fork worker — returns arrays via pickle.

    Kept as fallback when SharedMemory is unavailable (e.g., older Python).
    """
    start_idx, end_idx = args
    entries = _fork_entries
    max_moves_per_sample = _fork_max_moves
    n = end_idx - start_idx

    boards = np.zeros((n, BOARD_PLANES, 8, 8), dtype=np.float32)
    all_move_features = np.zeros((n, max_moves_per_sample, MOVE_FEATURE_SIZE), dtype=np.float32)
    move_counts = np.zeros(n, dtype=np.int32)
    targets = np.zeros(n, dtype=np.int32)
    reward_weights = np.zeros(n, dtype=np.float32)
    value_targets = np.zeros(n, dtype=np.float32)
    scores_arr = np.zeros(n, dtype=np.float32)

    if _HAS_CYTHON:
        _cy_preprocess_chunk(
            entries, start_idx, end_idx, max_moves_per_sample,
            boards, all_move_features, move_counts, targets, scores_arr, value_targets,
        )
    else:
        for i in range(n):
            entry = entries[start_idx + i]
            state_dict = entry.state
            _encode_board_fast(state_dict, boards[i])

            num_moves = _encode_moves_fast(state_dict, entry.legal_moves, all_move_features[i])

            move_counts[i] = num_moves
            chosen_idx = entry.chosen_index
            if num_moves > 0:
                targets[i] = min(chosen_idx, num_moves - 1)
            else:
                targets[i] = 0
            scores_arr[i] = entry.score
            value_targets[i] = float(entry.result)

    reward_weights[:] = compute_reward_weights_batch(scores_arr)

    return boards, all_move_features, move_counts, targets, reward_weights, value_targets


def _preprocess_dicts_fork_shm(args: Tuple[int, int]) -> None:
    """Fork worker for dict entries with SharedMemory output.

    Same as _preprocess_chunk_fork_shm but reads dict entries (from
    _fork_entries stored as dicts) using key access instead of attribute
    access.  Used by CachedTensorDataset.from_dicts() on Linux.
    """
    from multiprocessing.shared_memory import SharedMemory as _SHM

    start_idx, end_idx = args
    entries = _fork_entries
    max_moves_per_sample = _fork_max_moves
    n = end_idx - start_idx
    total_n = _fork_total_n
    names = _fork_shm_names

    shm_boards = _SHM(name=names['boards'], create=False)
    shm_mf = _SHM(name=names['move_features'], create=False)
    shm_mc = _SHM(name=names['move_counts'], create=False)
    shm_tgt = _SHM(name=names['targets'], create=False)
    shm_rw = _SHM(name=names['reward_weights'], create=False)
    shm_vt = _SHM(name=names['value_targets'], create=False)

    try:
        boards = np.ndarray((total_n, BOARD_PLANES, 8, 8), dtype=np.float32, buffer=shm_boards.buf)
        all_mf = np.ndarray((total_n, max_moves_per_sample, MOVE_FEATURE_SIZE), dtype=np.float32, buffer=shm_mf.buf)
        move_counts = np.ndarray(total_n, dtype=np.int32, buffer=shm_mc.buf)
        targets = np.ndarray(total_n, dtype=np.int32, buffer=shm_tgt.buf)
        reward_weights = np.ndarray(total_n, dtype=np.float32, buffer=shm_rw.buf)
        value_targets = np.ndarray(total_n, dtype=np.float32, buffer=shm_vt.buf)

        b_slice = boards[start_idx:end_idx]
        mf_slice = all_mf[start_idx:end_idx]
        mc_slice = move_counts[start_idx:end_idx]
        tgt_slice = targets[start_idx:end_idx]
        vt_slice = value_targets[start_idx:end_idx]

        scores_arr = np.zeros(n, dtype=np.float32)

        if _HAS_CYTHON_DICTS:
            # Single Cython call — eliminates Python per-entry loop.
            _cy_preprocess_dicts_chunk(
                entries, start_idx, end_idx, max_moves_per_sample,
                b_slice, mf_slice, mc_slice, tgt_slice, scores_arr, vt_slice,
            )
        else:
            _cy_board = _cy_encode_board if _HAS_CYTHON else None
            _cy_moves = _cy_encode_moves if _HAS_CYTHON else None

            for i in range(n):
                ed = entries[start_idx + i]
                state_dict = ed['state']
                if _cy_board is not None:
                    _cy_board(state_dict, b_slice[i])
                else:
                    _encode_board_fast(state_dict, b_slice[i])
                if _cy_moves is not None:
                    num_moves = _cy_moves(state_dict, ed['legal_moves'], mf_slice[i])
                else:
                    num_moves = _encode_moves_fast(state_dict, ed['legal_moves'], mf_slice[i])
                mc_slice[i] = num_moves
                chosen_idx = ed['chosen_index']
                tgt_slice[i] = min(chosen_idx, num_moves - 1) if num_moves > 0 else 0
                scores_arr[i] = ed.get('score', 0.0)
                vt_slice[i] = float(ed.get('result', 0))

        reward_weights[start_idx:end_idx] = compute_reward_weights_batch(scores_arr)
    finally:
        # Close (detach) shared memory handles — parent still owns them
        shm_boards.close()
        shm_mf.close()
        shm_mc.close()
        shm_tgt.close()
        shm_rw.close()
        shm_vt.close()


def preprocess_entries_to_tensors(
    entries: List[ReplayEntry],
    max_moves_per_sample: int = 32,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pre-process replay entries into pre-computed tensors for fast training.

    Uses multiprocessing to parallelize across CPU cores when the dataset
    is large enough to amortize the IPC overhead.

    Args:
        entries: List of replay entries to process
        max_moves_per_sample: Maximum number of moves to pad to (for fixed-size batching)
        show_progress: Whether to print progress updates

    Returns:
        Tuple of:
        - boards: (N, BOARD_PLANES, 8, 8) float32 tensor
        - move_features: (N, max_moves, MOVE_FEATURE_SIZE) float32 tensor (padded)
        - move_counts: (N,) int32 tensor (actual number of moves per sample)
        - targets: (N,) int32 tensor (chosen move index)
        - reward_weights: (N,) float32 tensor (reward-based weights)
        - value_targets: (N,) float32 tensor (game result: +1, -1, 0)
    """
    n = len(entries)
    if n == 0:
        return (
            torch.empty(0, BOARD_PLANES, 8, 8),
            torch.empty(0, max_moves_per_sample, MOVE_FEATURE_SIZE),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
        )

    # Parallelize for large datasets where IPC cost is amortized.
    # Scale workers with core count, tiered caps to avoid IPC bottlenecks.
    # Fork+SharedMemory path has near-zero IPC: workers read from inherited
    # globals and write directly to shared memory — no pickle serialization
    # of input OR output. Higher worker counts pay off on high-core machines.
    _cores = os.cpu_count() or 1
    _worker_cap = 64 if _cores >= 128 else (48 if _cores >= 96 else (24 if _cores >= 48 else 16))
    num_workers = max(1, min(_worker_cap, _cores // 2))

    # Fork path (Linux) has near-zero serialization cost — lower threshold.
    # Spawn path (Windows/macOS) has pickle overhead — keep higher threshold.
    import multiprocessing as _mp
    _use_fork = False
    try:
        _use_fork = _mp.get_start_method() == 'fork'
    except RuntimeError:
        pass
    _parallel_threshold = 2000 if _use_fork else 5000
    use_parallel = n >= _parallel_threshold and num_workers > 1

    if use_parallel:
        # Ensure minimum chunk size (500) to avoid tiny chunks when n is
        # barely above threshold and num_workers is high.
        _MIN_CHUNK = 500
        chunk_size = max(_MIN_CHUNK, (n + num_workers - 1) // num_workers)
        # Reduce worker count if chunks are large enough to need fewer workers
        num_workers = min(num_workers, (n + chunk_size - 1) // chunk_size)

        if show_progress:
            print(f"  Pre-processing {n} entries with {num_workers} workers...")

        if _use_fork:
            # Fork path: zero input AND output serialization via SharedMemory.
            # Workers read entries from inherited global, write to pre-allocated
            # shared memory blocks.  No pickle, no concatenate.
            global _fork_entries, _fork_max_moves, _fork_shm_names, _fork_total_n
            _fork_entries = entries
            _fork_max_moves = max_moves_per_sample
            _fork_total_n = n

            args = [
                (start, min(start + chunk_size, n))
                for start in range(0, n, chunk_size)
            ]

            _shm_ok = True
            try:
                from multiprocessing.shared_memory import SharedMemory as _SHM
            except ImportError:
                _shm_ok = False
            _shm_tensors = False  # set True if shm path produces tensors directly

            if _shm_ok:
                # Pre-allocate shared memory for all output arrays
                _boards_sz = n * BOARD_PLANES * 8 * 8 * 4  # float32
                _mf_sz = n * max_moves_per_sample * MOVE_FEATURE_SIZE * 4
                _mc_sz = n * 4  # int32
                _tgt_sz = n * 4
                _rw_sz = n * 4  # float32
                _vt_sz = n * 4

                shm_list = []
                try:
                    # Append incrementally so partially-allocated segments are
                    # cleaned up by the finally block if a later allocation fails.
                    shm_boards = _SHM(create=True, size=max(1, _boards_sz)); shm_list.append(shm_boards)
                    shm_mf = _SHM(create=True, size=max(1, _mf_sz)); shm_list.append(shm_mf)
                    shm_mc = _SHM(create=True, size=max(1, _mc_sz)); shm_list.append(shm_mc)
                    shm_tgt = _SHM(create=True, size=max(1, _tgt_sz)); shm_list.append(shm_tgt)
                    shm_rw = _SHM(create=True, size=max(1, _rw_sz)); shm_list.append(shm_rw)
                    shm_vt = _SHM(create=True, size=max(1, _vt_sz)); shm_list.append(shm_vt)

                    _fork_shm_names = {
                        'boards': shm_boards.name,
                        'move_features': shm_mf.name,
                        'move_counts': shm_mc.name,
                        'targets': shm_tgt.name,
                        'reward_weights': shm_rw.name,
                        'value_targets': shm_vt.name,
                    }

                    # Workers write directly to shared memory — return nothing
                    with ProcessPoolExecutor(max_workers=num_workers,
                                     initializer=_preprocess_pool_init) as pool:
                        list(pool.map(_preprocess_chunk_fork_shm, args))

                    # Create torch tensors from shared memory views, then clone
                    # to own memory.  clone() is one copy (shm → tensor); the
                    # alternative (np.copy + from_numpy) would be two copies
                    # (shm → numpy copy → tensor share).
                    boards = torch.from_numpy(np.ndarray((n, BOARD_PLANES, 8, 8), dtype=np.float32, buffer=shm_boards.buf)).clone()
                    all_move_features = torch.from_numpy(np.ndarray((n, max_moves_per_sample, MOVE_FEATURE_SIZE), dtype=np.float32, buffer=shm_mf.buf)).clone()
                    move_counts = torch.from_numpy(np.ndarray(n, dtype=np.int32, buffer=shm_mc.buf)).clone()
                    targets = torch.from_numpy(np.ndarray(n, dtype=np.int32, buffer=shm_tgt.buf)).clone()
                    reward_weights = torch.from_numpy(np.ndarray(n, dtype=np.float32, buffer=shm_rw.buf)).clone()
                    value_targets = torch.from_numpy(np.ndarray(n, dtype=np.float32, buffer=shm_vt.buf)).clone()
                    # Mark as using shm path — skip from_numpy below
                    _shm_tensors = True

                except Exception as e:
                    # SharedMemory failed — fall back to legacy fork path
                    if show_progress:
                        print(f"  SharedMemory failed ({e}), using legacy fork path...")
                    _shm_ok = False
                finally:
                    _fork_shm_names = None
                    for shm in shm_list:
                        try:
                            shm.close()
                            shm.unlink()
                        except Exception:
                            pass

            if not _shm_ok:
                # Legacy fork path: workers return arrays via pickle
                with ProcessPoolExecutor(max_workers=num_workers,
                                     initializer=_preprocess_pool_init) as pool:
                    results = list(pool.map(_preprocess_chunk_fork, args))

                boards = np.concatenate([r[0] for r in results], axis=0)
                all_move_features = np.concatenate([r[1] for r in results], axis=0)
                move_counts = np.concatenate([r[2] for r in results])
                targets = np.concatenate([r[3] for r in results])
                reward_weights = np.concatenate([r[4] for r in results])
                value_targets = np.concatenate([r[5] for r in results])

            _fork_entries = None  # Release reference (runs for both shm and legacy paths)
            _fork_total_n = 0
        else:
            # Spawn path: serialize entries to dicts for pickling across processes.
            entry_dicts = [e.to_dict() for e in entries]

            chunks = []
            for start in range(0, n, chunk_size):
                chunk = entry_dicts[start:start + chunk_size]
                chunks.append((chunk, max_moves_per_sample))

            with ProcessPoolExecutor(max_workers=num_workers,
                                     initializer=_preprocess_pool_init) as pool:
                results = list(pool.map(_preprocess_chunk, chunks))

            boards = np.concatenate([r[0] for r in results], axis=0)
            all_move_features = np.concatenate([r[1] for r in results], axis=0)
            move_counts = np.concatenate([r[2] for r in results])
            targets = np.concatenate([r[3] for r in results])
            reward_weights = np.concatenate([r[4] for r in results])
            value_targets = np.concatenate([r[5] for r in results])

        if show_progress:
            print(f"  Pre-processing complete: {n} entries")

        # shm path already produced torch tensors; numpy paths need conversion
        if _use_fork and _shm_tensors:
            return (boards, all_move_features, move_counts, targets, reward_weights, value_targets)
        return (
            torch.from_numpy(boards),
            torch.from_numpy(all_move_features),
            torch.from_numpy(move_counts),
            torch.from_numpy(targets),
            torch.from_numpy(reward_weights),
            torch.from_numpy(value_targets),
        )

    # Sequential fallback for small datasets — uses fast-path encoding
    boards = np.zeros((n, BOARD_PLANES, 8, 8), dtype=np.float32)
    all_move_features = np.zeros((n, max_moves_per_sample, MOVE_FEATURE_SIZE), dtype=np.float32)
    move_counts = np.zeros(n, dtype=np.int32)
    targets = np.zeros(n, dtype=np.int32)
    scores_arr = np.zeros(n, dtype=np.float32)
    value_targets = np.zeros(n, dtype=np.float32)

    if _HAS_CYTHON and n > 0:
        if show_progress:
            print(f"  Pre-processing {n} entries (Cython fast path)...")
        _cy_preprocess_chunk(
            entries, 0, n, max_moves_per_sample,
            boards, all_move_features, move_counts, targets, scores_arr, value_targets,
        )
    else:
        log_interval = max(1, n // 20)  # Log every 5%
        for i, entry in enumerate(entries):
            if show_progress and i > 0 and i % log_interval == 0:
                print(f"  Pre-processing: {i}/{n} ({100*i/n:.1f}%)")

            _encode_board_fast(entry.state, boards[i])
            num_moves = _encode_moves_fast(entry.state, entry.legal_moves, all_move_features[i])

            move_counts[i] = num_moves
            if num_moves > 0:
                if entry.chosen_index >= num_moves:
                    print(f"  Warning: chosen_index {entry.chosen_index} out of range for {num_moves} moves (clipped)")
                targets[i] = min(entry.chosen_index, num_moves - 1)
            else:
                print(f"  Warning: entry {i} has zero legal moves — skipping target (move_counts=0 will mask this entry)")
                targets[i] = 0
            scores_arr[i] = entry.score
            value_targets[i] = float(entry.result)

    # Vectorized reward weight computation.
    reward_weights = compute_reward_weights_batch(scores_arr)

    if show_progress:
        print(f"  Pre-processing complete: {n} entries")

    return (
        torch.from_numpy(boards),
        torch.from_numpy(all_move_features),
        torch.from_numpy(move_counts),
        torch.from_numpy(targets),
        torch.from_numpy(reward_weights),
        torch.from_numpy(value_targets),
    )


class CachedTensorDataset(Dataset):
    """
    Dataset backed by pre-computed tensors in RAM.
    
    This is the fastest dataset implementation when you have sufficient RAM
    to hold all training data. Eliminates all CPU preprocessing during training.
    """
    
    def __init__(
        self,
        boards: torch.Tensor,
        move_features: torch.Tensor,
        move_counts: torch.Tensor,
        targets: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
        value_targets: Optional[torch.Tensor] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Args:
            boards: (N, BOARD_PLANES, 8, 8) tensor
            move_features: (N, max_moves, MOVE_FEATURE_SIZE) tensor
            move_counts: (N,) tensor with actual move counts
            targets: (N,) tensor with target indices
            reward_weights: (N,) tensor with reward-based weights (optional)
            value_targets: (N,) tensor with game results for value head (optional)
        """
        self.boards = boards
        self.move_features = move_features
        self.move_counts = move_counts
        self.targets = targets
        # Default to uniform weights if not provided (backward compat)
        if reward_weights is not None:
            self.reward_weights = reward_weights
        else:
            self.reward_weights = torch.ones(len(boards), dtype=torch.float32)
        # Default to zero value targets if not provided (backward compat)
        if value_targets is not None:
            self.value_targets = value_targets
        else:
            self.value_targets = torch.zeros(len(boards), dtype=torch.float32)
        self.metadata = dict(metadata or {})
    
    def __len__(self) -> int:
        return len(self.boards)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, float, float]:
        """
        Returns:
            - board: (BOARD_PLANES, 8, 8) tensor
            - move_features: (max_moves, MOVE_FEATURE_SIZE) tensor
            - move_count: int (actual number of valid moves)
            - target: int (chosen move index)
            - reward_weight: float (reward-based weight for loss)
            - value_target: float (game result for value head)
        """
        return (
            self.boards[idx],
            self.move_features[idx],
            int(self.move_counts[idx]),
            int(self.targets[idx]),
            float(self.reward_weights[idx]),
            float(self.value_targets[idx]),
        )
    
    @classmethod
    def from_entries(
        cls,
        entries: List[ReplayEntry],
        max_moves_per_sample: int = 32,
        show_progress: bool = True,
    ) -> 'CachedTensorDataset':
        """Create a CachedTensorDataset from replay entries."""
        boards, move_features, move_counts, targets, reward_weights, value_targets = preprocess_entries_to_tensors(
            entries, max_moves_per_sample, show_progress
        )
        return cls(boards, move_features, move_counts, targets, reward_weights, value_targets)

    @classmethod
    def from_dicts(
        cls,
        entry_dicts: List[dict],
        max_moves_per_sample: int = 32,
        show_progress: bool = True,
    ) -> 'CachedTensorDataset':
        """Create a CachedTensorDataset directly from raw dicts.

        Skips the ReplayEntry.from_dict() → to_dict() round-trip that
        from_entries() requires when the caller already has dicts (e.g.
        incremental self-play updates).  Uses _preprocess_chunk which
        natively accepts dicts.

        For large batches (>5000 dicts), parallelizes across CPU cores
        using ProcessPoolExecutor with chunk splitting.
        """
        n = len(entry_dicts)
        if n == 0:
            return cls(
                torch.empty(0, BOARD_PLANES, 8, 8),
                torch.empty(0, max_moves_per_sample, MOVE_FEATURE_SIZE),
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )

        # Check if fork start method is available for zero-copy input
        import multiprocessing as _mp
        _use_fork = False
        try:
            _use_fork = _mp.get_start_method() == 'fork'
        except RuntimeError:
            pass
        _parallel_threshold = 2000 if _use_fork else 5000

        if n >= _parallel_threshold:
            _cores = os.cpu_count() or 1
            _worker_cap = 64 if _cores >= 128 else (48 if _cores >= 96 else (24 if _cores >= 48 else 16))
            num_workers = max(1, min(_worker_cap, _cores // 2))
            _MIN_CHUNK = 500
            chunk_size = max(_MIN_CHUNK, (n + num_workers - 1) // num_workers)
            num_workers = min(num_workers, (n + chunk_size - 1) // chunk_size)

            if show_progress:
                print(f"  Pre-processing {n} dicts with {num_workers} workers"
                      f" ({'fork+shm' if _use_fork else 'spawn'})...")

            if _use_fork:
                # Fork path: zero input serialization + SharedMemory output
                global _fork_entries, _fork_max_moves, _fork_shm_names, _fork_total_n
                _fork_entries = entry_dicts
                _fork_max_moves = max_moves_per_sample
                _fork_total_n = n

                args = [
                    (start, min(start + chunk_size, n))
                    for start in range(0, n, chunk_size)
                ]

                _shm_ok = True
                try:
                    from multiprocessing.shared_memory import SharedMemory as _SHM
                except ImportError:
                    _shm_ok = False

                if _shm_ok:
                    _boards_sz = n * BOARD_PLANES * 8 * 8 * 4
                    _mf_sz = n * max_moves_per_sample * MOVE_FEATURE_SIZE * 4
                    _mc_sz = n * 4  # int32
                    _tgt_sz = n * 4
                    _rw_sz = n * 4
                    _vt_sz = n * 4

                    shm_list = []
                    try:
                        # Append incrementally so partially-allocated segments
                        # are cleaned up by the finally block on failure.
                        shm_boards = _SHM(create=True, size=max(1, _boards_sz)); shm_list.append(shm_boards)
                        shm_mf = _SHM(create=True, size=max(1, _mf_sz)); shm_list.append(shm_mf)
                        shm_mc = _SHM(create=True, size=max(1, _mc_sz)); shm_list.append(shm_mc)
                        shm_tgt = _SHM(create=True, size=max(1, _tgt_sz)); shm_list.append(shm_tgt)
                        shm_rw = _SHM(create=True, size=max(1, _rw_sz)); shm_list.append(shm_rw)
                        shm_vt = _SHM(create=True, size=max(1, _vt_sz)); shm_list.append(shm_vt)

                        _fork_shm_names = {
                            'boards': shm_boards.name,
                            'move_features': shm_mf.name,
                            'move_counts': shm_mc.name,
                            'targets': shm_tgt.name,
                            'reward_weights': shm_rw.name,
                            'value_targets': shm_vt.name,
                        }

                        with ProcessPoolExecutor(max_workers=num_workers,
                                     initializer=_preprocess_pool_init) as pool:
                            list(pool.map(_preprocess_dicts_fork_shm, args))

                        boards = torch.from_numpy(np.ndarray(
                            (n, BOARD_PLANES, 8, 8), dtype=np.float32,
                            buffer=shm_boards.buf)).clone()
                        mf = torch.from_numpy(np.ndarray(
                            (n, max_moves_per_sample, MOVE_FEATURE_SIZE),
                            dtype=np.float32, buffer=shm_mf.buf)).clone()
                        mc = torch.from_numpy(np.ndarray(
                            n, dtype=np.int32, buffer=shm_mc.buf)).clone()
                        tgt = torch.from_numpy(np.ndarray(
                            n, dtype=np.int32, buffer=shm_tgt.buf)).clone()
                        rw = torch.from_numpy(np.ndarray(
                            n, dtype=np.float32, buffer=shm_rw.buf)).clone()
                        vt = torch.from_numpy(np.ndarray(
                            n, dtype=np.float32, buffer=shm_vt.buf)).clone()

                        if show_progress:
                            print(f"  Pre-processing complete: {n} entries")
                        return cls(boards, mf, mc, tgt, rw, vt)

                    except Exception as e:
                        if show_progress:
                            print(f"  SharedMemory failed ({e}), using spawn path...")
                        _shm_ok = False
                    finally:
                        _fork_shm_names = None
                        _fork_entries = None
                        _fork_total_n = 0
                        for shm in shm_list:
                            try:
                                shm.close()
                                shm.unlink()
                            except Exception:
                                pass

                _fork_entries = None
                _fork_total_n = 0

            # Spawn path: pickle dict chunks across processes
            chunks = [
                (entry_dicts[i:i + chunk_size], max_moves_per_sample)
                for i in range(0, n, chunk_size)
            ]
            with ProcessPoolExecutor(max_workers=num_workers,
                                     initializer=_preprocess_pool_init) as pool:
                results = list(pool.map(_preprocess_chunk, chunks))

            boards = np.concatenate([r[0] for r in results], axis=0)
            mf = np.concatenate([r[1] for r in results], axis=0)
            mc = np.concatenate([r[2] for r in results])
            tgt = np.concatenate([r[3] for r in results])
            rw = np.concatenate([r[4] for r in results])
            vt = np.concatenate([r[5] for r in results])
        else:
            if show_progress:
                print(f"  Pre-processing {n} dicts (direct path)...")
            boards, mf, mc, tgt, rw, vt = _preprocess_chunk(
                (entry_dicts, max_moves_per_sample))

        if show_progress:
            print(f"  Pre-processing complete: {n} entries")
        return cls(
            torch.from_numpy(boards),
            torch.from_numpy(mf),
            torch.from_numpy(mc),
            torch.from_numpy(tgt),
            torch.from_numpy(rw),
            torch.from_numpy(vt),
        )
    
    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        """Save the cached tensors to a .pt file."""
        payload = {
            'boards': self.boards,
            'move_features': self.move_features,
            'move_counts': self.move_counts,
            'targets': self.targets,
            'reward_weights': self.reward_weights,
            'value_targets': self.value_targets,
            'metadata': {
                **self.metadata,
                **(metadata or {}),
                'created_at': time.time(),
            },
        }
        torch.save(payload, path)
        print(f"Saved cached dataset to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CachedTensorDataset':
        """Load cached tensors from a .pt file."""
        data = torch.load(path, weights_only=True)
        return cls(
            data['boards'],
            data['move_features'],
            data['move_counts'],
            data['targets'],
            data.get('reward_weights', None),  # Backward compat with old caches
            data.get('value_targets', None),   # Backward compat with old caches
            metadata=data.get('metadata', None),
        )

    def concat(self, other: 'CachedTensorDataset', max_entries: int = 0) -> 'CachedTensorDataset':
        """Concatenate another dataset onto this one, returning a new dataset.

        When max_entries > 0, keeps the most recent entries (tail) by trimming
        the oldest (head) entries from self.  New entries from ``other`` are
        always kept in full.

        This enables incremental dataset updates: preprocess only new self-play
        entries and concatenate with the existing preprocessed dataset instead of
        re-preprocessing the entire replay buffer.
        """
        # How many old entries to keep (trim oldest from self)
        if max_entries > 0 and len(self) + len(other) > max_entries:
            keep_old = max(0, max_entries - len(other))
            offset = len(self) - keep_old
            old_boards = self.boards[offset:]
            old_mf = self.move_features[offset:]
            old_mc = self.move_counts[offset:]
            old_t = self.targets[offset:]
            old_rw = self.reward_weights[offset:]
            old_vt = self.value_targets[offset:]
        else:
            old_boards = self.boards
            old_mf = self.move_features
            old_mc = self.move_counts
            old_t = self.targets
            old_rw = self.reward_weights
            old_vt = self.value_targets

        return CachedTensorDataset(
            boards=torch.cat([old_boards, other.boards], dim=0),
            move_features=torch.cat([old_mf, other.move_features], dim=0),
            move_counts=torch.cat([old_mc, other.move_counts], dim=0),
            targets=torch.cat([old_t, other.targets], dim=0),
            reward_weights=torch.cat([old_rw, other.reward_weights], dim=0),
            value_targets=torch.cat([old_vt, other.value_targets], dim=0),
        )


class CUDAPrefetcher:
    """Prefetches next batch to GPU in a separate CUDA stream.

    Overlaps H2D data transfer with GPU computation for the current batch,
    eliminating transfer latency from the critical path.
    """

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        self._iter = iter(self.dataloader)
        self._prefetch()
        return self

    def _prefetch(self):
        try:
            self._next_batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self._next_batch = tuple(
                t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in self._next_batch
            )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next_batch
        if batch is None:
            raise StopIteration
        for t in batch:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(torch.cuda.current_stream())
        self._prefetch()
        return batch

    def __len__(self):
        return len(self.dataloader)


class FastBatchIterator:
    """Direct tensor-indexing batch iterator for CachedTensorDataset.

    Bypasses DataLoader entirely — shuffles indices once per epoch and
    slices contiguous tensors with a single fancy-index per field.
    This eliminates per-sample __getitem__, collation, worker IPC, and
    Python type-conversion overhead that DataLoader imposes.

    **GPU-resident mode** (``device`` is a CUDA device): all dataset tensors
    are moved to GPU once at construction.  Batch indexing then happens on
    GPU with zero H2D transfer, no pin_memory, and no CUDAPrefetcher.
    Enabled automatically when the dataset fits comfortably in VRAM.

    **CPU+pin mode** (``pin_memory=True``, no ``device``): output tensors
    are pinned so that CUDAPrefetcher's ``non_blocking=True`` transfers
    are truly asynchronous.

    Yields a 6-tuple so the training loop works unchanged:
      (boards, move_features, move_counts, targets, reward_weights, value_targets)

    Exposes a `.dataset` attribute for compatibility with trainer code that
    checks ``isinstance(loader.dataset, CachedTensorDataset)``.
    """

    def __init__(
        self,
        dataset: CachedTensorDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = False,
        device: Optional[torch.device] = None,
        capacity: int = 0,
        amp_enabled: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = len(dataset)

        # GPU-resident mode: move entire dataset to VRAM once.
        # Eliminates all pin_memory + CUDAPrefetcher + H2D transfer overhead.
        self.on_gpu = False

        # When AMP is enabled, store boards and move_features as float16 on GPU.
        # Board values are {0.0, 1.0} and move features are in [0.0, 1.0] — all
        # exactly representable in float16.  AMP autocast handles float16→compute_dtype
        # at kernel entry, so no manual casting needed.  Halves VRAM for these tensors.
        _use_half = amp_enabled and device is not None and device.type == 'cuda'
        self._storage_dtype = torch.float16 if _use_half else torch.float32

        if device is not None and device.type == 'cuda':
            # Use capacity for VRAM budget check (pre-allocate for future updates).
            # Compute per-entry bytes using the actual storage dtype.
            _budget_n = max(self.n, capacity) if capacity > 0 else self.n
            _float_bytes = 2 if _use_half else 4
            if self.n > 0:
                _b_elems = dataset.boards[0].nelement()
                _mf_elems = dataset.move_features[0].nelement()
                _per_entry = (_b_elems + _mf_elems) * _float_bytes + (4 + 4 + 4 + 4)
            else:
                _per_entry = 1168 if _use_half else 2320
            budget_bytes = _budget_n * _per_entry

            total_vram = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            available = total_vram - allocated
            # Use GPU cache if dataset fits in <55% of remaining VRAM.
            # The model is small (~1MB params + ~2MB optimizer), so most VRAM
            # headroom is for activations + gradients which scale with batch_size.
            # 55% leaves ample room for batch_size up to 32K on 64GB GPUs.
            if budget_bytes < available * 0.55:
                alloc_n = _budget_n if capacity > 0 else self.n
                b_shape = dataset.boards.shape[1:]
                mf_shape = dataset.move_features.shape[1:]
                _sd = self._storage_dtype

                if alloc_n > self.n:
                    # Pre-allocate to capacity and fill the first n entries
                    self._boards = torch.empty(alloc_n, *b_shape, device=device,
                                               dtype=_sd).contiguous(
                                                   memory_format=torch.channels_last)
                    self._boards[:self.n] = dataset.boards.to(
                        device, dtype=_sd, memory_format=torch.channels_last)
                    self._move_features = torch.empty(alloc_n, *mf_shape, device=device,
                                                      dtype=_sd)
                    self._move_features[:self.n] = dataset.move_features.to(device, dtype=_sd)
                    self._move_counts = torch.empty(alloc_n, device=device,
                                                    dtype=dataset.move_counts.dtype)
                    self._move_counts[:self.n] = dataset.move_counts.to(device)
                    self._targets = torch.empty(alloc_n, device=device,
                                                dtype=dataset.targets.dtype)
                    self._targets[:self.n] = dataset.targets.to(device)
                    self._reward_weights = torch.empty(alloc_n, device=device,
                                                       dtype=dataset.reward_weights.dtype)
                    self._reward_weights[:self.n] = dataset.reward_weights.to(device)
                    self._value_targets = torch.empty(alloc_n, device=device,
                                                      dtype=dataset.value_targets.dtype)
                    self._value_targets[:self.n] = dataset.value_targets.to(device)
                    _dtype_label = "fp16" if _use_half else "fp32"
                    print(f"  GPU-resident dataset: {self.n} entries in "
                          f"{alloc_n}-capacity buffer "
                          f"({budget_bytes / 1e6:.0f}MB reserved, "
                          f"{available / 1e6:.0f}MB available, {_dtype_label}, channels_last)")
                else:
                    self._boards = dataset.boards.to(device, dtype=_sd,
                                                     memory_format=torch.channels_last)
                    self._move_features = dataset.move_features.to(device, dtype=_sd)
                    self._move_counts = dataset.move_counts.to(device)
                    self._targets = dataset.targets.to(device)
                    self._reward_weights = dataset.reward_weights.to(device)
                    self._value_targets = dataset.value_targets.to(device)
                    dataset_bytes = self.n * _per_entry
                    _dtype_label = "fp16" if _use_half else "fp32"
                    print(f"  GPU-resident dataset: {dataset_bytes / 1e6:.0f}MB on GPU "
                          f"({available / 1e6:.0f}MB available, {_dtype_label}, channels_last)")
                self.on_gpu = True
                self._device = device
                # Cache the pre-shuffle VRAM decision so __iter__ doesn't re-check
                # VRAM every epoch.  Invalidated by update_data() when buffer grows.
                self._can_preshuffle = self._check_preshuffle_budget()
            else:
                print(f"  Dataset too large for GPU cache ({budget_bytes / 1e6:.0f}MB, "
                      f"{available / 1e6:.0f}MB available) — using CPU+pin")

        if not self.on_gpu:
            # Pin memory once at construction — eliminates per-batch pin_memory()
            # overhead (~5μs/tensor/batch).  Pinned pages enable truly asynchronous
            # H2D transfers via CUDAPrefetcher's non_blocking=True.
            _should_pin = pin_memory and torch.cuda.is_available()
            if _should_pin:
                self._boards = dataset.boards.pin_memory()
                self._move_features = dataset.move_features.pin_memory()
                self._move_counts = dataset.move_counts.pin_memory()
                self._targets = dataset.targets.pin_memory()
                self._reward_weights = dataset.reward_weights.pin_memory()
                self._value_targets = dataset.value_targets.pin_memory()
            else:
                self._boards = dataset.boards
                self._move_features = dataset.move_features
                self._move_counts = dataset.move_counts
                self._targets = dataset.targets
                self._reward_weights = dataset.reward_weights
                self._value_targets = dataset.value_targets
            self._device = None
            self._can_preshuffle = False  # CPU data — no GPU pre-shuffle

        # Already pinned at construction (or GPU-resident) — no per-batch pinning needed
        self.pin_memory = False

    def _check_preshuffle_budget(self) -> bool:
        """Check if VRAM allows epoch-start pre-shuffle (gather all tensors).

        Pre-shuffle creates a full copy of the active data (self.n entries).
        When the buffer has pre-allocated capacity > n, the tensors are larger
        than the active region.  Use per-entry element counts * self.n to
        estimate the actual copy size, not t.nelement() which includes slack.
        """
        if not self.on_gpu or self.n == 0:
            return False
        _dev = self._boards.device
        # Per-entry bytes for the active region (self.n entries, not full capacity)
        _per_entry_bytes = sum(
            (t.nelement() // max(1, t.shape[0])) * t.element_size()
            for t in (self._boards, self._move_features, self._move_counts,
                      self._targets, self._reward_weights, self._value_targets)
        )
        _data_bytes = self.n * _per_entry_bytes
        _total_vram = torch.cuda.get_device_properties(_dev).total_memory
        _allocated = torch.cuda.memory_allocated(_dev)
        _free = _total_vram - _allocated
        return _data_bytes < _free * 0.4

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Generate indices on the same device as data — GPU randperm is faster
        _dev = self._boards.device

        if self.shuffle and self._can_preshuffle:
            # Pre-shuffle: gather all data once per epoch with a random
            # permutation, then iterate with contiguous slices (zero-copy views).
            # Trades 6 large gather ops at epoch start for ~1464 per-batch
            # fancy-index kernel launches (batch_size=4096, 1M entries).
            # Contiguous slicing is a view — no kernel launch, no data copy.
            # _can_preshuffle is cached at init / update_data() to avoid
            # re-checking VRAM every epoch.
            b = mf = mc = tgt = rw = vt = None
            try:
                perm = torch.randperm(self.n, device=_dev)
                # Fancy indexing may not preserve channels_last memory format.
                # Explicit conversion here (1× per epoch) avoids the model's
                # forward pass converting every batch (61× per epoch).
                b = self._boards[perm].contiguous(memory_format=torch.channels_last)
                mf = self._move_features[perm]
                mc = self._move_counts[perm]
                tgt = self._targets[perm]
                rw = self._reward_weights[perm]
                vt = self._value_targets[perm]
                del perm

                for start in range(0, self.n, self.batch_size):
                    end = min(start + self.batch_size, self.n)
                    if self.drop_last and (end - start) < self.batch_size:
                        break
                    yield (b[start:end], mf[start:end], mc[start:end],
                           tgt[start:end], rw[start:end], vt[start:end])
            finally:
                # Free shuffled copies immediately — they can be ~1.2GB on GPU.
                # Without explicit del, generator frame locals persist until GC.
                # Also covers OOM during gather: partially-allocated tensors freed.
                del b, mf, mc, tgt, rw, vt
            return

        # Fallback: CPU data, non-shuffle, or low VRAM
        if self.shuffle:
            # Shuffle requires fancy indexing (gather kernel per batch)
            indices = torch.randperm(self.n, device=_dev)
            for start in range(0, self.n, self.batch_size):
                end = min(start + self.batch_size, self.n)
                if self.drop_last and (end - start) < self.batch_size:
                    break
                idx = indices[start:end]
                yield (
                    self._boards[idx].contiguous(memory_format=torch.channels_last),
                    self._move_features[idx],
                    self._move_counts[idx],
                    self._targets[idx],
                    self._reward_weights[idx],
                    self._value_targets[idx],
                )
        else:
            # Non-shuffle: direct slicing produces views (zero-copy, no kernel launch).
            # Avoids torch.arange allocation + per-batch gather that the old path used.
            for start in range(0, self.n, self.batch_size):
                end = min(start + self.batch_size, self.n)
                if self.drop_last and (end - start) < self.batch_size:
                    break
                yield (
                    self._boards[start:end].contiguous(memory_format=torch.channels_last),
                    self._move_features[start:end],
                    self._move_counts[start:end],
                    self._targets[start:end],
                    self._reward_weights[start:end],
                    self._value_targets[start:end],
                )

    def update_data(
        self,
        new_dataset: CachedTensorDataset,
        max_entries: int = 0,
    ) -> None:
        """Update buffers in-place with new data, avoiding full GPU re-upload.

        Keeps the most recent ``max_entries`` entries by trimming the oldest
        from the existing buffer and appending all entries from *new_dataset*.
        When GPU-resident, only the new entries are transferred via PCIe —
        existing GPU data is shifted in-place (GPU→GPU memcpy, ~10× faster
        than CPU→GPU upload for the same size).

        Falls back to full replacement when:
        - Not GPU-resident (CPU pinned path — rebuild is cheap anyway)
        - Buffer capacity is insufficient and reallocation is needed
        """
        new_n = len(new_dataset)
        if new_n == 0:
            return

        if max_entries > 0 and self.n + new_n > max_entries:
            keep_old = max(0, max_entries - new_n)
        else:
            keep_old = self.n

        total = keep_old + new_n
        offset = self.n - keep_old  # how many old entries to trim from head

        if self.on_gpu:
            dev = self._device
            buf_cap = self._boards.shape[0]

            if total <= buf_cap:
                # Buffer is large enough — shift old data left, append new.
                # GPU→GPU copy: ~10× faster than equivalent CPU→GPU transfer.
                if offset > 0 and keep_old > 0:
                    # Left-shift by `offset`: copy [offset:offset+keep_old] → [0:keep_old].
                    # Source and destination overlap — previous approach used .clone()
                    # which allocated ~1.8GB temporary on GPU.  Instead, copy in
                    # non-overlapping chunks of size `offset` (left-to-right memmove):
                    #   chunk 0: [offset:2*offset] → [0:offset]        (disjoint)
                    #   chunk 1: [2*offset:3*offset] → [offset:2*offset] (disjoint)
                    #   ...
                    # Zero temporary allocation, ~19 small GPU memcpy ops.
                    _tensors = [self._boards, self._move_features, self._move_counts,
                                self._targets, self._reward_weights, self._value_targets]
                    for dst_start in range(0, keep_old, offset):
                        dst_end = min(dst_start + offset, keep_old)
                        src_start = dst_start + offset
                        src_end = src_start + (dst_end - dst_start)
                        for t in _tensors:
                            t[dst_start:dst_end] = t[src_start:src_end]

                # Upload only new entries (small PCIe transfer).
                # Match storage dtype (float16 when AMP) for consistency.
                _sd = self._storage_dtype
                self._boards[keep_old:total] = new_dataset.boards.to(
                    dev, dtype=_sd, memory_format=torch.channels_last, non_blocking=True)
                self._move_features[keep_old:total] = new_dataset.move_features.to(
                    dev, dtype=_sd, non_blocking=True)
                self._move_counts[keep_old:total] = new_dataset.move_counts.to(
                    dev, non_blocking=True)
                self._targets[keep_old:total] = new_dataset.targets.to(
                    dev, non_blocking=True)
                self._reward_weights[keep_old:total] = new_dataset.reward_weights.to(
                    dev, non_blocking=True)
                self._value_targets[keep_old:total] = new_dataset.value_targets.to(
                    dev, non_blocking=True)
                # No explicit synchronize() needed: all non_blocking transfers are
                # enqueued on the default stream.  The next GPU operation (training
                # forward pass) is also on the default stream and will automatically
                # wait for these transfers to complete.  The CPU-side metadata
                # updates below (self.n, print) only access tensor shapes/dtypes
                # which don't require the data transfer to finish.
                self.n = total
                self.drop_last = total > self.batch_size
                _kb_per = sum(t.element_size() * (t.nelement() // max(1, buf_cap))
                              for t in [self._boards, self._move_features, self._move_counts,
                                        self._targets, self._reward_weights, self._value_targets]
                              ) / 1024 if buf_cap > 0 else 1.9
                print(f"  GPU buffer updated in-place: kept {keep_old} old + "
                      f"{new_n} new = {total} entries "
                      f"(uploaded {new_n * _kb_per / 1024:.1f}MB, "
                      f"avoided {keep_old * _kb_per / 1024:.1f}MB re-upload)")
                self._can_preshuffle = self._check_preshuffle_budget()
                return

            # Buffer too small — must reallocate.  Allocate with slack to
            # reduce future reallocations.
            new_cap = max(total, int(max_entries * 1.1)) if max_entries > 0 else total
            print(f"  GPU buffer realloc: {buf_cap} → {new_cap} capacity")
            b_shape = self._boards.shape[1:]
            mf_shape = self._move_features.shape[1:]
            new_boards = torch.empty(new_cap, *b_shape, device=dev,
                                     dtype=self._boards.dtype).contiguous(
                                         memory_format=torch.channels_last)
            new_mf = torch.empty(new_cap, *mf_shape, device=dev,
                                 dtype=self._move_features.dtype)
            new_mc = torch.empty(new_cap, device=dev, dtype=self._move_counts.dtype)
            new_tgt = torch.empty(new_cap, device=dev, dtype=self._targets.dtype)
            new_rw = torch.empty(new_cap, device=dev, dtype=self._reward_weights.dtype)
            new_vt = torch.empty(new_cap, device=dev, dtype=self._value_targets.dtype)

            # Copy old data (GPU→GPU, fast)
            if keep_old > 0:
                new_boards[:keep_old] = self._boards[offset:offset + keep_old]
                new_mf[:keep_old] = self._move_features[offset:offset + keep_old]
                new_mc[:keep_old] = self._move_counts[offset:offset + keep_old]
                new_tgt[:keep_old] = self._targets[offset:offset + keep_old]
                new_rw[:keep_old] = self._reward_weights[offset:offset + keep_old]
                new_vt[:keep_old] = self._value_targets[offset:offset + keep_old]

            # Upload new entries (small PCIe transfer, match storage dtype)
            _sd = self._storage_dtype
            new_boards[keep_old:total] = new_dataset.boards.to(
                dev, dtype=_sd, memory_format=torch.channels_last, non_blocking=True)
            new_mf[keep_old:total] = new_dataset.move_features.to(dev, dtype=_sd,
                                                                   non_blocking=True)
            new_mc[keep_old:total] = new_dataset.move_counts.to(dev, non_blocking=True)
            new_tgt[keep_old:total] = new_dataset.targets.to(dev, non_blocking=True)
            new_rw[keep_old:total] = new_dataset.reward_weights.to(dev, non_blocking=True)
            new_vt[keep_old:total] = new_dataset.value_targets.to(dev, non_blocking=True)
            # No explicit sync — same-stream ordering guarantees (see in-place path).

            # Swap buffers — old ones freed by refcount
            self._boards = new_boards
            self._move_features = new_mf
            self._move_counts = new_mc
            self._targets = new_tgt
            self._reward_weights = new_rw
            self._value_targets = new_vt
            self.n = total
            self.drop_last = total > self.batch_size
            self._can_preshuffle = self._check_preshuffle_budget()
            return

        # CPU path: rebuild from scratch (pinning is fast, no PCIe bottleneck)
        merged = self.dataset.concat(new_dataset, max_entries=max_entries)
        self.dataset = merged
        self.n = len(merged)
        _should_pin = torch.cuda.is_available()
        if _should_pin:
            self._boards = merged.boards.pin_memory()
            self._move_features = merged.move_features.pin_memory()
            self._move_counts = merged.move_counts.pin_memory()
            self._targets = merged.targets.pin_memory()
            self._reward_weights = merged.reward_weights.pin_memory()
            self._value_targets = merged.value_targets.pin_memory()
        else:
            self._boards = merged.boards
            self._move_features = merged.move_features
            self._move_counts = merged.move_counts
            self._targets = merged.targets
            self._reward_weights = merged.reward_weights
            self._value_targets = merged.value_targets
        self.drop_last = self.n > self.batch_size


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, float, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.

    Returns:
    - boards: (batch_size, BOARD_PLANES, 8, 8)
    - all_move_features: (total_moves, MOVE_FEATURE_SIZE)
    - move_counts: (batch_size,) - number of moves per sample
    - targets: (batch_size,) - index of chosen move for each sample
    - reward_weights: (batch_size,) - reward-based weights for loss
    - value_targets: (batch_size,) - game result for value head
    """
    boards = []
    all_move_features = []
    move_counts = []
    targets = []
    reward_weights = []
    value_targets = []

    for board, move_features, target, rw, vt in batch:
        boards.append(board)
        all_move_features.append(move_features)
        move_counts.append(move_features.shape[0])
        targets.append(target)
        reward_weights.append(rw)
        value_targets.append(vt)

    return (
        torch.stack(boards),
        torch.cat(all_move_features, dim=0).contiguous(),
        torch.tensor(move_counts, dtype=torch.int32),
        torch.tensor(targets, dtype=torch.int32),
        torch.tensor(reward_weights, dtype=torch.float32),
        torch.tensor(value_targets, dtype=torch.float32),
    )


def create_dataloader(
    entries: List[ReplayEntry],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_ram_cache: bool = True,
    ram_threshold_gb: float = 16.0,
    cache_file: Optional[str] = None,
    device: Optional[torch.device] = None,
    capacity: int = 0,
    max_moves_per_sample: int = 32,
    amp_enabled: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from replay entries.

    Args:
        entries: List of replay entries
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        use_ram_cache: If True and sufficient RAM available, pre-process to tensors
        ram_threshold_gb: Minimum available RAM (GB) required for caching
        device: Target device — when CUDA, tries GPU-resident caching for zero H2D overhead

    Returns:
        DataLoader instance
    """
    available_ram = get_available_ram_gb()
    total_ram = get_total_ram_gb()
    
    # Estimate memory needed for cached dataset (RAM: always float32; GPU may use float16)
    # Each entry: ~5*8*8*B (board) + M*8*B (moves) + 16 (counts/targets/weights)
    # B=4 (fp32): M=32→~1.9KB. B=2 (fp16 on GPU w/ AMP): M=32→~1.2KB
    _per_entry_kb = (5 * 8 * 8 * 4 + max_moves_per_sample * 8 * 4 + 16) / 1024
    estimated_size_gb = len(entries) * _per_entry_kb / (1024 ** 2)
    
    # Use RAM caching if:
    # 1. Explicitly enabled
    # 2. Sufficient RAM available (with safety margin)
    # 3. Total RAM is high (e.g., server with 1TB RAM)
    should_cache = (
        use_ram_cache 
        and available_ram > ram_threshold_gb 
        and (available_ram > estimated_size_gb * 2 or total_ram > 64)
    )
    
    if should_cache:
        print(f"RAM Caching enabled ({available_ram:.1f}GB available, {estimated_size_gb:.2f}GB needed)")
        print("Pre-processing entries to tensors...")

        cache_meta = {
            'cache_version': 1,
            'entry_count': len(entries),
            'entry_signature': _entry_signature(entries),
            'max_moves_per_sample': max_moves_per_sample,
            'ram_threshold_gb': ram_threshold_gb,
        }

        cache_path = Path(cache_file).expanduser() if cache_file else None
        if cache_path is not None:
            if cache_path.exists():
                print(f"RAM cache file found: {cache_path}")
                try:
                    cached_dataset = CachedTensorDataset.load(str(cache_path))
                    cached_meta = cached_dataset.metadata
                    if (
                        cached_meta.get('entry_count') == cache_meta['entry_count']
                        and cached_meta.get('entry_signature') == cache_meta['entry_signature']
                        and cached_meta.get('max_moves_per_sample') == cache_meta['max_moves_per_sample']
                    ):
                        print("Loaded matching RAM cache from file.")
                        return FastBatchIterator(
                            cached_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=len(cached_dataset) > batch_size,
                            pin_memory=pin_memory,
                            device=device,
                            capacity=capacity,
                            amp_enabled=amp_enabled,
                        )
                    print("RAM cache file metadata mismatch; rebuilding cache.")
                except Exception as e:
                    print(f"RAM cache file invalid ({e}); rebuilding cache.")

            print(f"RAM cache enabled; saving computed dataset to {cache_path}")
        cached_dataset = CachedTensorDataset.from_entries(
            entries,
            max_moves_per_sample=max_moves_per_sample,
            show_progress=True
        )
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cached_dataset.save(str(cache_path), metadata=cache_meta)
            except Exception as e:
                print(f"Warning: could not save RAM cache file {cache_path}: {e}")

        # FastBatchIterator: bypass DataLoader entirely for pre-tensorized data.
        # Direct tensor[indices] is orders of magnitude faster than per-sample
        # __getitem__ + collation + worker IPC that DataLoader imposes.
        # When device is CUDA, tries GPU-resident caching for zero H2D overhead.
        # Only drop incomplete last batch when there are at least 2 full batches;
        # otherwise the single partial batch would be dropped → 0 batches.
        _drop = len(cached_dataset) > batch_size
        return FastBatchIterator(
            cached_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=_drop,
            pin_memory=pin_memory,
            device=device,
            capacity=capacity,
            amp_enabled=amp_enabled,
        )

    # Fall back to standard dataset
    print(f"Using standard dataset (available RAM: {available_ram:.1f}GB)")
    dataset = DamaDataset(entries)
    
    # persistent_workers can cause hangs - only use when we have enough batches
    # to make it worthwhile
    use_persistent = num_workers > 0 and len(entries) > batch_size * 100
    
    # Scale prefetch factor based on batch size for better GPU utilization
    # Larger batches benefit from more prefetching
    if num_workers > 0:
        if batch_size >= 4096:
            prefetch = 4  # More prefetching for large batches
        elif batch_size >= 1024:
            prefetch = 3
        else:
            prefetch = 2
    else:
        prefetch = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory and num_workers > 0,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
        drop_last=len(entries) > batch_size,
    )


def create_dataloader_from_dataset(
    dataset: CachedTensorDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
    capacity: int = 0,
    amp_enabled: bool = False,
) -> FastBatchIterator:
    """Create a fast batch iterator from a pre-built CachedTensorDataset.

    Uses direct tensor indexing instead of DataLoader for maximum throughput.
    Used when dataset has been pre-processed in a background thread
    to avoid blocking the training loop.

    When device is CUDA, tries GPU-resident caching for zero H2D overhead.
    When capacity > 0, pre-allocates GPU buffers to that size so future
    update_data() calls avoid reallocation.
    """
    return FastBatchIterator(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=len(dataset) > batch_size,
        pin_memory=pin_memory,
        device=device,
        capacity=capacity,
        amp_enabled=amp_enabled,
    )


def prepare_training_data(
    replay_buffer: ReplayBuffer,
    max_entries: int = 100000,
    val_split: float = 0.1,
) -> Tuple[List[ReplayEntry], List[ReplayEntry]]:
    """
    Prepare training and validation data from replay buffer.

    Args:
        replay_buffer: Source of training data
        max_entries: Maximum entries to use
        val_split: Fraction for validation

    Returns:
        (train_entries, val_entries)
    """
    # Collect entries
    entries = replay_buffer.sample_entries(max_entries)

    if not entries:
        return [], []

    # Shuffle
    random.shuffle(entries)

    # Split
    val_size = int(len(entries) * val_split)
    val_entries = entries[:val_size]
    train_entries = entries[val_size:]

    return train_entries, val_entries
