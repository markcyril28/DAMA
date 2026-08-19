"""ML model inference for move selection."""

import time
import warnings
import threading
from typing import Optional, List, Tuple
from pathlib import Path

import torch
import numpy as np

from ...types import Move
from ...game_state import GameState
from .model import MoveScorerNet, load_model, create_model, fold_batchnorm
from .move_encoder import encode_board, encode_moves


# Global model cache
_model_cache: dict = {}
_model_cache_lock = threading.Lock()

# Project root directory (5 levels up from this file: inference.py -> ml -> ai -> dama -> src -> PROJECT_ROOT)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def _resolve_model_path(model_path: str) -> Path:
    """
    Resolve a model path to an absolute path.

    If the path is relative, it is resolved relative to the project root,
    not the current working directory.

    Args:
        model_path: Path to the model (absolute or relative)

    Returns:
        Absolute path to the model
    """
    path = Path(model_path)
    if path.is_absolute():
        return path
    # Resolve relative paths against project root
    return _PROJECT_ROOT / path


def _stat_signature(path: Path) -> Tuple[int, int]:
    """Return (st_mtime_ns, st_size) identifying the file's current contents."""
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


def get_model(model_path: str, device: Optional[torch.device] = None) -> MoveScorerNet:
    """
    Get a model, loading from cache if available.

    The cache is keyed on (model_path, device) and stores the file's
    mtime/size signature, so a checkpoint overwritten at the same path
    (e.g. models/latest.pt during training) is reloaded automatically.

    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on

    Returns:
        Loaded model ready for inference
    """
    global _model_cache

    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn(
                "CUDA not available, falling back to CPU inference. "
                "This may be slower. Install PyTorch with CUDA support for GPU acceleration."
            )
            device = torch.device('cpu')

    cache_key = (model_path, str(device))
    path = _resolve_model_path(model_path)

    try:
        signature = _stat_signature(path)
    except FileNotFoundError:
        # Checkpoints are written to a temp file then os.rename()'d into
        # place, so the file can be briefly missing. Serve the cached model
        # if one exists; otherwise retry the stat briefly before giving up.
        with _model_cache_lock:
            entry = _model_cache.get(cache_key)
        if entry is not None:
            return entry[0]
        signature = None
        for _ in range(5):
            time.sleep(0.05)
            try:
                signature = _stat_signature(path)
                break
            except FileNotFoundError:
                pass
        if signature is None:
            raise FileNotFoundError(f"Model not found: {path}")

    with _model_cache_lock:
        entry = _model_cache.get(cache_key)
    if entry is not None and entry[1] == signature:
        return entry[0]

    # Cache miss or stale checkpoint: load outside the lock so other
    # threads are not blocked behind torch.load.
    model = load_model(str(path), device)
    # Fold BatchNorm into Conv2d on CPU for ~10-20% faster
    # inference.  GPU inference benefits less (cuDNN already
    # fuses BN+Conv internally), so only fold on CPU.
    if device.type == 'cpu':
        try:
            fold_batchnorm(model)
        except Exception:
            pass  # Folding is optional; fall back to unfused

    with _model_cache_lock:
        entry = _model_cache.get(cache_key)
        if entry is not None and entry[1] == signature:
            # Another thread loaded the same checkpoint first; reuse it.
            return entry[0]
        # Replace any existing entry for this (path, device) pair in
        # place so the cache holds at most one model per key.
        _model_cache[cache_key] = (model, signature)

    return model


def clear_model_cache() -> None:
    """Clear the model cache."""
    global _model_cache
    with _model_cache_lock:
        _model_cache.clear()


def _validate_inference_depth(depth: int) -> int:
    """Validate and return the supported ML inference depth."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth not in (1, 2, 3):
        raise ValueError("ML inference depth must be one of 1, 2, or 3")
    return depth


def _inference_device(model: MoveScorerNet, device) -> torch.device:
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _score_policy_moves(
    state: GameState,
    moves: List[Move],
    model: MoveScorerNet,
    device: torch.device,
) -> torch.Tensor:
    """Return policy logits for ``moves`` without selecting an action."""
    board_tensor = torch.from_numpy(encode_board(state)).unsqueeze(0)
    move_tensor = torch.from_numpy(encode_moves(state, moves))
    if device.type != 'cpu':
        board_tensor = board_tensor.to(device)
        move_tensor = move_tensor.to(device)
    with torch.inference_mode():
        return model.score_single(board_tensor, move_tensor)


def _value_for_state(
    state: GameState,
    model: MoveScorerNet,
    device: torch.device,
) -> float:
    """Evaluate a state from the current player's perspective."""
    if getattr(model, 'value_head', None) is None:
        raise ValueError("ML inference depth 2 or 3 requires a model value head")
    board_tensor = torch.from_numpy(encode_board(state)).unsqueeze(0)
    if device.type != 'cpu':
        board_tensor = board_tensor.to(device)
    with torch.inference_mode():
        embedding = model.board_encoder(board_tensor)
        value = model.value_head(embedding)
    return float(value.reshape(-1)[0].item())


def _negamax_value(
    state: GameState,
    model: MoveScorerNet,
    device: torch.device,
    remaining_plies: int,
    alpha: float,
    beta: float,
) -> float:
    """Return shallow negamax value from the side-to-move perspective."""
    moves = state.legal_moves()
    if not moves:
        return -1.0
    if remaining_plies <= 0:
        return _value_for_state(state, model, device)

    best = float('-inf')
    for move in moves:
        child = state.apply_move(move)
        score = -_negamax_value(
            child,
            model,
            device,
            remaining_plies - 1,
            -beta,
            -alpha,
        )
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def _select_move_with_model(
    state: GameState,
    moves: List[Move],
    model: MoveScorerNet,
    device=None,
    depth: int = 1,
) -> Optional[Move]:
    """Select a move with policy argmax or value-head shallow negamax."""
    depth = _validate_inference_depth(depth)
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]

    resolved_device = _inference_device(model, device)
    if depth == 1:
        scores = _score_policy_moves(state, moves, model, resolved_device)
        return moves[int(scores.argmax().item())]

    if getattr(model, 'value_head', None) is None:
        raise ValueError("ML inference depth 2 or 3 requires a model value head")

    best_move = moves[0]
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    for move in moves:
        score = -_negamax_value(
            state.apply_move(move),
            model,
            resolved_device,
            depth - 1,
            -beta,
            -alpha,
        )
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
    return best_move


def get_ml_move(
    state: GameState,
    model_path: str = "models/latest.pt",
    device: Optional[torch.device] = None,
    depth: int = 1,
) -> Optional[Move]:
    """
    Get the best move according to the ML model.

    Args:
        state: Current game state
        model_path: Path to the model checkpoint
        device: Device to run inference on
        depth: ML inference depth. Depth 1 preserves policy argmax. Depths 2
            and 3 use shallow negamax and require a model value head.

    Returns:
        The best move, or None if no legal moves
    """
    depth = _validate_inference_depth(depth)
    moves = state.legal_moves()
    if not moves:
        return None

    if len(moves) == 1:
        return moves[0]

    try:
        model = get_model(model_path, device)
    except FileNotFoundError:
        resolved_path = _resolve_model_path(model_path)
        warnings.warn(f"Model not found at {resolved_path}")
        raise

    return _select_move_with_model(state, moves, model, device=device, depth=depth)


def get_ml_move_idx(
    state: GameState,
    legal_moves: List[Move],
    model_path: str = "models/latest.pt",
    device: Optional[torch.device] = None,
    depth: int = 1,
) -> Optional[int]:
    """Get index of the best move according to the ML model.

    Like get_ml_move() but accepts pre-computed legal_moves and returns
    the index directly, avoiding:
      1. Redundant state.legal_moves() call (~0.2ms)
      2. O(n) legal_moves.index(move) search
      3. Move object comparison overhead

    Args:
        state: Current game state
        legal_moves: Pre-computed legal moves for this position
        model_path: Path to the model checkpoint
        device: Device to run inference on
        depth: ML inference depth in the inclusive range 1 to 3.

    Returns:
        Index of the best move, or None if no legal moves
    """
    depth = _validate_inference_depth(depth)
    if not legal_moves:
        return None

    if len(legal_moves) == 1:
        return 0

    try:
        model = get_model(model_path, device)
    except FileNotFoundError:
        raise

    selected = _select_move_with_model(
        state,
        legal_moves,
        model,
        device=device,
        depth=depth,
    )
    return legal_moves.index(selected) if selected is not None else None


def get_move_scores(
    state: GameState,
    model_path: str = "models/latest.pt",
    device: Optional[torch.device] = None
) -> List[Tuple[Move, float]]:
    """
    Get scores for all legal moves.

    Args:
        state: Current game state
        model_path: Path to the model checkpoint
        device: Device to run inference on

    Returns:
        List of (move, score) tuples sorted by score descending
    """
    moves = state.legal_moves()
    if not moves:
        return []

    try:
        model = get_model(model_path, device)
    except FileNotFoundError:
        return [(m, 0.0) for m in moves]

    # Encode state and moves
    board_tensor = torch.from_numpy(encode_board(state)).unsqueeze(0)
    move_tensor = torch.from_numpy(encode_moves(state, moves))

    # Move to device (skip no-op .to() when already on CPU)
    if device is None or isinstance(device, str):
        device = next(model.parameters()).device
    if device.type != 'cpu':
        board_tensor = board_tensor.to(device)
        move_tensor = move_tensor.to(device)

    # inference_mode is faster than no_grad
    with torch.inference_mode():
        scores = model.score_single(board_tensor, move_tensor)

    # Convert to list
    scores_list = scores.cpu().numpy().tolist()
    move_scores = list(zip(moves, scores_list))

    # Sort by score descending
    move_scores.sort(key=lambda x: x[1], reverse=True)

    return move_scores


def create_dummy_model(save_path: str = "models/latest.pt") -> None:
    """
    Create and save a dummy model for testing.

    This creates a randomly initialized model that can be used
    before any training has been done.
    """
    from .model import create_model, save_model

    model = create_model()

    # Ensure directory exists
    path = _resolve_model_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_model(model, str(path), step=0, loss=float('inf'))
    print(f"Created dummy model at {path}")
