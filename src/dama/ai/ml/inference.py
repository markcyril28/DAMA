"""ML model inference for move selection."""

import warnings
import threading
from typing import Optional, List, Tuple
from pathlib import Path

import torch
import numpy as np

from ...types import Move
from ...game_state import GameState
from .model import MoveScorerNet, load_model, create_model
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


def get_model(model_path: str, device: Optional[torch.device] = None) -> MoveScorerNet:
    """
    Get a model, loading from cache if available.

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

    if cache_key not in _model_cache:
        with _model_cache_lock:
            if cache_key not in _model_cache:
                path = _resolve_model_path(model_path)
                if not path.exists():
                    raise FileNotFoundError(f"Model not found: {path}")

                model = load_model(str(path), device)
                _model_cache[cache_key] = model

    return _model_cache[cache_key]


def clear_model_cache() -> None:
    """Clear the model cache."""
    global _model_cache
    _model_cache.clear()


def get_ml_move(
    state: GameState,
    model_path: str = "models/latest.pt",
    device: Optional[torch.device] = None
) -> Optional[Move]:
    """
    Get the best move according to the ML model.

    Args:
        state: Current game state
        model_path: Path to the model checkpoint
        device: Device to run inference on

    Returns:
        The best move, or None if no legal moves
    """
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

    # Encode state and moves
    board_tensor = torch.from_numpy(encode_board(state)).unsqueeze(0)
    move_tensor = torch.from_numpy(encode_moves(state, moves))

    # Move to device (skip no-op .to() when already on CPU)
    if device is None or isinstance(device, str):
        device = next(model.parameters()).device
    if device.type != 'cpu':
        board_tensor = board_tensor.to(device)
        move_tensor = move_tensor.to(device)

    # inference_mode is faster than no_grad: also disables version counting
    # and autograd dispatch overhead (~5-10% faster for small forward passes)
    with torch.inference_mode():
        scores = model.score_single(board_tensor, move_tensor)

    # Select best move
    best_idx = scores.argmax().item()
    return moves[best_idx]


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
