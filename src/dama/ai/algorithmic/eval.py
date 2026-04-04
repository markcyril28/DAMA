"""Board evaluation for algorithmic AI."""

from typing import Dict, Optional

from ...types import Player, Piece, PieceType, Position
from ...game_state import GameState
from ...board import Board
from ...movegen import generate_all_moves


# Default evaluation weights
DEFAULT_WEIGHTS = {
    'man': 100,
    'king': 200,
    'mobility': 5,
    'advancement': 2,
    'center_control': 3,
    'back_rank': 10,
}

# Current active weights (can be customized)
WEIGHTS = DEFAULT_WEIGHTS.copy()


def set_custom_weights(custom_weights: Optional[Dict[str, int]]) -> None:
    """
    Set custom evaluation weights.
    
    Args:
        custom_weights: Dictionary of weight values, or None to reset to defaults.
    """
    global WEIGHTS
    if custom_weights is None:
        WEIGHTS = DEFAULT_WEIGHTS.copy()
    else:
        WEIGHTS = DEFAULT_WEIGHTS.copy()
        WEIGHTS.update(custom_weights)


def get_weights() -> Dict[str, int]:
    """Get the current evaluation weights."""
    return WEIGHTS.copy()

# Center squares (more valuable for control)
CENTER_SQUARES = {
    (3, 2), (3, 4), (3, 6),
    (4, 1), (4, 3), (4, 5), (4, 7),
}


def evaluate(state: GameState) -> float:
    """
    Evaluate a game state from the perspective of the current player.

    Returns a score where positive is good for current player,
    negative is good for opponent.
    """
    if state.is_terminal():
        winner = state.winner()
        if winner == state.current_player:
            return 10000  # Win
        elif winner is not None:
            return -10000  # Loss
        return 0  # Draw

    board = state.board
    current = state.current_player
    opponent = current.opponent()

    score = 0.0

    # Material evaluation
    score += _evaluate_material(board, current, opponent)

    # Mobility evaluation
    score += _evaluate_mobility(state, current, opponent)

    # Positional evaluation
    score += _evaluate_position(board, current, opponent)

    return score


def _evaluate_material(board: Board, current: Player, opponent: Player) -> float:
    """Evaluate material advantage."""
    score = 0.0

    # Count pieces
    current_men, current_kings = board.count_pieces(current)
    opponent_men, opponent_kings = board.count_pieces(opponent)

    # Material score
    score += (current_men - opponent_men) * WEIGHTS['man']
    score += (current_kings - opponent_kings) * WEIGHTS['king']

    return score


def _evaluate_mobility(state: GameState, current: Player, opponent: Player) -> float:
    """Evaluate mobility (number of legal moves)."""
    # Current player's moves
    current_moves = len(state.legal_moves())

    # Create opponent's state to count their moves
    # This is expensive, so we approximate
    opponent_state = GameState(state.board, opponent, state.move_count)
    opponent_moves = len(opponent_state.legal_moves())

    return (current_moves - opponent_moves) * WEIGHTS['mobility']


def _evaluate_position(board: Board, current: Player, opponent: Player) -> float:
    """Evaluate positional factors."""
    score = 0.0

    for pos, piece in board.get_pieces():
        multiplier = 1 if piece.player == current else -1
        row, col = pos

        # Advancement (for men only)
        if not piece.is_king:
            if piece.player == Player.ONE:
                # Player 1 advances toward row 7
                advancement = row
            else:
                # Player 2 advances toward row 0
                advancement = 7 - row

            score += advancement * WEIGHTS['advancement'] * multiplier

        # Center control
        if pos in CENTER_SQUARES:
            score += WEIGHTS['center_control'] * multiplier

        # Back rank defense (kings protecting back rank)
        if piece.is_king:
            if piece.player == Player.ONE and row == 0:
                score += WEIGHTS['back_rank'] * multiplier
            elif piece.player == Player.TWO and row == 7:
                score += WEIGHTS['back_rank'] * multiplier

    return score


def quick_evaluate(state: GameState) -> float:
    """
    Quick evaluation for move ordering.
    Only considers material, much faster than full evaluation.
    """
    board = state.board
    current = state.current_player
    opponent = current.opponent()

    current_men, current_kings = board.count_pieces(current)
    opponent_men, opponent_kings = board.count_pieces(opponent)

    score = (current_men - opponent_men) * WEIGHTS['man']
    score += (current_kings - opponent_kings) * WEIGHTS['king']

    return score
