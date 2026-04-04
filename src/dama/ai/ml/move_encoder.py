"""Move encoding for ML model."""

from typing import List
import numpy as np

from ...types import Move, Player, Position, Piece, PieceType
from ...game_state import GameState
from ...board import Board


def encode_board(state: GameState) -> np.ndarray:
    """
    Encode board state as a tensor for the neural network.

    Returns:
        numpy array of shape (5, 8, 8):
        - Plane 0: Current player's men
        - Plane 1: Current player's kings
        - Plane 2: Opponent's men
        - Plane 3: Opponent's kings
        - Plane 4: All ones (side to move indicator)
    """
    board = state.board
    current = state.current_player
    opponent = current.opponent()

    # Initialize planes
    planes = np.zeros((5, 8, 8), dtype=np.float32)

    flip = current == Player.TWO

    for pos, piece in board.get_pieces():
        row, col = pos
        if flip:
            row = 7 - row

        if piece.player == current:
            if piece.is_king:
                planes[1, row, col] = 1.0
            else:
                planes[0, row, col] = 1.0
        else:
            if piece.is_king:
                planes[3, row, col] = 1.0
            else:
                planes[2, row, col] = 1.0

    # Side to move indicator (all ones)
    planes[4, :, :] = 1.0

    return planes


def encode_move(move: Move, piece: Piece, current_player: Player = Player.ONE) -> np.ndarray:
    """
    Encode a move as a feature vector.

    When current_player is Player.TWO, rows are flipped so both sides
    see the board from the same canonical orientation (pieces at top,
    advancing downward).

    Returns:
        numpy array of shape (8,):
        - 0: from_row (normalized)
        - 1: from_col (normalized)
        - 2: to_row (normalized)
        - 3: to_col (normalized)
        - 4: is_capture (0 or 1)
        - 5: num_captures (normalized)
        - 6: promotion (0 or 1)
        - 7: piece_is_king (0 or 1)
    """
    from_row, from_col = move.start
    to_row, to_col = move.end

    if current_player == Player.TWO:
        from_row = 7 - from_row
        to_row = 7 - to_row

    features = np.array([
        from_row / 7.0,           # Normalize to [0, 1]
        from_col / 7.0,
        to_row / 7.0,
        to_col / 7.0,
        1.0 if move.is_capture else 0.0,
        min(move.num_captures / 4.0, 1.0),  # Normalize (max typical is 4)
        1.0 if move.promotion else 0.0,
        1.0 if piece.is_king else 0.0,
    ], dtype=np.float32)

    return features


def encode_moves(state: GameState, moves: List[Move]) -> np.ndarray:
    """
    Encode a list of moves as feature vectors.

    Returns:
        numpy array of shape (num_moves, 8)
    """
    features = []
    board = state.board
    current_player = state.current_player

    for move in moves:
        piece = board.get_piece(move.start)
        if piece is not None:
            features.append(encode_move(move, piece, current_player))
        else:
            # Fallback for edge cases
            features.append(np.zeros(8, dtype=np.float32))

    if not features:
        return np.zeros((0, 8), dtype=np.float32)

    return np.stack(features)


def decode_board(planes: np.ndarray, current_player: Player = Player.ONE) -> GameState:
    """
    Decode a board tensor back to a GameState.

    Args:
        planes: numpy array of shape (5, 8, 8)
        current_player: The player to move

    Returns:
        GameState
    """
    board = Board()
    opponent = current_player.opponent()

    for row in range(8):
        for col in range(8):
            if planes[0, row, col] > 0.5:
                board.set_piece((row, col), Piece(current_player, PieceType.MAN))
            elif planes[1, row, col] > 0.5:
                board.set_piece((row, col), Piece(current_player, PieceType.KING))
            elif planes[2, row, col] > 0.5:
                board.set_piece((row, col), Piece(opponent, PieceType.MAN))
            elif planes[3, row, col] > 0.5:
                board.set_piece((row, col), Piece(opponent, PieceType.KING))

    return GameState(board, current_player)


# Number of move features
MOVE_FEATURE_SIZE = 8

# Number of board planes
BOARD_PLANES = 5
