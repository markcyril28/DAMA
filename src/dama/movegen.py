"""Move generation for Filipino Dama."""

from typing import List, Tuple, Optional, Set

from .types import Move, Player, Position, Piece, PieceType
from .board import Board
from .rules import (
    FORWARD_DIRECTIONS_P1,
    FORWARD_DIRECTIONS_P2,
    BACKWARD_DIRECTIONS_P1,
    BACKWARD_DIRECTIONS_P2,
    ALL_DIRECTIONS,
    FORCED_CAPTURE,
)
from .config import get_config


def get_forward_directions(player: Player) -> List[Tuple[int, int]]:
    """Get the forward diagonal directions for a player."""
    return FORWARD_DIRECTIONS_P1 if player == Player.ONE else FORWARD_DIRECTIONS_P2


def get_backward_directions(player: Player) -> List[Tuple[int, int]]:
    """Get the backward diagonal directions for a player."""
    return BACKWARD_DIRECTIONS_P1 if player == Player.ONE else BACKWARD_DIRECTIONS_P2


def get_move_directions(piece: Piece) -> List[Tuple[int, int]]:
    """Get all valid move directions for a piece."""
    if piece.is_king:
        return ALL_DIRECTIONS
    return get_forward_directions(piece.player)


def get_capture_directions(piece: Piece, config=None, _rules=None) -> List[Tuple[int, int]]:
    """Get all valid capture directions for a piece (considering backward capture rule)."""
    if piece.is_king:
        return ALL_DIRECTIONS

    if _rules is None:
        if config is None:
            config = get_config()
        _rules = config.game.rules
    forward = get_forward_directions(piece.player)

    if _rules.backward_capture:
        # Include backward directions for capture
        backward = get_backward_directions(piece.player)
        return forward + backward

    return forward


def generate_simple_moves(board: Board, pos: Position, piece: Piece, config=None) -> List[Move]:
    """Generate non-capture moves for a piece at a position."""
    moves = []
    row, col = pos
    directions = get_move_directions(piece)
    if config is None:
        config = get_config()
    rules = config.game.rules

    for dr, dc in directions:
        if piece.is_king and rules.king_flying_capture:
            # Flying king: can move any number of squares along diagonal
            distance = 1
            while True:
                new_row, new_col = row + distance * dr, col + distance * dc
                new_pos = (new_row, new_col)

                if not Board.in_bounds(new_row, new_col):
                    break

                if not board.is_empty(new_pos):
                    # Blocked by a piece
                    break

                moves.append(Move(
                    path=(pos, new_pos),
                    captures=(),
                    promotion=False,  # Kings don't promote
                ))
                distance += 1
        else:
            # Standard move: one square
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)

            if Board.in_bounds(new_row, new_col) and board.is_empty(new_pos):
                # Check for promotion
                promotion = (
                    not piece.is_king
                    and new_row == Board.promotion_row(piece.player)
                )
                moves.append(Move(
                    path=(pos, new_pos),
                    captures=(),
                    promotion=promotion,
                ))

    return moves


def generate_captures_from_position(
    board: Board,
    pos: Position,
    piece: Piece,
    already_captured: Set[Position],
    path: List[Position],
    config=None,
) -> List[Move]:
    """
    Recursively generate all capture sequences from a position.

    Uses in-place board mutation with undo/redo to avoid O(num_pieces)
    dict copies per capture branch. The board is always restored before
    returning, so the caller sees no mutation.

    Args:
        board: The current board state (mutated temporarily, restored on return).
        pos: Current position of the piece.
        piece: The piece making the capture.
        already_captured: Set of positions already captured in this sequence.
        path: Path taken so far (list of positions).

    Returns:
        List of all possible capture moves (including multi-jumps).
    """
    row, col = pos
    if config is None:
        config = get_config()
    rules = config.game.rules

    directions = get_capture_directions(piece, config, _rules=rules)
    captures_found = []
    # Direct dict access for speed in hot loop
    _pieces = board._pieces

    for dr, dc in directions:
        if piece.is_king and rules.king_flying_capture:
            # Flying king: scan along the diagonal for capture opportunities
            captures_found.extend(
                _generate_flying_king_captures(
                    board, pos, piece, dr, dc, already_captured, path, config
                )
            )
        else:
            # Standard capture: jump one square over enemy
            capture_row, capture_col = row + dr, col + dc
            capture_pos = (capture_row, capture_col)
            land_row, land_col = row + 2 * dr, col + 2 * dc
            land_pos = (land_row, land_col)

            if not (0 <= land_row < 8 and 0 <= land_col < 8):
                continue

            if capture_pos in already_captured:
                continue

            captured_piece = _pieces.get(capture_pos)
            if captured_piece is None or captured_piece.player == piece.player:
                continue

            if land_pos in _pieces and land_pos != path[0]:
                continue

            # Valid capture found — mutate board in-place, recurse, then undo
            new_path = path + [land_pos]
            new_captured = already_captured | {capture_pos}

            # Apply: move piece from pos to land_pos, remove captured
            _pieces.pop(pos, None)
            _pieces.pop(capture_pos, None)
            _pieces[land_pos] = piece

            further_captures = generate_captures_from_position(
                board,
                land_pos,
                piece,
                new_captured,
                new_path,
                config,
            )

            # Undo: restore board to pre-mutation state
            _pieces.pop(land_pos, None)
            _pieces[pos] = piece
            _pieces[capture_pos] = captured_piece

            if further_captures:
                captures_found.extend(further_captures)
            else:
                promotion = (
                    not piece.is_king
                    and land_pos[0] == Board.promotion_row(piece.player)
                )
                captures_found.append(Move(
                    path=tuple(new_path),
                    captures=tuple(new_captured),
                    promotion=promotion,
                ))

    return captures_found


def _generate_flying_king_captures(
    board: Board,
    pos: Position,
    piece: Piece,
    dr: int,
    dc: int,
    already_captured: Set[Position],
    path: List[Position],
    config=None,
) -> List[Move]:
    """
    Generate captures for a flying king along a single diagonal direction.

    Uses in-place board mutation with undo/redo to avoid board cloning.

    A flying king can:
    - Move any number of squares along a diagonal
    - Capture an enemy piece at any distance
    - Land on any empty square beyond the captured piece

    Args:
        board: The current board state (mutated temporarily, restored on return).
        pos: Current position of the king.
        piece: The king piece.
        dr, dc: Direction to scan.
        already_captured: Set of positions already captured in this sequence.
        path: Path taken so far.

    Returns:
        List of capture moves from this direction.
    """
    row, col = pos
    captures_found = []
    _pieces = board._pieces

    # Scan along the diagonal to find an enemy piece
    distance = 1
    while True:
        scan_row, scan_col = row + distance * dr, col + distance * dc
        scan_pos = (scan_row, scan_col)

        if not (0 <= scan_row < 8 and 0 <= scan_col < 8):
            break

        scanned_piece = _pieces.get(scan_pos)

        if scanned_piece is not None:
            # Found a piece - check if it's capturable
            if scanned_piece.player == piece.player:
                # Friendly piece blocks the path
                break

            if scan_pos in already_captured:
                # Already captured in this sequence - blocks further movement
                break

            # Enemy piece found - look for landing squares beyond it
            land_distance = 1
            while True:
                land_row = scan_row + land_distance * dr
                land_col = scan_col + land_distance * dc
                land_pos = (land_row, land_col)

                if not (0 <= land_row < 8 and 0 <= land_col < 8):
                    break

                # Check if landing square is valid
                if land_pos in _pieces and land_pos != path[0]:
                    # Blocked by another piece
                    break

                if land_pos not in _pieces or land_pos == path[0]:
                    # Valid landing square — mutate in-place, recurse, undo
                    new_path = path + [land_pos]
                    new_captured = already_captured | {scan_pos}

                    # Apply
                    _pieces.pop(pos, None)
                    _pieces.pop(scan_pos, None)
                    _pieces[land_pos] = piece

                    # Look for further captures
                    further_captures = generate_captures_from_position(
                        board,
                        land_pos,
                        piece,
                        new_captured,
                        new_path,
                        config,
                    )

                    # Undo
                    _pieces.pop(land_pos, None)
                    _pieces[pos] = piece
                    _pieces[scan_pos] = scanned_piece

                    if further_captures:
                        captures_found.extend(further_captures)
                    else:
                        # End of capture sequence
                        captures_found.append(Move(
                            path=tuple(new_path),
                            captures=tuple(new_captured),
                            promotion=False,  # Kings don't promote
                        ))

                land_distance += 1

            # After finding an enemy piece, stop scanning in this direction
            break

        distance += 1

    return captures_found


def generate_captures(board: Board, pos: Position, piece: Piece, config=None) -> List[Move]:
    """Generate all capture moves for a piece at a position."""
    return generate_captures_from_position(
        board, pos, piece, set(), [pos], config
    )


def generate_all_moves(board: Board, player: Player) -> List[Move]:
    """
    Generate all legal moves for a player.

    If captures are available and forced_capture is enabled,
    only capture moves are returned.

    Returns:
        List of legal Move objects.
    """
    capture_moves = []
    config = get_config()
    rules = config.game.rules

    # Materialize pieces list: capture generation now mutates the board
    # temporarily (undo/redo), so iterating a live generator would break.
    pieces = list(board.get_pieces(player))

    for pos, piece in pieces:
        # Generate captures for this piece
        captures = generate_captures(board, pos, piece, config)
        capture_moves.extend(captures)

    # Early exit: when forced_capture is enabled (always true for Filipino
    # Dama) and any captures exist, simple moves are illegal.  Skip
    # generate_simple_moves entirely — avoids ~50% of Move object
    # allocations in mid/late game where captures are common.
    if rules.forced_capture and capture_moves:
        return capture_moves

    simple_moves = []
    for pos, piece in pieces:
        # Generate simple moves
        simple = generate_simple_moves(board, pos, piece, config)
        simple_moves.extend(simple)

    return simple_moves + capture_moves


def has_legal_moves(board: Board, player: Player) -> bool:
    """Check if a player has any legal moves."""
    # Quick check: does the player have any pieces?
    if not board.has_pieces(player):
        return False

    config = get_config()

    # Materialize pieces list: capture generation mutates board temporarily.
    pieces = list(board.get_pieces(player))

    # Check for at least one legal move
    for pos, piece in pieces:
        # Check for captures
        if generate_captures(board, pos, piece, config):
            return True
        # Check for simple moves
        if generate_simple_moves(board, pos, piece, config):
            return True

    return False
