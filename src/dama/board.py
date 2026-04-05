"""Board state representation for Filipino Dama."""

from typing import Optional, Dict, Iterator, Tuple

from .types import Piece, Player, PieceType, Position


class Board:
    """
    8x8 board for Filipino Dama.

    Only dark squares are used: those where (row + col) % 2 == 1.
    Row 0 is the top; Player 1 starts on rows 0-2, Player 2 on rows 5-7.
    """

    SIZE = 8

    def __init__(self):
        """Create an empty board."""
        # Maps position (row, col) -> Piece
        self._pieces: Dict[Position, Piece] = {}

    def clone(self) -> "Board":
        """Create a copy of this board."""
        new_board = Board()
        new_board._pieces = dict(self._pieces)
        return new_board

    @classmethod
    def initial(cls) -> "Board":
        """Create a board with the standard initial setup."""
        board = cls()

        # Player 1 pieces on rows 0-2 (top), dark squares
        for row in range(3):
            for col in range(cls.SIZE):
                if cls.is_playable(row, col):
                    board.set_piece((row, col), Piece(Player.ONE, PieceType.MAN))

        # Player 2 pieces on rows 5-7 (bottom), dark squares
        for row in range(5, 8):
            for col in range(cls.SIZE):
                if cls.is_playable(row, col):
                    board.set_piece((row, col), Piece(Player.TWO, PieceType.MAN))

        return board

    @staticmethod
    def is_playable(row: int, col: int) -> bool:
        """Check if a square is a playable (dark) square."""
        return (row + col) % 2 == 1

    @staticmethod
    def in_bounds(row: int, col: int) -> bool:
        """Check if a position is within the board."""
        return 0 <= row < Board.SIZE and 0 <= col < Board.SIZE

    def get_piece(self, pos: Position) -> Optional[Piece]:
        """Get the piece at a position, or None if empty."""
        return self._pieces.get(pos)

    def set_piece(self, pos: Position, piece: Optional[Piece]) -> None:
        """Set or remove a piece at a position."""
        if piece is None:
            self._pieces.pop(pos, None)
        else:
            self._pieces[pos] = piece

    def remove_piece(self, pos: Position) -> Optional[Piece]:
        """Remove and return the piece at a position."""
        return self._pieces.pop(pos, None)

    def move_piece(self, from_pos: Position, to_pos: Position) -> None:
        """Move a piece from one position to another."""
        piece = self.remove_piece(from_pos)
        if piece is not None:
            self.set_piece(to_pos, piece)

    def get_pieces(self, player: Optional[Player] = None) -> Iterator[Tuple[Position, Piece]]:
        """Iterate over all pieces, optionally filtered by player."""
        for pos, piece in self._pieces.items():
            if player is None or piece.player == player:
                yield pos, piece

    def count_pieces(self, player: Player) -> Tuple[int, int]:
        """Count (men, kings) for a player."""
        men = 0
        kings = 0
        for _, piece in self.get_pieces(player):
            if piece.is_king:
                kings += 1
            else:
                men += 1
        return men, kings

    def is_empty(self, pos: Position) -> bool:
        """Check if a position is empty."""
        return pos not in self._pieces

    def has_pieces(self, player: Player) -> bool:
        """Check if a player has any pieces on the board."""
        return any(p.player == player for p in self._pieces.values())

    @staticmethod
    def promotion_row(player: Player) -> int:
        """Get the promotion row for a player."""
        # Player 1 moves down, promotes on row 7
        # Player 2 moves up, promotes on row 0
        return 7 if player == Player.ONE else 0

    def to_compact(self) -> dict:
        """Convert board to compact JSON-serializable format.

        Uses tuples for positions instead of lists — tuples are cheaper to
        create (~20ns vs ~40ns each, ×24 pieces ×1500 calls/batch = ~0.7ms
        savings per self-play batch).  Both serialize identically to JSON
        arrays, and all consumers (Cython encode, scoring, from_compact)
        read positions via [0]/[1] indexing which works for both types.
        """
        p1_men = []
        p1_kings = []
        p2_men = []
        p2_kings = []

        for pos, piece in self._pieces.items():
            if piece.player == Player.ONE:
                if piece.is_king:
                    p1_kings.append(pos)
                else:
                    p1_men.append(pos)
            else:
                if piece.is_king:
                    p2_kings.append(pos)
                else:
                    p2_men.append(pos)

        return {
            "p1_men": p1_men,
            "p1_kings": p1_kings,
            "p2_men": p2_men,
            "p2_kings": p2_kings,
        }

    @classmethod
    def from_compact(cls, data: dict) -> "Board":
        """Create a board from compact format."""
        board = cls()

        for pos in data.get("p1_men", []):
            board.set_piece(tuple(pos), Piece(Player.ONE, PieceType.MAN))
        for pos in data.get("p1_kings", []):
            board.set_piece(tuple(pos), Piece(Player.ONE, PieceType.KING))
        for pos in data.get("p2_men", []):
            board.set_piece(tuple(pos), Piece(Player.TWO, PieceType.MAN))
        for pos in data.get("p2_kings", []):
            board.set_piece(tuple(pos), Piece(Player.TWO, PieceType.KING))

        return board

    def __str__(self) -> str:
        """String representation of the board."""
        lines = []
        lines.append("  0 1 2 3 4 5 6 7")
        for row in range(self.SIZE):
            row_str = f"{row} "
            for col in range(self.SIZE):
                piece = self.get_piece((row, col))
                if piece is None:
                    row_str += ". " if self.is_playable(row, col) else "  "
                elif piece.player == Player.ONE:
                    row_str += "O " if piece.is_king else "o "
                else:
                    row_str += "X " if piece.is_king else "x "
            lines.append(row_str)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Board({len(self._pieces)} pieces)"
