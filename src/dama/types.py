"""Type definitions for Filipino Dama."""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Tuple, Optional


class Player(IntEnum):
    """Player identifiers."""
    ONE = 1  # Starts on rows 0-2, moves downward (increasing row)
    TWO = 2  # Starts on rows 5-7, moves upward (decreasing row)

    def opponent(self) -> "Player":
        """Return the opposing player."""
        return Player.TWO if self == Player.ONE else Player.ONE


class PieceType(Enum):
    """Types of pieces."""
    MAN = "man"
    KING = "king"


@dataclass(frozen=True)
class Piece:
    """A game piece on the board."""
    player: Player
    piece_type: PieceType

    @property
    def is_king(self) -> bool:
        """Check if this piece is a king."""
        return self.piece_type == PieceType.KING

    def promote(self) -> "Piece":
        """Return a promoted (king) version of this piece."""
        return Piece(self.player, PieceType.KING)


# Type alias for board positions
Position = Tuple[int, int]


@dataclass(frozen=True)
class Move:
    """
    Represents a move in Filipino Dama.

    Attributes:
        path: List of positions from start to end. A simple move has len(path) == 2;
              a multi-jump has len(path) >= 3.
        captures: List of positions of captured pieces (in order).
        promotion: True if the moving piece promotes at the end of this move.
    """
    path: Tuple[Position, ...]
    captures: Tuple[Position, ...]
    promotion: bool = False

    @property
    def start(self) -> Position:
        """Starting position of the move."""
        return self.path[0]

    @property
    def end(self) -> Position:
        """Ending position of the move."""
        return self.path[-1]

    @property
    def is_capture(self) -> bool:
        """Check if this move involves any captures."""
        return len(self.captures) > 0

    @property
    def num_captures(self) -> int:
        """Number of pieces captured in this move."""
        return len(self.captures)

    def to_dict(self) -> dict:
        """Convert move to a JSON-serializable dict."""
        return {
            "path": [list(p) for p in self.path],
            "captures": [list(c) for c in self.captures],
            "promotion": self.promotion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Move":
        """Create a Move from a dict representation."""
        return cls(
            path=tuple(tuple(p) for p in data["path"]),
            captures=tuple(tuple(c) for c in data["captures"]),
            promotion=data.get("promotion", False),
        )

    def __repr__(self) -> str:
        path_str = "->".join(f"({r},{c})" for r, c in self.path)
        if self.captures:
            return f"Move({path_str}, captures={len(self.captures)}, promo={self.promotion})"
        return f"Move({path_str})"
