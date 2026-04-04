"""Game state management for Filipino Dama."""

from typing import Optional, List
from dataclasses import dataclass

from .types import Move, Player, Position, Piece, PieceType
from .board import Board
from .movegen import generate_all_moves, has_legal_moves


@dataclass
class GameState:
    """
    Complete game state including board and turn information.

    This class is treated as immutable - apply_move returns a new state.
    """
    board: Board
    current_player: Player
    move_count: int = 0

    @classmethod
    def initial(cls) -> "GameState":
        """Create the initial game state."""
        return cls(
            board=Board.initial(),
            current_player=Player.ONE,
            move_count=0,
        )

    def legal_moves(self) -> List[Move]:
        """Get all legal moves for the current player."""
        return generate_all_moves(self.board, self.current_player)

    def apply_move(self, move: Move) -> "GameState":
        """
        Apply a move and return the new game state.

        The original state is not modified.
        """
        new_board = self.board.clone()

        # Get the piece that's moving
        piece = new_board.get_piece(move.start)
        if piece is None:
            raise ValueError(f"No piece at {move.start}")

        # Remove captured pieces
        for capture_pos in move.captures:
            new_board.remove_piece(capture_pos)

        # Move the piece
        new_board.remove_piece(move.start)

        # Handle promotion
        if move.promotion:
            piece = piece.promote()

        new_board.set_piece(move.end, piece)

        # Switch to the other player
        return GameState(
            board=new_board,
            current_player=self.current_player.opponent(),
            move_count=self.move_count + 1,
        )

    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return not has_legal_moves(self.board, self.current_player)

    def winner(self) -> Optional[Player]:
        """
        Get the winner of the game, or None if not terminal or draw.

        In Filipino Dama, a player with no legal moves loses.
        """
        if not self.is_terminal():
            return None

        # Current player has no moves, so they lose
        return self.current_player.opponent()

    def to_compact(self) -> dict:
        """Convert game state to compact JSON-serializable format."""
        data = self.board.to_compact()
        data["turn"] = int(self.current_player)
        data["move_count"] = self.move_count
        return data

    @classmethod
    def from_compact(cls, data: dict) -> "GameState":
        """Create a game state from compact format."""
        board = Board.from_compact(data)
        current_player = Player(data["turn"])
        move_count = data.get("move_count", 0)
        return cls(board, current_player, move_count)

    def __str__(self) -> str:
        lines = [
            f"Turn: Player {self.current_player} | Move #{self.move_count}",
            str(self.board),
        ]
        if self.is_terminal():
            winner = self.winner()
            if winner:
                lines.append(f"Game Over! Winner: Player {winner}")
            else:
                lines.append("Game Over! Draw")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GameState(player={self.current_player}, move={self.move_count})"
