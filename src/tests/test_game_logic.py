"""
Tests for the core game logic.

This module tests the fundamental game mechanics including:
- Board representation and operations
- Move generation
- Game state management
- Rule enforcement
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dama.board import Board
from dama.types import Player, Piece, PieceType, Position, Move
from dama.game_state import GameState
from dama.movegen import generate_all_moves, has_legal_moves


class TestBoard:
    """Tests for Board class."""
    
    def test_initial_board_setup(self):
        """Test that initial board has correct piece placement."""
        board = Board.initial()
        
        # Check Player 1 pieces at top (rows 0-2)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:  # Dark squares only
                    piece = board.get_piece((row, col))
                    assert piece is not None, f"Expected piece at ({row}, {col})"
                    assert piece.player == Player.ONE
                    assert piece.is_king is False
        
        # Check Player 2 pieces at bottom (rows 5-7)
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:  # Dark squares only
                    piece = board.get_piece((row, col))
                    assert piece is not None, f"Expected piece at ({row}, {col})"
                    assert piece.player == Player.TWO
                    assert piece.is_king is False
    
    def test_empty_center(self):
        """Test that center rows are empty initially."""
        board = Board.initial()
        
        for row in range(3, 5):
            for col in range(8):
                piece = board.get_piece((row, col))
                assert piece is None, f"Expected empty at ({row}, {col})"
    
    def test_clone_independence(self):
        """Test that cloned board is independent."""
        board = Board.initial()
        clone = board.clone()
        
        # Modify original
        board.remove_piece((0, 1))
        
        # Clone should be unaffected
        assert clone.get_piece((0, 1)) is not None


class TestGameState:
    """Tests for GameState class."""
    
    def test_initial_state(self):
        """Test initial game state."""
        state = GameState.initial()
        
        assert state.current_player == Player.ONE
        assert state.move_count == 0
        assert not state.is_terminal()
    
    def test_legal_moves_available(self):
        """Test that initial state has legal moves."""
        state = GameState.initial()
        moves = state.legal_moves()
        
        assert len(moves) > 0, "Initial state should have legal moves"
    
    def test_apply_move_switches_player(self):
        """Test that applying a move switches the current player."""
        state = GameState.initial()
        moves = state.legal_moves()
        
        assert len(moves) > 0
        new_state = state.apply_move(moves[0])
        
        assert new_state.current_player == Player.TWO
        assert new_state.move_count == 1
    
    def test_compact_serialization(self):
        """Test compact serialization round-trip."""
        state = GameState.initial()
        
        compact = state.to_compact()
        restored = GameState.from_compact(compact)
        
        assert restored.current_player == state.current_player
        assert restored.move_count == state.move_count


class TestMoveGeneration:
    """Tests for move generation."""
    
    def test_has_legal_moves_initial(self):
        """Test has_legal_moves for initial board."""
        board = Board.initial()
        
        assert has_legal_moves(board, Player.ONE)
        assert has_legal_moves(board, Player.TWO)
    
    def test_generate_moves_initial(self):
        """Test move generation for initial board."""
        board = Board.initial()
        
        moves_p1 = generate_all_moves(board, Player.ONE)
        moves_p2 = generate_all_moves(board, Player.TWO)
        
        # Both players should have moves
        assert len(moves_p1) > 0
        assert len(moves_p2) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
