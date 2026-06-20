"""
Tests for the ML training components.

This module tests:
- Dataset loading and preprocessing
- Move encoding/decoding
- Model architecture
- Training loop components
"""

import pytest
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dama.types import Player, Move, Position
from dama.board import Board
from dama.game_state import GameState
from dama.ai.ml.move_encoder import (
    encode_board,
    encode_move,
    encode_moves,
    decode_board,
    BOARD_PLANES,
    MOVE_FEATURE_SIZE
)
from dama.ai.ml.dataset import (
    DamaDataset,
    CachedTensorDataset,
    _encode_board_fast,
    _encode_moves_fast,
    preprocess_entries_to_tensors,
    collate_batch,
    get_available_ram_gb,
    get_total_ram_gb,
)
from dama.ai.ml.replay import ReplayEntry
from dama.ai.ml.scoring import (
    compute_material_score,
    compute_material_advantage,
    compute_positional_score,
    compute_game_score,
    compute_per_move_score,
    compute_reward_weight,
    score_game_entries,
)


class TestMoveEncoder:
    """Tests for move encoding."""
    
    def test_encode_board_shape(self):
        """Test that board encoding has correct shape."""
        state = GameState.initial()
        encoded = encode_board(state)
        
        assert encoded.shape == (BOARD_PLANES, 8, 8)
        assert encoded.dtype == np.float32
    
    def test_encode_board_planes(self):
        """Test that board planes contain valid values."""
        state = GameState.initial()
        encoded = encode_board(state)
        
        # Values should be 0 or 1
        assert np.all((encoded >= 0) & (encoded <= 1))
        
        # Last plane (side to move) should be all 1s
        assert np.all(encoded[4] == 1.0)
    
    def test_encode_moves_shape(self):
        """Test that move encoding has correct shape."""
        state = GameState.initial()
        moves = state.legal_moves()

        encoded = encode_moves(state, moves)

        assert encoded.shape == (len(moves), MOVE_FEATURE_SIZE)
        assert encoded.dtype == np.float32

    def test_encode_board_p2_rotation(self):
        """Test that P2 board encoding rotates positions onto playable squares."""
        state = GameState.initial()

        # P1 encoding: P1 men should be in rows 0-2 (top)
        p1_enc = encode_board(state)
        # P1 men (plane 0) should have pieces in rows 0-2
        assert p1_enc[0, :3, :].sum() > 0
        assert p1_enc[0, 5:, :].sum() == 0

        # Switch to P2's turn by making a move
        moves = state.legal_moves()
        state_p2 = state.apply_move(moves[0])
        assert state_p2.current_player == Player.TWO

        # P2 encoding: P2's own men (plane 0) should appear in rows 0-2
        # because positions are rotated — P2's actual rows 5-7 map to 0-2
        p2_enc = encode_board(state_p2)
        # Current player's men (P2) should be at top after rotation
        assert p2_enc[0, :3, :].sum() > 0
        # Opponent's men (P1) should be at bottom after rotation
        assert p2_enc[2, 5:, :].sum() > 0
        # A row-only mirror changes square parity.  A 180-degree rotation keeps
        # every encoded piece on the same playable-square color as P1.
        occupied = np.argwhere(p2_enc[:4].sum(axis=0) > 0)
        assert all((row + col) % 2 == 1 for row, col in occupied)

    def test_canonical_board_is_identical_for_equivalent_sides(self):
        """Equivalent P1/P2 positions must have one canonical tensor."""
        p1_state = GameState.initial()
        # The initial board is invariant under a 180-degree rotation plus a
        # player swap, so changing the side to move creates its P2 equivalent.
        p2_state = GameState(Board.initial(), Player.TWO)

        assert np.array_equal(encode_board(p1_state), encode_board(p2_state))

    def test_encode_move_p2_rotation(self):
        """Test that P2 move encoding rotates both rows and columns."""
        from dama.types import Piece, PieceType
        # A P2 move from (5,0) to (4,1) rotates to (2,7) -> (3,6).
        move = Move(path=((5, 0), (4, 1)), captures=())
        piece = Piece(Player.TWO, PieceType.MAN)

        enc_p1 = encode_move(move, piece, Player.ONE)
        enc_p2 = encode_move(move, piece, Player.TWO)

        # P1: from_row=5/7, to_row=4/7
        assert abs(enc_p1[0] - 5 / 7.0) < 1e-6
        assert abs(enc_p1[2] - 4 / 7.0) < 1e-6

        # P2: from_row=(7-5)/7=2/7, to_row=(7-4)/7=3/7
        assert abs(enc_p2[0] - 2 / 7.0) < 1e-6
        assert abs(enc_p2[2] - 3 / 7.0) < 1e-6
        assert abs(enc_p2[1] - 7 / 7.0) < 1e-6
        assert abs(enc_p2[3] - 6 / 7.0) < 1e-6

    def test_decode_board_p2_roundtrip(self):
        """Test that encode -> decode round-trips correctly for P2."""
        state = GameState.initial()
        moves = state.legal_moves()
        state_p2 = state.apply_move(moves[0])

        encoded = encode_board(state_p2)
        decoded = decode_board(encoded, state_p2.current_player)

        # All pieces should be in the same positions after round-trip
        for pos, piece in state_p2.board.get_pieces():
            decoded_piece = decoded.board.get_piece(pos)
            assert decoded_piece is not None, f"Missing piece at {pos}"
            assert decoded_piece.player == piece.player
            assert decoded_piece.is_king == piece.is_king

    def test_fast_encoders_match_reference_for_p2(self):
        """Optimized P2 encoding must retain the reference rotation semantics."""
        state = GameState(Board.initial(), Player.TWO)
        moves = state.legal_moves()
        state_dict = state.to_compact()
        move_dicts = [move.to_dict() for move in moves]

        board_fast = np.empty((BOARD_PLANES, 8, 8), dtype=np.float32)
        _encode_board_fast(state_dict, board_fast)
        assert np.array_equal(board_fast, encode_board(state))

        moves_fast = np.zeros((len(moves), MOVE_FEATURE_SIZE), dtype=np.float32)
        count = _encode_moves_fast(state_dict, move_dicts, moves_fast)
        assert count == len(moves)
        assert np.array_equal(moves_fast, encode_moves(state, moves))

        try:
            from dama.ai.ml._fast_encode import (
                encode_board_fast_cy,
                encode_moves_fast_cy,
            )
        except ImportError:
            return

        board_cy = np.empty_like(board_fast)
        moves_cy = np.zeros_like(moves_fast)
        encode_board_fast_cy(state_dict, board_cy)
        cy_count = encode_moves_fast_cy(state_dict, move_dicts, moves_cy)
        assert cy_count == len(moves)
        assert np.array_equal(board_cy, board_fast)
        assert np.array_equal(moves_cy, moves_fast)


class TestDataset:
    """Tests for dataset classes."""
    
    @pytest.fixture
    def sample_entries(self):
        """Create sample replay entries for testing."""
        state = GameState.initial()
        moves = state.legal_moves()
        
        entries = []
        for i in range(10):
            entry = ReplayEntry(
                state=state.to_compact(),
                legal_moves=[m.to_dict() for m in moves[:5]],
                chosen_index=0,
                result=1,
                score=2.5,
            )
            entries.append(entry)
        
        return entries
    
    def test_dama_dataset_length(self, sample_entries):
        """Test DamaDataset length."""
        dataset = DamaDataset(sample_entries)
        assert len(dataset) == len(sample_entries)
    
    def test_dama_dataset_getitem(self, sample_entries):
        """Test DamaDataset item retrieval."""
        dataset = DamaDataset(sample_entries)
        board, move_features, target, reward_weight, value_target = dataset[0]

        assert isinstance(board, torch.Tensor)
        assert board.shape == (BOARD_PLANES, 8, 8)
        assert isinstance(move_features, torch.Tensor)
        assert isinstance(target, int)
        assert isinstance(reward_weight, float)
        assert reward_weight > 0
        assert isinstance(value_target, float)

    def test_preprocess_entries_to_tensors(self, sample_entries):
        """Test tensor preprocessing."""
        boards, move_features, move_counts, targets, reward_weights, value_targets = preprocess_entries_to_tensors(
            sample_entries, max_moves_per_sample=64, show_progress=False
        )

        assert boards.shape == (len(sample_entries), BOARD_PLANES, 8, 8)
        assert move_features.shape == (len(sample_entries), 64, MOVE_FEATURE_SIZE)
        assert move_counts.shape == (len(sample_entries),)
        assert targets.shape == (len(sample_entries),)
        assert reward_weights.shape == (len(sample_entries),)
        assert (reward_weights > 0).all()
        assert value_targets.shape == (len(sample_entries),)

    def test_sample_weight_scales_reward_weights(self, sample_entries):
        """Test extra sample weights are applied to loss weights."""
        sample_entries[0].sample_weight = 0.25

        dataset = DamaDataset(sample_entries)
        _, _, _, reward_weight, _ = dataset[0]
        expected = compute_reward_weight(sample_entries[0].score) * 0.25
        assert reward_weight == pytest.approx(expected)

        _, _, _, _, reward_weights, _ = preprocess_entries_to_tensors(
            sample_entries, max_moves_per_sample=64, show_progress=False
        )
        assert reward_weights[0].item() == pytest.approx(expected)

    def test_cached_tensor_dataset(self, sample_entries):
        """Test CachedTensorDataset creation."""
        cached_dataset = CachedTensorDataset.from_entries(
            sample_entries, max_moves_per_sample=64, show_progress=False
        )

        assert len(cached_dataset) == len(sample_entries)

        # Test item retrieval
        board, move_features, move_count, target, reward_weight, value_target = cached_dataset[0]
        assert board.shape == (BOARD_PLANES, 8, 8)
        assert move_features.shape == (64, MOVE_FEATURE_SIZE)
        assert isinstance(reward_weight, float)
        assert reward_weight > 0
        assert isinstance(value_target, float)

    def test_collate_batch(self, sample_entries):
        """Test batch collation."""
        dataset = DamaDataset(sample_entries)
        batch = [dataset[i] for i in range(3)]

        boards, all_moves, move_counts, targets, reward_weights, value_targets = collate_batch(batch)

        assert boards.shape[0] == 3
        assert len(move_counts) == 3
        assert len(targets) == 3
        assert len(reward_weights) == 3
        assert (reward_weights > 0).all()
        assert len(value_targets) == 3
    
    def test_ram_detection(self):
        """Test RAM detection functions."""
        available = get_available_ram_gb()
        total = get_total_ram_gb()
        
        assert available >= 0
        assert total >= available


class TestModel:
    """Tests for the ML model."""
    
    def test_model_creation(self):
        """Test model creation."""
        from dama.ai.ml.model import create_model, MoveScorerNet
        
        model = create_model()
        assert isinstance(model, MoveScorerNet)
    
    def test_model_forward(self):
        """Test model forward pass."""
        from dama.ai.ml.model import create_model
        
        model = create_model()
        model.eval()
        
        batch_size = 4
        num_moves = 10
        
        boards = torch.randn(batch_size, BOARD_PLANES, 8, 8)
        move_features = torch.randn(num_moves, MOVE_FEATURE_SIZE)
        move_counts = torch.tensor([3, 2, 3, 2], dtype=torch.long)
        
        with torch.no_grad():
            scores = model(boards, move_features, move_counts)
        
        assert scores.shape == (num_moves,)

    def test_checkpoint_encoding_version_migrates_with_warning(self, tmp_path):
        """Legacy weights remain loadable while the encoding transition is explicit."""
        from dama.ai.ml.model import create_model, load_model, save_model
        from dama.ai.ml.move_encoder import ENCODING_VERSION

        model = create_model()
        current_path = tmp_path / 'current.pt'
        save_model(model, str(current_path))
        checkpoint = torch.load(current_path, weights_only=True)
        assert checkpoint['encoding_version'] == ENCODING_VERSION
        assert load_model(str(current_path), torch.device('cpu')) is not None

        legacy_path = tmp_path / 'legacy.pt'
        checkpoint.pop('encoding_version')
        torch.save(checkpoint, legacy_path)
        with pytest.warns(RuntimeWarning, match='encoding version 1'):
            assert load_model(str(legacy_path), torch.device('cpu')) is not None

    def test_teacher_difficulties_are_assigned_to_both_sides(self):
        """Mixed teacher strengths must be generated in both orientations."""
        from dama.ai.ml.trainer import _ordered_difficulty_matchups

        assert _ordered_difficulty_matchups(['hard', 'super_hard']) == [
            ('hard', 'hard'),
            ('hard', 'super_hard'),
            ('super_hard', 'hard'),
            ('super_hard', 'super_hard'),
        ]


class TestScoring:
    """Tests for the scoring system."""

    def test_material_score_initial(self):
        """Test material score at game start."""
        state = GameState.initial()
        score_p1 = compute_material_score(state.board, Player.ONE)
        score_p2 = compute_material_score(state.board, Player.TWO)
        # Each player starts with 12 men
        assert score_p1 == 12.0
        assert score_p2 == 12.0

    def test_material_advantage_initial(self):
        """Test material advantage is zero at game start."""
        state = GameState.initial()
        adv = compute_material_advantage(state.board, Player.ONE)
        assert adv == 0.0

    def test_positional_score_nonnegative(self):
        """Positional score should be non-negative at game start."""
        state = GameState.initial()
        pos = compute_positional_score(state, Player.ONE)
        # Should have some positive value from advancement/center/back-row
        assert pos >= 0.0

    def test_game_score_win(self):
        """Winning score should be large positive."""
        state = GameState.initial()
        score = compute_game_score(
            player=Player.ONE,
            winner=Player.ONE,
            total_moves=50,
            max_moves=200,
            final_state=state,
            captures_made=5,
        )
        assert score > 5.0  # Should be significantly positive

    def test_game_score_loss(self):
        """Losing score should be large negative."""
        state = GameState.initial()
        score = compute_game_score(
            player=Player.ONE,
            winner=Player.TWO,
            total_moves=50,
            max_moves=200,
            final_state=state,
            captures_made=2,
        )
        assert score < -5.0  # Should be significantly negative

    def test_game_score_draw(self):
        """Draw score should be negative (to discourage draws)."""
        state = GameState.initial()
        score = compute_game_score(
            player=Player.ONE,
            winner=None,
            total_moves=200,
            max_moves=200,
            final_state=state,
            captures_made=3,
        )
        assert score < 0  # Draws are penalized

    def test_reward_weight_positive_score(self):
        """Positive scores should give weight > 1."""
        w = compute_reward_weight(5.0)
        assert w > 1.0

    def test_reward_weight_negative_score(self):
        """Negative scores should give weight < 1."""
        w = compute_reward_weight(-5.0)
        assert w < 1.0

    def test_reward_weight_zero_score(self):
        """Zero score should give weight close to 1."""
        w = compute_reward_weight(0.0)
        assert abs(w - 1.0) < 0.01

    def test_reward_weight_bounded(self):
        """Reward weight should be bounded."""
        w_high = compute_reward_weight(100.0)
        w_low = compute_reward_weight(-100.0)
        assert w_high <= 2.0
        assert w_low >= 0.1

    def test_per_move_score_blending(self):
        """Per-move score should blend position and outcome."""
        state = GameState.initial()
        # Early move with positive game score
        early = compute_per_move_score(state, Player.ONE, game_score=10.0, move_index=0, total_moves=100)
        # Late move with same game score
        late = compute_per_move_score(state, Player.ONE, game_score=10.0, move_index=99, total_moves=100)
        # Late move should be more influenced by game_score
        # Both should be positive with a positive game_score
        assert early > 0
        assert late > 0

    def test_score_game_entries(self):
        """Test that score_game_entries populates scores on all entries."""
        state = GameState.initial()
        moves = state.legal_moves()
        entries = []
        for _ in range(5):
            entry = ReplayEntry(
                state=state.to_compact(),
                legal_moves=[m.to_dict() for m in moves[:3]],
                chosen_index=0,
                result=1,
                score=0.0,
            )
            entries.append(entry)

        score_game_entries(
            entries=entries,
            winner=Player.ONE,
            total_moves=5,
            max_moves=200,
            final_state=state,
            player_captures={Player.ONE: 3, Player.TWO: 1},
        )

        for entry in entries:
            assert entry.score != 0.0  # Scores should have been populated

    def test_replay_entry_score_serialization(self):
        """Test that ReplayEntry score field roundtrips through JSON."""
        entry = ReplayEntry(
            state={'turn': 1, 'p1_men': [], 'p1_kings': [], 'p2_men': [], 'p2_kings': []},
            legal_moves=[],
            chosen_index=0,
            result=1,
            score=3.14159,
        )
        d = entry.to_dict()
        assert 'score' in d
        
        restored = ReplayEntry.from_dict(d)
        assert abs(restored.score - 3.14159) < 0.001

    def test_replay_entry_sample_weight_serialization(self):
        """Test that ReplayEntry sample_weight roundtrips through JSON."""
        entry = ReplayEntry(
            state={'turn': 1, 'p1_men': [], 'p1_kings': [], 'p2_men': [], 'p2_kings': []},
            legal_moves=[],
            chosen_index=0,
            result=1,
            score=3.14159,
            sample_weight=0.75,
        )
        d = entry.to_dict()
        assert d['sample_weight'] == pytest.approx(0.75)

        restored = ReplayEntry.from_dict(d)
        assert restored.sample_weight == pytest.approx(0.75)

    def test_replay_entry_backward_compat(self):
        """Test backward compatibility - old entries without score field."""
        d = {
            'state': {'turn': 1, 'p1_men': [], 'p1_kings': [], 'p2_men': [], 'p2_kings': []},
            'legal_moves': [],
            'chosen_index': 0,
            'result': 0,
        }
        entry = ReplayEntry.from_dict(d)
        assert entry.score == 0.0  # Default value
        assert entry.sample_weight == 1.0


class TestTrainingSideBalance:
    """Tests for side weight balancing during self-play ingestion."""

    def test_side_balance_equalizes_reward_mass(self):
        """Trainer normalizer should equalize total P1 and P2 loss weight."""
        from dama.ai.ml.trainer import Trainer

        entries = [
            {'state': {'turn': 1}, 'score': 10.0},
            {'state': {'turn': 1}, 'score': 10.0},
            {'state': {'turn': 2}, 'score': -10.0},
        ]

        stats = Trainer._balance_side_sample_weights(entries)
        assert stats is not None

        def side_mass(side):
            return sum(
                compute_reward_weight(e.get('score', 0.0)) * e.get('sample_weight', 1.0)
                for e in entries
                if e['state']['turn'] == side
            )

        assert side_mass(1) == pytest.approx(side_mass(2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
