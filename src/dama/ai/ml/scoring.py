"""
Scoring system for Filipino Dama game positions.

Computes detailed scores during play that serve as shaped rewards
for more targeted training. The scoring captures:

1. Material advantage (piece counts with king bonus)
2. Positional quality (center control, advancement, back-row defense)
3. Tactical features (capture threats, mobility)
4. Game outcome (win/loss/draw with efficiency bonuses)

These scores are stored in replay entries and used as reward weights
during training so that the model learns more from good play and
winning strategies, and less from losing or drawing moves.
"""

import math
from typing import Optional, Dict, Any, Tuple

from ...types import Player, PieceType
from ...game_state import GameState
from ...board import Board


# ────────────────────────────────────────────────────────────────
# Tunable constants
# ────────────────────────────────────────────────────────────────

# Piece material values
MAN_VALUE = 1.0
KING_VALUE = 1.5

# Positional bonuses (per piece)
CENTER_BONUS = 0.1          # Pieces on center 4x4 squares
ADVANCE_BONUS_PER_ROW = 0.03  # Bonus per row advanced toward promotion
BACK_ROW_BONUS = 0.05       # Pieces on own back row (defensive)
EDGE_PENALTY = -0.02        # Slight penalty for edge pieces (less mobile)

# Tactical bonuses
MOBILITY_WEIGHT = 0.02      # Per legal move available
CAPTURE_MOVE_BONUS = 0.05   # Per legal capture move available
KING_MOBILITY_WEIGHT = 0.03  # Extra weight for king mobility

# Outcome scoring
WIN_SCORE = 10.0
LOSS_SCORE = -10.0
DRAW_SCORE = -3.0           # Penalize draws to encourage decisive play

# Efficiency modifiers
QUICK_WIN_BONUS_MAX = 3.0   # Maximum bonus for winning quickly
QUICK_WIN_HALF_MOVES = 60   # Moves at which bonus is halved
DOMINATION_BONUS = 2.0      # Bonus for winning with large material lead
CAPTURE_EFFICIENCY = 0.15   # Bonus per capture in a winning game

# Reward weighting parameters (for training loss)
REWARD_WEIGHT_MIN = 0.1     # Minimum weight (losing games still contribute)
REWARD_WEIGHT_MAX = 2.0     # Maximum weight (dominant wins)
REWARD_WEIGHT_DRAW = 0.5    # Weight for draw games


# ────────────────────────────────────────────────────────────────
# Scoring functions
# ────────────────────────────────────────────────────────────────

def compute_material_score(board: Board, player: Player) -> float:
    """
    Compute the raw material score for a player.

    Men count as MAN_VALUE, kings count as KING_VALUE.
    """
    men, kings = board.count_pieces(player)
    return men * MAN_VALUE + kings * KING_VALUE


def compute_material_advantage(board: Board, player: Player) -> float:
    """
    Compute relative material advantage: player score minus opponent score.
    """
    my_score = compute_material_score(board, player)
    opp_score = compute_material_score(board, player.opponent())
    return my_score - opp_score


def compute_positional_score(state: GameState, player: Player) -> float:
    """
    Evaluate positional quality for a player.

    Considers:
    - Center control: pieces on center squares (rows 2-5, cols 2-5)
    - Advancement: how far each man has advanced toward promotion
    - Back-row defense: pieces on own back row
    - Edge penalty: pieces on edge columns are slightly less mobile
    """
    board = state.board
    score = 0.0

    promotion_row = board.promotion_row(player)
    # Start row depends on player direction
    start_row = 0 if player == Player.ONE else 7

    for pos, piece in board.get_pieces(player):
        row, col = pos

        # Center control bonus (center 4x4 zone)
        if 2 <= row <= 5 and 2 <= col <= 5:
            score += CENTER_BONUS

        # Advancement bonus for men (kings already promoted)
        if not piece.is_king:
            if player == Player.ONE:
                # Player ONE moves downward (row 0 -> 7)
                advancement = row  # 0 to 7
            else:
                # Player TWO moves upward (row 7 -> 0)
                advancement = 7 - row
            score += advancement * ADVANCE_BONUS_PER_ROW

        # Back-row defense bonus
        if row == start_row:
            score += BACK_ROW_BONUS

        # Edge penalty
        if col == 0 or col == 7:
            score += EDGE_PENALTY

    return score


def compute_mobility_score(state: GameState, player: Player) -> float:
    """
    Compute a mobility-based score.

    More legal moves = more options = generally better position.
    Capture moves are weighted slightly higher.
    """
    from ...movegen import generate_all_moves

    # We need to evaluate from the perspective of this player's turn
    # Build a temporary state if needed
    if state.current_player == player:
        moves = state.legal_moves()
    else:
        # Create a state where it's this player's turn for evaluation
        temp_state = GameState(
            board=state.board,
            current_player=player,
            move_count=state.move_count,
        )
        moves = temp_state.legal_moves()

    score = 0.0
    for move in moves:
        if move.is_capture:
            score += CAPTURE_MOVE_BONUS
            # Bonus for multi-captures
            score += (move.num_captures - 1) * CAPTURE_MOVE_BONUS * 0.5
        else:
            score += MOBILITY_WEIGHT

        # Extra weight for king moves
        piece = state.board.get_piece(move.start)
        if piece and piece.is_king:
            score += KING_MOBILITY_WEIGHT

    return score


def compute_position_total(state: GameState, player: Player) -> float:
    """
    Compute the total positional evaluation for a player at a given state.

    This is the in-game score that captures how good the position is.
    """
    material = compute_material_score(state.board, player)
    material_adv = compute_material_advantage(state.board, player)
    positional = compute_positional_score(state, player)
    mobility = compute_mobility_score(state, player)

    # Weighted combination
    return material + material_adv * 0.5 + positional + mobility


def compute_game_score(
    player: Player,
    winner: Optional[Player],
    total_moves: int,
    max_moves: int,
    final_state: GameState,
    captures_made: int = 0,
) -> float:
    """
    Compute the final game score for a player after game completion.

    This combines the outcome with efficiency metrics to produce
    a shaped reward that provides richer training signal than
    simple +1/-1/0.

    Args:
        player: The player to score for
        winner: Who won (None for draw)
        total_moves: Total moves in the game
        max_moves: Maximum allowed moves
        final_state: The final game state
        captures_made: Number of captures this player made during the game

    Returns:
        A float score combining outcome and quality metrics.
    """
    score = 0.0

    # ── Outcome component ──
    if winner is None:
        score += DRAW_SCORE
    elif winner == player:
        score += WIN_SCORE
    else:
        score += LOSS_SCORE

    # ── Efficiency bonus/penalty (only for wins) ──
    if winner == player:
        # Quick-win bonus: exponential decay based on game length
        # Shorter games get higher bonus
        decay = math.exp(-total_moves / QUICK_WIN_HALF_MOVES * math.log(2))
        score += QUICK_WIN_BONUS_MAX * decay

        # Domination bonus: winning with large material lead
        final_adv = compute_material_advantage(final_state.board, player)
        if final_adv > 3:
            score += DOMINATION_BONUS * min(final_adv / 6.0, 1.0)

        # Capture efficiency: reward aggressive winning play
        score += captures_made * CAPTURE_EFFICIENCY

    elif winner is not None:
        # Loss penalty proportional to how badly we lost
        final_adv = compute_material_advantage(final_state.board, player)
        # final_adv will be negative for the loser
        score += final_adv * 0.2  # Additional penalty for material deficit

    # ── Final positional component ──
    # Include positional quality at game end (attenuated for outcomes)
    pos_score = compute_positional_score(final_state, player)
    score += pos_score * 0.3

    return score


def compute_per_move_score(
    state: GameState,
    player: Player,
    game_score: float,
    move_index: int,
    total_moves: int,
) -> float:
    """
    Compute a per-move score that blends in-game position evaluation
    with the final game outcome.

    Earlier moves get more of the positional evaluation;
    later moves get weighted more toward the game outcome.
    This creates a smooth gradient of signal from position to outcome.

    Args:
        state: The game state at this move
        player: The player who moved
        game_score: The final game score for this player
        move_index: Which move this is (0-indexed)
        total_moves: Total moves in the game

    Returns:
        A blended per-move score.
    """
    # In-game positional score (normalized to roughly [-5, 5] range)
    pos_score = compute_position_total(state, player)

    # Blend factor: linearly interpolate from positional to outcome
    # Early game: more position-driven
    # Late game: more outcome-driven
    if total_moves > 0:
        progress = move_index / total_moves
    else:
        progress = 0.5

    # Smooth blending with sigmoid-like curve
    outcome_weight = 0.3 + 0.7 * progress  # 30% outcome at start, 100% at end
    position_weight = 1.0 - outcome_weight

    blended = position_weight * pos_score + outcome_weight * game_score

    return blended


def compute_reward_weight(score: float) -> float:
    """
    Convert a per-move score into a training loss weight.

    The weight determines how much the model should learn from
    this particular move:
    - High weight for moves in winning/good positions
    - Low weight for moves in losing positions
    - Medium weight for draws

    Uses a sigmoid-like mapping to keep weights bounded.

    Args:
        score: The per-move score (from compute_per_move_score)

    Returns:
        A weight in [REWARD_WEIGHT_MIN, REWARD_WEIGHT_MAX]
    """
    # Sigmoid mapping centered at 0
    # score > 0 -> weight > 1.0 (learn more from good play)
    # score < 0 -> weight < 1.0 (learn less from bad play)
    # score = 0 -> weight = 1.0 (neutral)
    normalized = 2.0 / (1.0 + math.exp(-score / 5.0))  # Maps to (0, 2)

    # Clamp to configured range
    return max(REWARD_WEIGHT_MIN, min(REWARD_WEIGHT_MAX, normalized))


def score_game_entries(
    entries: list,
    winner: Optional[Player],
    total_moves: int,
    max_moves: int,
    final_state: GameState,
    player_captures: Optional[Dict[Player, int]] = None,
) -> list:
    """
    Compute and assign scores to all replay entries from a completed game.

    This should be called after a game finishes to populate the 'score'
    field in each ReplayEntry.

    Args:
        entries: List of ReplayEntry objects from the game
        winner: Who won (None for draw)
        total_moves: Total moves in the game
        max_moves: Maximum allowed moves
        final_state: The final game state
        player_captures: Dict mapping player -> total captures made

    Returns:
        The entries list (modified in-place with scores set)
    """
    if player_captures is None:
        player_captures = {Player.ONE: 0, Player.TWO: 0}

    # Pre-compute game-level scores per player
    game_scores = {}
    for player in [Player.ONE, Player.TWO]:
        game_scores[player] = compute_game_score(
            player=player,
            winner=winner,
            total_moves=total_moves,
            max_moves=max_moves,
            final_state=final_state,
            captures_made=player_captures.get(player, 0),
        )

    # Assign per-move scores
    for i, entry in enumerate(entries):
        player = Player(entry.state['turn'])

        # Reconstruct state for positional evaluation
        entry_state = GameState.from_compact(entry.state)

        entry.score = compute_per_move_score(
            state=entry_state,
            player=player,
            game_score=game_scores[player],
            move_index=i,
            total_moves=total_moves,
        )

    return entries


def get_scoring_config() -> Dict[str, Any]:
    """Return the current scoring configuration as a dict (for logging)."""
    return {
        'man_value': MAN_VALUE,
        'king_value': KING_VALUE,
        'center_bonus': CENTER_BONUS,
        'advance_bonus_per_row': ADVANCE_BONUS_PER_ROW,
        'back_row_bonus': BACK_ROW_BONUS,
        'edge_penalty': EDGE_PENALTY,
        'mobility_weight': MOBILITY_WEIGHT,
        'capture_move_bonus': CAPTURE_MOVE_BONUS,
        'king_mobility_weight': KING_MOBILITY_WEIGHT,
        'win_score': WIN_SCORE,
        'loss_score': LOSS_SCORE,
        'draw_score': DRAW_SCORE,
        'quick_win_bonus_max': QUICK_WIN_BONUS_MAX,
        'quick_win_half_moves': QUICK_WIN_HALF_MOVES,
        'domination_bonus': DOMINATION_BONUS,
        'capture_efficiency': CAPTURE_EFFICIENCY,
        'reward_weight_min': REWARD_WEIGHT_MIN,
        'reward_weight_max': REWARD_WEIGHT_MAX,
        'reward_weight_draw': REWARD_WEIGHT_DRAW,
    }
