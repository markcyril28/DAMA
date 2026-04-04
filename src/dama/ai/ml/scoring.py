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
import numpy as np
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


def compute_reward_weights_batch(scores: np.ndarray) -> np.ndarray:
    """Vectorized batch version of compute_reward_weight.

    Operates on an entire numpy array at once using vectorized exp/clip,
    avoiding Python-level per-element function call overhead.

    Args:
        scores: 1-D float32 numpy array of per-move scores.

    Returns:
        1-D float32 numpy array of weights in [REWARD_WEIGHT_MIN, REWARD_WEIGHT_MAX].
    """
    normalized = 2.0 / (1.0 + np.exp(-scores / 5.0))
    return np.clip(normalized, REWARD_WEIGHT_MIN, REWARD_WEIGHT_MAX).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Fast-path scoring: works directly on compact dicts
# ────────────────────────────────────────────────────────────────
# Avoids GameState/Board/Piece object creation and eliminates
# the expensive legal_moves() call by using pre-stored move counts.

def _material_from_compact(state_dict: dict, player_int: int) -> float:
    """Compute material score directly from compact dict."""
    if player_int == 1:
        men = len(state_dict.get('p1_men', ()))
        kings = len(state_dict.get('p1_kings', ()))
    else:
        men = len(state_dict.get('p2_men', ()))
        kings = len(state_dict.get('p2_kings', ()))
    return men * MAN_VALUE + kings * KING_VALUE


def _material_advantage_from_compact(state_dict: dict, player_int: int) -> float:
    """Compute material advantage directly from compact dict."""
    opp = 2 if player_int == 1 else 1
    return _material_from_compact(state_dict, player_int) - _material_from_compact(state_dict, opp)


def _positional_score_from_compact(state_dict: dict, player_int: int) -> float:
    """Compute positional score directly from compact dict (no object creation)."""
    score = 0.0
    start_row = 0 if player_int == 1 else 7

    # Men
    men_key = 'p1_men' if player_int == 1 else 'p2_men'
    for pos in state_dict.get(men_key, ()):
        row, col = pos[0], pos[1]
        if 2 <= row <= 5 and 2 <= col <= 5:
            score += CENTER_BONUS
        if player_int == 1:
            advancement = row
        else:
            advancement = 7 - row
        score += advancement * ADVANCE_BONUS_PER_ROW
        if row == start_row:
            score += BACK_ROW_BONUS
        if col == 0 or col == 7:
            score += EDGE_PENALTY

    # Kings
    king_key = 'p1_kings' if player_int == 1 else 'p2_kings'
    for pos in state_dict.get(king_key, ()):
        row, col = pos[0], pos[1]
        if 2 <= row <= 5 and 2 <= col <= 5:
            score += CENTER_BONUS
        if row == start_row:
            score += BACK_ROW_BONUS
        if col == 0 or col == 7:
            score += EDGE_PENALTY

    return score


def _mobility_score_from_moves(legal_moves: list, state_dict: dict, player_int: int) -> float:
    """Approximate mobility score from pre-stored legal moves (no move generation).

    Uses the move dicts already stored in the replay entry instead of
    re-running the expensive legal_moves() computation.
    """
    king_key = 'p1_kings' if player_int == 1 else 'p2_kings'
    king_set = {(p[0], p[1]) for p in state_dict.get(king_key, ())}

    score = 0.0
    for m in legal_moves:
        captures = m.get('captures', ())
        if captures:
            n_cap = len(captures)
            score += CAPTURE_MOVE_BONUS
            score += (n_cap - 1) * CAPTURE_MOVE_BONUS * 0.5
        else:
            score += MOBILITY_WEIGHT
        start = m['path'][0]
        if (start[0], start[1]) in king_set:
            score += KING_MOBILITY_WEIGHT

    return score


def _position_total_fast(state_dict: dict, legal_moves: list, player_int: int) -> float:
    """Fast compute_position_total using compact dicts (no GameState/Board objects)."""
    material = _material_from_compact(state_dict, player_int)
    material_adv = _material_advantage_from_compact(state_dict, player_int)
    positional = _positional_score_from_compact(state_dict, player_int)
    mobility = _mobility_score_from_moves(legal_moves, state_dict, player_int)
    return material + material_adv * 0.5 + positional + mobility


def _per_move_score_fast(
    state_dict: dict,
    legal_moves: list,
    player_int: int,
    game_score: float,
    move_index: int,
    total_moves: int,
) -> float:
    """Fast per-move score using compact dicts (no GameState reconstruction)."""
    pos_score = _position_total_fast(state_dict, legal_moves, player_int)

    if total_moves > 0:
        progress = move_index / total_moves
    else:
        progress = 0.5

    outcome_weight = 0.3 + 0.7 * progress
    position_weight = 1.0 - outcome_weight

    return position_weight * pos_score + outcome_weight * game_score


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

    Uses fast-path scoring that works directly on compact dicts and
    pre-stored legal moves, avoiding expensive GameState reconstruction
    and redundant move generation.

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

    # Assign per-move scores using fast path (no GameState reconstruction)
    game_scores_by_int = {int(p): s for p, s in game_scores.items()}
    for i, entry in enumerate(entries):
        player_int = entry.state['turn']
        entry.score = _per_move_score_fast(
            state_dict=entry.state,
            legal_moves=entry.legal_moves,
            player_int=player_int,
            game_score=game_scores_by_int[player_int],
            move_index=i,
            total_moves=total_moves,
        )

    return entries


def _compute_game_score_from_compact(
    player_int: int,
    winner_int,
    total_moves: int,
    max_moves: int,
    final_state_dict: dict,
    captures_made: int = 0,
) -> float:
    """Compute game score directly from compact dict (no GameState/Board objects).

    Drop-in replacement for compute_game_score that works with the output of
    play_full_game_cy.
    """
    score = 0.0

    if winner_int is None:
        score += DRAW_SCORE
    elif winner_int == player_int:
        score += WIN_SCORE
    else:
        score += LOSS_SCORE

    if winner_int == player_int:
        decay = math.exp(-total_moves / QUICK_WIN_HALF_MOVES * math.log(2))
        score += QUICK_WIN_BONUS_MAX * decay

        final_adv = _material_advantage_from_compact(final_state_dict, player_int)
        if final_adv > 3:
            score += DOMINATION_BONUS * min(final_adv / 6.0, 1.0)

        score += captures_made * CAPTURE_EFFICIENCY

    elif winner_int is not None:
        final_adv = _material_advantage_from_compact(final_state_dict, player_int)
        score += final_adv * 0.2

    pos_score = _positional_score_from_compact(final_state_dict, player_int)
    score += pos_score * 0.3

    return score


def score_game_dicts(
    entry_dicts: list,
    winner_int,
    total_moves: int,
    max_moves: int,
    final_state_dict: dict,
    p1_captures: int = 0,
    p2_captures: int = 0,
) -> None:
    """Score a list of entry dicts in place (no ReplayEntry or GameState needed).

    Works with the output of play_full_game_cy. Modifies each dict's 'score'
    key using the same scoring logic as score_game_entries.

    Inlines the entire scoring chain (material, positional, mobility, blend)
    into a single loop to eliminate ~7 Python function calls per entry.
    """
    # Pre-compute game-level scores for both players (2 calls, outside the loop)
    game_scores = {}
    for pi in (1, 2):
        caps = p1_captures if pi == 1 else p2_captures
        game_scores[pi] = _compute_game_score_from_compact(
            player_int=pi,
            winner_int=winner_int,
            total_moves=total_moves,
            max_moves=max_moves,
            final_state_dict=final_state_dict,
            captures_made=caps,
        )

    # Cache constants as locals to avoid global lookups in the tight loop
    _MAN_VALUE = MAN_VALUE
    _KING_VALUE = KING_VALUE
    _CENTER_BONUS = CENTER_BONUS
    _ADVANCE_PER_ROW = ADVANCE_BONUS_PER_ROW
    _BACK_ROW = BACK_ROW_BONUS
    _EDGE_PEN = EDGE_PENALTY
    _MOB_W = MOBILITY_WEIGHT
    _CAP_BONUS = CAPTURE_MOVE_BONUS
    _KING_MOB = KING_MOBILITY_WEIGHT
    _inv_total = 1.0 / total_moves if total_moves > 0 else 0.0

    for i, ed in enumerate(entry_dicts):
        state_dict = ed['state']
        player_int = state_dict['turn']
        game_score = game_scores[player_int]
        legal_moves = ed['legal_moves']

        # ── Inline material ──
        if player_int == 1:
            my_men_list = state_dict.get('p1_men', ())
            my_kings_list = state_dict.get('p1_kings', ())
            opp_men_n = len(state_dict.get('p2_men', ()))
            opp_kings_n = len(state_dict.get('p2_kings', ()))
        else:
            my_men_list = state_dict.get('p2_men', ())
            my_kings_list = state_dict.get('p2_kings', ())
            opp_men_n = len(state_dict.get('p1_men', ()))
            opp_kings_n = len(state_dict.get('p1_kings', ()))

        my_mat = len(my_men_list) * _MAN_VALUE + len(my_kings_list) * _KING_VALUE
        opp_mat = opp_men_n * _MAN_VALUE + opp_kings_n * _KING_VALUE
        material_adv = my_mat - opp_mat

        # ── Inline positional ──
        pos_score = 0.0
        start_row = 0 if player_int == 1 else 7
        for pos in my_men_list:
            row = pos[0]; col = pos[1]
            if 2 <= row <= 5 and 2 <= col <= 5:
                pos_score += _CENTER_BONUS
            pos_score += (row if player_int == 1 else 7 - row) * _ADVANCE_PER_ROW
            if row == start_row:
                pos_score += _BACK_ROW
            if col == 0 or col == 7:
                pos_score += _EDGE_PEN

        king_set = set()
        for pos in my_kings_list:
            row = pos[0]; col = pos[1]
            king_set.add((row, col))
            if 2 <= row <= 5 and 2 <= col <= 5:
                pos_score += _CENTER_BONUS
            if row == start_row:
                pos_score += _BACK_ROW
            if col == 0 or col == 7:
                pos_score += _EDGE_PEN

        # ── Inline mobility ──
        mob_score = 0.0
        for m in legal_moves:
            captures = m.get('captures', ())
            if captures:
                n_cap = len(captures)
                mob_score += _CAP_BONUS + (n_cap - 1) * _CAP_BONUS * 0.5
            else:
                mob_score += _MOB_W
            start = m['path'][0]
            if (start[0], start[1]) in king_set:
                mob_score += _KING_MOB

        # ── Combine position total ──
        total_pos = my_mat + material_adv * 0.5 + pos_score + mob_score

        # ── Blend with game outcome ──
        progress = i * _inv_total if total_moves > 0 else 0.5
        outcome_weight = 0.3 + 0.7 * progress
        ed['score'] = (1.0 - outcome_weight) * total_pos + outcome_weight * game_score


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
