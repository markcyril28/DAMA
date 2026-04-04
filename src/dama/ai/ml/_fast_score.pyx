# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython-accelerated game scoring for self-play data.

Drop-in replacement for score_game_dicts() in scoring.py.
Eliminates Python interpreter overhead in the per-entry scoring loop
(~100 entries per game × thousands of games per self-play cycle).
"""

from libc.math cimport exp, log

cimport cython


# ── Scoring constants (must match scoring.py) ──────────────────────

cdef double MAN_VALUE = 1.0
cdef double KING_VALUE = 1.5
cdef double CENTER_BONUS = 0.1
cdef double ADVANCE_BONUS_PER_ROW = 0.03
cdef double BACK_ROW_BONUS = 0.05
cdef double EDGE_PENALTY = -0.02
cdef double MOBILITY_WEIGHT = 0.02
cdef double CAPTURE_MOVE_BONUS = 0.05
cdef double KING_MOBILITY_WEIGHT = 0.03
cdef double WIN_SCORE = 10.0
cdef double LOSS_SCORE = -10.0
cdef double DRAW_SCORE = -3.0
cdef double QUICK_WIN_BONUS_MAX = 3.0
cdef double QUICK_WIN_HALF_MOVES = 60.0
cdef double DOMINATION_BONUS = 2.0
cdef double CAPTURE_EFFICIENCY = 0.15
cdef double LN2 = 0.6931471805599453  # log(2)


# ── Helper: game-level score from compact dict ──────────────────────

cdef double _game_score_from_compact(
    int player_int,
    object winner_int,
    int total_moves,
    int max_moves,
    dict final_state_dict,
    int captures_made,
):
    """Compute game score from compact dict (C-level, no Python calls)."""
    cdef double score = 0.0
    cdef double decay, final_adv, pos_score
    cdef int is_winner, is_loser

    # Outcome base score
    if winner_int is None:
        score = DRAW_SCORE
    elif <int>winner_int == player_int:
        score = WIN_SCORE
    else:
        score = LOSS_SCORE

    # Winner bonuses
    if winner_int is not None and <int>winner_int == player_int:
        decay = exp(-total_moves / QUICK_WIN_HALF_MOVES * LN2)
        score += QUICK_WIN_BONUS_MAX * decay
        final_adv = _material_adv_compact(final_state_dict, player_int)
        if final_adv > 3.0:
            score += DOMINATION_BONUS * (final_adv / 6.0 if final_adv < 6.0 else 1.0)
        score += captures_made * CAPTURE_EFFICIENCY

    # Loser adjustment
    elif winner_int is not None:
        final_adv = _material_adv_compact(final_state_dict, player_int)
        score += final_adv * 0.2

    # Positional adjustment
    pos_score = _positional_compact(final_state_dict, player_int)
    score += pos_score * 0.3

    return score


# ── Helper: material advantage ──────────────────────────────────────

cdef double _material_adv_compact(dict state_dict, int player_int):
    """Material advantage from compact dict."""
    cdef object my_men, my_kings, opp_men, opp_kings
    cdef double my_mat, opp_mat

    if player_int == 1:
        my_men = state_dict.get('p1_men', ())
        my_kings = state_dict.get('p1_kings', ())
        opp_men = state_dict.get('p2_men', ())
        opp_kings = state_dict.get('p2_kings', ())
    else:
        my_men = state_dict.get('p2_men', ())
        my_kings = state_dict.get('p2_kings', ())
        opp_men = state_dict.get('p1_men', ())
        opp_kings = state_dict.get('p1_kings', ())

    my_mat = len(my_men) * MAN_VALUE + len(my_kings) * KING_VALUE
    opp_mat = len(opp_men) * MAN_VALUE + len(opp_kings) * KING_VALUE
    return my_mat - opp_mat


# ── Helper: positional score ────────────────────────────────────────

cdef double _positional_compact(dict state_dict, int player_int):
    """Positional score from compact dict."""
    cdef double score = 0.0
    cdef int start_row, row, col, advancement
    cdef object men_list, kings_list

    start_row = 0 if player_int == 1 else 7

    if player_int == 1:
        men_list = state_dict.get('p1_men', ())
        kings_list = state_dict.get('p1_kings', ())
    else:
        men_list = state_dict.get('p2_men', ())
        kings_list = state_dict.get('p2_kings', ())

    for pos in men_list:
        row = pos[0]
        col = pos[1]
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

    for pos in kings_list:
        row = pos[0]
        col = pos[1]
        if 2 <= row <= 5 and 2 <= col <= 5:
            score += CENTER_BONUS
        if row == start_row:
            score += BACK_ROW_BONUS
        if col == 0 or col == 7:
            score += EDGE_PENALTY

    return score


# ── Main entry point: score_game_dicts ──────────────────────────────

def score_game_dicts_cy(
    list entry_dicts,
    object winner_int,
    int total_moves,
    int max_moves,
    dict final_state_dict,
    int p1_captures=0,
    int p2_captures=0,
):
    """Score a list of entry dicts in place (Cython-accelerated).

    Drop-in replacement for scoring.score_game_dicts().  Inlines all
    scoring computations with C-level arithmetic and avoids Python
    set/tuple creation in the per-entry loop.
    """
    cdef double game_score_p1, game_score_p2, game_score
    cdef double inv_total
    cdef int i, n, player_int, start_row, row, col
    cdef int num_captures, num_kings
    cdef double my_mat, opp_mat, material_adv
    cdef double pos_score, mob_score, total_pos, progress, outcome_weight
    cdef dict ed, state_dict, m
    # Use object (not list) — dict.get() may return tuple () as default
    cdef object my_men_list, my_kings_list, legal_moves, path, captures

    # King lookup: use a flat array of (row, col) pairs instead of Python set.
    # Max 12 kings per player (all pieces promoted). 24 ints for 12 pairs.
    cdef int king_rows[12]
    cdef int king_cols[12]
    cdef int j, sr, sc
    cdef bint is_king

    # Pre-compute game scores for both players (2 calls total)
    game_score_p1 = _game_score_from_compact(
        1, winner_int, total_moves, max_moves, final_state_dict, p1_captures)
    game_score_p2 = _game_score_from_compact(
        2, winner_int, total_moves, max_moves, final_state_dict, p2_captures)

    inv_total = 1.0 / total_moves if total_moves > 0 else 0.0
    n = len(entry_dicts)

    for i in range(n):
        ed = entry_dicts[i]
        state_dict = ed['state']
        player_int = state_dict['turn']
        game_score = game_score_p1 if player_int == 1 else game_score_p2
        legal_moves = ed['legal_moves']

        # ── Material ──
        if player_int == 1:
            my_men_list = state_dict.get('p1_men', ())
            my_kings_list = state_dict.get('p1_kings', ())
            opp_mat = (len(state_dict.get('p2_men', ())) * MAN_VALUE
                       + len(state_dict.get('p2_kings', ())) * KING_VALUE)
        else:
            my_men_list = state_dict.get('p2_men', ())
            my_kings_list = state_dict.get('p2_kings', ())
            opp_mat = (len(state_dict.get('p1_men', ())) * MAN_VALUE
                       + len(state_dict.get('p1_kings', ())) * KING_VALUE)

        my_mat = len(my_men_list) * MAN_VALUE + len(my_kings_list) * KING_VALUE
        material_adv = my_mat - opp_mat

        # ── Positional ──
        pos_score = 0.0
        start_row = 0 if player_int == 1 else 7

        for pos in my_men_list:
            row = pos[0]
            col = pos[1]
            if 2 <= row <= 5 and 2 <= col <= 5:
                pos_score += CENTER_BONUS
            if player_int == 1:
                pos_score += row * ADVANCE_BONUS_PER_ROW
            else:
                pos_score += (7 - row) * ADVANCE_BONUS_PER_ROW
            if row == start_row:
                pos_score += BACK_ROW_BONUS
            if col == 0 or col == 7:
                pos_score += EDGE_PENALTY

        # Build king position array (replaces Python set)
        num_kings = len(my_kings_list)
        if num_kings > 12:
            num_kings = 12
        for j in range(num_kings):
            pos = my_kings_list[j]
            king_rows[j] = pos[0]
            king_cols[j] = pos[1]
            row = pos[0]
            col = pos[1]
            if 2 <= row <= 5 and 2 <= col <= 5:
                pos_score += CENTER_BONUS
            if row == start_row:
                pos_score += BACK_ROW_BONUS
            if col == 0 or col == 7:
                pos_score += EDGE_PENALTY

        # ── Mobility ──
        mob_score = 0.0
        for m in legal_moves:
            captures = m.get('captures', ())
            if captures:
                num_captures = len(captures)
                mob_score += CAPTURE_MOVE_BONUS + (num_captures - 1) * CAPTURE_MOVE_BONUS * 0.5
            else:
                mob_score += MOBILITY_WEIGHT
            # Check if start position is a king (C-array scan, no set/tuple)
            path = m['path']
            sr = path[0][0]
            sc = path[0][1]
            is_king = False
            for j in range(num_kings):
                if king_rows[j] == sr and king_cols[j] == sc:
                    is_king = True
                    break
            if is_king:
                mob_score += KING_MOBILITY_WEIGHT

        # ── Combine ──
        total_pos = my_mat + material_adv * 0.5 + pos_score + mob_score

        # ── Blend with game outcome ──
        progress = i * inv_total if total_moves > 0 else 0.5
        outcome_weight = 0.3 + 0.7 * progress
        ed['score'] = (1.0 - outcome_weight) * total_pos + outcome_weight * game_score
