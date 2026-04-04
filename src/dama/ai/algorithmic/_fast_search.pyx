# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython-accelerated alpha-beta search for Filipino Dama.

Implements the entire minimax search tree in C, eliminating Python object
creation/destruction overhead at every node. This is the #1 optimization
for self-play throughput since minimax search is 99% of game time.

Board representation: int8[64] flat array, index = row * 8 + col.
  EMPTY=0, P1_MAN=1, P1_KING=2, P2_MAN=3, P2_KING=4

Rule flags are read from the Python config once per search call and
passed as ints through the C call tree — zero per-node Python overhead.
"""

from libc.string cimport memcpy, memset
from libc.math cimport fabs
from libc.time cimport clock, CLOCKS_PER_SEC
from libc.stdlib cimport malloc, free, calloc

# ── Board cell values ──
DEF EMPTY = 0
DEF P1_MAN = 1
DEF P1_KING = 2
DEF P2_MAN = 3
DEF P2_KING = 4

# ── Players ──
DEF PLAYER_ONE = 1
DEF PLAYER_TWO = 2

# ── Move limits ──
DEF MAX_PATH = 8
DEF MAX_CAPTURES = 7
DEF MAX_MOVES = 64

cdef struct CMove:
    int path_r[MAX_PATH]
    int path_c[MAX_PATH]
    int path_len
    int cap_r[MAX_CAPTURES]
    int cap_c[MAX_CAPTURES]
    int num_captures
    bint promotion

cdef struct CMoveList:
    CMove moves[MAX_MOVES]
    int count

# ── Rule flags (packed into a single struct for efficient passing) ──
cdef struct Rules:
    bint forced_capture
    bint backward_capture
    bint king_flying_capture

# ── Directions ──
cdef int FWD_P1_DR[2]
cdef int FWD_P1_DC[2]
cdef int FWD_P2_DR[2]
cdef int FWD_P2_DC[2]
cdef int ALL_DR[4]
cdef int ALL_DC[4]

FWD_P1_DR[0] = 1; FWD_P1_DC[0] = -1
FWD_P1_DR[1] = 1; FWD_P1_DC[1] = 1
FWD_P2_DR[0] = -1; FWD_P2_DC[0] = -1
FWD_P2_DR[1] = -1; FWD_P2_DC[1] = 1
ALL_DR[0] = -1; ALL_DC[0] = -1
ALL_DR[1] = -1; ALL_DC[1] = 1
ALL_DR[2] = 1;  ALL_DC[2] = -1
ALL_DR[3] = 1;  ALL_DC[3] = 1

# ── Evaluation weights ──
DEF W_MAN = 100
DEF W_KING = 200
DEF W_MOBILITY = 5
DEF W_ADVANCEMENT = 2
DEF W_CENTER = 3
DEF W_BACK_RANK = 10

# Center squares lookup
cdef bint CENTER_SQ[64]

cdef void _init_center():
    cdef int i
    for i in range(64):
        CENTER_SQ[i] = 0
    CENTER_SQ[3*8+2] = 1; CENTER_SQ[3*8+4] = 1; CENTER_SQ[3*8+6] = 1
    CENTER_SQ[4*8+1] = 1; CENTER_SQ[4*8+3] = 1; CENTER_SQ[4*8+5] = 1; CENTER_SQ[4*8+7] = 1

_init_center()


# ═══════════════════════════════════════════════════════════════════════
# Zobrist hashing + Transposition table
# ═══════════════════════════════════════════════════════════════════════
# Zobrist hashing maps board positions to 64-bit pseudo-random keys.
# The transposition table caches search results so that positions reached
# via different move orders (transpositions) reuse prior work.
# Typical hit rates are 30-60%, giving ~2-4x speedup at deeper depths.

DEF TT_SIZE_BITS = 20
DEF TT_SIZE = 1048576          # 1 << 20 = 1M entries (~16MB)
DEF TT_MASK = 1048575          # TT_SIZE - 1

# TT entry flags
DEF TT_EXACT = 0
DEF TT_LOWERBOUND = 1
DEF TT_UPPERBOUND = 2

cdef struct TTEntry:
    unsigned long long hash_key  # Full hash for collision verification
    float score
    short depth
    unsigned char flag           # TT_EXACT / TT_LOWERBOUND / TT_UPPERBOUND
    unsigned char padding

# Zobrist keys: 5 piece types × 64 squares, plus side-to-move toggle.
# Piece indices: 0=EMPTY (unused), 1=P1_MAN, 2=P1_KING, 3=P2_MAN, 4=P2_KING
cdef unsigned long long ZOBRIST_PIECES[5][64]
cdef unsigned long long ZOBRIST_SIDE

# Module-level TT allocated on first use (persists across searches within a process).
cdef TTEntry *_tt_table = NULL

cdef void _init_zobrist():
    """Initialize Zobrist hash keys with a deterministic PRNG."""
    # LCG with constants from Knuth's MMIX
    cdef unsigned long long state = 0x12345678DEADBEEF
    cdef int piece, sq
    for piece in range(5):
        for sq in range(64):
            state = state * 6364136223846793005ULL + 1442695040888963407ULL
            ZOBRIST_PIECES[piece][sq] = state
    state = state * 6364136223846793005ULL + 1442695040888963407ULL
    ZOBRIST_SIDE = state

_init_zobrist()

cdef void _ensure_tt():
    """Allocate TT on first use (zero-initialized = all entries empty)."""
    global _tt_table
    if _tt_table == NULL:
        _tt_table = <TTEntry *>calloc(TT_SIZE, sizeof(TTEntry))

cdef inline unsigned long long compute_hash(
    signed char *board, int player
) noexcept nogil:
    """Compute full Zobrist hash for a board position."""
    cdef unsigned long long h = 0
    cdef int sq, piece
    for sq in range(64):
        piece = board[sq]
        if piece != EMPTY:
            h = h ^ ZOBRIST_PIECES[piece][sq]
    if player == PLAYER_TWO:
        h = h ^ ZOBRIST_SIDE
    return h

cdef inline void tt_store(
    unsigned long long hash_key, float score, int depth, int flag
) noexcept nogil:
    """Store a search result in the transposition table (always-replace)."""
    cdef unsigned long long idx = hash_key & TT_MASK
    cdef TTEntry *entry = &_tt_table[idx]
    # Always-replace: simpler than depth-preferred and works well with
    # iterative deepening (newer results from deeper searches overwrite).
    entry.hash_key = hash_key
    entry.score = score
    entry.depth = <short>depth
    entry.flag = <unsigned char>flag
    entry.padding = 0

cdef inline bint tt_probe(
    unsigned long long hash_key, int depth,
    float *score, float *alpha, float *beta
) noexcept nogil:
    """Probe the transposition table. Returns True if a cutoff occurred.

    On a hit with sufficient depth, adjusts alpha/beta or returns exact score.
    The caller should check if alpha >= beta after this returns True.
    """
    cdef unsigned long long idx = hash_key & TT_MASK
    cdef TTEntry *entry = &_tt_table[idx]
    if entry.hash_key != hash_key:
        return False
    if entry.depth < depth:
        return False
    if entry.flag == TT_EXACT:
        score[0] = entry.score
        return True
    elif entry.flag == TT_LOWERBOUND:
        if entry.score > alpha[0]:
            alpha[0] = entry.score
    elif entry.flag == TT_UPPERBOUND:
        if entry.score < beta[0]:
            beta[0] = entry.score
    if alpha[0] >= beta[0]:
        score[0] = entry.score
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# Board helpers
# ═══════════════════════════════════════════════════════════════════════

cdef inline bint in_bounds(int r, int c) noexcept nogil:
    return 0 <= r < 8 and 0 <= c < 8

cdef inline int cell(signed char *board, int r, int c) noexcept nogil:
    return board[r * 8 + c]

cdef inline void set_cell(signed char *board, int r, int c, int val) noexcept nogil:
    board[r * 8 + c] = val

cdef inline bint is_player(int cell_val, int player) noexcept nogil:
    if player == PLAYER_ONE:
        return cell_val == P1_MAN or cell_val == P1_KING
    else:
        return cell_val == P2_MAN or cell_val == P2_KING

cdef inline bint is_opponent(int cell_val, int player) noexcept nogil:
    if player == PLAYER_ONE:
        return cell_val == P2_MAN or cell_val == P2_KING
    else:
        return cell_val == P1_MAN or cell_val == P1_KING

cdef inline bint is_king(int cell_val) noexcept nogil:
    return cell_val == P1_KING or cell_val == P2_KING

cdef inline int promotion_row(int player) noexcept nogil:
    return 7 if player == PLAYER_ONE else 0

cdef inline int promote_piece(int cell_val) noexcept nogil:
    if cell_val == P1_MAN:
        return P1_KING
    elif cell_val == P2_MAN:
        return P2_KING
    return cell_val

cdef inline int opponent(int player) noexcept nogil:
    return PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE


# ═══════════════════════════════════════════════════════════════════════
# Move generation
# ═══════════════════════════════════════════════════════════════════════

cdef void _get_move_dirs(int piece, int player, Rules *rules,
                         int **out_dr, int **out_dc, int *out_n) noexcept nogil:
    """Get movement directions for simple (non-capture) moves."""
    if is_king(piece):
        out_dr[0] = ALL_DR; out_dc[0] = ALL_DC; out_n[0] = 4
    elif player == PLAYER_ONE:
        out_dr[0] = FWD_P1_DR; out_dc[0] = FWD_P1_DC; out_n[0] = 2
    else:
        out_dr[0] = FWD_P2_DR; out_dc[0] = FWD_P2_DC; out_n[0] = 2


cdef void _get_capture_dirs(int piece, int player, Rules *rules,
                            int **out_dr, int **out_dc, int *out_n) noexcept nogil:
    """Get capture directions — all 4 for kings, or forward+backward if enabled."""
    if is_king(piece) or rules.backward_capture:
        out_dr[0] = ALL_DR; out_dc[0] = ALL_DC; out_n[0] = 4
    elif player == PLAYER_ONE:
        out_dr[0] = FWD_P1_DR; out_dc[0] = FWD_P1_DC; out_n[0] = 2
    else:
        out_dr[0] = FWD_P2_DR; out_dc[0] = FWD_P2_DC; out_n[0] = 2


cdef void generate_simple_moves(
    signed char *board, int r, int c, int piece, int player,
    Rules *rules, CMoveList *out
) noexcept nogil:
    """Generate non-capture moves for a piece."""
    cdef int nr, nc, d, dist
    cdef int *dirs_r
    cdef int *dirs_c
    cdef int ndirs
    cdef bint is_k = is_king(piece)

    _get_move_dirs(piece, player, rules, &dirs_r, &dirs_c, &ndirs)

    for d in range(ndirs):
        if is_k and rules.king_flying_capture:
            # Flying king: slide along diagonal
            dist = 1
            while True:
                nr = r + dist * dirs_r[d]
                nc = c + dist * dirs_c[d]
                if not in_bounds(nr, nc):
                    break
                if cell(board, nr, nc) != EMPTY:
                    break
                if out.count < MAX_MOVES:
                    _add_simple_move(out, r, c, nr, nc, 0)
                dist += 1
        else:
            nr = r + dirs_r[d]
            nc = c + dirs_c[d]
            if in_bounds(nr, nc) and cell(board, nr, nc) == EMPTY:
                if out.count < MAX_MOVES:
                    _add_simple_move(out, r, c, nr, nc,
                                     nr == promotion_row(player) and not is_k)


cdef inline void _add_simple_move(
    CMoveList *out, int sr, int sc, int er, int ec, bint promo
) noexcept nogil:
    cdef CMove *m = &out.moves[out.count]
    m.path_r[0] = sr; m.path_c[0] = sc
    m.path_r[1] = er; m.path_c[1] = ec
    m.path_len = 2
    m.num_captures = 0
    m.promotion = promo
    out.count += 1


# ── Capture generation (recursive, in-place mutation with undo) ──

ctypedef unsigned long long uint64

cdef inline bint bit_test(uint64 bits, int r, int c) noexcept nogil:
    return (bits >> (r * 8 + c)) & 1

cdef inline uint64 bit_set(uint64 bits, int r, int c) noexcept nogil:
    return bits | ((<uint64>1) << (r * 8 + c))


cdef int _generate_captures_recursive(
    signed char *board, int r, int c, int piece, int player,
    uint64 captured_bits,
    int *path_r, int *path_c, int path_len,
    int *cap_r, int *cap_c, int num_caps,
    int start_r, int start_c,
    Rules *rules, CMoveList *out
) noexcept nogil:
    """Recursively generate capture sequences. Returns count found."""
    cdef int d, cr, cc, lr, lc
    cdef int found = 0
    cdef int captured_piece, further
    cdef int *dirs_r
    cdef int *dirs_c
    cdef int ndirs

    _get_capture_dirs(piece, player, rules, &dirs_r, &dirs_c, &ndirs)

    for d in range(ndirs):
        if is_king(piece) and rules.king_flying_capture:
            found += _flying_king_capture(
                board, r, c, piece, player,
                dirs_r[d], dirs_c[d],
                captured_bits, path_r, path_c, path_len,
                cap_r, cap_c, num_caps, start_r, start_c, rules, out
            )
        else:
            cr = r + dirs_r[d]
            cc = c + dirs_c[d]
            lr = r + 2 * dirs_r[d]
            lc = c + 2 * dirs_c[d]

            if not in_bounds(lr, lc):
                continue
            if bit_test(captured_bits, cr, cc):
                continue
            captured_piece = cell(board, cr, cc)
            if not is_opponent(captured_piece, player):
                continue
            if cell(board, lr, lc) != EMPTY:
                if not (lr == start_r and lc == start_c):
                    continue

            path_r[path_len] = lr
            path_c[path_len] = lc
            cap_r[num_caps] = cr
            cap_c[num_caps] = cc

            set_cell(board, r, c, EMPTY)
            set_cell(board, cr, cc, EMPTY)
            set_cell(board, lr, lc, piece)

            further = _generate_captures_recursive(
                board, lr, lc, piece, player,
                bit_set(captured_bits, cr, cc),
                path_r, path_c, path_len + 1,
                cap_r, cap_c, num_caps + 1,
                start_r, start_c, rules, out
            )

            set_cell(board, lr, lc, EMPTY)
            set_cell(board, r, c, piece)
            set_cell(board, cr, cc, captured_piece)

            if further > 0:
                found += further
            else:
                if out.count < MAX_MOVES:
                    _copy_capture_move(
                        out, path_r, path_c, path_len + 1,
                        cap_r, cap_c, num_caps + 1,
                        not is_king(piece) and lr == promotion_row(player)
                    )
                    found += 1

    return found


cdef int _flying_king_capture(
    signed char *board, int r, int c, int piece, int player,
    int dr, int dc,
    uint64 captured_bits,
    int *path_r, int *path_c, int path_len,
    int *cap_r, int *cap_c, int num_caps,
    int start_r, int start_c,
    Rules *rules, CMoveList *out
) noexcept nogil:
    """Generate captures for a flying king along one diagonal."""
    cdef int sr, sc, dist, found = 0
    cdef int scan_piece
    cdef int lr, lc, land_dist, further
    cdef int captured_piece

    dist = 1
    while True:
        sr = r + dist * dr
        sc = c + dist * dc
        if not in_bounds(sr, sc):
            break
        scan_piece = cell(board, sr, sc)
        if scan_piece != EMPTY:
            if is_player(scan_piece, player):
                break
            if bit_test(captured_bits, sr, sc):
                break
            captured_piece = scan_piece
            land_dist = 1
            while True:
                lr = sr + land_dist * dr
                lc = sc + land_dist * dc
                if not in_bounds(lr, lc):
                    break
                if cell(board, lr, lc) != EMPTY:
                    if lr == start_r and lc == start_c:
                        pass
                    else:
                        break

                if cell(board, lr, lc) == EMPTY or (lr == start_r and lc == start_c):
                    path_r[path_len] = lr
                    path_c[path_len] = lc
                    cap_r[num_caps] = sr
                    cap_c[num_caps] = sc

                    set_cell(board, r, c, EMPTY)
                    set_cell(board, sr, sc, EMPTY)
                    set_cell(board, lr, lc, piece)

                    further = _generate_captures_recursive(
                        board, lr, lc, piece, player,
                        bit_set(captured_bits, sr, sc),
                        path_r, path_c, path_len + 1,
                        cap_r, cap_c, num_caps + 1,
                        start_r, start_c, rules, out
                    )

                    set_cell(board, lr, lc, EMPTY)
                    set_cell(board, r, c, piece)
                    set_cell(board, sr, sc, captured_piece)

                    if further > 0:
                        found += further
                    else:
                        if out.count < MAX_MOVES:
                            _copy_capture_move(
                                out, path_r, path_c, path_len + 1,
                                cap_r, cap_c, num_caps + 1, 0)
                            found += 1

                land_dist += 1
            break
        dist += 1

    return found


cdef inline void _copy_capture_move(
    CMoveList *out,
    int *path_r, int *path_c, int path_len,
    int *cap_r, int *cap_c, int num_caps,
    bint promo
) noexcept nogil:
    cdef CMove *m = &out.moves[out.count]
    cdef int i
    m.path_len = path_len
    for i in range(path_len):
        m.path_r[i] = path_r[i]
        m.path_c[i] = path_c[i]
    m.num_captures = num_caps
    for i in range(num_caps):
        m.cap_r[i] = cap_r[i]
        m.cap_c[i] = cap_c[i]
    m.promotion = promo
    out.count += 1


cdef void generate_captures(
    signed char *board, int r, int c, int piece, int player,
    Rules *rules, CMoveList *out
) noexcept nogil:
    cdef int path_r[MAX_PATH]
    cdef int path_c[MAX_PATH]
    cdef int cap_r[MAX_CAPTURES]
    cdef int cap_c[MAX_CAPTURES]

    path_r[0] = r; path_c[0] = c

    _generate_captures_recursive(
        board, r, c, piece, player,
        0, path_r, path_c, 1, cap_r, cap_c, 0,
        r, c, rules, out
    )


cdef void generate_all_moves_c(
    signed char *board, int player, Rules *rules, CMoveList *out
) noexcept nogil:
    """Generate all legal moves for a player. Forced capture applied if rules say so."""
    cdef CMoveList simple_moves
    cdef CMoveList capture_moves
    cdef int r, c, piece, i

    simple_moves.count = 0
    capture_moves.count = 0

    for r in range(8):
        for c in range(8):
            if (r + c) % 2 != 1:
                continue
            piece = cell(board, r, c)
            if piece == EMPTY or not is_player(piece, player):
                continue
            generate_captures(board, r, c, piece, player, rules, &capture_moves)
            generate_simple_moves(board, r, c, piece, player, rules, &simple_moves)

    if rules.forced_capture and capture_moves.count > 0:
        out.count = capture_moves.count
        for i in range(capture_moves.count):
            out.moves[i] = capture_moves.moves[i]
    else:
        out.count = simple_moves.count + capture_moves.count
        for i in range(capture_moves.count):
            out.moves[i] = capture_moves.moves[i]
        for i in range(simple_moves.count):
            out.moves[capture_moves.count + i] = simple_moves.moves[i]


# ═══════════════════════════════════════════════════════════════════════
# Apply move
# ═══════════════════════════════════════════════════════════════════════

cdef void apply_move_c(
    signed char *board, signed char *new_board, CMove *move, int player
) noexcept nogil:
    cdef int sr, sc, er, ec, i, piece

    memcpy(new_board, board, 64)
    sr = move.path_r[0]; sc = move.path_c[0]
    er = move.path_r[move.path_len - 1]; ec = move.path_c[move.path_len - 1]
    piece = new_board[sr * 8 + sc]
    for i in range(move.num_captures):
        new_board[move.cap_r[i] * 8 + move.cap_c[i]] = EMPTY
    new_board[sr * 8 + sc] = EMPTY
    if move.promotion:
        piece = promote_piece(piece)
    new_board[er * 8 + ec] = piece


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

cdef float evaluate_c(signed char *board, int player) noexcept nogil:
    """Evaluate board from perspective of `player` (material + position, no mobility)."""
    cdef int r, c, piece, idx
    cdef int cur_men = 0, cur_kings = 0, opp_men = 0, opp_kings = 0
    cdef float score = 0.0
    cdef int advancement
    cdef float mult

    for r in range(8):
        for c in range(8):
            if (r + c) % 2 != 1:
                continue
            piece = cell(board, r, c)
            if piece == EMPTY:
                continue

            idx = r * 8 + c
            if is_player(piece, player):
                mult = 1.0
                if is_king(piece):
                    cur_kings += 1
                else:
                    cur_men += 1
            else:
                mult = -1.0
                if is_king(piece):
                    opp_kings += 1
                else:
                    opp_men += 1

            if not is_king(piece):
                advancement = r if piece == P1_MAN else 7 - r
                score += advancement * W_ADVANCEMENT * mult

            if CENTER_SQ[idx]:
                score += W_CENTER * mult

            if is_king(piece):
                if piece == P1_KING and r == 0:
                    score += W_BACK_RANK * mult
                elif piece == P2_KING and r == 7:
                    score += W_BACK_RANK * mult

    score += (cur_men - opp_men) * W_MAN
    score += (cur_kings - opp_kings) * W_KING
    return score


cdef float evaluate_with_mobility(
    signed char *board, int player, Rules *rules
) noexcept nogil:
    """Full evaluation including mobility."""
    cdef CMoveList cur_moves, opp_moves
    cdef float score
    cdef int opp = opponent(player)

    cur_moves.count = 0
    generate_all_moves_c(board, player, rules, &cur_moves)
    if cur_moves.count == 0:
        return -10000.0

    opp_moves.count = 0
    generate_all_moves_c(board, opp, rules, &opp_moves)
    if opp_moves.count == 0:
        return 10000.0

    score = evaluate_c(board, player)
    score += (cur_moves.count - opp_moves.count) * W_MOBILITY
    return score


# ═══════════════════════════════════════════════════════════════════════
# Alpha-beta search
# ═══════════════════════════════════════════════════════════════════════

cdef struct SearchState:
    double deadline
    int nodes
    bint timeout

cdef float alphabeta(
    signed char *board, int player, int depth, float alpha, float beta,
    Rules *rules, SearchState *ss
) noexcept nogil:
    """Negamax alpha-beta search with transposition table."""
    cdef CMoveList moves
    cdef signed char new_board[64]
    cdef float score, best, orig_alpha
    cdef int i, opp, tt_flag
    cdef unsigned long long h

    ss.nodes += 1

    if (ss.nodes & 1023) == 0:
        if _check_deadline(ss):
            return 0.0

    if depth == 0:
        return evaluate_with_mobility(board, player, rules)

    # ── TT probe ──
    h = compute_hash(board, player)
    orig_alpha = alpha
    if _tt_table != NULL:
        if tt_probe(h, depth, &score, &alpha, &beta):
            return score

    moves.count = 0
    generate_all_moves_c(board, player, rules, &moves)

    if moves.count == 0:
        return -10000.0

    _order_moves(&moves)

    opp = opponent(player)
    best = -100000.0

    for i in range(moves.count):
        apply_move_c(board, new_board, &moves.moves[i], player)
        score = -alphabeta(new_board, opp, depth - 1, -beta, -alpha, rules, ss)

        if ss.timeout:
            return 0.0

        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    # ── TT store ──
    if _tt_table != NULL and not ss.timeout:
        if best <= orig_alpha:
            tt_flag = TT_UPPERBOUND
        elif best >= beta:
            tt_flag = TT_LOWERBOUND
        else:
            tt_flag = TT_EXACT
        tt_store(h, best, depth, tt_flag)

    return best


cdef bint _check_deadline(SearchState *ss) noexcept nogil:
    cdef double now = <double>clock() / <double>CLOCKS_PER_SEC
    if now >= ss.deadline:
        ss.timeout = True
        return True
    return False


cdef void _order_moves(CMoveList *moves) noexcept nogil:
    """Selection sort by priority (captures > promotions > center)."""
    cdef int i, j, best_idx
    cdef int scores[MAX_MOVES]
    cdef int er, ec, s
    cdef CMove temp

    for i in range(moves.count):
        er = moves.moves[i].path_r[moves.moves[i].path_len - 1]
        ec = moves.moves[i].path_c[moves.moves[i].path_len - 1]
        s = moves.moves[i].num_captures * 10
        if moves.moves[i].promotion:
            s += 5
        s += 7 - <int>(fabs(3.5 - er) + fabs(3.5 - ec))
        scores[i] = s

    for i in range(moves.count - 1):
        best_idx = i
        for j in range(i + 1, moves.count):
            if scores[j] > scores[best_idx]:
                best_idx = j
        if best_idx != i:
            temp = moves.moves[i]
            moves.moves[i] = moves.moves[best_idx]
            moves.moves[best_idx] = temp
            s = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = s


cdef int search_root(
    signed char *board, int player, CMoveList *moves, int depth,
    Rules *rules, SearchState *ss
) noexcept nogil:
    """Search at root level. Returns index of best move."""
    cdef float alpha = -100000.0
    cdef float beta = 100000.0
    cdef float best_score = -100000.0
    cdef float score
    cdef int best_idx = 0
    cdef int i, opp
    cdef signed char new_board[64]

    opp = opponent(player)

    for i in range(moves.count):
        apply_move_c(board, new_board, &moves.moves[i], player)
        score = -alphabeta(new_board, opp, depth - 1, -beta, -alpha, rules, ss)

        if ss.timeout:
            return best_idx

        if score > best_score:
            best_score = score
            best_idx = i
        if score > alpha:
            alpha = score
        if best_score >= 9000:
            break

    return best_idx


cdef dict cmove_to_dict(CMove *m):
    """Convert CMove to Python dict matching Move.to_dict() format."""
    cdef list path = []
    cdef list captures = []
    cdef int i
    for i in range(m.path_len):
        path.append([m.path_r[i], m.path_c[i]])
    for i in range(m.num_captures):
        captures.append([m.cap_r[i], m.cap_c[i]])
    return {"path": path, "captures": captures, "promotion": bool(m.promotion)}


cdef void _load_board(object state, signed char *board):
    """Load a GameState's board into a flat array."""
    memset(board, 0, 64)
    py_board = state.board
    for (r, c), piece in py_board._pieces.items():
        if piece.player.value == 1:
            board[r * 8 + c] = P1_KING if piece.is_king else P1_MAN
        else:
            board[r * 8 + c] = P2_KING if piece.is_king else P2_MAN


cdef Rules _load_rules():
    """Load rules from the Python config singleton (called once per search)."""
    cdef Rules rules
    from dama.config import get_config
    cfg = get_config()
    rules.forced_capture = cfg.game.rules.forced_capture
    rules.backward_capture = cfg.game.rules.backward_capture
    rules.king_flying_capture = cfg.game.rules.king_flying_capture
    return rules


# ═══════════════════════════════════════════════════════════════════════
# Public Python API
# ═══════════════════════════════════════════════════════════════════════

def fast_search(
    object state,
    str difficulty = 'medium',
    double time_budget_override = 0.0,
    int max_depth_override = 0,
) -> dict:
    """Run iterative-deepening alpha-beta search entirely in C.

    Args:
        state: GameState object
        difficulty: 'easy', 'medium', 'hard'
        time_budget_override: Override time budget (seconds), 0 = use default
        max_depth_override: Override max depth, 0 = use default

    Returns:
        dict with keys: 'move' (Move-compatible dict or None), 'score', 'depth', 'nodes'
    """
    cdef signed char board[64]
    cdef int player
    cdef double time_budget, deadline
    cdef int max_depth
    cdef CMoveList moves
    cdef SearchState ss
    cdef Rules rules
    cdef int best_idx, best_depth
    cdef int depth

    # Parse difficulty
    if time_budget_override > 0:
        time_budget = time_budget_override
    elif difficulty == 'easy':
        time_budget = 0.2
    elif difficulty == 'hard':
        time_budget = 2.5
    else:
        time_budget = 0.8

    if max_depth_override > 0:
        max_depth = max_depth_override
    elif difficulty == 'easy':
        max_depth = 3
    elif difficulty == 'hard':
        max_depth = 8
    else:
        max_depth = 5

    _load_board(state, board)
    player = int(state.current_player)
    rules = _load_rules()

    moves.count = 0
    generate_all_moves_c(board, player, &rules, &moves)

    if moves.count == 0:
        return {'move': None, 'score': -10000, 'depth': 0, 'nodes': 0}
    if moves.count == 1:
        return {'move': cmove_to_dict(&moves.moves[0]), 'score': 0, 'depth': 0, 'nodes': 1}

    _order_moves(&moves)

    deadline = <double>clock() / <double>CLOCKS_PER_SEC + time_budget
    ss.deadline = deadline
    ss.nodes = 0
    ss.timeout = False

    best_idx = 0
    best_depth = 0

    for depth in range(1, max_depth + 1):
        ss.timeout = False
        idx = search_root(board, player, &moves, depth, &rules, &ss)
        if not ss.timeout:
            best_idx = idx
            best_depth = depth
        if ss.timeout or _check_deadline(&ss):
            break

    return {
        'move': cmove_to_dict(&moves.moves[best_idx]),
        'score': 0,
        'depth': best_depth,
        'nodes': ss.nodes,
    }


def fast_generate_moves(object state) -> list:
    """Generate all legal moves for a GameState. Returns list of move dicts."""
    cdef signed char board[64]
    cdef int player
    cdef CMoveList moves
    cdef Rules rules
    cdef int i

    _load_board(state, board)
    player = int(state.current_player)
    rules = _load_rules()

    moves.count = 0
    generate_all_moves_c(board, player, &rules, &moves)

    result = []
    for i in range(moves.count):
        result.append(cmove_to_dict(&moves.moves[i]))
    return result


# ═══════════════════════════════════════════════════════════════════════
# Full game simulation (for self-play)
# ═══════════════════════════════════════════════════════════════════════
# Runs entire algo-vs-algo games in C, avoiding all GameState/Board/Piece/
# Move Python object creation. Only the output (compact dicts for replay)
# touches Python. This eliminates the #1 remaining self-play bottleneck
# after fast_search: the Python game loop overhead.

cdef void init_standard_board(signed char *board) noexcept nogil:
    """Initialize board with the standard Filipino Dama starting position."""
    cdef int r, c
    memset(board, 0, 64)
    # Player 1 on rows 0-2, dark squares
    for r in range(3):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r * 8 + c] = P1_MAN
    # Player 2 on rows 5-7, dark squares
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r * 8 + c] = P2_MAN


cdef dict board_to_compact_dict(signed char *board, int player, int move_count):
    """Convert flat board array to compact Python dict (same as Board.to_compact + turn)."""
    cdef list p1_men = [], p1_kings = [], p2_men = [], p2_kings = []
    cdef int r, c, piece
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 != 1:
                continue
            piece = board[r * 8 + c]
            if piece == P1_MAN:
                p1_men.append([r, c])
            elif piece == P1_KING:
                p1_kings.append([r, c])
            elif piece == P2_MAN:
                p2_men.append([r, c])
            elif piece == P2_KING:
                p2_kings.append([r, c])
    return {
        'p1_men': p1_men,
        'p1_kings': p1_kings,
        'p2_men': p2_men,
        'p2_kings': p2_kings,
        'turn': player,
        'move_count': move_count,
    }


def play_full_game_cy(
    str p1_difficulty = 'medium',
    str p2_difficulty = 'medium',
    int max_moves = 100,
    double noise_prob = 0.1,
    int start_player = 1,
) -> dict:
    """Play a complete algorithmic game entirely in C.

    Both players use iterative-deepening alpha-beta search. The entire game
    loop stays in C — no GameState, Board, Move, or Piece Python objects are
    created during gameplay. Only the output (compact dicts) touches Python.

    Returns:
        dict with 'entries' (list of replay-entry dicts), 'winner' (1/2/None),
        'num_moves', 'p1_captures', 'p2_captures', 'final_state' (compact dict).
    """
    import random as _rng

    cdef signed char board[64]
    cdef signed char new_board[64]
    cdef CMoveList moves
    cdef Rules rules
    cdef int player, chosen_idx, move_num, depth, idx, i
    cdef double time_budget
    cdef int max_depth, best_idx
    cdef SearchState ss
    cdef int p1_caps = 0, p2_caps = 0
    cdef bint game_over = False

    rules = _load_rules()
    init_standard_board(board)
    player = start_player

    cdef list entries = []
    cdef list moves_list
    cdef dict state_dict
    cdef int actual_moves = 0
    cdef str cur_diff

    for move_num in range(max_moves):
        # Generate legal moves in C
        moves.count = 0
        generate_all_moves_c(board, player, &rules, &moves)

        if moves.count == 0:
            game_over = True
            break

        # Order moves for search efficiency + consistent ordering
        _order_moves(&moves)

        # Convert board and moves to Python dicts for replay recording
        state_dict = board_to_compact_dict(board, player, move_num)
        moves_list = []
        for i in range(moves.count):
            moves_list.append(cmove_to_dict(&moves.moves[i]))

        # Choose move: noise (exploration) or alpha-beta search
        if moves.count == 1:
            chosen_idx = 0
        elif _rng.random() < noise_prob:
            chosen_idx = _rng.randint(0, moves.count - 1)
        else:
            cur_diff = p1_difficulty if player == PLAYER_ONE else p2_difficulty

            if cur_diff == 'easy':
                time_budget = 0.2; max_depth = 3
            elif cur_diff == 'hard':
                time_budget = 2.5; max_depth = 8
            else:
                time_budget = 0.8; max_depth = 5

            ss.deadline = <double>clock() / <double>CLOCKS_PER_SEC + time_budget
            ss.nodes = 0
            ss.timeout = False

            best_idx = 0
            for depth in range(1, max_depth + 1):
                ss.timeout = False
                idx = search_root(board, player, &moves, depth, &rules, &ss)
                if not ss.timeout:
                    best_idx = idx
                if ss.timeout or _check_deadline(&ss):
                    break
            chosen_idx = best_idx

        # Track captures per player
        if moves.moves[chosen_idx].num_captures > 0:
            if player == PLAYER_ONE:
                p1_caps += moves.moves[chosen_idx].num_captures
            else:
                p2_caps += moves.moves[chosen_idx].num_captures

        # Record entry (format matches ReplayEntry.to_dict())
        entries.append({
            'state': state_dict,
            'legal_moves': moves_list,
            'chosen_index': chosen_idx,
            'result': 0,
            'score': 0.0,
        })

        # Apply move in C (board → new_board, then copy back)
        apply_move_c(board, new_board, &moves.moves[chosen_idx], player)
        memcpy(board, new_board, 64)
        player = opponent(player)
        actual_moves += 1

    # Determine winner
    cdef int winner_int = 0  # 0 = no winner (draw)
    if game_over:
        # Current player had no moves — opponent wins
        winner_int = opponent(player)
    elif actual_moves >= max_moves:
        # Max moves reached — check if current player is stuck
        moves.count = 0
        generate_all_moves_c(board, player, &rules, &moves)
        if moves.count == 0:
            winner_int = opponent(player)

    # Set results for each entry
    winner_py = winner_int if winner_int != 0 else None
    for entry_d in entries:
        turn = entry_d['state']['turn']
        if winner_int == 0:
            entry_d['result'] = 0  # Draw
        elif turn == winner_int:
            entry_d['result'] = 1  # Win
        else:
            entry_d['result'] = -1  # Loss

    # Final state for scoring
    final_state_dict = board_to_compact_dict(board, player, actual_moves)

    return {
        'entries': entries,
        'winner': winner_py,
        'num_moves': actual_moves,
        'p1_captures': p1_caps,
        'p2_captures': p2_caps,
        'final_state': final_state_dict,
    }
