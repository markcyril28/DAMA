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
from libc.math cimport fabs, sqrt
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
DEF MAX_PLY = 32
DEF NUM_KILLERS = 2

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
    # Dark squares in the 4x4 center zone (rows 2-5, cols 2-5),
    # matching the scoring system's CENTER_BONUS definition.
    CENTER_SQ[2*8+3] = 1; CENTER_SQ[2*8+5] = 1   # (2,3), (2,5)
    CENTER_SQ[3*8+2] = 1; CENTER_SQ[3*8+4] = 1   # (3,2), (3,4)
    CENTER_SQ[4*8+3] = 1; CENTER_SQ[4*8+5] = 1   # (4,3), (4,5)
    CENTER_SQ[5*8+2] = 1; CENTER_SQ[5*8+4] = 1   # (5,2), (5,4)

_init_center()

# Precomputed dark square indices — only dark squares ((r+c)%2==1) can hold
# pieces. Iterating 32 dark squares instead of 64 total squares halves the
# loop count in evaluate_c, _has_pieces, _count_player_pieces, compute_hash,
# and move generation (which all skipped light squares via branch anyway).
DEF NUM_DARK_SQ = 32
cdef int DARK_SQ[32]       # flat index (r*8+c)
cdef int DARK_SQ_R[32]     # row
cdef int DARK_SQ_C[32]     # col

cdef void _init_dark_sq():
    cdef int i = 0, r, c
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                DARK_SQ[i] = r * 8 + c
                DARK_SQ_R[i] = r
                DARK_SQ_C[i] = c
                i += 1

_init_dark_sq()


# ── Precomputed LMR (Late Move Reduction) table ──
# LMR_TABLE[depth][move_index] = reduction amount.
# Graduated reductions replace the fixed R=1: later moves at deeper depths
# get stronger reductions. Formula: R = sqrt(depth-1) * sqrt(i-1) / 3.0,
# clamped to [1, depth-2].  Typical values:
#   depth 3, move 3: R=1   depth 8, move 5: R=1   depth 12, move 8: R=2
#   depth 3, move 8: R=1   depth 8, move 10: R=2  depth 12, move 16: R=4
DEF LMR_MAX_D = 33
DEF LMR_MAX_M = 65
cdef int LMR_TABLE[33][65]

cdef void _init_lmr_table() noexcept nogil:
    cdef int d, m, r
    for d in range(LMR_MAX_D):
        for m in range(LMR_MAX_M):
            if d < 3 or m < 3:
                LMR_TABLE[d][m] = 0
            else:
                r = <int>(sqrt(<double>(d - 1)) * sqrt(<double>(m - 1)) / 3.0)
                if r < 1:
                    r = 1
                if r > d - 2:
                    r = d - 2
                LMR_TABLE[d][m] = r

_init_lmr_table()


# ═══════════════════════════════════════════════════════════════════════
# Zobrist hashing + Transposition table
# ═══════════════════════════════════════════════════════════════════════
# Zobrist hashing maps board positions to 64-bit pseudo-random keys.
# The transposition table caches search results so that positions reached
# via different move orders (transpositions) reuse prior work.
# Typical hit rates are 30-60%, giving ~2-4x speedup at deeper depths.

DEF TT_SIZE_BITS = 22
DEF TT_SIZE = 4194304          # 1 << 22 = 4M entries (~64MB)
DEF TT_MASK = 4194303          # TT_SIZE - 1

# TT entry flags
DEF TT_EXACT = 0
DEF TT_LOWERBOUND = 1
DEF TT_UPPERBOUND = 2

# Packed generation+flag byte: (generation << 2) | flag
# 6-bit generation (0-63) wraps more often but fits best-move in 16 bytes.
DEF TT_GEN_SHIFT = 2
DEF TT_GEN_MASK = 63            # 0x3F — 6 bits for generation
DEF TT_FLAG_MASK = 3            # 0x03 — 2 bits for flag

cdef struct TTEntry:
    unsigned long long hash_key  # 8 bytes: full hash for collision verification
    float score                  # 4 bytes
    unsigned char depth          # 1 byte: search depth (max 32, was short)
    unsigned char gen_flag       # 1 byte: (generation << 2) | flag
    unsigned char best_from      # 1 byte: best move from-square (0-63), 0xFF = none
    unsigned char best_to        # 1 byte: best move to-square (0-63), 0xFF = none
    # Total: 16 bytes — same as before, no padding increase

# Zobrist keys: 5 piece types × 64 squares, plus side-to-move toggle.
# Piece indices: 0=EMPTY (unused), 1=P1_MAN, 2=P1_KING, 3=P2_MAN, 4=P2_KING
cdef unsigned long long ZOBRIST_PIECES[5][64]
cdef unsigned long long ZOBRIST_SIDE

# Module-level TT allocated on first use (persists across searches within a process).
cdef TTEntry *_tt_table = NULL
# Generation counter: incremented on each fast_search() call. Entries with a
# different generation are treated as stale (logically empty) without needing
# a 16MB memset to physically clear the table. Wraps at 256 — worst case, a
# 256-search-old entry passes the generation check, which is harmless (just a
# rare false TT hit that the hash verification catches).
cdef unsigned char _tt_generation = 0

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
    cdef int i, sq, piece
    for i in range(NUM_DARK_SQ):
        sq = DARK_SQ[i]
        piece = board[sq]
        if piece != EMPTY:
            h = h ^ ZOBRIST_PIECES[piece][sq]
    if player == PLAYER_TWO:
        h = h ^ ZOBRIST_SIDE
    return h

cdef inline void tt_store(
    unsigned long long hash_key, float score, int depth, int flag,
    int best_from, int best_to
) noexcept nogil:
    """Store a search result in the transposition table (always-replace)."""
    cdef unsigned long long idx = hash_key & TT_MASK
    cdef TTEntry *entry = &_tt_table[idx]
    # Always-replace: simpler than depth-preferred and works well with
    # iterative deepening (newer results from deeper searches overwrite).
    entry.hash_key = hash_key
    entry.score = score
    entry.depth = <unsigned char>depth
    entry.gen_flag = (_tt_generation << TT_GEN_SHIFT) | (<unsigned char>flag & TT_FLAG_MASK)
    entry.best_from = <unsigned char>best_from
    entry.best_to = <unsigned char>best_to

cdef inline bint tt_probe(
    unsigned long long hash_key, int depth,
    float *score, float *alpha, float *beta,
    int *tt_from, int *tt_to
) noexcept nogil:
    """Probe the transposition table. Returns True if a cutoff occurred.

    On a hit with sufficient depth, adjusts alpha/beta or returns exact score.
    Always sets tt_from/tt_to when hash matches (even if depth insufficient)
    so the caller can use the TT move for ordering.
    """
    cdef unsigned long long idx = hash_key & TT_MASK
    cdef TTEntry *entry = &_tt_table[idx]
    cdef unsigned char gen, flag

    tt_from[0] = -1
    tt_to[0] = -1

    gen = (entry.gen_flag >> TT_GEN_SHIFT) & TT_GEN_MASK
    if gen != _tt_generation:
        return False
    if entry.hash_key != hash_key:
        return False

    # Hash match — extract best move for ordering (even if depth insufficient)
    if entry.best_from != 0xFF:
        tt_from[0] = <int>entry.best_from
        tt_to[0] = <int>entry.best_to

    if entry.depth < depth:
        return False

    flag = entry.gen_flag & TT_FLAG_MASK
    if flag == TT_EXACT:
        score[0] = entry.score
        return True
    elif flag == TT_LOWERBOUND:
        if entry.score > alpha[0]:
            alpha[0] = entry.score
    elif flag == TT_UPPERBOUND:
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
    cdef int r, c, piece, i, sq

    simple_moves.count = 0
    capture_moves.count = 0

    for i in range(NUM_DARK_SQ):
        sq = DARK_SQ[i]
        piece = board[sq]
        if piece == EMPTY or not is_player(piece, player):
            continue
        r = DARK_SQ_R[i]
        c = DARK_SQ_C[i]
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


cdef int generate_captures_only_c(
    signed char *board, int player, Rules *rules, CMoveList *out
) noexcept nogil:
    """Generate only capture moves (no simple moves). For quiescence search."""
    cdef int i, sq, piece
    out.count = 0
    for i in range(NUM_DARK_SQ):
        sq = DARK_SQ[i]
        piece = board[sq]
        if piece == EMPTY or not is_player(piece, player):
            continue
        generate_captures(board, DARK_SQ_R[i], DARK_SQ_C[i], piece, player, rules, out)
    return out.count


cdef inline bint _has_pieces(signed char *board, int player) noexcept nogil:
    """Fast check if player has any pieces. O(32) worst, early-exit on first find."""
    cdef int i, piece
    for i in range(NUM_DARK_SQ):
        piece = board[DARK_SQ[i]]
        if piece != EMPTY and is_player(piece, player):
            return True
    return False


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
    cdef int i, r, c, piece, idx
    cdef int cur_men = 0, cur_kings = 0, opp_men = 0, opp_kings = 0
    cdef float score = 0.0
    cdef int advancement
    cdef float mult

    for i in range(NUM_DARK_SQ):
        idx = DARK_SQ[i]
        piece = board[idx]
        if piece == EMPTY:
            continue

        r = DARK_SQ_R[i]
        c = DARK_SQ_C[i]

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


# Maximum depth for quiescence capture-only search (prevents explosion)
DEF MAX_QS_DEPTH = 6

cdef float quiescence(
    signed char *board, int player, float alpha, float beta,
    Rules *rules, SearchState *ss, unsigned long long h, int qs_depth
) noexcept nogil:
    """Quiescence search: extend search with captures until position is quiet.

    At depth 0, instead of calling evaluate_with_mobility() which generates
    ALL moves for BOTH sides (expensive), quiescence does:
    1. Stand-pat eval using evaluate_c() (material+positional, no mobility)
    2. Generate captures only (much cheaper than full move generation)
    3. If no captures: position is quiet, return stand-pat
    4. If captures: search them recursively (typically 1-3 captures)

    Cost: ~1 capture-only generation per QS node vs 2 full generations
    in evaluate_with_mobility(). Most leaves are quiet (0 captures),
    so the common case is O(32 squares scan) + O(64 squares eval) ≈ O(96).
    """
    cdef float stand_pat, score
    cdef CMoveList captures
    cdef signed char new_board[64]
    cdef unsigned long long child_h
    cdef int i, j, opp

    ss.nodes += 1

    # Deadline check (shared counter with main search)
    if (ss.nodes & 4095) == 0:
        if _check_deadline(ss):
            return 0.0

    # Terminal detection: if current player has no pieces, they lost
    if not _has_pieces(board, player):
        return -10000.0

    # Stand-pat: use material+positional eval (no mobility — too expensive)
    stand_pat = evaluate_c(board, player)

    # Beta cutoff: standing pat is already good enough
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    # Depth limit: stop extending captures at MAX_QS_DEPTH
    if qs_depth <= 0:
        return alpha

    # Generate captures only (skip simple moves entirely)
    captures.count = 0
    generate_captures_only_c(board, player, rules, &captures)

    if captures.count == 0:
        # No captures — position is quiet
        return alpha

    opp = opponent(player)

    # Search each capture with delta pruning.
    # Delta pruning: if the stand-pat score plus the maximum possible material
    # gain from a capture (sum of captured piece values + safety margin) is
    # still below alpha, skip the capture — it can't improve our position.
    cdef int cap_sq_dp, cap_gain
    cdef int DELTA_MARGIN = 50  # Small safety margin for positional gains

    for i in range(captures.count):
        # Delta prune: estimate maximum material gain from this capture
        cap_gain = 0
        for j in range(captures.moves[i].num_captures):
            cap_sq_dp = captures.moves[i].cap_r[j] * 8 + captures.moves[i].cap_c[j]
            if is_king(board[cap_sq_dp]):
                cap_gain += W_KING
            else:
                cap_gain += W_MAN
        # Add promotion bonus if applicable
        if captures.moves[i].promotion:
            cap_gain += W_KING - W_MAN  # Gaining king value from promotion
        if stand_pat + cap_gain + DELTA_MARGIN <= alpha:
            continue  # This capture can't raise alpha — prune it

        apply_move_c(board, new_board, &captures.moves[i], player)
        child_h = _hash_after_move(h, board, &captures.moves[i], player)

        score = -quiescence(new_board, opp, -beta, -alpha,
                            rules, ss, child_h, qs_depth - 1)

        if ss.timeout:
            return 0.0

        if score >= beta:
            return beta  # Beta cutoff
        if score > alpha:
            alpha = score

    return alpha


# ═══════════════════════════════════════════════════════════════════════
# Alpha-beta search
# ═══════════════════════════════════════════════════════════════════════

cdef struct SearchState:
    double deadline
    int nodes
    bint timeout
    float root_score         # Best score from most recent search_root call
    # Killer moves: 2 per ply (moves that caused beta cutoffs)
    int killers_from[MAX_PLY][NUM_KILLERS]
    int killers_to[MAX_PLY][NUM_KILLERS]
    # History heuristic: history[from_sq][to_sq] — cutoff frequency
    int history[64][64]
    # Countermove heuristic: when opponent plays prev_from→prev_to, the
    # stored response move is a good candidate (caused cutoffs historically).
    int countermove_from[64][64]
    int countermove_to[64][64]

cdef inline void _init_search_tables(SearchState *ss) noexcept nogil:
    """Zero killer, history, and countermove tables for a new search."""
    memset(ss.killers_from, 0xFF, MAX_PLY * NUM_KILLERS * sizeof(int))  # -1
    memset(ss.killers_to, 0xFF, MAX_PLY * NUM_KILLERS * sizeof(int))
    memset(ss.history, 0, 64 * 64 * sizeof(int))
    memset(ss.countermove_from, 0xFF, 64 * 64 * sizeof(int))  # -1
    memset(ss.countermove_to, 0xFF, 64 * 64 * sizeof(int))


cdef inline void _store_killer(SearchState *ss, int ply, CMove *m) noexcept nogil:
    """Store a quiet move that caused beta cutoff as a killer."""
    if ply >= MAX_PLY:
        return
    cdef int from_sq = m.path_r[0] * 8 + m.path_c[0]
    cdef int to_sq = m.path_r[m.path_len - 1] * 8 + m.path_c[m.path_len - 1]
    # Don't store duplicate
    if ss.killers_from[ply][0] == from_sq and ss.killers_to[ply][0] == to_sq:
        return
    # Shift slot 0 to slot 1, insert new at slot 0
    ss.killers_from[ply][1] = ss.killers_from[ply][0]
    ss.killers_to[ply][1] = ss.killers_to[ply][0]
    ss.killers_from[ply][0] = from_sq
    ss.killers_to[ply][0] = to_sq


cdef inline void _update_history(SearchState *ss, CMove *m, int depth) noexcept nogil:
    """Increment history table on cutoff (depth^2 weighting)."""
    cdef int from_sq = m.path_r[0] * 8 + m.path_c[0]
    cdef int to_sq = m.path_r[m.path_len - 1] * 8 + m.path_c[m.path_len - 1]
    cdef int bonus = depth * depth
    ss.history[from_sq][to_sq] += bonus
    if ss.history[from_sq][to_sq] > 10000:
        ss.history[from_sq][to_sq] = 10000


cdef inline void _update_history_malus(SearchState *ss, CMove *m, int depth) noexcept nogil:
    """Decrease history score for a quiet move that failed to cause cutoff.
    Standard complement to _update_history: moves that were searched but didn't
    cut should be ordered lower in future searches."""
    cdef int from_sq = m.path_r[0] * 8 + m.path_c[0]
    cdef int to_sq = m.path_r[m.path_len - 1] * 8 + m.path_c[m.path_len - 1]
    cdef int malus = depth * depth
    ss.history[from_sq][to_sq] -= malus
    if ss.history[from_sq][to_sq] < -10000:
        ss.history[from_sq][to_sq] = -10000


cdef inline void _store_countermove(
    SearchState *ss, int prev_from, int prev_to, CMove *m
) noexcept nogil:
    """Store a response move as countermove for the opponent's previous move.
    On beta cutoff, the current move is a good response to prev_from→prev_to."""
    cdef int from_sq = m.path_r[0] * 8 + m.path_c[0]
    cdef int to_sq = m.path_r[m.path_len - 1] * 8 + m.path_c[m.path_len - 1]
    ss.countermove_from[prev_from][prev_to] = from_sq
    ss.countermove_to[prev_from][prev_to] = to_sq


cdef inline void _age_history(SearchState *ss) noexcept nogil:
    """Halve all history table entries to prevent saturation over long games.
    Called once per move. Without aging, all entries eventually reach the cap
    (10000) and the history heuristic loses its discriminative power."""
    cdef int i, j
    for i in range(64):
        for j in range(64):
            ss.history[i][j] >>= 1  # Right-shift by 1 = halve


cdef inline unsigned long long _hash_after_move(
    unsigned long long h, signed char *board, CMove *m, int player
) noexcept nogil:
    """Compute Zobrist hash after move, incrementally (O(captures) not O(64))."""
    cdef int from_sq = m.path_r[0] * 8 + m.path_c[0]
    cdef int to_sq = m.path_r[m.path_len - 1] * 8 + m.path_c[m.path_len - 1]
    cdef int piece = board[from_sq]
    cdef int i, cap_sq, cap_piece, end_piece

    # Remove piece from start square
    h = h ^ ZOBRIST_PIECES[piece][from_sq]

    # Remove captured pieces
    for i in range(m.num_captures):
        cap_sq = m.cap_r[i] * 8 + m.cap_c[i]
        cap_piece = board[cap_sq]
        h = h ^ ZOBRIST_PIECES[cap_piece][cap_sq]

    # Add piece to end square (with possible promotion)
    end_piece = promote_piece(piece) if m.promotion else piece
    h = h ^ ZOBRIST_PIECES[end_piece][to_sq]

    # Toggle side to move
    h = h ^ ZOBRIST_SIDE

    return h


cdef inline int _count_player_pieces(
    signed char *board, int player
) noexcept nogil:
    """Count pieces for the given player (for NMP zugzwang guard).
    Early-exits once threshold (4) is met — avoids scanning remaining squares."""
    cdef int count = 0, i, piece
    for i in range(NUM_DARK_SQ):
        piece = board[DARK_SQ[i]]
        if piece != EMPTY and is_player(piece, player):
            count += 1
            if count >= 4:
                return count
    return count


cdef void _order_moves_full(
    CMoveList *moves, int ply, SearchState *ss, int tt_from, int tt_to,
    signed char *board, int prev_from, int prev_to
) noexcept nogil:
    """Enhanced move ordering: TT > captures (by value) > killers > countermove > history > position."""
    cdef int i, j, best_idx
    cdef int scores[MAX_MOVES]
    cdef int er, ec, s, from_sq, to_sq, k
    cdef int cap_sq, cap_value
    cdef CMove temp

    for i in range(moves.count):
        from_sq = moves.moves[i].path_r[0] * 8 + moves.moves[i].path_c[0]
        er = moves.moves[i].path_r[moves.moves[i].path_len - 1]
        ec = moves.moves[i].path_c[moves.moves[i].path_len - 1]
        to_sq = er * 8 + ec

        # TT move: absolute highest priority — search this first
        if tt_from >= 0 and from_sq == tt_from and to_sq == tt_to:
            scores[i] = 50000
            continue

        s = 0

        # Captures: highest priority, scored by material value of captured pieces.
        if moves.moves[i].num_captures > 0:
            cap_value = 0
            for j in range(moves.moves[i].num_captures):
                cap_sq = moves.moves[i].cap_r[j] * 8 + moves.moves[i].cap_c[j]
                if is_king(board[cap_sq]):
                    cap_value += W_KING
                else:
                    cap_value += W_MAN
            s = 20000 + cap_value
        else:
            # Killer move bonus
            if ply < MAX_PLY:
                for k in range(NUM_KILLERS):
                    if (ss.killers_from[ply][k] == from_sq and
                            ss.killers_to[ply][k] == to_sq):
                        s = 15000
                        break
            # Countermove bonus: if opponent just played prev_from→prev_to,
            # the stored response that previously caused a cutoff gets priority.
            if s == 0 and prev_from >= 0:
                if (ss.countermove_from[prev_from][prev_to] == from_sq and
                        ss.countermove_to[prev_from][prev_to] == to_sq):
                    s = 12000
            # History heuristic for quiet moves
            if s == 0:
                s = ss.history[from_sq][to_sq]

        # Promotion bonus
        if moves.moves[i].promotion:
            s += 10000

        # Center bias
        s += 7 - <int>(fabs(3.5 - er) + fabs(3.5 - ec))
        scores[i] = s

    # Insertion sort — O(n) best case on nearly-ordered input (common with
    # TT/killer pre-ordering), O(n²) worst case. Better than selection sort
    # for the typical 10-20 move lists in Dama.
    for i in range(1, moves.count):
        s = scores[i]
        temp = moves.moves[i]
        j = i - 1
        while j >= 0 and scores[j] < s:
            scores[j + 1] = scores[j]
            moves.moves[j + 1] = moves.moves[j]
            j -= 1
        scores[j + 1] = s
        moves.moves[j + 1] = temp


cdef float alphabeta(
    signed char *board, int player, int depth, float alpha, float beta,
    Rules *rules, SearchState *ss, unsigned long long h, int ply,
    bint allow_null, int prev_from, int prev_to
) noexcept nogil:
    """Negamax alpha-beta with TT, NMP, PVS, LMR, LMP, IID, countermove, killers, history, and futility pruning."""
    cdef CMoveList moves
    cdef signed char new_board[64]
    cdef float score, best, orig_alpha, static_eval, null_score
    cdef int i, j, opp, tt_flag, reduced, nmp_R, lmp_limit
    cdef unsigned long long child_h, null_h
    cdef bint futility_ok
    cdef int tt_from_sq, tt_to_sq, best_move_idx
    cdef int best_from_sq, best_to_sq
    cdef int mv_from, mv_to  # from/to squares of each move for countermove passing
    # IID variables (Internal Iterative Deepening)
    cdef float _iid_score, _iid_alpha, _iid_beta

    ss.nodes += 1

    if (ss.nodes & 4095) == 0:
        if _check_deadline(ss):
            return 0.0

    if depth == 0:
        return quiescence(board, player, alpha, beta, rules, ss, h, MAX_QS_DEPTH)

    # ── TT probe ──
    orig_alpha = alpha
    tt_from_sq = -1
    tt_to_sq = -1
    if _tt_table != NULL:
        if tt_probe(h, depth, &score, &alpha, &beta, &tt_from_sq, &tt_to_sq):
            return score

    # ── Internal Iterative Deepening (IID) ──
    # When no TT move is available for ordering at depth >= 6, do a shallow
    # search first to populate the TT. The resulting best move dramatically
    # improves ordering for the full-depth search, making PVS and LMR more
    # effective. Cost: one depth-(depth-3) search. Payoff: better move
    # ordering reduces the full-depth tree by 20-40% at these depths.
    if tt_from_sq < 0 and depth >= 6 and not ss.timeout:
        alphabeta(board, player, depth - 3, alpha, beta, rules, ss, h, ply, True, prev_from, prev_to)
        if _tt_table != NULL and not ss.timeout:
            _iid_alpha = -100000.0; _iid_beta = 100000.0
            tt_probe(h, 0, &_iid_score, &_iid_alpha, &_iid_beta,
                     &tt_from_sq, &tt_to_sq)

    moves.count = 0
    generate_all_moves_c(board, player, rules, &moves)

    if moves.count == 0:
        return -10000.0

    opp = opponent(player)

    # ── Null Move Pruning ──
    # In quiet positions (no forced captures), pass the turn and search at
    # reduced depth with a null window around beta. If the opponent can't
    # beat beta even with a free move, our position is strong enough to prune.
    # Guards: depth >= 4 (overhead exceeds savings at shallower depths),
    # not near mate, no captures (quiet position), not after a null move
    # (prevent double-null), >= 4 pieces (avoid zugzwang in endgames).
    # No eval guard: in balanced positions NMP still cuts effectively at depth 4+
    # and the straggler reduction on hard games outweighs the rare wasted search.
    if (allow_null and depth >= 4
            and not (alpha > 9000 or alpha < -9000)
            and moves.moves[0].num_captures == 0
            and _count_player_pieces(board, player) >= 4):
        nmp_R = 2 + depth // 6  # Adaptive reduction: deeper → more aggressive
        null_h = h ^ ZOBRIST_SIDE
        null_score = -alphabeta(board, opp, depth - 1 - nmp_R, -beta, -beta + 1,
                                rules, ss, null_h, ply + 1, False, -1, -1)
        if not ss.timeout and null_score >= beta:
            # Don't trust mate scores from null move
            if null_score >= 9000:
                return beta
            return null_score

    # ── Futility pruning ──
    # At shallow depth (1-2), if the static eval is far below alpha even with
    # an optimistic margin, quiet moves (non-captures, non-promotions) are
    # unlikely to raise the score above alpha. Skip them.
    futility_ok = False
    if depth <= 2 and not (alpha > 9000 or alpha < -9000):
        static_eval = evaluate_c(board, player)
        if static_eval + depth * 100 <= alpha:
            futility_ok = True

    # ── Late Move Pruning (LMP) limit ──
    # At shallow depths, skip late quiet moves entirely. With good move
    # ordering (TT > captures > killers > history), moves beyond the LMP
    # threshold are almost certainly bad. More aggressive than LMR (which
    # just reduces depth). Thresholds chosen conservatively — captures and
    # promotions always searched regardless.
    # depth 1: 6, depth 2: 8, depth 3: 12, depth 4: 16
    if depth <= 4 and not (alpha > 9000 or alpha < -9000):
        lmp_limit = 2 + depth * depth * 2  # 4,10,20,34 → but capped below
        if depth == 1: lmp_limit = 6
        elif depth == 2: lmp_limit = 8
        elif depth == 3: lmp_limit = 12
        elif depth == 4: lmp_limit = 16
    else:
        lmp_limit = 999  # No LMP at deeper depths

    # Enhanced move ordering with killers, countermove, history, and TT best move
    _order_moves_full(&moves, ply, ss, tt_from_sq, tt_to_sq, board, prev_from, prev_to)

    best = -100000.0
    best_move_idx = 0

    for i in range(moves.count):
        # Futility prune: skip quiet moves when static eval + margin <= alpha.
        # Always search the first move (we need at least one legal move result)
        # and always search captures and promotions.
        if (futility_ok and i > 0
                and moves.moves[i].num_captures == 0
                and not moves.moves[i].promotion):
            continue

        # LMP: skip late quiet moves at shallow depths
        if (i >= lmp_limit
                and moves.moves[i].num_captures == 0
                and not moves.moves[i].promotion
                and best > -9000):  # Don't prune if we haven't found any good move yet
            continue

        apply_move_c(board, new_board, &moves.moves[i], player)
        # Incremental hash update (O(captures) not O(64))
        child_h = _hash_after_move(h, board, &moves.moves[i], player)

        # Compute from/to for countermove passing to child
        mv_from = moves.moves[i].path_r[0] * 8 + moves.moves[i].path_c[0]
        mv_to = (moves.moves[i].path_r[moves.moves[i].path_len - 1] * 8
                 + moves.moves[i].path_c[moves.moves[i].path_len - 1])

        if i == 0:
            # First move (expected best): full window, full depth
            score = -alphabeta(new_board, opp, depth - 1, -beta, -alpha,
                               rules, ss, child_h, ply + 1, True, mv_from, mv_to)
        else:
            # Graduated LMR: reduce depth for late quiet moves using precomputed
            # table. Later moves at deeper depths get stronger reductions.
            reduced = 0
            if (i >= 3 and depth >= 3
                    and moves.moves[i].num_captures == 0
                    and not moves.moves[i].promotion):
                reduced = LMR_TABLE[depth][i] if depth < LMR_MAX_D and i < LMR_MAX_M else 1

            # PVS + LMR: scout with null window at (potentially reduced) depth
            score = -alphabeta(new_board, opp, depth - 1 - reduced,
                               -alpha - 1, -alpha,
                               rules, ss, child_h, ply + 1, True, mv_from, mv_to)
            if score > alpha and not ss.timeout:
                # Promising — re-search at full depth, full window
                score = -alphabeta(new_board, opp, depth - 1, -beta, -alpha,
                                   rules, ss, child_h, ply + 1, True, mv_from, mv_to)

        if ss.timeout:
            return 0.0

        if score > best:
            best = score
            best_move_idx = i
        if score > alpha:
            alpha = score
        if alpha >= beta:
            # Beta cutoff — update killer, history, and countermove for quiet moves
            if moves.moves[i].num_captures == 0:
                _store_killer(ss, ply, &moves.moves[i])
                _update_history(ss, &moves.moves[i], depth)
                if prev_from >= 0:
                    _store_countermove(ss, prev_from, prev_to, &moves.moves[i])
            # History malus: penalize previously searched quiet moves that
            # didn't cause a cutoff. Only at depth >= 3 where history ordering
            # matters and the penalty is meaningful.
            if depth >= 3:
                for j in range(i):
                    if (moves.moves[j].num_captures == 0
                            and not moves.moves[j].promotion):
                        _update_history_malus(ss, &moves.moves[j], depth)
            break

    # ── TT store (with best move from/to for future ordering) ──
    if _tt_table != NULL and not ss.timeout:
        if best <= orig_alpha:
            tt_flag = TT_UPPERBOUND
        elif best >= beta:
            tt_flag = TT_LOWERBOUND
        else:
            tt_flag = TT_EXACT
        best_from_sq = moves.moves[best_move_idx].path_r[0] * 8 + moves.moves[best_move_idx].path_c[0]
        best_to_sq = (moves.moves[best_move_idx].path_r[moves.moves[best_move_idx].path_len - 1] * 8
                      + moves.moves[best_move_idx].path_c[moves.moves[best_move_idx].path_len - 1])
        tt_store(h, best, depth, tt_flag, best_from_sq, best_to_sq)

    return best


cdef bint _check_deadline(SearchState *ss) noexcept nogil:
    cdef double now = <double>clock() / <double>CLOCKS_PER_SEC
    if now >= ss.deadline:
        ss.timeout = True
        return True
    return False


cdef int search_root(
    signed char *board, int player, CMoveList *moves, int depth,
    float alpha, float beta,
    Rules *rules, SearchState *ss, unsigned long long h
) noexcept nogil:
    """Search at root level with PVS. Returns index of best move.

    Stores the best score in ss.root_score for aspiration window logic.
    """
    cdef float best_score = -100000.0
    cdef float score
    cdef int best_idx = 0
    cdef int i, opp
    cdef signed char new_board[64]
    cdef unsigned long long child_h
    cdef int mv_from, mv_to

    opp = opponent(player)

    for i in range(moves.count):
        apply_move_c(board, new_board, &moves.moves[i], player)
        child_h = _hash_after_move(h, board, &moves.moves[i], player)
        mv_from = moves.moves[i].path_r[0] * 8 + moves.moves[i].path_c[0]
        mv_to = (moves.moves[i].path_r[moves.moves[i].path_len - 1] * 8
                 + moves.moves[i].path_c[moves.moves[i].path_len - 1])

        if i == 0:
            score = -alphabeta(new_board, opp, depth - 1, -beta, -alpha,
                               rules, ss, child_h, 1, True, mv_from, mv_to)
        else:
            # PVS: null window scout
            score = -alphabeta(new_board, opp, depth - 1, -alpha - 1, -alpha,
                               rules, ss, child_h, 1, True, mv_from, mv_to)
            if score > alpha and score < beta and not ss.timeout:
                score = -alphabeta(new_board, opp, depth - 1, -beta, -alpha,
                                   rules, ss, child_h, 1, True, mv_from, mv_to)

        if ss.timeout:
            ss.root_score = best_score
            return best_idx

        if score > best_score:
            best_score = score
            best_idx = i
        if score > alpha:
            alpha = score
        if alpha >= beta or best_score >= 9000:
            break

    ss.root_score = best_score
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
    cdef int best_idx, best_depth, idx
    cdef int depth

    # Parse difficulty
    if time_budget_override > 0:
        time_budget = time_budget_override
    elif difficulty == 'easy':
        time_budget = 0.2
    elif difficulty == 'hard':
        time_budget = 2.5
    elif difficulty == 'super_hard':
        time_budget = 5.0
    else:
        time_budget = 0.8

    if max_depth_override > 0:
        max_depth = max_depth_override
    elif difficulty == 'easy':
        max_depth = 3
    elif difficulty == 'hard':
        max_depth = 8
    elif difficulty == 'super_hard':
        max_depth = 12
    else:
        max_depth = 5

    cdef unsigned long long h

    _load_board(state, board)
    player = int(state.current_player)
    rules = _load_rules()

    moves.count = 0
    generate_all_moves_c(board, player, &rules, &moves)

    if moves.count == 0:
        return {'move': None, 'score': -10000, 'depth': 0, 'nodes': 0}
    if moves.count == 1:
        return {'move': cmove_to_dict(&moves.moves[0]), 'score': 0, 'depth': 0, 'nodes': 1}

    # Allocate TT on first use. Bump generation counter to logically invalidate
    # all stale entries — avoids a 16MB memset (~1ms) per search call.
    _ensure_tt()
    global _tt_generation
    _tt_generation = (_tt_generation + 1) & TT_GEN_MASK

    # Declare C variables before the nogil block
    cdef int _tt_from = -1, _tt_to = -1
    cdef float _d_score, _d_alpha, _d_beta
    cdef CMove _tmp_move
    cdef float prev_score = 0.0
    cdef float asp_delta, asp_alpha, asp_beta

    best_idx = 0
    best_depth = 0

    # Release the GIL for the entire compute-intensive section.
    # All operations below are pure C — no Python objects touched.
    # This enables true multi-threading when fast_search is called from
    # concurrent threads (e.g. ThreadPoolExecutor in self-play workers).
    # The transposition table (_tt_table) uses lockless sharing — concurrent
    # reads/writes may occasionally corrupt entries, but this is a well-
    # established technique in parallel game-tree search (cf. Stockfish).
    # Corrupted TT entries only reduce search efficiency, never correctness
    # (move generation is deterministic, TT probes have hash verification).
    with nogil:
        # Compute initial Zobrist hash and initialize killer/history tables
        h = compute_hash(board, player)
        _init_search_tables(&ss)

        # Order moves using full heuristic (TT move from prior searches + killers + history)
        if _tt_table != NULL:
            _d_alpha = -100000.0; _d_beta = 100000.0
            tt_probe(h, 0, &_d_score, &_d_alpha, &_d_beta, &_tt_from, &_tt_to)
        _order_moves_full(&moves, 0, &ss, _tt_from, _tt_to, board, -1, -1)

        deadline = <double>clock() / <double>CLOCKS_PER_SEC + time_budget
        ss.deadline = deadline
        ss.nodes = 0
        ss.timeout = False

        # Iterative deepening with aspiration windows.
        # At depth < 5 (easy), the tree is small enough that full-window PVS
        # is faster than the overhead of fail/retry cycles.
        for depth in range(1, max_depth + 1):
            ss.timeout = False

            if depth < 5:
                # Full window for shallow depths
                idx = search_root(board, player, &moves, depth,
                                  -100000.0, 100000.0, &rules, &ss, h)
            else:
                # Aspiration window with progressive widening on fail
                asp_delta = 75.0  # ~3/4 man value
                while True:
                    asp_alpha = prev_score - asp_delta
                    asp_beta = prev_score + asp_delta
                    idx = search_root(board, player, &moves, depth,
                                      asp_alpha, asp_beta, &rules, &ss, h)
                    if ss.timeout:
                        break
                    if ss.root_score > asp_alpha and ss.root_score < asp_beta:
                        break  # Score within window — accept result
                    # Fail-low or fail-high: widen window
                    asp_delta = asp_delta * 2.0
                    if asp_delta >= 5000.0:
                        # Window is wide enough — fall back to full window
                        ss.timeout = False
                        idx = search_root(board, player, &moves, depth,
                                          -100000.0, 100000.0, &rules, &ss, h)
                        break

            if not ss.timeout:
                best_idx = idx
                best_depth = depth
                prev_score = ss.root_score
                # Swap best move to position 0 for next ID iteration
                if best_idx != 0:
                    _tmp_move = moves.moves[0]
                    moves.moves[0] = moves.moves[best_idx]
                    moves.moves[best_idx] = _tmp_move
                    best_idx = 0
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
    cdef int i, r, c, piece
    for i in range(NUM_DARK_SQ):
        piece = board[DARK_SQ[i]]
        if piece == EMPTY:
            continue
        r = DARK_SQ_R[i]
        c = DARK_SQ_C[i]
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
    cdef int player, chosen_idx, apply_idx, move_num, depth, idx, i, j
    cdef double time_budget
    cdef int max_depth, best_idx
    cdef SearchState ss
    cdef int p1_caps = 0, p2_caps = 0
    cdef bint game_over = False
    cdef unsigned long long h
    cdef float _prev_score, _asp_a, _asp_b
    cdef int root_order[MAX_MOVES]  # Maps current pos → original pos for replay
    cdef CMove temp_move
    cdef float dummy_score, dummy_alpha, dummy_beta
    cdef int tt_from_sq, tt_to_sq

    rules = _load_rules()
    init_standard_board(board)
    player = start_player

    # Allocate TT once; bump generation once at game start. During the game,
    # TT entries from prior moves are still useful — positions reached from
    # move N's search overlap with move N+1's search tree. Same generation
    # means entries from prior moves in THIS game are accepted (free hits).
    _ensure_tt()
    global _tt_generation
    _tt_generation = (_tt_generation + 1) & TT_GEN_MASK

    # Compute initial hash and initialize killer/history tables.
    # Killers and history persist across moves within a game — moves that
    # cause cutoffs at ply N tend to be good at the same ply in later positions.
    h = compute_hash(board, player)
    _init_search_tables(&ss)

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

        # Order moves using full heuristic (TT move, killers, history).
        # TT probe extracts best move from prior search of this position
        # (iterative deepening reuses TT entries). Killers/history from prior
        # moves in this game provide additional ordering signal.
        tt_from_sq = -1; tt_to_sq = -1
        if _tt_table != NULL:
            dummy_alpha = -100000.0; dummy_beta = 100000.0
            tt_probe(h, 0, &dummy_score, &dummy_alpha, &dummy_beta,
                     &tt_from_sq, &tt_to_sq)
        _order_moves_full(&moves, 0, &ss, tt_from_sq, tt_to_sq, board, -1, -1)
        # Initialize index mapping (current pos → original pos for replay)
        for i in range(moves.count):
            root_order[i] = i

        # Convert board and moves to Python dicts for replay recording
        state_dict = board_to_compact_dict(board, player, move_num)
        moves_list = []
        for i in range(moves.count):
            moves_list.append(cmove_to_dict(&moves.moves[i]))

        # Choose move: noise (exploration) or alpha-beta search
        # chosen_idx: index into moves_list (original pre-ID-swap order) for replay
        # apply_idx:  index into moves.moves[] (current, possibly swapped order)
        if moves.count == 1:
            chosen_idx = 0
            apply_idx = 0
        elif _rng.random() < noise_prob:
            chosen_idx = _rng.randint(0, moves.count - 1)
            apply_idx = chosen_idx  # no ID swaps in noise path
        else:
            cur_diff = p1_difficulty if player == PLAYER_ONE else p2_difficulty

            if cur_diff == 'easy':
                time_budget = 0.2; max_depth = 3
            elif cur_diff == 'hard':
                time_budget = 2.5; max_depth = 8
            elif cur_diff == 'super_hard':
                time_budget = 5.0; max_depth = 12
            else:
                time_budget = 0.8; max_depth = 5

            # TT is NOT cleared between moves — entries from prior searches are
            # still valid (same game, related positions) and provide free hits.
            # Age history table: halve all entries to prevent long-game saturation.
            # Without aging, entries cap at 10000 and all quiet moves look equally
            # good after ~50 moves. Halving preserves relative ordering while
            # letting recent cutoffs dominate older ones.
            _age_history(&ss)
            ss.deadline = <double>clock() / <double>CLOCKS_PER_SEC + time_budget
            ss.nodes = 0
            ss.timeout = False

            best_idx = 0
            _prev_score = 0.0
            for depth in range(1, max_depth + 1):
                ss.timeout = False
                if depth < 5:
                    idx = search_root(board, player, &moves, depth,
                                      -100000.0, 100000.0, &rules, &ss, h)
                else:
                    _asp_a = _prev_score - 75.0
                    _asp_b = _prev_score + 75.0
                    idx = search_root(board, player, &moves, depth,
                                      _asp_a, _asp_b, &rules, &ss, h)
                    if not ss.timeout and (ss.root_score <= _asp_a or ss.root_score >= _asp_b):
                        ss.timeout = False
                        idx = search_root(board, player, &moves, depth,
                                          -100000.0, 100000.0, &rules, &ss, h)
                if not ss.timeout:
                    best_idx = idx
                    _prev_score = ss.root_score
                    # Swap best move to position 0 for next ID iteration.
                    # PVS searches move 0 with full window — putting the
                    # best move from depth N at position 0 for depth N+1
                    # maximizes PVS effectiveness (the PV move is usually
                    # still best at the next depth).
                    if best_idx != 0:
                        temp_move = moves.moves[0]
                        moves.moves[0] = moves.moves[best_idx]
                        moves.moves[best_idx] = temp_move
                        j = root_order[0]
                        root_order[0] = root_order[best_idx]
                        root_order[best_idx] = j
                        best_idx = 0
                if ss.timeout or _check_deadline(&ss):
                    break
            # chosen_idx: original index for replay entry (maps to moves_list)
            # apply_idx:  actual position in the (swapped) C array
            chosen_idx = root_order[best_idx]
            apply_idx = best_idx

        # Track captures per player (use apply_idx for swapped C array)
        if moves.moves[apply_idx].num_captures > 0:
            if player == PLAYER_ONE:
                p1_caps += moves.moves[apply_idx].num_captures
            else:
                p2_caps += moves.moves[apply_idx].num_captures

        # Record entry (format matches ReplayEntry.to_dict())
        # chosen_idx indexes into moves_list (original pre-swap order)
        entries.append({
            'state': state_dict,
            'legal_moves': moves_list,
            'chosen_index': chosen_idx,
            'result': 0,
            'score': 0.0,
        })

        # Apply move in C (board → new_board, then copy back)
        # Use apply_idx for the actual move in the (possibly swapped) C array
        h = _hash_after_move(h, board, &moves.moves[apply_idx], player)
        apply_move_c(board, new_board, &moves.moves[apply_idx], player)
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
