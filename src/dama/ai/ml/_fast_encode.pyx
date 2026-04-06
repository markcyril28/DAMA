# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython-accelerated board and move encoding for training data preprocessing.

Drop-in replacement for _encode_board_fast / _encode_moves_fast in dataset.py.
Eliminates Python interpreter overhead in the per-entry inner loop.

[Pass 82] Uses CPython C API (PyDict_GetItem, PyTuple_GET_ITEM,
PyLong_AsLong) to bypass Python method dispatch and sequence protocol.
Unrolled board-plane encoding eliminates mapping list allocation per entry.
"""

import numpy as np
cimport numpy as np
cimport cython
from cpython.dict cimport PyDict_GetItem
from cpython.tuple cimport PyTuple_GET_ITEM
from cpython.list cimport PyList_GET_ITEM
from cpython.long cimport PyLong_AsLong
from cpython.ref cimport PyObject


# [Pass 83] Safe element accessor for tuples AND lists.
# board_to_compact_dict (Cython) creates tuples, but JSON deserialization
# (replay buffer loading) creates lists.  PyTuple_GET_ITEM on a list
# segfaults (reads list's ob_item pointer as an element — wrong struct layout).
# This helper adds one type check (~1ns) to avoid the crash while keeping
# C-level access (~3ns) instead of Python __getitem__ protocol (~20ns).
cdef inline object _pos_item(object pos, Py_ssize_t i):
    if type(pos) is tuple:
        return <object>PyTuple_GET_ITEM(pos, i)
    return <object>PyList_GET_ITEM(pos, i)

np.import_array()

ctypedef np.float32_t DTYPE_f
ctypedef np.int32_t DTYPE_i

# ── Pre-intern dict key strings ────────────────────────────────────
# PyDict_GetItem uses pointer comparison for interned strings before
# falling back to hash+strcmp.  Module-level allocation avoids per-call
# string creation.
cdef object _K_P1_MEN = 'p1_men'
cdef object _K_P1_KINGS = 'p1_kings'
cdef object _K_P2_MEN = 'p2_men'
cdef object _K_P2_KINGS = 'p2_kings'
cdef object _K_TURN = 'turn'
cdef object _K_PATH = 'path'
cdef object _K_CAPTURES = 'captures'
cdef object _K_PROMOTION = 'promotion'
cdef object _K_STATE = 'state'
cdef object _K_LEGAL_MOVES = 'legal_moves'
cdef object _K_CHOSEN_INDEX = 'chosen_index'
cdef object _K_SCORE = 'score'
cdef object _K_RESULT = 'result'


cdef inline object _dict_get(dict d, object key):
    """Get value from dict, return empty tuple if key missing.
    PyDict_GetItem is ~40% faster than d.get(key, ()) — no method dispatch."""
    cdef PyObject* result = PyDict_GetItem(d, key)
    if result == NULL:
        return ()
    return <object>result


cdef inline object _dict_get_false(dict d, object key):
    """Get value from dict, return False if missing."""
    cdef PyObject* result = PyDict_GetItem(d, key)
    if result == NULL:
        return False
    return <object>result


cdef inline object _dict_get_zero(dict d, object key):
    """Get value from dict, return 0 if missing."""
    cdef PyObject* result = PyDict_GetItem(d, key)
    if result == NULL:
        return 0
    return <object>result


cdef inline object _dict_get_fzero(dict d, object key):
    """Get value from dict, return 0.0 if missing."""
    cdef PyObject* result = PyDict_GetItem(d, key)
    if result == NULL:
        return 0.0
    return <object>result


# ── Inline: encode positions into a board plane ───────────────────

cdef inline void _encode_positions(
    object positions,
    np.ndarray[DTYPE_f, ndim=3] planes,
    int plane_idx,
    bint flip,
):
    """Write 1.0 into planes[plane_idx, row, col] for each position tuple."""
    cdef int row, col
    cdef object pos
    for pos in positions:
        row = PyLong_AsLong(_pos_item(pos, 0))
        if flip:
            row = 7 - row
        col = PyLong_AsLong(_pos_item(pos, 1))
        planes[plane_idx, row, col] = 1.0


# ── Inline: encode positions into a 4D board array at entry i ─────

cdef inline void _encode_positions_4d(
    object positions,
    np.ndarray[DTYPE_f, ndim=4] boards,
    int i,
    int plane_idx,
    bint flip,
):
    """Write 1.0 into boards[i, plane_idx, row, col] for each position tuple."""
    cdef int row, col
    cdef object pos
    for pos in positions:
        row = PyLong_AsLong(_pos_item(pos, 0))
        if flip:
            row = 7 - row
        col = PyLong_AsLong(_pos_item(pos, 1))
        boards[i, plane_idx, row, col] = 1.0


def encode_board_fast_cy(dict state_dict, np.ndarray[DTYPE_f, ndim=3] planes):
    """Encode board state directly from compact dict into pre-allocated planes.

    Cython version: ~3-5x faster than pure Python for large batches.
    Flips rows for P2 so both sides see canonical orientation.

    [Pass 82] Unrolled plane encoding (no mapping list allocation) +
    C API dict/tuple access.
    """
    cdef int turn = state_dict[_K_TURN]
    cdef bint flip = (turn == 2)

    # Zero the planes
    planes[:, :, :] = 0.0

    # Unrolled: eliminates mapping list [('p1_men', 0), ...] allocation per call
    if turn == 1:
        _encode_positions(_dict_get(state_dict, _K_P1_MEN), planes, 0, flip)
        _encode_positions(_dict_get(state_dict, _K_P1_KINGS), planes, 1, flip)
        _encode_positions(_dict_get(state_dict, _K_P2_MEN), planes, 2, flip)
        _encode_positions(_dict_get(state_dict, _K_P2_KINGS), planes, 3, flip)
    else:
        _encode_positions(_dict_get(state_dict, _K_P2_MEN), planes, 0, flip)
        _encode_positions(_dict_get(state_dict, _K_P2_KINGS), planes, 1, flip)
        _encode_positions(_dict_get(state_dict, _K_P1_MEN), planes, 2, flip)
        _encode_positions(_dict_get(state_dict, _K_P1_KINGS), planes, 3, flip)

    # Plane 4: all ones (bias plane)
    planes[4, :, :] = 1.0


def encode_moves_fast_cy(
    dict state_dict,
    list legal_moves,
    np.ndarray[DTYPE_f, ndim=2] out,
) -> int:
    """Encode moves directly from dicts into pre-allocated array.

    Cython version: ~3-5x faster than pure Python for typical move counts.
    Returns the number of valid moves encoded.

    [Pass 82] C API dict/tuple access for ~30-40% improvement.
    """
    cdef int turn = state_dict[_K_TURN]
    cdef int n, i
    cdef dict m
    cdef object path, captures, pos
    cdef bint promotion, is_king, flip
    cdef int start_r, start_c, end_r, end_c
    cdef int num_captures, path_len
    cdef float cap_ratio

    # C-array king lookup (avoids Python set + tuple construction)
    cdef int king_rows[12]
    cdef int king_cols[12]
    cdef int num_kings, k
    cdef object kings_list

    if turn == 1:
        kings_list = _dict_get(state_dict, _K_P1_KINGS)
    else:
        kings_list = _dict_get(state_dict, _K_P2_KINGS)
    num_kings = len(kings_list)
    if num_kings > 12:
        num_kings = 12
    for k in range(num_kings):
        pos = kings_list[k]
        king_rows[k] = PyLong_AsLong(_pos_item(pos, 0))
        king_cols[k] = PyLong_AsLong(_pos_item(pos, 1))
    flip = (turn == 2)

    n = len(legal_moves)
    if n > out.shape[0]:
        n = out.shape[0]

    for i in range(n):
        m = legal_moves[i]
        path = m[_K_PATH]
        captures = _dict_get(m, _K_CAPTURES)
        promotion = _dict_get_false(m, _K_PROMOTION)

        pos = path[0]
        start_r = PyLong_AsLong(_pos_item(pos, 0))
        start_c = PyLong_AsLong(_pos_item(pos, 1))
        path_len = len(path)
        pos = path[path_len - 1]
        end_r = PyLong_AsLong(_pos_item(pos, 0))
        end_c = PyLong_AsLong(_pos_item(pos, 1))

        # C-array scan for king check (no Python set/tuple overhead)
        is_king = False
        for k in range(num_kings):
            if king_rows[k] == start_r and king_cols[k] == start_c:
                is_king = True
                break

        if flip:
            start_r = 7 - start_r
            end_r = 7 - end_r

        out[i, 0] = start_r / 7.0
        out[i, 1] = start_c / 7.0
        out[i, 2] = end_r / 7.0
        out[i, 3] = end_c / 7.0
        out[i, 4] = 1.0 if captures else 0.0
        num_captures = len(captures)
        cap_ratio = num_captures / 4.0
        out[i, 5] = cap_ratio if cap_ratio < 1.0 else 1.0
        out[i, 6] = 1.0 if promotion else 0.0
        out[i, 7] = 1.0 if is_king else 0.0

    return n


def preprocess_chunk_cy(
    list entries,
    int start_idx,
    int end_idx,
    int max_moves_per_sample,
    np.ndarray[DTYPE_f, ndim=4] boards,
    np.ndarray[DTYPE_f, ndim=3] all_move_features,
    np.ndarray[DTYPE_i, ndim=1] move_counts,
    np.ndarray[DTYPE_i, ndim=1] targets,
    np.ndarray[DTYPE_f, ndim=1] scores_arr,
    np.ndarray[DTYPE_f, ndim=1] value_targets,
):
    """Process a chunk of entries into pre-allocated arrays.

    Combined encode_board + encode_moves in a single Cython function
    to minimize Python-to-C transition overhead per entry.

    [Pass 82] Unrolled board encoding + C API dict/tuple access.
    """
    cdef int n = end_idx - start_idx
    cdef int i, j, turn, num_moves, chosen_idx
    cdef int row, col, start_r, start_c, end_r, end_c, num_captures
    cdef int path_len
    cdef dict state_dict, m_dict
    cdef object path, captures, pos
    cdef list legal_moves_list
    cdef bint promotion, is_king, flip
    cdef float cap_ratio

    # C-array king lookup (avoids Python set + tuple per entry)
    cdef int king_rows[12]
    cdef int king_cols[12]
    cdef int num_kings, k
    cdef object kings_list

    # [Pass 72] Bulk board init — single numpy memset + fill for the entire
    # chunk instead of n individual Python-level slice assignments per entry.
    # boards[:n] zeros all 5 planes; then plane 4 gets bias fill.
    boards[:n, :, :, :] = 0.0
    boards[:n, 4, :, :] = 1.0

    for i in range(n):
        entry = entries[start_idx + i]
        state_dict = entry.state
        turn = state_dict[_K_TURN]
        flip = (turn == 2)

        # ── Encode board (unrolled — no mapping list allocation) ──
        if turn == 1:
            _encode_positions_4d(_dict_get(state_dict, _K_P1_MEN), boards, i, 0, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P1_KINGS), boards, i, 1, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P2_MEN), boards, i, 2, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P2_KINGS), boards, i, 3, flip)
        else:
            _encode_positions_4d(_dict_get(state_dict, _K_P2_MEN), boards, i, 0, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P2_KINGS), boards, i, 1, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P1_MEN), boards, i, 2, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P1_KINGS), boards, i, 3, flip)

        # ── Encode moves ── (C-array king lookup)
        if turn == 1:
            kings_list = _dict_get(state_dict, _K_P1_KINGS)
        else:
            kings_list = _dict_get(state_dict, _K_P2_KINGS)
        num_kings = len(kings_list)
        if num_kings > 12:
            num_kings = 12
        for k in range(num_kings):
            pos = kings_list[k]
            king_rows[k] = PyLong_AsLong(_pos_item(pos, 0))
            king_cols[k] = PyLong_AsLong(_pos_item(pos, 1))

        legal_moves_list = entry.legal_moves
        num_moves = len(legal_moves_list)
        if num_moves > max_moves_per_sample:
            num_moves = max_moves_per_sample

        for j in range(num_moves):
            m_dict = legal_moves_list[j]
            path = m_dict[_K_PATH]
            captures = _dict_get(m_dict, _K_CAPTURES)
            promotion = _dict_get_false(m_dict, _K_PROMOTION)

            pos = path[0]
            start_r = PyLong_AsLong(_pos_item(pos, 0))
            start_c = PyLong_AsLong(_pos_item(pos, 1))
            path_len = len(path)
            pos = path[path_len - 1]
            end_r = PyLong_AsLong(_pos_item(pos, 0))
            end_c = PyLong_AsLong(_pos_item(pos, 1))
            is_king = False
            for k in range(num_kings):
                if king_rows[k] == start_r and king_cols[k] == start_c:
                    is_king = True
                    break

            if flip:
                start_r = 7 - start_r
                end_r = 7 - end_r

            all_move_features[i, j, 0] = start_r / 7.0
            all_move_features[i, j, 1] = start_c / 7.0
            all_move_features[i, j, 2] = end_r / 7.0
            all_move_features[i, j, 3] = end_c / 7.0
            all_move_features[i, j, 4] = 1.0 if captures else 0.0
            num_captures = len(captures)
            cap_ratio = num_captures / 4.0
            all_move_features[i, j, 5] = cap_ratio if cap_ratio < 1.0 else 1.0
            all_move_features[i, j, 6] = 1.0 if promotion else 0.0
            all_move_features[i, j, 7] = 1.0 if is_king else 0.0

        move_counts[i] = num_moves
        chosen_idx = entry.chosen_index
        if num_moves > 0:
            targets[i] = chosen_idx if chosen_idx < num_moves else num_moves - 1
        else:
            targets[i] = 0
        scores_arr[i] = entry.score
        value_targets[i] = <float>entry.result


def preprocess_dicts_chunk_cy(
    list entry_dicts,
    int start_idx,
    int end_idx,
    int max_moves_per_sample,
    np.ndarray[DTYPE_f, ndim=4] boards,
    np.ndarray[DTYPE_f, ndim=3] all_move_features,
    np.ndarray[DTYPE_i, ndim=1] move_counts,
    np.ndarray[DTYPE_i, ndim=1] targets,
    np.ndarray[DTYPE_f, ndim=1] scores_arr,
    np.ndarray[DTYPE_f, ndim=1] value_targets,
):
    """Process a chunk of dict entries into pre-allocated arrays.

    Same as preprocess_chunk_cy but uses dict key access (ed['state'])
    instead of attribute access (entry.state).  Used by _preprocess_chunk,
    _preprocess_dicts_fork_shm, and inline preprocessing paths where
    entries are raw dicts from self-play workers.

    [Pass 82] Unrolled board encoding + C API dict/tuple access.
    """
    cdef int n = end_idx - start_idx
    cdef int i, j, turn, num_moves, chosen_idx
    cdef int row, col, start_r, start_c, end_r, end_c, num_captures
    cdef int path_len
    cdef dict state_dict, m_dict, ed
    cdef object path, captures, pos, positions
    cdef list legal_moves_list
    cdef bint promotion, is_king, flip
    cdef float cap_ratio

    # C-array king lookup (avoids Python set + tuple per entry)
    cdef int king_rows[12]
    cdef int king_cols[12]
    cdef int num_kings, k
    cdef object kings_list

    # [Pass 72] Bulk board init — single numpy memset + fill for the entire
    # chunk instead of n individual Python-level slice assignments per entry.
    boards[:n, :, :, :] = 0.0
    boards[:n, 4, :, :] = 1.0

    for i in range(n):
        ed = entry_dicts[start_idx + i]
        state_dict = ed[_K_STATE]
        turn = state_dict[_K_TURN]
        flip = (turn == 2)

        # ── Encode board (unrolled — no mapping list allocation) ──
        if turn == 1:
            _encode_positions_4d(_dict_get(state_dict, _K_P1_MEN), boards, i, 0, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P1_KINGS), boards, i, 1, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P2_MEN), boards, i, 2, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P2_KINGS), boards, i, 3, flip)
        else:
            _encode_positions_4d(_dict_get(state_dict, _K_P2_MEN), boards, i, 0, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P2_KINGS), boards, i, 1, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P1_MEN), boards, i, 2, flip)
            _encode_positions_4d(_dict_get(state_dict, _K_P1_KINGS), boards, i, 3, flip)

        # ── Encode moves ── (C-array king lookup)
        if turn == 1:
            kings_list = _dict_get(state_dict, _K_P1_KINGS)
        else:
            kings_list = _dict_get(state_dict, _K_P2_KINGS)
        num_kings = len(kings_list)
        if num_kings > 12:
            num_kings = 12
        for k in range(num_kings):
            pos = kings_list[k]
            king_rows[k] = PyLong_AsLong(_pos_item(pos, 0))
            king_cols[k] = PyLong_AsLong(_pos_item(pos, 1))

        legal_moves_list = ed[_K_LEGAL_MOVES]
        num_moves = len(legal_moves_list)
        if num_moves > max_moves_per_sample:
            num_moves = max_moves_per_sample

        for j in range(num_moves):
            m_dict = legal_moves_list[j]
            path = m_dict[_K_PATH]
            captures = _dict_get(m_dict, _K_CAPTURES)
            promotion = _dict_get_false(m_dict, _K_PROMOTION)

            pos = path[0]
            start_r = PyLong_AsLong(_pos_item(pos, 0))
            start_c = PyLong_AsLong(_pos_item(pos, 1))
            path_len = len(path)
            pos = path[path_len - 1]
            end_r = PyLong_AsLong(_pos_item(pos, 0))
            end_c = PyLong_AsLong(_pos_item(pos, 1))
            is_king = False
            for k in range(num_kings):
                if king_rows[k] == start_r and king_cols[k] == start_c:
                    is_king = True
                    break

            if flip:
                start_r = 7 - start_r
                end_r = 7 - end_r

            all_move_features[i, j, 0] = start_r / 7.0
            all_move_features[i, j, 1] = start_c / 7.0
            all_move_features[i, j, 2] = end_r / 7.0
            all_move_features[i, j, 3] = end_c / 7.0
            all_move_features[i, j, 4] = 1.0 if captures else 0.0
            num_captures = len(captures)
            cap_ratio = num_captures / 4.0
            all_move_features[i, j, 5] = cap_ratio if cap_ratio < 1.0 else 1.0
            all_move_features[i, j, 6] = 1.0 if promotion else 0.0
            all_move_features[i, j, 7] = 1.0 if is_king else 0.0

        move_counts[i] = num_moves
        chosen_idx = ed[_K_CHOSEN_INDEX]
        if num_moves > 0:
            targets[i] = chosen_idx if chosen_idx < num_moves else num_moves - 1
        else:
            targets[i] = 0
        scores_arr[i] = _dict_get_fzero(ed, _K_SCORE)
        value_targets[i] = <float>_dict_get_zero(ed, _K_RESULT)
