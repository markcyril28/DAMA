# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython-accelerated board and move encoding for training data preprocessing.

Drop-in replacement for _encode_board_fast / _encode_moves_fast in dataset.py.
Eliminates Python interpreter overhead in the per-entry inner loop.
"""

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

ctypedef np.float32_t DTYPE_f
ctypedef np.int64_t DTYPE_i


def encode_board_fast_cy(dict state_dict, np.ndarray[DTYPE_f, ndim=3] planes):
    """Encode board state directly from compact dict into pre-allocated planes.

    Cython version: ~3-5x faster than pure Python for large batches.
    """
    cdef int turn = state_dict['turn']
    cdef list mapping
    cdef str key
    cdef int plane_idx
    cdef list positions
    cdef int row, col

    if turn == 1:
        mapping = [('p1_men', 0), ('p1_kings', 1), ('p2_men', 2), ('p2_kings', 3)]
    else:
        mapping = [('p2_men', 0), ('p2_kings', 1), ('p1_men', 2), ('p1_kings', 3)]

    # Zero the planes
    planes[:, :, :] = 0.0

    for key, plane_idx in mapping:
        positions = state_dict.get(key, ())
        for pos in positions:
            row = pos[0]
            col = pos[1]
            planes[plane_idx, row, col] = 1.0

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
    """
    cdef int turn = state_dict['turn']
    cdef str king_key
    cdef set king_set
    cdef int n, i
    cdef dict m
    cdef list path, captures
    cdef bint promotion, is_king
    cdef int start_r, start_c, end_r, end_c
    cdef int num_captures
    cdef float cap_ratio

    king_key = 'p1_kings' if turn == 1 else 'p2_kings'
    king_set = {(pos[0], pos[1]) for pos in state_dict.get(king_key, ())}

    n = len(legal_moves)
    if n > out.shape[0]:
        n = out.shape[0]

    for i in range(n):
        m = legal_moves[i]
        path = m['path']
        captures = m.get('captures', ())
        promotion = m.get('promotion', False)
        start_r = path[0][0]
        start_c = path[0][1]
        end_r = path[len(path) - 1][0]
        end_c = path[len(path) - 1][1]
        is_king = (start_r, start_c) in king_set

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
    """
    cdef int n = end_idx - start_idx
    cdef int i, j, turn, plane_idx, num_moves, chosen_idx
    cdef int row, col, start_r, start_c, end_r, end_c, num_captures
    cdef int path_len
    cdef dict state_dict, m_dict
    cdef list positions, path, captures, legal_moves_list, mapping
    cdef str key, king_key
    cdef bint promotion, is_king
    cdef set king_set
    cdef float cap_ratio

    for i in range(n):
        entry = entries[start_idx + i]
        state_dict = entry.state
        turn = state_dict['turn']

        # ── Encode board ──
        if turn == 1:
            mapping = [('p1_men', 0), ('p1_kings', 1), ('p2_men', 2), ('p2_kings', 3)]
        else:
            mapping = [('p2_men', 0), ('p2_kings', 1), ('p1_men', 2), ('p1_kings', 3)]

        # Zero this entry's planes
        boards[i, :, :, :] = 0.0

        for key, plane_idx in mapping:
            positions = state_dict.get(key, ())
            for pos in positions:
                row = pos[0]
                col = pos[1]
                boards[i, plane_idx, row, col] = 1.0
        boards[i, 4, :, :] = 1.0

        # ── Encode moves ──
        king_key = 'p1_kings' if turn == 1 else 'p2_kings'
        king_set = {(pos[0], pos[1]) for pos in state_dict.get(king_key, ())}

        legal_moves_list = entry.legal_moves
        num_moves = len(legal_moves_list)
        if num_moves > max_moves_per_sample:
            num_moves = max_moves_per_sample

        for j in range(num_moves):
            m_dict = legal_moves_list[j]
            path = m_dict['path']
            captures = m_dict.get('captures', ())
            promotion = m_dict.get('promotion', False)
            start_r = path[0][0]
            start_c = path[0][1]
            path_len = len(path)
            end_r = path[path_len - 1][0]
            end_c = path[path_len - 1][1]
            is_king = (start_r, start_c) in king_set

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
