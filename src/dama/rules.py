"""Game rules constants for Filipino Dama."""

# Board dimensions
BOARD_SIZE = 8

# Starting rows for each player
PLAYER_ONE_ROWS = range(0, 3)  # Rows 0, 1, 2
PLAYER_TWO_ROWS = range(5, 8)  # Rows 5, 6, 7

# Movement directions
# Player 1 moves downward (increasing row)
# Player 2 moves upward (decreasing row)
PLAYER_ONE_FORWARD = 1   # +1 row
PLAYER_TWO_FORWARD = -1  # -1 row

# Diagonal directions for moves
# (row_delta, col_delta)
FORWARD_DIRECTIONS_P1 = [(1, -1), (1, 1)]   # Down-left, Down-right
FORWARD_DIRECTIONS_P2 = [(-1, -1), (-1, 1)] # Up-left, Up-right
BACKWARD_DIRECTIONS_P1 = [(-1, -1), (-1, 1)]  # Up-left, Up-right (backward for P1)
BACKWARD_DIRECTIONS_P2 = [(1, -1), (1, 1)]    # Down-left, Down-right (backward for P2)
ALL_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # For kings

# Rule flags (configurable in future versions)
FORCED_CAPTURE = True
MULTI_JUMP = True
MAXIMUM_CAPTURE_REQUIRED = False  # Not enforced in this version
BACKWARD_CAPTURE = False  # Allow regular pieces to capture backwards
KING_FLYING_CAPTURE = True  # Kings can capture pieces at any distance along diagonal

# Promotion
# Player 1 promotes on row 7
# Player 2 promotes on row 0
PROMOTION_ROW_P1 = 7
PROMOTION_ROW_P2 = 0
