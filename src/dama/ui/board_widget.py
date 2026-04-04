"""Board widget for rendering the game board."""

from typing import Optional, List, Set, Tuple
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QMouseEvent, QPaintEvent, QResizeEvent, QPixmap

from ..types import Move, Player, Position, Piece, PieceType
from ..game_state import GameState
from ..board import Board
from ..config import get_config


class BoardWidget(QWidget):
    """Widget for displaying and interacting with the game board."""

    # Signal emitted when a move is made
    move_made = pyqtSignal(object)  # Move object

    def __init__(self, parent=None):
        super().__init__(parent)

        self.state: Optional[GameState] = None
        self.legal_moves: List[Move] = []

        # Selection state
        self.selected_pos: Optional[Position] = None
        self.highlighted_positions: Set[Position] = set()
        self.highlighted_path: List[Position] = []

        # Board perspective (False = Player 1 at bottom, True = Player 2 at bottom)
        self.flipped = False

        # Visual settings
        self._update_colors()

        # Cached board pixmap
        self._board_cache_pixmap: Optional[QPixmap] = None
        self._board_cache_key: Optional[Tuple[int, int, str, str]] = None

        # Widget setup
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        # Interaction state
        self.interactive = True
        self.hover_pos: Optional[Position] = None

    def _update_colors(self) -> None:
        """Update colors from config."""
        config = get_config()
        self.light_color = QColor(config.ui.board.light_color)
        self.dark_color = QColor(config.ui.board.dark_color)
        self.highlight_color = QColor(config.ui.board.highlight_color)
        self.p1_color = QColor(config.ui.pieces.p1_color)
        self.p2_color = QColor(config.ui.pieces.p2_color)
        self.piece_style = config.ui.pieces.style

        # Cached brushes/pens
        self._light_brush = QBrush(self.light_color)
        self._dark_brush = QBrush(self.dark_color)
        self._highlight_brush = QBrush(self.highlight_color)
        self._p1_fill_brush = QBrush(self.p1_color)
        self._p2_fill_brush = QBrush(self.p2_color)
        self._p1_outline_pen = QPen(QColor(0, 0, 0))
        self._p1_outline_pen.setWidth(2)
        self._p2_outline_pen = QPen(QColor(255, 255, 255))
        self._p2_outline_pen.setWidth(2)

        self._p1_ring_pen = QPen(self.p1_color)
        self._p1_ring_pen.setWidth(3)
        self._p2_ring_pen = QPen(self.p2_color)
        self._p2_ring_pen.setWidth(3)

        self._hover_pen = QPen(QColor(100, 100, 100, 100))
        self._hover_pen.setWidth(2)

        self._invalidate_board_cache()

    def _invalidate_board_cache(self) -> None:
        """Invalidate the cached board pixmap."""
        self._board_cache_pixmap = None
        self._board_cache_key = None

    def _ensure_board_cache(self) -> None:
        """Ensure the board pixmap cache is valid."""
        key = (self.width(), self.height(), self.light_color.name(), self.dark_color.name())
        if self._board_cache_pixmap is not None and self._board_cache_key == key:
            return

        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        square_size = self._get_square_size()

        for row in range(Board.SIZE):
            for col in range(Board.SIZE):
                rect = self._square_to_rect((row, col))
                brush = self._dark_brush if Board.is_playable(row, col) else self._light_brush
                painter.fillRect(rect, brush)

        painter.end()

        self._board_cache_pixmap = pixmap
        self._board_cache_key = key

    def refresh_colors(self) -> None:
        """Refresh colors and repaint."""
        self._update_colors()
        self.update()

    def flip_board(self) -> None:
        """Toggle the board perspective."""
        self.flipped = not self.flipped
        self.update()

    def set_flipped(self, flipped: bool) -> None:
        """Set the board perspective."""
        self.flipped = flipped
        self.update()

    def is_flipped(self) -> bool:
        """Return whether the board is flipped."""
        return self.flipped

    def set_state(self, state: GameState) -> None:
        """Set the game state to display."""
        self.state = state
        self.legal_moves = state.legal_moves() if state else []
        self.clear_selection()
        self.update()

    def set_interactive(self, interactive: bool) -> None:
        """Set whether the board accepts user input."""
        self.interactive = interactive
        if not interactive:
            self.clear_selection()

    def clear_selection(self) -> None:
        """Clear the current selection."""
        self.selected_pos = None
        self.highlighted_positions.clear()
        self.highlighted_path.clear()
        self.update()

    def _get_square_size(self) -> float:
        """Get the size of each square."""
        return min(self.width(), self.height()) / Board.SIZE

    def _get_board_offset(self) -> Tuple[float, float]:
        """Get the offset to center the board."""
        square_size = self._get_square_size()
        board_size = square_size * Board.SIZE
        x_offset = (self.width() - board_size) / 2
        y_offset = (self.height() - board_size) / 2
        return x_offset, y_offset

    def _pos_to_square(self, x: float, y: float) -> Optional[Position]:
        """Convert widget coordinates to board position."""
        square_size = self._get_square_size()
        x_offset, y_offset = self._get_board_offset()

        col = int((x - x_offset) / square_size)
        row = int((y - y_offset) / square_size)

        if 0 <= row < Board.SIZE and 0 <= col < Board.SIZE:
            # Flip coordinates if board is flipped
            if self.flipped:
                row = Board.SIZE - 1 - row
                col = Board.SIZE - 1 - col
            return (row, col)
        return None

    def _square_to_rect(self, pos: Position) -> QRectF:
        """Convert board position to widget rectangle."""
        square_size = self._get_square_size()
        x_offset, y_offset = self._get_board_offset()

        row, col = pos
        
        # Flip coordinates if board is flipped
        if self.flipped:
            row = Board.SIZE - 1 - row
            col = Board.SIZE - 1 - col
        
        x = x_offset + col * square_size
        y = y_offset + row * square_size

        return QRectF(x, y, square_size, square_size)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the board."""
        self._ensure_board_cache()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        square_size = self._get_square_size()

        if self._board_cache_pixmap is not None:
            painter.drawPixmap(0, 0, self._board_cache_pixmap)

        # Draw highlights
        for pos in self.highlighted_positions:
            rect = self._square_to_rect(pos)
            painter.fillRect(rect, self._highlight_brush)

        # Draw selected square
        if self.selected_pos:
            rect = self._square_to_rect(self.selected_pos)
            highlight = QColor(self.highlight_color)
            highlight.setAlpha(180)
            painter.fillRect(rect, highlight)

        # Draw pieces
        if self.state:
            for pos, piece in self.state.board.get_pieces():
                self._draw_piece(painter, pos, piece, square_size)

        # Draw hover indicator
        if self.hover_pos and self.interactive:
            rect = self._square_to_rect(self.hover_pos)
            painter.setPen(self._hover_pen)
            painter.drawRect(rect)

        painter.end()

    def _draw_piece(self, painter: QPainter, pos: Position, piece: Piece, square_size: float) -> None:
        """Draw a piece at a position."""
        rect = self._square_to_rect(pos)
        center = rect.center()
        radius = square_size * 0.35

        # Choose color
        if piece.player == Player.ONE:
            fill_brush = self._p1_fill_brush
            outline_pen = self._p1_outline_pen
            ring_pen = self._p1_ring_pen
        else:
            fill_brush = self._p2_fill_brush
            outline_pen = self._p2_outline_pen
            ring_pen = self._p2_ring_pen

        # Draw based on style
        if self.piece_style == "outlined":
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.setPen(ring_pen)
        else:  # flat
            painter.setBrush(fill_brush)
            painter.setPen(outline_pen)

        painter.drawEllipse(center, radius, radius)

        # Draw king indicator
        if piece.is_king:
            crown_radius = radius * 0.5
            painter.setBrush(QBrush(outline_pen.color()))
            painter.setPen(QPen(fill_brush.color()))
            painter.drawEllipse(center, crown_radius, crown_radius)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        if not self.interactive or not self.state:
            return

        # Get mouse position (compatible with PyQt6)
        mouse_pos = event.position()
        pos = self._pos_to_square(mouse_pos.x(), mouse_pos.y())
        if pos is None:
            return

        # If clicking on a highlighted destination, make the move
        if pos in self.highlighted_positions and self.selected_pos:
            # Find the move that ends at this position
            for move in self.legal_moves:
                if move.start == self.selected_pos and move.end == pos:
                    self.move_made.emit(move)
                    self.clear_selection()
                    return

        # If clicking on own piece, select it
        piece = self.state.board.get_piece(pos)
        if piece and piece.player == self.state.current_player:
            self.selected_pos = pos

            # Find all legal destinations from this piece
            self.highlighted_positions.clear()
            for move in self.legal_moves:
                if move.start == pos:
                    self.highlighted_positions.add(move.end)

            self.update()
        else:
            # Clicking elsewhere clears selection
            self.clear_selection()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for hover effects."""
        if not self.interactive:
            return

        # Get mouse position (compatible with PyQt6)
        mouse_pos = event.position()
        pos = self._pos_to_square(mouse_pos.x(), mouse_pos.y())
        if pos != self.hover_pos:
            self.hover_pos = pos
            self.update()

    def leaveEvent(self, event) -> None:
        """Handle mouse leaving the widget."""
        self.hover_pos = None
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle resize."""
        super().resizeEvent(event)
        self._invalidate_board_cache()
        self.update()
