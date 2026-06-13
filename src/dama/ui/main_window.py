"""Main window for Filipino Dama."""

import sys
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QMenuBar, QMenu, QStatusBar, QMessageBox, QPushButton,
    QDockWidget, QTabWidget, QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction

from ..game_state import GameState
from ..types import Move, Player
from ..engine import Engine, PlayerType, GameResult
from ..config import get_config, save_config
from .board_widget import BoardWidget
from .settings_dialog import SettingsDialog
from .training_panel import TrainingPanel
from .ai_settings_sidebar import AISettingsSidebar


class AIWorker(QThread):
    """Worker thread for AI move computation."""

    move_ready = pyqtSignal(object)  # Move
    error = pyqtSignal(str)

    def __init__(self, engine: Engine, player_type: PlayerType):
        super().__init__()
        self.engine = engine
        self.player_type = player_type

    def run(self):
        try:
            move = self.engine.get_ai_move(self.player_type)
            if move:
                self.move_ready.emit(move)
            else:
                self.error.emit("AI could not find a valid move")
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Filipino Dama")
        self.setMinimumSize(600, 650)

        # Game engine
        self.engine = Engine()
        self.engine.on_state_changed = self._on_state_changed
        self.engine.on_game_over = self._on_game_over
        self.engine.on_move_request = self._on_move_request

        # AI worker
        self.ai_worker: Optional[AIWorker] = None
        # True while an AI worker is thinking; covers the gap between the
        # worker's completion signal and isRunning() turning false
        self._ai_thinking = False
        
        # Self-play control
        self.self_play_paused = False
        self.ai_move_delay_ms = 500  # Delay between AI moves for visibility

        # Initialize UI
        self._init_ui()
        self._init_ai_sidebar()
        self._init_menu()
        self._init_status_bar()

        # Load player types from config
        self._load_player_types()

        # Start new game
        self.engine.new_game()

    def _init_ui(self) -> None:
        """Initialize the main UI."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # Info bar
        info_layout = QHBoxLayout()

        self.turn_label = QLabel("Turn: White")
        self.turn_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        info_layout.addWidget(self.turn_label)

        info_layout.addStretch()

        # Opponent selection
        opponent_layout = QHBoxLayout()
        
        # White player (Player 1) label and combo
        white_label = QLabel("⚪ White:")
        white_label.setStyleSheet("font-weight: bold;")
        opponent_layout.addWidget(white_label)
        
        self.player1_type_combo = QComboBox()
        self.player1_type_combo.addItems(["Human", "Algorithm", "AI Model"])
        self.player1_type_combo.setToolTip("Select White player type (starts first, moves down)")
        self.player1_type_combo.currentTextChanged.connect(self._on_player1_type_changed)
        opponent_layout.addWidget(self.player1_type_combo)
        
        vs_label = QLabel(" vs ")
        vs_label.setStyleSheet("font-weight: bold;")
        opponent_layout.addWidget(vs_label)
        
        # Black player (Player 2) label and combo
        black_label = QLabel("⚫ Black:")
        black_label.setStyleSheet("font-weight: bold;")
        opponent_layout.addWidget(black_label)
        
        self.opponent_combo = QComboBox()
        self.opponent_combo.addItems(["Human", "Algorithm", "AI Model"])
        self.opponent_combo.setToolTip("Select Black player type (moves up)")
        self.opponent_combo.currentTextChanged.connect(self._on_opponent_changed)
        opponent_layout.addWidget(self.opponent_combo)
        
        info_layout.addLayout(opponent_layout)

        layout.addLayout(info_layout)

        # Board widget
        self.board_widget = BoardWidget()
        self.board_widget.move_made.connect(self._on_human_move)
        layout.addWidget(self.board_widget, stretch=1)

        # Control buttons
        btn_layout = QHBoxLayout()

        new_game_btn = QPushButton("New Game")
        new_game_btn.clicked.connect(self._new_game)
        btn_layout.addWidget(new_game_btn)

        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self._undo)
        btn_layout.addWidget(undo_btn)

        self.flip_btn = QPushButton("Flip Board")
        self.flip_btn.setToolTip("Switch board perspective (view from other player's side)")
        self.flip_btn.clicked.connect(self._flip_board)
        btn_layout.addWidget(self.flip_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setToolTip("Pause/Resume AI self-play")
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.setEnabled(False)  # Enabled when AI vs AI
        btn_layout.addWidget(self.pause_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)

    def _init_ai_sidebar(self) -> None:
        """Initialize the AI Settings sidebar dock."""
        self.ai_sidebar = AISettingsSidebar()
        self.ai_sidebar_dock = QDockWidget("AI Settings", self)
        self.ai_sidebar_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.ai_sidebar_dock.setWidget(self.ai_sidebar)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.ai_sidebar_dock)

        self.ai_sidebar.settings_changed.connect(self._on_settings_changed)
        self.ai_sidebar.open_full_settings_requested.connect(self._show_settings)

    def _init_menu(self) -> None:
        """Initialize the menu bar."""
        menubar = self.menuBar()

        # Game menu
        game_menu = menubar.addMenu("&Game")

        new_action = QAction("&New Game", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_game)
        game_menu.addAction(new_action)

        game_menu.addSeparator()

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._undo)
        game_menu.addAction(undo_action)

        game_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        game_menu.addAction(quit_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")

        preferences_action = QAction("&Preferences...", self)
        preferences_action.triggered.connect(self._show_settings)
        settings_menu.addAction(preferences_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        flip_action = QAction("&Flip Board", self)
        flip_action.setShortcut("F")
        flip_action.triggered.connect(self._flip_board)
        view_menu.addAction(flip_action)

        view_menu.addSeparator()

        ai_sidebar_action = self.ai_sidebar_dock.toggleViewAction()
        ai_sidebar_action.setText("&AI Settings Sidebar")
        ai_sidebar_action.setShortcut("Ctrl+Shift+A")
        view_menu.addAction(ai_sidebar_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        rules_action = QAction("&Rules", self)
        rules_action.triggered.connect(self._show_rules)
        help_menu.addAction(rules_action)

        # Training menu
        training_menu = menubar.addMenu("&Training")

        training_panel_action = QAction("&Training Panel...", self)
        training_panel_action.triggered.connect(self._show_training_panel)
        training_menu.addAction(training_panel_action)
        
        training_menu.addSeparator()
        
        # Quick setup for AI vs Algorithm test
        self.ai_vs_algo_action = QAction("Watch: AI Model vs Algorithm", self)
        self.ai_vs_algo_action.setToolTip("Set up a game with AI Model (Player 1) vs Algorithm (Player 2)")
        self.ai_vs_algo_action.triggered.connect(self._setup_ai_vs_algo)
        training_menu.addAction(self.ai_vs_algo_action)
        
        self.algo_vs_ai_action = QAction("Watch: Algorithm vs AI Model", self)
        self.algo_vs_ai_action.setToolTip("Set up a game with Algorithm (Player 1) vs AI Model (Player 2)")
        self.algo_vs_ai_action.triggered.connect(self._setup_algo_vs_ai)
        training_menu.addAction(self.algo_vs_ai_action)

    def _init_status_bar(self) -> None:
        """Initialize the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _load_player_types(self) -> None:
        """Load player types from config."""
        config = get_config()
        self.engine.set_player_type(Player.ONE, PlayerType(config.players.p1_type))
        self.engine.set_player_type(Player.TWO, PlayerType(config.players.p2_type))
        
        # Update combo boxes to match config (without triggering signals)
        self.player1_type_combo.blockSignals(True)
        self.opponent_combo.blockSignals(True)
        
        p1_display = self._type_to_display(config.players.p1_type)
        p2_display = self._type_to_display(config.players.p2_type)
        self.player1_type_combo.setCurrentText(p1_display)
        self.opponent_combo.setCurrentText(p2_display)
        
        self.player1_type_combo.blockSignals(False)
        self.opponent_combo.blockSignals(False)

    def _type_to_display(self, type_str: str) -> str:
        """Convert internal type string to display string."""
        mapping = {"human": "Human", "algorithmic": "Algorithm", "ml": "AI Model"}
        return mapping.get(type_str, "Human")
    
    def _display_to_type(self, display_str: str) -> str:
        """Convert display string to internal type string."""
        mapping = {"Human": "human", "Algorithm": "algorithmic", "AI Model": "ml"}
        return mapping.get(display_str, "human")
    
    def _on_player1_type_changed(self, text: str) -> None:
        """Handle White (Player 1) type selection change."""
        type_str = self._display_to_type(text)
        self.engine.set_player_type(Player.ONE, PlayerType(type_str))
        
        # Update config
        config = get_config()
        config.players.p1_type = type_str
        save_config()
        
        self.status_bar.showMessage(f"White set to {text}")
        
        # Update pause button and refresh game state
        self._update_pause_button()
        self._refresh_current_turn()
    
    def _on_opponent_changed(self, text: str) -> None:
        """Handle Black (Player 2) type selection change."""
        type_str = self._display_to_type(text)
        self.engine.set_player_type(Player.TWO, PlayerType(type_str))
        
        # Update config
        config = get_config()
        config.players.p2_type = type_str
        save_config()
        
        self.status_bar.showMessage(f"Black set to {text}")
        
        # Update pause button and refresh game state
        self._update_pause_button()
        self._refresh_current_turn()
    
    def _refresh_current_turn(self) -> None:
        """Refresh the current turn to trigger AI if needed."""
        state = self.engine.state
        if state and not state.is_terminal():
            current_type = self.engine.get_current_player_type()
            self.board_widget.set_interactive(current_type == PlayerType.HUMAN)
            
            # If it's an AI's turn, trigger move request
            if current_type != PlayerType.HUMAN:
                current_player = state.current_player
                self._on_move_request(current_player, current_type)

    def _new_game(self) -> None:
        """Start a new game."""
        # Cancel any running AI
        if self.ai_worker and self.ai_worker.isRunning():
            self.ai_worker.terminate()
            self.ai_worker.wait()
        self._ai_thinking = False

        # Reset pause state
        self.self_play_paused = False
        self._update_pause_button()

        self.engine.new_game()
        self.status_bar.showMessage("New game started")

    def _undo(self) -> None:
        """Undo the last move."""
        if self.engine.undo():
            self.status_bar.showMessage("Move undone")
        else:
            self.status_bar.showMessage("Nothing to undo")

    def _toggle_pause(self) -> None:
        """Toggle pause state for AI self-play."""
        self.self_play_paused = not self.self_play_paused
        self._update_pause_button()
        
        if self.self_play_paused:
            self.status_bar.showMessage("Self-play paused")
        else:
            self.status_bar.showMessage("Self-play resumed")
            # Resume AI play if it's an AI's turn
            self._refresh_current_turn()
    
    def _update_pause_button(self) -> None:
        """Update the pause button state based on player types."""
        p1_type = self.engine.get_player_type(Player.ONE)
        p2_type = self.engine.get_player_type(Player.TWO)
        is_self_play = (p1_type != PlayerType.HUMAN and p2_type != PlayerType.HUMAN)
        
        self.pause_btn.setEnabled(is_self_play)
        self.pause_btn.setText("Resume" if self.self_play_paused else "Pause")

    def _flip_board(self) -> None:
        """Flip the board perspective."""
        self.board_widget.flip_board()
        perspective = "Black" if self.board_widget.is_flipped() else "White"
        self.status_bar.showMessage(f"Board flipped - viewing from {perspective}'s perspective")

    def _on_state_changed(self, state: GameState) -> None:
        """Handle game state changes."""
        self.board_widget.set_state(state)
        color_name = "White" if state.current_player == Player.ONE else "Black"
        self.turn_label.setText(f"Turn: {color_name}")

        # Update interactivity
        current_type = self.engine.get_current_player_type()
        self.board_widget.set_interactive(current_type == PlayerType.HUMAN)

    def _on_game_over(self, result: GameResult) -> None:
        """Handle game over."""
        if result.winner:
            winner_color = "White" if result.winner == Player.ONE else "Black"
            message = f"{winner_color} wins!"
        else:
            message = "Game ended in a draw!"

        message += f"\n\nTotal moves: {result.total_moves}"

        self.status_bar.showMessage(message.split("\n")[0])

        QMessageBox.information(self, "Game Over", message)

    def _on_move_request(self, player: Player, player_type: PlayerType) -> None:
        """Handle AI move request."""
        # Don't spawn a second worker while one is still thinking
        # (e.g. _refresh_current_turn firing during an in-flight think)
        if self._ai_thinking or (self.ai_worker is not None and self.ai_worker.isRunning()):
            return

        color_name = "White" if player == Player.ONE else "Black"
        # Check if paused
        if self.self_play_paused:
            self.status_bar.showMessage(f"Paused - {color_name} ({player_type.value}) waiting...")
            return
        
        self.status_bar.showMessage(f"{color_name} ({player_type.value}) is thinking...")
        
        # Update pause button state
        self._update_pause_button()

        # Start AI worker thread
        self._ai_thinking = True
        self.ai_worker = AIWorker(self.engine, player_type)
        self.ai_worker.move_ready.connect(self._on_ai_move_ready)
        self.ai_worker.error.connect(self._on_ai_error)
        self.ai_worker.start()

    def _on_human_move(self, move: Move) -> None:
        """Handle move from human player."""
        if self.engine.make_move(move):
            self.status_bar.showMessage(f"Move: {move}")
        else:
            self.status_bar.showMessage("Invalid move")

    def _on_ai_move_ready(self, move: Move) -> None:
        """Handle move from AI."""
        self._ai_thinking = False
        # Check if both players are AI (self-play mode)
        p1_type = self.engine.get_player_type(Player.ONE)
        p2_type = self.engine.get_player_type(Player.TWO)
        is_self_play = (p1_type != PlayerType.HUMAN and p2_type != PlayerType.HUMAN)
        
        if is_self_play and self.ai_move_delay_ms > 0:
            # Delay the move to allow user to see the game progress
            QTimer.singleShot(self.ai_move_delay_ms, lambda: self._apply_ai_move(move))
        else:
            self._apply_ai_move(move)
    
    def _apply_ai_move(self, move: Move) -> None:
        """Apply the AI move to the game."""
        if self.engine.make_move(move):
            self.status_bar.showMessage(f"AI played: {move}")
        else:
            self.status_bar.showMessage("AI made invalid move")

    def _on_ai_error(self, error: str) -> None:
        """Handle AI error."""
        self._ai_thinking = False
        self.status_bar.showMessage(f"AI error: {error}")
        QMessageBox.warning(self, "AI Error", error)

    def _show_settings(self) -> None:
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.exec()

    def _on_settings_changed(self) -> None:
        """Handle settings change."""
        self._load_player_types()
        self.board_widget.refresh_colors()
        self.ai_sidebar.refresh_from_config()
        self.status_bar.showMessage("Settings updated")
        
        # Refresh current turn in case player types changed
        self._refresh_current_turn()

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Filipino Dama",
            "<h2>Filipino Dama</h2>"
            "<p>Version 0.1.0</p>"
            "<p>A traditional Filipino checkers game with AI opponents.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Human vs Human gameplay</li>"
            "<li>Algorithmic AI (minimax with alpha-beta)</li>"
            "<li>ML-based AI opponent</li>"
            "<li>Customizable appearance</li>"
            "</ul>"
        )

    def _show_rules(self) -> None:
        """Show rules dialog."""
        rules = """
<h2>Filipino Dama Rules</h2>

<h3>Setup</h3>
<p>Each player starts with 12 pieces on the dark squares of their first three rows.</p>

<h3>Movement</h3>
<ul>
<li><b>Men</b> move diagonally forward by one square.</li>
<li><b>Kings</b> move diagonally in any direction by one square.</li>
</ul>

<h3>Capturing</h3>
<ul>
<li>Capture by jumping over an adjacent opponent piece to an empty square beyond.</li>
<li><b>Forced capture:</b> If a capture is available, you must take it.</li>
<li><b>Multi-jump:</b> If after capturing you can capture again, you must continue.</li>
</ul>

<h3>Promotion</h3>
<p>A man reaching the opponent's back rank becomes a king.</p>

<h3>Winning</h3>
<p>You win when your opponent has no legal moves (no pieces or all blocked).</p>
"""
        QMessageBox.information(self, "Rules", rules)

    def _show_training_panel(self) -> None:
        """Show the training panel in a dock widget."""
        # Check if dock already exists
        for dock in self.findChildren(QDockWidget):
            if dock.windowTitle() == "ML Training":
                dock.show()
                dock.raise_()
                return

        # Create new dock
        dock = QDockWidget("ML Training", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea |
                            Qt.DockWidgetArea.LeftDockWidgetArea)

        training_panel = TrainingPanel()
        dock.setWidget(training_panel)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _setup_ai_vs_algo(self) -> None:
        """Set up AI Model vs Algorithm game."""
        # Set player types
        self.player1_type_combo.setCurrentText("AI Model")
        self.opponent_combo.setCurrentText("Algorithm")
        
        # Start new game
        self._new_game()
        self.status_bar.showMessage("AI Model (White) vs Algorithm (Black) - Watch mode")

    def _setup_algo_vs_ai(self) -> None:
        """Set up Algorithm vs AI Model game."""
        # Set player types
        self.player1_type_combo.setCurrentText("Algorithm")
        self.opponent_combo.setCurrentText("AI Model")
        
        # Start new game
        self._new_game()
        self.status_bar.showMessage("Algorithm (White) vs AI Model (Black) - Watch mode")

    def closeEvent(self, event) -> None:
        """Handle window close."""
        # Stop AI worker
        if self.ai_worker and self.ai_worker.isRunning():
            self.ai_worker.terminate()
            self.ai_worker.wait()
        self._ai_thinking = False

        # Shut down training/testing subprocesses and their monitor threads.
        # Docked panels never receive closeEvent themselves, so without this
        # the trainer process would be orphaned and keep running.
        for panel in self.findChildren(TrainingPanel):
            panel.shutdown()

        # Save config
        save_config()

        event.accept()


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Filipino Dama")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
