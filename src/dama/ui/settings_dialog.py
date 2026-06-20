"""Settings dialog for Filipino Dama."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QComboBox, QPushButton, QColorDialog, QGroupBox,
    QFormLayout, QDialogButtonBox, QFileDialog, QSpinBox,
    QDoubleSpinBox, QScrollArea, QFrame, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from pathlib import Path
import multiprocessing

from ..config import get_config, save_config, Config


def discover_models() -> list[str]:
    """Scan model roots and all named checkpoint directories for models."""
    models_dirs = [
        Path("models"),
        Path("models/checkpoints"),
        Path.cwd() / "models",
        Path.cwd() / "models" / "checkpoints",
    ]
    found = set()
    for d in models_dirs:
        if d.exists():
            for pt_file in d.glob("*.pt"):
                found.add(str(pt_file))
    for models_root in {Path("models"), Path.cwd() / "models"}:
        if models_root.exists():
            for pt_file in models_root.glob("checkpoints*/*.pt"):
                found.add(str(pt_file))
    return sorted(found)


class ColorButton(QPushButton):
    """Button that displays and allows selection of a color."""

    color_changed = pyqtSignal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        self._update_style()
        self.clicked.connect(self._pick_color)
        self.setFixedSize(60, 30)

    def _update_style(self) -> None:
        """Update button style to show the color."""
        self.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid black;"
        )

    def _pick_color(self) -> None:
        """Open color picker dialog."""
        color = QColorDialog.getColor(QColor(self._color), self)
        if color.isValid():
            self._color = color.name()
            self._update_style()
            self.color_changed.emit(self._color)

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, value: str) -> None:
        self._color = value
        self._update_style()


class SettingsDialog(QDialog):
    """Dialog for configuring game settings."""

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        self.config = get_config()
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize the UI."""
        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()

        # Appearance tab
        appearance_tab = self._create_appearance_tab()
        tabs.addTab(appearance_tab, "Appearance")

        # Game Rules tab
        rules_tab = self._create_rules_tab()
        tabs.addTab(rules_tab, "Game Rules")

        # Players tab
        players_tab = self._create_players_tab()
        tabs.addTab(players_tab, "Players")

        # AI tab
        ai_tab = self._create_ai_tab()
        tabs.addTab(ai_tab, "AI Settings")

        layout.addWidget(tabs)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)

        layout.addWidget(buttons)

    def _create_appearance_tab(self) -> QWidget:
        """Create the appearance settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Board colors
        board_group = QGroupBox("Board Colors")
        board_layout = QFormLayout(board_group)

        self.light_color_btn = ColorButton(self.config.ui.board.light_color)
        board_layout.addRow("Light squares:", self.light_color_btn)

        self.dark_color_btn = ColorButton(self.config.ui.board.dark_color)
        board_layout.addRow("Dark squares:", self.dark_color_btn)

        self.highlight_color_btn = ColorButton(self.config.ui.board.highlight_color)
        board_layout.addRow("Highlight:", self.highlight_color_btn)

        layout.addWidget(board_group)

        # Piece colors
        piece_group = QGroupBox("Piece Colors")
        piece_layout = QFormLayout(piece_group)

        self.p1_color_btn = ColorButton(self.config.ui.pieces.p1_color)
        piece_layout.addRow("Player 1:", self.p1_color_btn)

        self.p2_color_btn = ColorButton(self.config.ui.pieces.p2_color)
        piece_layout.addRow("Player 2:", self.p2_color_btn)

        self.piece_style_combo = QComboBox()
        self.piece_style_combo.addItems(["flat", "outlined"])
        self.piece_style_combo.setCurrentText(self.config.ui.pieces.style)
        piece_layout.addRow("Piece style:", self.piece_style_combo)

        layout.addWidget(piece_group)

        layout.addStretch()
        return widget

    def _create_rules_tab(self) -> QWidget:
        """Create the game rules settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Capture Rules
        capture_group = QGroupBox("Capture Rules")
        capture_layout = QFormLayout(capture_group)

        self.forced_capture_check = QCheckBox("Forced Capture")
        self.forced_capture_check.setChecked(self.config.game.rules.forced_capture)
        self.forced_capture_check.setToolTip("Players must capture when a capture is available")
        capture_layout.addRow("", self.forced_capture_check)

        self.multi_jump_check = QCheckBox("Multi-Jump")
        self.multi_jump_check.setChecked(self.config.game.rules.multi_jump)
        self.multi_jump_check.setToolTip("Allow multiple consecutive captures in one turn")
        capture_layout.addRow("", self.multi_jump_check)

        self.backward_capture_check = QCheckBox("Backward Capture")
        self.backward_capture_check.setChecked(self.config.game.rules.backward_capture)
        self.backward_capture_check.setToolTip("Allow regular pieces (non-kings) to capture backwards")
        capture_layout.addRow("", self.backward_capture_check)

        layout.addWidget(capture_group)

        # King Rules
        king_group = QGroupBox("King Rules")
        king_layout = QFormLayout(king_group)

        self.king_flying_check = QCheckBox("Flying Kings (Long-Range Capture)")
        self.king_flying_check.setChecked(self.config.game.rules.king_flying_capture)
        self.king_flying_check.setToolTip(
            "Kings can capture pieces at any distance along a diagonal\n"
            "and land on any empty square beyond the captured piece"
        )
        king_layout.addRow("", self.king_flying_check)

        layout.addWidget(king_group)

        # Note about rule changes
        note_label = QLabel(
            "<i>Note: Rule changes will apply to new games only.</i>"
        )
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

        layout.addStretch()
        return widget

    def _create_players_tab(self) -> QWidget:
        """Create the players settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        players_group = QGroupBox("Player Types")
        players_layout = QFormLayout(players_group)

        self.p1_type_combo = QComboBox()
        self.p1_type_combo.addItems(["human", "algorithmic", "ml"])
        self.p1_type_combo.setCurrentText(self.config.players.p1_type)
        players_layout.addRow("Player 1:", self.p1_type_combo)

        self.p2_type_combo = QComboBox()
        self.p2_type_combo.addItems(["human", "algorithmic", "ml"])
        self.p2_type_combo.setCurrentText(self.config.players.p2_type)
        players_layout.addRow("Player 2:", self.p2_type_combo)

        layout.addWidget(players_group)
        layout.addStretch()
        return widget

    def _create_ai_tab(self) -> QWidget:
        """Create the AI settings tab."""
        # Use scroll area for the AI tab
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Algorithmic AI
        algo_group = QGroupBox("Algorithmic AI")
        algo_layout = QFormLayout(algo_group)

        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["easy", "medium", "hard", "super_hard", "custom"])
        self.difficulty_combo.setCurrentText(self.config.ai.algorithmic.difficulty)
        self.difficulty_combo.currentTextChanged.connect(self._on_difficulty_changed)
        algo_layout.addRow("Difficulty:", self.difficulty_combo)

        layout.addWidget(algo_group)

        # Multithreading Settings
        threading_group = QGroupBox("Performance (Multithreading)")
        threading_layout = QFormLayout(threading_group)

        self.use_parallel_check = QCheckBox("Enable Parallel Search")
        self.use_parallel_check.setChecked(self.config.ai.algorithmic.use_parallel)
        self.use_parallel_check.setToolTip("Use multiple CPU cores for faster search")
        self.use_parallel_check.stateChanged.connect(self._on_parallel_changed)
        threading_layout.addRow("", self.use_parallel_check)

        # Number of threads
        cpu_count = multiprocessing.cpu_count()
        self.num_threads_spin = QSpinBox()
        self.num_threads_spin.setRange(0, cpu_count * 2)
        self.num_threads_spin.setSpecialValueText(f"Auto ({cpu_count} cores)")
        self.num_threads_spin.setValue(self.config.ai.algorithmic.num_threads)
        self.num_threads_spin.setToolTip(f"Number of threads (0 = auto, detected {cpu_count} cores)")
        threading_layout.addRow("Threads:", self.num_threads_spin)

        layout.addWidget(threading_group)

        # Custom Algorithm Parameters
        self.custom_params_group = QGroupBox("Custom Algorithm Parameters")
        custom_layout = QFormLayout(self.custom_params_group)

        # Time budget
        self.time_budget_spin = QDoubleSpinBox()
        self.time_budget_spin.setRange(0.1, 30.0)
        self.time_budget_spin.setSingleStep(0.1)
        self.time_budget_spin.setSuffix(" sec")
        self.time_budget_spin.setValue(self.config.ai.algorithmic.time_budget)
        self.time_budget_spin.setToolTip("Maximum time for AI to think per move")
        custom_layout.addRow("Time Budget:", self.time_budget_spin)

        # Max depth
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 20)
        self.max_depth_spin.setValue(self.config.ai.algorithmic.max_depth)
        self.max_depth_spin.setToolTip("Maximum search depth for the alpha-beta algorithm")
        custom_layout.addRow("Max Depth:", self.max_depth_spin)

        layout.addWidget(self.custom_params_group)

        # Evaluation Weights
        self.weights_group = QGroupBox("Evaluation Weights")
        weights_layout = QFormLayout(self.weights_group)

        self.weight_man_spin = QSpinBox()
        self.weight_man_spin.setRange(0, 1000)
        self.weight_man_spin.setValue(self.config.ai.algorithmic.weight_man)
        self.weight_man_spin.setToolTip("Value of a regular piece (man)")
        weights_layout.addRow("Man Value:", self.weight_man_spin)

        self.weight_king_spin = QSpinBox()
        self.weight_king_spin.setRange(0, 1000)
        self.weight_king_spin.setValue(self.config.ai.algorithmic.weight_king)
        self.weight_king_spin.setToolTip("Value of a king piece")
        weights_layout.addRow("King Value:", self.weight_king_spin)

        self.weight_mobility_spin = QSpinBox()
        self.weight_mobility_spin.setRange(0, 100)
        self.weight_mobility_spin.setValue(self.config.ai.algorithmic.weight_mobility)
        self.weight_mobility_spin.setToolTip("Importance of having more available moves")
        weights_layout.addRow("Mobility:", self.weight_mobility_spin)

        self.weight_advancement_spin = QSpinBox()
        self.weight_advancement_spin.setRange(0, 100)
        self.weight_advancement_spin.setValue(self.config.ai.algorithmic.weight_advancement)
        self.weight_advancement_spin.setToolTip("Importance of advancing pieces toward promotion")
        weights_layout.addRow("Advancement:", self.weight_advancement_spin)

        self.weight_center_spin = QSpinBox()
        self.weight_center_spin.setRange(0, 100)
        self.weight_center_spin.setValue(self.config.ai.algorithmic.weight_center_control)
        self.weight_center_spin.setToolTip("Importance of controlling center squares")
        weights_layout.addRow("Center Control:", self.weight_center_spin)

        self.weight_back_rank_spin = QSpinBox()
        self.weight_back_rank_spin.setRange(0, 100)
        self.weight_back_rank_spin.setValue(self.config.ai.algorithmic.weight_back_rank)
        self.weight_back_rank_spin.setToolTip("Importance of protecting the back rank")
        weights_layout.addRow("Back Rank:", self.weight_back_rank_spin)

        # Reset weights button
        reset_weights_btn = QPushButton("Reset to Defaults")
        reset_weights_btn.clicked.connect(self._reset_weights)
        weights_layout.addRow("", reset_weights_btn)

        layout.addWidget(self.weights_group)

        # ML AI
        ml_group = QGroupBox("ML AI")
        ml_layout = QFormLayout(ml_group)

        # Model selection combo box
        self.model_combo = QComboBox()
        self._populate_model_list()
        ml_layout.addRow("Select Model:", self.model_combo)

        # Manual model path
        model_row = QHBoxLayout()
        self.model_path_label = QLabel(self.config.ai.ml.model_path)
        self.model_path_label.setWordWrap(True)
        model_row.addWidget(self.model_path_label)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(browse_btn)

        ml_layout.addRow("Custom Model:", model_row)
        
        # Refresh models button
        refresh_btn = QPushButton("Refresh Model List")
        refresh_btn.clicked.connect(self._populate_model_list)
        ml_layout.addRow("", refresh_btn)

        layout.addWidget(ml_group)

        layout.addStretch()
        
        # Update custom params visibility
        self._on_difficulty_changed(self.difficulty_combo.currentText())
        self._on_parallel_changed(self.use_parallel_check.checkState().value)
        
        scroll.setWidget(widget)
        return scroll

    def _browse_model(self) -> None:
        """Browse for model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Model (*.pt);;All Files (*)"
        )
        if path:
            self.model_path_label.setText(path)
            # Also add to combo if not already present
            if self.model_combo.findText(path) == -1:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)

    def _on_difficulty_changed(self, difficulty: str) -> None:
        """Handle difficulty selection change."""
        is_custom = difficulty == "custom"
        self.custom_params_group.setVisible(is_custom)
        self.weights_group.setVisible(is_custom)

    def _on_parallel_changed(self, state: int) -> None:
        """Handle parallel search checkbox change."""
        is_enabled = state != 0
        self.num_threads_spin.setEnabled(is_enabled)

    def _reset_weights(self) -> None:
        """Reset evaluation weights to defaults."""
        self.weight_man_spin.setValue(100)
        self.weight_king_spin.setValue(200)
        self.weight_mobility_spin.setValue(5)
        self.weight_advancement_spin.setValue(2)
        self.weight_center_spin.setValue(3)
        self.weight_back_rank_spin.setValue(10)
        self.time_budget_spin.setValue(1.0)
        self.max_depth_spin.setValue(6)

    def _populate_model_list(self) -> None:
        """Populate the model combo box with available models."""
        self.model_combo.clear()

        # Add found models
        for model_path in discover_models():
            self.model_combo.addItem(model_path)
        
        # Add current config model if not in list
        current_model = self.config.ai.ml.model_path
        if current_model and self.model_combo.findText(current_model) == -1:
            self.model_combo.addItem(current_model)
        
        # Select current model
        if current_model:
            idx = self.model_combo.findText(current_model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        
        # If no models found, add a placeholder
        if self.model_combo.count() == 0:
            self.model_combo.addItem("No models found - use Browse...")

    def _apply(self) -> None:
        """Apply settings without closing."""
        self._save_settings()
        self.settings_changed.emit()

    def _accept(self) -> None:
        """Save settings and close."""
        self._save_settings()
        self.settings_changed.emit()
        self.accept()

    def _save_settings(self) -> None:
        """Save current settings to config."""
        # Board colors
        self.config.ui.board.light_color = self.light_color_btn.color
        self.config.ui.board.dark_color = self.dark_color_btn.color
        self.config.ui.board.highlight_color = self.highlight_color_btn.color

        # Piece colors
        self.config.ui.pieces.p1_color = self.p1_color_btn.color
        self.config.ui.pieces.p2_color = self.p2_color_btn.color
        self.config.ui.pieces.style = self.piece_style_combo.currentText()

        # Game rules
        self.config.game.rules.forced_capture = self.forced_capture_check.isChecked()
        self.config.game.rules.multi_jump = self.multi_jump_check.isChecked()
        self.config.game.rules.backward_capture = self.backward_capture_check.isChecked()
        self.config.game.rules.king_flying_capture = self.king_flying_check.isChecked()

        # Players
        self.config.players.p1_type = self.p1_type_combo.currentText()
        self.config.players.p2_type = self.p2_type_combo.currentText()

        # AI settings - difficulty
        self.config.ai.algorithmic.difficulty = self.difficulty_combo.currentText()
        
        # AI settings - multithreading
        self.config.ai.algorithmic.use_parallel = self.use_parallel_check.isChecked()
        self.config.ai.algorithmic.num_threads = self.num_threads_spin.value()
        
        # AI settings - custom algorithm parameters
        self.config.ai.algorithmic.time_budget = self.time_budget_spin.value()
        self.config.ai.algorithmic.max_depth = self.max_depth_spin.value()
        
        # AI settings - evaluation weights
        self.config.ai.algorithmic.weight_man = self.weight_man_spin.value()
        self.config.ai.algorithmic.weight_king = self.weight_king_spin.value()
        self.config.ai.algorithmic.weight_mobility = self.weight_mobility_spin.value()
        self.config.ai.algorithmic.weight_advancement = self.weight_advancement_spin.value()
        self.config.ai.algorithmic.weight_center_control = self.weight_center_spin.value()
        self.config.ai.algorithmic.weight_back_rank = self.weight_back_rank_spin.value()
        
        # ML model path - prefer combo selection, fallback to label
        combo_model = self.model_combo.currentText()
        if combo_model and not combo_model.startswith("No models"):
            self.config.ai.ml.model_path = combo_model
        else:
            self.config.ai.ml.model_path = self.model_path_label.text()

        save_config()
