"""AI Settings sidebar for quick access during gameplay."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel
)
from PyQt6.QtCore import pyqtSignal

from ..config import get_config, save_config
from .settings_dialog import discover_models


class AISettingsSidebar(QWidget):
    """Compact sidebar exposing frequently-used AI settings."""

    settings_changed = pyqtSignal()
    open_full_settings_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        config = get_config()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Primary controls (flat, top-level) ---
        top_layout = QFormLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["easy", "medium", "hard", "super_hard", "custom"])
        self.difficulty_combo.setCurrentText(config.ai.algorithmic.difficulty)
        self.difficulty_combo.currentTextChanged.connect(self._on_difficulty_changed)
        self.difficulty_combo.currentTextChanged.connect(self._save)
        top_layout.addRow("Difficulty:", self.difficulty_combo)

        self.model_combo = QComboBox()
        self._populate_model_list()
        self.model_combo.currentTextChanged.connect(self._save)
        top_layout.addRow("ML Model:", self.model_combo)

        layout.addLayout(top_layout)

        # --- Search tuning (grouped) ---
        tuning_group = QGroupBox("Search Tuning")
        tuning_layout = QFormLayout(tuning_group)

        self.time_budget_spin = QDoubleSpinBox()
        self.time_budget_spin.setRange(0.1, 30.0)
        self.time_budget_spin.setSingleStep(0.1)
        self.time_budget_spin.setSuffix(" sec")
        self.time_budget_spin.setValue(config.ai.algorithmic.time_budget)
        self.time_budget_spin.setToolTip("Max time per move")
        self.time_budget_spin.valueChanged.connect(self._save)
        tuning_layout.addRow("Time Budget:", self.time_budget_spin)

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 20)
        self.max_depth_spin.setValue(config.ai.algorithmic.max_depth)
        self.max_depth_spin.setToolTip("Search depth for alpha-beta")
        self.max_depth_spin.valueChanged.connect(self._save)
        self._max_depth_label = QLabel("Max Depth:")
        tuning_layout.addRow(self._max_depth_label, self.max_depth_spin)

        layout.addWidget(tuning_group)

        # --- Actions ---
        actions_layout = QHBoxLayout()

        refresh_btn = QPushButton("Refresh Models")
        refresh_btn.setToolTip("Rescan for available .pt models")
        refresh_btn.clicked.connect(self._populate_model_list)
        actions_layout.addWidget(refresh_btn)

        full_btn = QPushButton("All Settings\u2026")
        full_btn.setToolTip("Open the full AI Settings dialog")
        full_btn.clicked.connect(self.open_full_settings_requested.emit)
        actions_layout.addWidget(full_btn)

        layout.addLayout(actions_layout)

        layout.addStretch()

        # Initial visibility
        self._on_difficulty_changed(self.difficulty_combo.currentText())

    # -------------------------------------------------------------- Slots

    def _on_difficulty_changed(self, difficulty: str) -> None:
        is_custom = difficulty == "custom"
        self.max_depth_spin.setVisible(is_custom)
        self._max_depth_label.setVisible(is_custom)

    def _save(self) -> None:
        """Persist current widget values to config and notify."""
        config = get_config()
        config.ai.algorithmic.difficulty = self.difficulty_combo.currentText()
        config.ai.algorithmic.time_budget = self.time_budget_spin.value()
        config.ai.algorithmic.max_depth = self.max_depth_spin.value()

        model = self.model_combo.currentText()
        if model and not model.startswith("No models"):
            config.ai.ml.model_path = model

        save_config()
        self.settings_changed.emit()

    # ------------------------------------------------------------ Public

    def refresh_from_config(self) -> None:
        """Re-read config and update widgets (blocks signals to avoid loops)."""
        config = get_config()

        for widget in (self.difficulty_combo, self.time_budget_spin,
                       self.max_depth_spin, self.model_combo):
            widget.blockSignals(True)

        self.difficulty_combo.setCurrentText(config.ai.algorithmic.difficulty)
        self.time_budget_spin.setValue(config.ai.algorithmic.time_budget)
        self.max_depth_spin.setValue(config.ai.algorithmic.max_depth)
        self._populate_model_list()

        for widget in (self.difficulty_combo, self.time_budget_spin,
                       self.max_depth_spin, self.model_combo):
            widget.blockSignals(False)

        self._on_difficulty_changed(self.difficulty_combo.currentText())

    # ----------------------------------------------------------- Helpers

    def _populate_model_list(self) -> None:
        self.model_combo.clear()
        for path in discover_models():
            self.model_combo.addItem(path)

        current = get_config().ai.ml.model_path
        if current and self.model_combo.findText(current) == -1:
            self.model_combo.addItem(current)
        if current:
            idx = self.model_combo.findText(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

        if self.model_combo.count() == 0:
            self.model_combo.addItem("No models found")
