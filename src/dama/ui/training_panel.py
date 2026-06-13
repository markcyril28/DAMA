"""Training panel for in-GUI ML training."""

import json
import multiprocessing as mp
from typing import Optional, Dict, Any, List
from pathlib import Path
from queue import Empty
from dataclasses import dataclass
import math

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QSpinBox, QProgressBar,
    QTextEdit, QComboBox, QFormLayout, QMessageBox,
    QTabWidget, QScrollArea, QFrame, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush

from ..config import get_config


# Use 'spawn' start method for CUDA compatibility
# This must be set before creating any multiprocessing objects
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Already set


# IPC Message types
MSG_START = 'START'
MSG_PAUSE = 'PAUSE'
MSG_RESUME = 'RESUME'
MSG_STOP = 'STOP'
MSG_STATUS = 'STATUS'
MSG_STATUS_REPLY = 'STATUS_REPLY'
MSG_CHECKPOINT = 'CHECKPOINT'
MSG_ERROR = 'ERROR'
MSG_LOG = 'LOG'


def _trainer_process(control_queue: mp.Queue, status_queue: mp.Queue, args: dict):
    """
    Training process that runs in the background.

    Communicates with the GUI via queues.
    """
    try:
        # Import here to avoid loading torch in main process
        from ..ai.ml.trainer import Trainer, TrainingConfig

        config = TrainingConfig(
            device=args.get('device', 'cuda'),
            amp=args.get('amp', True),
            cpu_workers=args.get('cpu_workers', 10),
            selfplay_games=args.get('selfplay_games', 500),
            batch_size=args.get('batch_size', 256),
            learning_rate=args.get('learning_rate', 3e-4),
            train_steps=args.get('train_steps', 10000),
            checkpoint_every=args.get('checkpoint_every', 1000),
            resume=args.get('resume'),
            pipeline_mode=args.get('pipeline_mode', 'simultaneous'),
            # Algo-vs-algo settings
            algo_vs_algo_enabled=args.get('algo_vs_algo_enabled', False),
            algo_vs_algo_games=args.get('algo_vs_algo_games', 100),
            algo_vs_algo_difficulties=args.get('algo_vs_algo_difficulties', ['easy', 'medium', 'hard']),
            # Model testing settings
            test_vs_algo=args.get('test_vs_algo', True),
            test_every=args.get('test_every', 5000),
            test_games=args.get('test_games', 50),
            test_difficulty=args.get('test_difficulty', 'medium'),
        )

        trainer = Trainer(config)

        # train() services control_queue internally (PAUSE/RESUME/STOP/
        # STATUS) and pushes STATUS_REPLY heartbeats plus CHECKPOINT
        # notifications onto status_queue while it runs.
        trainer.set_control_queues(control_queue, status_queue)
        trainer.train()

        # Send final status
        status_queue.put({
            'type': MSG_STATUS_REPLY,
            'step': trainer.step,
            'complete': True,
            'stopped': trainer.is_stopped,
        })

    except Exception as e:
        status_queue.put({'type': MSG_ERROR, 'message': str(e)})


class TrainingWorker(QThread):
    """
    Worker thread that monitors the training process.
    """

    status_update = pyqtSignal(dict)
    training_complete = pyqtSignal(bool)  # True when stopped by the user
    error_occurred = pyqtSignal(str)

    def __init__(self, status_queue: mp.Queue, process: Optional[mp.Process] = None):
        super().__init__()
        self.status_queue = status_queue
        self._process = process
        self._running = True

    def run(self):
        while self._running:
            try:
                msg = self.status_queue.get(timeout=0.5)
            except Empty:
                # Detect a trainer that died without a final message
                # (OOM kill, segfault, CUDA abort).  Allow a short grace
                # period for a last in-flight message before reporting.
                if self._process is not None and not self._process.is_alive():
                    try:
                        msg = self.status_queue.get(timeout=1.0)
                    except Empty:
                        self.error_occurred.emit(
                            "Training process exited unexpectedly "
                            f"(exit code {self._process.exitcode})")
                        break
                else:
                    continue

            msg_type = msg.get('type')

            if msg_type == MSG_STATUS_REPLY:
                self.status_update.emit(msg)
                if msg.get('complete') or msg.get('stopped'):
                    self.training_complete.emit(bool(msg.get('stopped')))
                    break

            elif msg_type == MSG_ERROR:
                self.error_occurred.emit(msg.get('message', 'Unknown error'))
                break

            elif msg_type == MSG_CHECKPOINT:
                self.status_update.emit(msg)

    def stop(self):
        self._running = False


# Message types for model testing
MSG_TEST_START = 'TEST_START'
MSG_TEST_PROGRESS = 'TEST_PROGRESS'
MSG_TEST_COMPLETE = 'TEST_COMPLETE'
MSG_TEST_STOP = 'TEST_STOP'
MSG_TEST_ERROR = 'TEST_ERROR'


def _tester_process(control_queue: mp.Queue, status_queue: mp.Queue, args: dict):
    """
    Testing process that runs model vs algorithm games in the background.
    """
    import sys
    import time
    
    try:
        from ..ai.ml.model_vs_algo import ModelVsAlgoTester
        
        tester = ModelVsAlgoTester(
            model_path=args.get('model_path', 'models/latest.pt'),
            algo_difficulty=args.get('algo_difficulty', 'medium'),
            num_workers=args.get('num_workers', 4),
            max_moves=args.get('max_moves', 200),
        )
        
        num_games = args.get('num_games', 100)
        
        def progress_callback(completed, total, stats):
            # Check for stop signal
            try:
                while True:
                    msg = control_queue.get_nowait()
                    if msg.get('type') == MSG_TEST_STOP:
                        tester.stop()
                        return
            except Empty:
                pass
            
            # Send progress update
            status_queue.put({
                'type': MSG_TEST_PROGRESS,
                'completed': completed,
                'total': total,
                'ml_wins': stats.ml_wins,
                'algo_wins': stats.algo_wins,
                'draws': stats.draws,
                'ml_win_rate': stats.ml_win_rate,
            })
        
        stats = tester.run_tests(num_games=num_games, callback=progress_callback)
        
        # Send completion
        status_queue.put({
            'type': MSG_TEST_COMPLETE,
            'stats': stats.to_dict(),
        })
    
    except Exception as e:
        status_queue.put({'type': MSG_TEST_ERROR, 'message': str(e)})


class TestWorker(QThread):
    """Worker thread that monitors the testing process."""
    
    progress_update = pyqtSignal(dict)
    test_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, status_queue: mp.Queue, process: Optional[mp.Process] = None):
        super().__init__()
        self.status_queue = status_queue
        self._process = process
        self._running = True

    def run(self):
        while self._running:
            try:
                msg = self.status_queue.get(timeout=0.5)
            except Empty:
                # Detect a tester that died without a final message.
                if self._process is not None and not self._process.is_alive():
                    try:
                        msg = self.status_queue.get(timeout=1.0)
                    except Empty:
                        self.error_occurred.emit(
                            "Test process exited unexpectedly "
                            f"(exit code {self._process.exitcode})")
                        break
                else:
                    continue

            msg_type = msg.get('type')

            if msg_type == MSG_TEST_PROGRESS:
                self.progress_update.emit(msg)

            elif msg_type == MSG_TEST_COMPLETE:
                self.test_complete.emit(msg.get('stats', {}))
                break

            elif msg_type == MSG_TEST_ERROR:
                self.error_occurred.emit(msg.get('message', 'Unknown error'))
                break

    def stop(self):
        self._running = False


class WinRateChartWidget(QWidget):
    """Widget to display win rate pie chart and bar chart."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ml_wins = 0
        self.algo_wins = 0
        self.draws = 0
        self.ml_as_p1_rate = 0.0
        self.ml_as_p2_rate = 0.0
        self.setMinimumHeight(200)
        self.setMinimumWidth(400)
    
    def set_data(self, ml_wins: int, algo_wins: int, draws: int,
                 ml_as_p1_rate: float = 0.0, ml_as_p2_rate: float = 0.0):
        """Set the chart data."""
        self.ml_wins = ml_wins
        self.algo_wins = algo_wins
        self.draws = draws
        self.ml_as_p1_rate = ml_as_p1_rate
        self.ml_as_p2_rate = ml_as_p2_rate
        self.update()
    
    def paintEvent(self, event):
        """Paint the charts."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        total = self.ml_wins + self.algo_wins + self.draws
        if total == 0:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No test data")
            return
        
        # Draw pie chart on left side
        pie_size = min(self.height() - 40, (self.width() // 2) - 40)
        pie_x = 20
        pie_y = (self.height() - pie_size) // 2
        
        # Colors
        ml_color = QColor(0, 200, 100)  # Green for ML
        algo_color = QColor(200, 80, 80)  # Red for Algorithm
        draw_color = QColor(150, 150, 150)  # Gray for draws
        
        # Calculate angles (Qt uses 16ths of a degree)
        ml_angle = int((self.ml_wins / total) * 360 * 16)
        algo_angle = int((self.algo_wins / total) * 360 * 16)
        draw_angle = 360 * 16 - ml_angle - algo_angle
        
        # Draw pie slices
        start_angle = 90 * 16  # Start at top
        
        painter.setBrush(QBrush(ml_color))
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        painter.drawPie(pie_x, pie_y, pie_size, pie_size, start_angle, ml_angle)
        start_angle += ml_angle
        
        painter.setBrush(QBrush(algo_color))
        painter.drawPie(pie_x, pie_y, pie_size, pie_size, start_angle, algo_angle)
        start_angle += algo_angle
        
        if self.draws > 0:
            painter.setBrush(QBrush(draw_color))
            painter.drawPie(pie_x, pie_y, pie_size, pie_size, start_angle, draw_angle)
        
        # Draw legend
        legend_x = pie_x + pie_size + 20
        legend_y = pie_y + 20
        
        painter.setFont(QFont("Arial", 10))
        
        painter.setBrush(QBrush(ml_color))
        painter.drawRect(legend_x, legend_y, 15, 15)
        painter.setPen(QColor(200, 200, 200))
        ml_pct = (self.ml_wins / total) * 100
        painter.drawText(legend_x + 20, legend_y + 12, f"ML Model: {self.ml_wins} ({ml_pct:.1f}%)")
        
        legend_y += 25
        painter.setBrush(QBrush(algo_color))
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        painter.drawRect(legend_x, legend_y, 15, 15)
        painter.setPen(QColor(200, 200, 200))
        algo_pct = (self.algo_wins / total) * 100
        painter.drawText(legend_x + 20, legend_y + 12, f"Algorithm: {self.algo_wins} ({algo_pct:.1f}%)")
        
        legend_y += 25
        painter.setBrush(QBrush(draw_color))
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        painter.drawRect(legend_x, legend_y, 15, 15)
        painter.setPen(QColor(200, 200, 200))
        draw_pct = (self.draws / total) * 100
        painter.drawText(legend_x + 20, legend_y + 12, f"Draws: {self.draws} ({draw_pct:.1f}%)")
        
        # Draw bar chart for P1 vs P2 performance on right side
        bar_x = self.width() // 2 + 40
        bar_width = 60
        max_bar_height = self.height() - 80
        
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(bar_x, 20, "Win Rate by Starting Position")
        
        # P1 bar
        p1_height = int(self.ml_as_p1_rate * max_bar_height)
        painter.setBrush(QBrush(QColor(100, 150, 255)))
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        painter.drawRect(bar_x, 40 + max_bar_height - p1_height, bar_width, p1_height)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(bar_x + 5, 40 + max_bar_height + 15, f"As P1")
        painter.drawText(bar_x + 5, 40 + max_bar_height + 30, f"{self.ml_as_p1_rate*100:.1f}%")
        
        # P2 bar
        p2_x = bar_x + bar_width + 20
        p2_height = int(self.ml_as_p2_rate * max_bar_height)
        painter.setBrush(QBrush(QColor(255, 150, 100)))
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        painter.drawRect(p2_x, 40 + max_bar_height - p2_height, bar_width, p2_height)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(p2_x + 5, 40 + max_bar_height + 15, f"As P2")
        painter.drawText(p2_x + 5, 40 + max_bar_height + 30, f"{self.ml_as_p2_rate*100:.1f}%")


class WinRateHistoryChart(QWidget):
    """Widget to display win rate over training steps."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: List[dict] = []  # [{step, ml_win_rate}, ...]
        self.setMinimumHeight(120)
        self.setMinimumWidth(300)
    
    def set_data(self, test_history: List[dict]):
        """Set the test history data."""
        self.data = test_history
        self.update()
    
    def paintEvent(self, event):
        """Paint the win rate history chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        if not self.data:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No test data")
            return
        
        # Margins
        margin = 40
        chart_width = self.width() - 2 * margin
        chart_height = self.height() - 2 * margin
        
        if chart_width <= 0 or chart_height <= 0:
            return
        
        # Draw 50% reference line
        mid_y = margin + chart_height // 2
        painter.setPen(QPen(QColor(100, 100, 100), 1, Qt.PenStyle.DashLine))
        painter.drawLine(margin, mid_y, self.width() - margin, mid_y)
        
        # Draw grid
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(5):
            y = margin + i * chart_height // 4
            if i != 2:  # Skip middle (already drawn)
                painter.drawLine(margin, y, self.width() - margin, y)
        
        # Draw axes
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawLine(margin, margin, margin, self.height() - margin)
        painter.drawLine(margin, self.height() - margin, self.width() - margin, self.height() - margin)
        
        # Axis labels
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(5, margin + 4, "100%")
        painter.drawText(5, mid_y + 4, "50%")
        painter.drawText(5, self.height() - margin + 4, "0%")
        
        # Title
        painter.setFont(QFont("Arial", 10))
        painter.drawText(margin + 10, margin - 5, "ML Win Rate vs Algorithm")
        
        # Draw win rate line
        if len(self.data) >= 1:
            painter.setPen(QPen(QColor(0, 200, 100), 2))
            
            max_step = max(d.get('step', 1) for d in self.data)
            min_step = min(d.get('step', 0) for d in self.data)
            step_range = max(max_step - min_step, 1)
            
            prev_x, prev_y = None, None
            for entry in self.data:
                step = entry.get('step', 0)
                win_rate = entry.get('ml_win_rate', 0.5)
                
                # Normalize x position
                x = margin + int(((step - min_step) / step_range) * chart_width)
                # Win rate to y (0% at bottom, 100% at top)
                y = margin + int((1 - win_rate) * chart_height)
                
                # Draw point
                painter.setBrush(QBrush(QColor(0, 200, 100)))
                painter.drawEllipse(x - 3, y - 3, 6, 6)
                
                # Draw line
                if prev_x is not None:
                    painter.drawLine(prev_x, prev_y, x, y)
                
                prev_x, prev_y = x, y


class LossChartWidget(QWidget):
    """Simple widget to display loss chart."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loss_data: List[float] = []
        self.val_loss_data: List[float] = []
        self.setMinimumHeight(150)
        self.setMinimumWidth(300)
    
    def set_data(self, loss_history: List[dict], val_loss_history: List[dict] = None):
        """Set the loss data to display."""
        self.loss_data = [entry.get('loss', 0) for entry in loss_history]
        if val_loss_history:
            self.val_loss_data = [entry.get('val_loss', 0) for entry in val_loss_history]
        else:
            self.val_loss_data = []
        self.update()
    
    def add_point(self, loss: float):
        """Add a single loss point."""
        import math
        # Skip NaN or infinite values
        if math.isnan(loss) or math.isinf(loss):
            return
        self.loss_data.append(loss)
        # Keep only last 500 points for performance
        if len(self.loss_data) > 500:
            self.loss_data = self.loss_data[-500:]
        self.update()
    
    def paintEvent(self, event):
        """Paint the loss chart."""
        if not self.loss_data:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        # Margins
        margin = 40
        chart_width = self.width() - 2 * margin
        chart_height = self.height() - 2 * margin
        
        if chart_width <= 0 or chart_height <= 0:
            return
        
        # Find data range (filter out NaN/inf values)
        import math
        all_data = [v for v in (self.loss_data + self.val_loss_data) 
                    if not (math.isnan(v) or math.isinf(v))]
        if not all_data:
            return
        
        max_loss = max(all_data) * 1.1
        min_loss = min(0, min(all_data))
        loss_range = max_loss - min_loss
        
        if loss_range == 0:
            loss_range = 1
        
        # Draw grid lines
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        for i in range(5):
            y = margin + i * chart_height // 4
            painter.drawLine(margin, y, self.width() - margin, y)
        
        # Draw axes
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawLine(margin, margin, margin, self.height() - margin)
        painter.drawLine(margin, self.height() - margin, self.width() - margin, self.height() - margin)
        
        # Draw axis labels
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor(200, 200, 200))
        for i in range(5):
            y = margin + i * chart_height // 4
            val = max_loss - (i * loss_range / 4)
            painter.drawText(5, y + 4, f"{val:.2f}")
        
        # Draw training loss line
        if len(self.loss_data) > 1:
            painter.setPen(QPen(QColor(0, 150, 255), 2))
            points_per_pixel = max(1, len(self.loss_data) // chart_width)
            
            prev_x, prev_y = None, None
            for i in range(0, len(self.loss_data), points_per_pixel):
                val = self.loss_data[i]
                # Skip NaN/inf values
                if math.isnan(val) or math.isinf(val):
                    prev_x, prev_y = None, None
                    continue
                x = margin + (i * chart_width) // len(self.loss_data)
                y = margin + int((max_loss - val) * chart_height / loss_range)
                y = max(margin, min(self.height() - margin, y))
                
                if prev_x is not None:
                    painter.drawLine(prev_x, prev_y, x, y)
                prev_x, prev_y = x, y
        
        # Draw validation loss line
        if len(self.val_loss_data) > 1:
            painter.setPen(QPen(QColor(255, 150, 0), 2))
            points_per_pixel = max(1, len(self.val_loss_data) // chart_width)
            
            prev_x, prev_y = None, None
            for i in range(0, len(self.val_loss_data), points_per_pixel):
                val = self.val_loss_data[i]
                # Skip NaN/inf values
                if math.isnan(val) or math.isinf(val):
                    prev_x, prev_y = None, None
                    continue
                x = margin + (i * chart_width) // len(self.val_loss_data)
                y = margin + int((max_loss - val) * chart_height / loss_range)
                y = max(margin, min(self.height() - margin, y))
                
                if prev_x is not None:
                    painter.drawLine(prev_x, prev_y, x, y)
                prev_x, prev_y = x, y
        
        # Legend
        painter.setFont(QFont("Arial", 9))
        painter.setPen(QColor(0, 150, 255))
        painter.drawText(margin + 10, margin + 15, "Training Loss")
        if self.val_loss_data:
            painter.setPen(QColor(255, 150, 0))
            painter.drawText(margin + 110, margin + 15, "Validation Loss")


class StatsPanel(QWidget):
    """Panel for displaying training statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Stats summary
        stats_group = QGroupBox("Training Statistics")
        stats_layout = QFormLayout(stats_group)
        
        self.total_steps_label = QLabel("0")
        stats_layout.addRow("Total Steps:", self.total_steps_label)
        
        self.epochs_label = QLabel("0")
        stats_layout.addRow("Epochs:", self.epochs_label)
        
        self.best_loss_label = QLabel("N/A")
        stats_layout.addRow("Best Loss:", self.best_loss_label)
        
        self.start_time_label = QLabel("N/A")
        stats_layout.addRow("Started:", self.start_time_label)
        
        self.end_time_label = QLabel("N/A")
        stats_layout.addRow("Ended:", self.end_time_label)
        
        self.latest_winrate_label = QLabel("N/A")
        stats_layout.addRow("Latest ML Win Rate:", self.latest_winrate_label)
        
        layout.addWidget(stats_group)
        
        # Loss chart
        chart_group = QGroupBox("Loss History")
        chart_layout = QVBoxLayout(chart_group)
        
        self.loss_chart = LossChartWidget()
        chart_layout.addWidget(self.loss_chart)
        
        layout.addWidget(chart_group)
        
        # Win rate history chart
        winrate_group = QGroupBox("Model Performance vs Algorithm")
        winrate_layout = QVBoxLayout(winrate_group)
        
        self.winrate_chart = WinRateHistoryChart()
        winrate_layout.addWidget(self.winrate_chart)
        
        layout.addWidget(winrate_group)
        
        # Refresh button
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Stats")
        self.refresh_btn.clicked.connect(self.load_stats)
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        layout.addStretch()
    
    def load_stats(self):
        """Load stats from file."""
        try:
            from ..ai.ml.trainer import load_training_stats
            stats = load_training_stats()
            if stats:
                self.update_stats(stats)
            else:
                self._log("No training stats found")
        except Exception as e:
            self._log(f"Error loading stats: {e}")
    
    def _log(self, message: str):
        """Log a message (for now just print)."""
        print(message)
    
    def update_stats(self, stats):
        """Update the display with stats."""
        self.total_steps_label.setText(str(stats.total_steps))
        self.epochs_label.setText(str(stats.epochs_completed))
        
        if stats.best_loss != float('inf'):
            self.best_loss_label.setText(f"{stats.best_loss:.4f}")
        else:
            self.best_loss_label.setText("N/A")
        
        if stats.start_time:
            self.start_time_label.setText(stats.start_time[:19].replace('T', ' '))
        
        if stats.end_time:
            self.end_time_label.setText(stats.end_time[:19].replace('T', ' '))
        
        # Update latest win rate
        if stats.test_history:
            latest_test = stats.test_history[-1]
            win_rate = latest_test.get('ml_win_rate', 0) * 100
            self.latest_winrate_label.setText(f"{win_rate:.1f}%")
        else:
            self.latest_winrate_label.setText("N/A")
        
        # Update charts
        self.loss_chart.set_data(stats.loss_history, stats.val_loss_history)
        self.winrate_chart.set_data(stats.test_history)


class TestPanel(QWidget):
    """
    Panel for running model vs algorithm tests and visualizing results.
    
    Features:
    - Run test games between ML model and algorithmic AI
    - Configure number of games and difficulty
    - Visualize win rates with charts
    - View test history
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Test process management
        self._test_process: Optional[mp.Process] = None
        self._test_control_queue: Optional[mp.Queue] = None
        self._test_status_queue: Optional[mp.Queue] = None
        self._test_worker: Optional[TestWorker] = None
        
        self._init_ui()
        self._load_latest_results()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Configuration group
        config_group = QGroupBox("Test Configuration")
        config_layout = QFormLayout(config_group)
        
        self.num_games_spin = QSpinBox()
        self.num_games_spin.setRange(10, 1000)
        self.num_games_spin.setValue(100)
        self.num_games_spin.setSingleStep(10)
        config_layout.addRow("Number of Games:", self.num_games_spin)
        
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(['easy', 'medium', 'hard', 'super_hard'])
        self.difficulty_combo.setCurrentText('medium')
        config_layout.addRow("Algorithm Difficulty:", self.difficulty_combo)
        
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        config_layout.addRow("Parallel Workers:", self.workers_spin)
        
        self.model_combo = QComboBox()
        self.model_combo.addItem("Latest (models/latest.pt)", "models/latest.pt")
        config_layout.addRow("Model:", self.model_combo)
        
        refresh_btn = QPushButton("Refresh Models")
        refresh_btn.clicked.connect(self._refresh_models)
        config_layout.addRow("", refresh_btn)
        
        layout.addWidget(config_group)
        
        # Progress group
        progress_group = QGroupBox("Test Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        status_layout = QHBoxLayout()
        self.games_label = QLabel("Games: 0 / 0")
        status_layout.addWidget(self.games_label)
        
        self.live_winrate_label = QLabel("ML Win Rate: --")
        status_layout.addWidget(self.live_winrate_label)
        
        self.test_state_label = QLabel("State: Idle")
        status_layout.addWidget(self.test_state_label)
        
        progress_layout.addLayout(status_layout)
        layout.addWidget(progress_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.start_test_btn = QPushButton("Start Test")
        self.start_test_btn.clicked.connect(self._start_test)
        btn_layout.addWidget(self.start_test_btn)
        
        self.stop_test_btn = QPushButton("Stop")
        self.stop_test_btn.clicked.connect(self._stop_test)
        self.stop_test_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_test_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Results visualization
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)
        
        self.win_rate_chart = WinRateChartWidget()
        results_layout.addWidget(self.win_rate_chart)
        
        # Statistics summary
        stats_layout = QFormLayout()
        
        self.total_games_label = QLabel("0")
        stats_layout.addRow("Total Games:", self.total_games_label)
        
        self.ml_wins_label = QLabel("0 (0.0%)")
        stats_layout.addRow("ML Wins:", self.ml_wins_label)
        
        self.algo_wins_label = QLabel("0 (0.0%)")
        stats_layout.addRow("Algorithm Wins:", self.algo_wins_label)
        
        self.draws_label = QLabel("0 (0.0%)")
        stats_layout.addRow("Draws:", self.draws_label)
        
        self.avg_length_label = QLabel("0")
        stats_layout.addRow("Avg Game Length:", self.avg_length_label)
        
        self.avg_time_label = QLabel("0 ms")
        stats_layout.addRow("Avg Game Time:", self.avg_time_label)
        
        results_layout.addLayout(stats_layout)
        layout.addWidget(results_group)
        
        # History group
        history_group = QGroupBox("Test History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Date/Time", "Games", "ML Win Rate", "Difficulty"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setMaximumHeight(150)
        history_layout.addWidget(self.history_table)
        
        history_btn_layout = QHBoxLayout()
        self.refresh_history_btn = QPushButton("Refresh History")
        self.refresh_history_btn.clicked.connect(self._refresh_history)
        history_btn_layout.addWidget(self.refresh_history_btn)
        
        self.load_result_btn = QPushButton("Load Selected")
        self.load_result_btn.clicked.connect(self._load_selected_result)
        history_btn_layout.addWidget(self.load_result_btn)
        
        history_btn_layout.addStretch()
        history_layout.addLayout(history_btn_layout)
        
        layout.addWidget(history_group)
        
        # Refresh history on init
        self._refresh_history()
    
    def _refresh_models(self):
        """Refresh the list of available models."""
        try:
            from ..ai.ml.trainer import list_checkpoints
            checkpoints = list_checkpoints()
            
            self.model_combo.clear()
            
            # Add latest first
            latest_path = Path('models/latest.pt')
            if latest_path.exists():
                self.model_combo.addItem("Latest (models/latest.pt)", str(latest_path))
            
            # Add checkpoints
            for cp in reversed(checkpoints):
                label = f"Step {cp['step']:,} ({cp['name']})"
                self.model_combo.addItem(label, cp['path'])
        
        except Exception as e:
            print(f"Error refreshing models: {e}")
    
    def _start_test(self):
        """Start the model vs algorithm test."""
        model_path = self.model_combo.currentData()
        if not model_path:
            model_path = "models/latest.pt"
        
        if not Path(model_path).exists():
            QMessageBox.warning(self, "Model Not Found",
                              f"Model not found at: {model_path}\n\n"
                              "Please train a model first or select a valid checkpoint.")
            return
        
        args = {
            'model_path': model_path,
            'algo_difficulty': self.difficulty_combo.currentText(),
            'num_workers': self.workers_spin.value(),
            'num_games': self.num_games_spin.value(),
            'max_moves': 200,
        }
        
        # Create queues
        self._test_control_queue = mp.Queue()
        self._test_status_queue = mp.Queue()
        
        # Start process
        self._test_process = mp.Process(
            target=_tester_process,
            args=(self._test_control_queue, self._test_status_queue, args)
        )
        self._test_process.start()
        
        # Start worker
        self._test_worker = TestWorker(self._test_status_queue, self._test_process)
        self._test_worker.progress_update.connect(self._on_test_progress)
        self._test_worker.test_complete.connect(self._on_test_complete)
        self._test_worker.error_occurred.connect(self._on_test_error)
        self._test_worker.start()
        
        # Update UI
        self._set_testing_state(True)
        self.test_state_label.setText("State: Running")
        self.progress_bar.setRange(0, self.num_games_spin.value())
        self.progress_bar.setValue(0)
    
    def _stop_test(self):
        """Stop the running test."""
        if self._test_control_queue:
            self._test_control_queue.put({'type': MSG_TEST_STOP})
        self.test_state_label.setText("State: Stopping...")
    
    def _on_test_progress(self, data: dict):
        """Handle test progress update."""
        completed = data.get('completed', 0)
        total = data.get('total', 0)
        ml_wins = data.get('ml_wins', 0)
        algo_wins = data.get('algo_wins', 0)
        draws = data.get('draws', 0)
        ml_win_rate = data.get('ml_win_rate', 0)
        
        self.progress_bar.setValue(completed)
        self.games_label.setText(f"Games: {completed} / {total}")
        self.live_winrate_label.setText(f"ML Win Rate: {ml_win_rate*100:.1f}%")
        
        # Update live chart
        self.win_rate_chart.set_data(ml_wins, algo_wins, draws)
    
    def _on_test_complete(self, stats: dict):
        """Handle test completion."""
        self._cleanup_test()
        self.test_state_label.setText("State: Complete")
        
        # Update display
        self._display_stats(stats)
        
        # Refresh history
        self._refresh_history()
        
        # Show summary
        total = stats.get('total_games', 0)
        ml_wins = stats.get('ml_wins', 0)
        ml_rate = stats.get('ml_win_rate', 0) * 100
        
        QMessageBox.information(self, "Test Complete",
                              f"Model vs Algorithm test complete!\n\n"
                              f"Total games: {total}\n"
                              f"ML Model wins: {ml_wins} ({ml_rate:.1f}%)\n\n"
                              f"Results saved to models/test_stats/")
    
    def _on_test_error(self, message: str):
        """Handle test error."""
        self._cleanup_test()
        self.test_state_label.setText("State: Error")
        QMessageBox.warning(self, "Test Error", message)
    
    def _set_testing_state(self, running: bool):
        """Update UI for testing state."""
        self.start_test_btn.setEnabled(not running)
        self.stop_test_btn.setEnabled(running)
        self.num_games_spin.setEnabled(not running)
        self.difficulty_combo.setEnabled(not running)
        self.workers_spin.setEnabled(not running)
        self.model_combo.setEnabled(not running)
    
    def _cleanup_test(self):
        """Clean up after test stops."""
        if self._test_worker:
            self._test_worker.stop()
            self._test_worker.wait()
            self._test_worker = None

        # Drain the status queue before terminating to avoid corrupted pipe state
        if self._test_status_queue:
            try:
                while not self._test_status_queue.empty():
                    self._test_status_queue.get_nowait()
            except Exception:
                pass

        if self._test_process and self._test_process.is_alive():
            self._test_process.terminate()
            self._test_process.join(timeout=5)
            if self._test_process.is_alive():
                self._test_process.kill()
                self._test_process.join(timeout=2)

        self._test_process = None
        self._test_control_queue = None
        self._test_status_queue = None
        
        self._set_testing_state(False)
    
    def _display_stats(self, stats: dict):
        """Display test statistics."""
        total = stats.get('total_games', 0)
        ml_wins = stats.get('ml_wins', 0)
        algo_wins = stats.get('algo_wins', 0)
        draws = stats.get('draws', 0)
        
        self.total_games_label.setText(str(total))
        
        if total > 0:
            ml_pct = (ml_wins / total) * 100
            algo_pct = (algo_wins / total) * 100
            draw_pct = (draws / total) * 100
            
            self.ml_wins_label.setText(f"{ml_wins} ({ml_pct:.1f}%)")
            self.algo_wins_label.setText(f"{algo_wins} ({algo_pct:.1f}%)")
            self.draws_label.setText(f"{draws} ({draw_pct:.1f}%)")
        else:
            self.ml_wins_label.setText("0 (0.0%)")
            self.algo_wins_label.setText("0 (0.0%)")
            self.draws_label.setText("0 (0.0%)")
        
        avg_length = stats.get('avg_game_length', 0)
        avg_time = stats.get('avg_game_time_ms', 0)
        
        self.avg_length_label.setText(f"{avg_length:.1f} moves")
        self.avg_time_label.setText(f"{avg_time:.1f} ms")
        
        # Update chart
        ml_as_p1_rate = stats.get('ml_as_p1_win_rate', 0)
        ml_as_p2_rate = stats.get('ml_as_p2_win_rate', 0)
        self.win_rate_chart.set_data(ml_wins, algo_wins, draws, ml_as_p1_rate, ml_as_p2_rate)
    
    def _refresh_history(self):
        """Refresh the test history table."""
        try:
            from ..ai.ml.model_vs_algo import list_test_results
            results = list_test_results()
            
            self.history_table.setRowCount(len(results))
            
            for i, result in enumerate(results):
                start_time = result.get('start_time', '')
                if start_time:
                    display_time = start_time[:19].replace('T', ' ')
                else:
                    display_time = 'Unknown'
                
                self.history_table.setItem(i, 0, QTableWidgetItem(display_time))
                self.history_table.setItem(i, 1, QTableWidgetItem(str(result.get('total_games', 0))))
                win_rate = result.get('ml_win_rate', 0) * 100
                self.history_table.setItem(i, 2, QTableWidgetItem(f"{win_rate:.1f}%"))
                
                # Store path in hidden data
                item = QTableWidgetItem(result.get('path', ''))
                self.history_table.setItem(i, 3, item)
        
        except Exception as e:
            print(f"Error refreshing history: {e}")
    
    def _load_selected_result(self):
        """Load the selected result from history."""
        row = self.history_table.currentRow()
        if row < 0:
            return
        
        path_item = self.history_table.item(row, 3)
        if not path_item:
            return
        
        path = path_item.text()
        
        try:
            from ..ai.ml.model_vs_algo import load_test_stats
            stats = load_test_stats(path)
            if stats:
                self._display_stats(stats.to_dict())
        except Exception as e:
            print(f"Error loading result: {e}")
    
    def _load_latest_results(self):
        """Load the latest test results on init."""
        try:
            from ..ai.ml.model_vs_algo import load_test_stats
            stats = load_test_stats()
            if stats:
                self._display_stats(stats.to_dict())
        except Exception:
            pass
    
    def closeEvent(self, event):
        """Handle panel close."""
        self._cleanup_test()
        event.accept()


class TrainingPanel(QWidget):
    """
    Panel for controlling ML training from the GUI.

    Features:
    - Start/Pause/Resume/Stop controls
    - Live progress display
    - Configuration options
    - Checkpoint management
    - Resume training from checkpoint
    - Training statistics visualization
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Process management
        self._process: Optional[mp.Process] = None
        self._control_queue: Optional[mp.Queue] = None
        self._status_queue: Optional[mp.Queue] = None
        self._worker: Optional[TrainingWorker] = None
        self._stop_requested = False

        # Live chart data
        self._live_loss_data: List[float] = []

        self._init_ui()
        self._refresh_checkpoints()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Training tab
        training_widget = QWidget()
        self.tabs.addTab(training_widget, "Training")
        self._init_training_tab(training_widget)

        # Statistics tab
        self.stats_panel = StatsPanel()
        self.tabs.addTab(self.stats_panel, "Statistics")
        
        # Model Testing tab
        self.test_panel = TestPanel()
        self.tabs.addTab(self.test_panel, "Model Test")

    def _init_training_tab(self, widget):
        layout = QVBoxLayout(widget)

        # Configuration group
        config_group = QGroupBox("Training Configuration")
        config_layout = QFormLayout(config_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda', 'cpu'])
        config_layout.addRow("Device:", self.device_combo)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 20)
        self.workers_spin.setValue(10)
        config_layout.addRow("CPU Workers:", self.workers_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(32, 512)
        self.batch_spin.setValue(256)
        self.batch_spin.setSingleStep(32)
        config_layout.addRow("Batch Size:", self.batch_spin)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 100000)
        self.steps_spin.setValue(10000)
        self.steps_spin.setSingleStep(1000)
        config_layout.addRow("Train Steps:", self.steps_spin)

        self.games_spin = QSpinBox()
        self.games_spin.setRange(50, 2000)
        self.games_spin.setValue(500)
        self.games_spin.setSingleStep(50)
        config_layout.addRow("Self-play Games:", self.games_spin)

        layout.addWidget(config_group)

        # Model Testing Configuration
        test_config_group = QGroupBox("Model Testing (during training)")
        test_config_layout = QFormLayout(test_config_group)
        self.test_enabled_check = QCheckBox()
        self.test_enabled_check.setChecked(True)
        self.test_enabled_check.setToolTip("Run test games against algorithm during training")
        test_config_layout.addRow("Enable Testing:", self.test_enabled_check)
        
        self.test_every_spin = QSpinBox()
        self.test_every_spin.setRange(1000, 20000)
        self.test_every_spin.setValue(5000)
        self.test_every_spin.setSingleStep(1000)
        self.test_every_spin.setToolTip("Run tests every N training steps")
        test_config_layout.addRow("Test Every (steps):", self.test_every_spin)
        
        self.test_games_spin = QSpinBox()
        self.test_games_spin.setRange(10, 200)
        self.test_games_spin.setValue(50)
        self.test_games_spin.setSingleStep(10)
        self.test_games_spin.setToolTip("Number of test games per evaluation")
        test_config_layout.addRow("Test Games:", self.test_games_spin)
        
        self.test_difficulty_combo = QComboBox()
        self.test_difficulty_combo.addItems(['easy', 'medium', 'hard', 'super_hard'])
        self.test_difficulty_combo.setCurrentText('medium')
        self.test_difficulty_combo.setToolTip("Algorithm difficulty for testing")
        test_config_layout.addRow("Test Difficulty:", self.test_difficulty_combo)
        
        layout.addWidget(test_config_group)

        # Resume from checkpoint group
        resume_group = QGroupBox("Resume Training")
        resume_layout = QFormLayout(resume_group)

        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.addItem("Start Fresh", None)
        resume_layout.addRow("Resume From:", self.checkpoint_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_checkpoints)
        resume_layout.addRow("", refresh_btn)

        layout.addWidget(resume_group)

        # Progress group
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Status labels
        status_layout = QHBoxLayout()
        self.step_label = QLabel("Step: 0")
        status_layout.addWidget(self.step_label)

        self.gpu_label = QLabel("GPU: N/A")
        status_layout.addWidget(self.gpu_label)

        self.state_label = QLabel("State: Idle")
        status_layout.addWidget(self.state_label)

        progress_layout.addLayout(status_layout)

        layout.addWidget(progress_group)

        # Log output
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        # Control buttons
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._pause_training)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)

        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(self._resume_training)
        self.resume_btn.setEnabled(False)
        btn_layout.addWidget(self.resume_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        layout.addWidget(QWidget())  # Spacer
        layout.addLayout(btn_layout)

        # Reload model button
        reload_layout = QHBoxLayout()
        self.reload_btn = QPushButton("Reload ML Model")
        self.reload_btn.clicked.connect(self._reload_model)
        reload_layout.addWidget(self.reload_btn)
        reload_layout.addStretch()

        layout.addLayout(reload_layout)

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _refresh_checkpoints(self):
        """Refresh the list of available checkpoints."""
        try:
            from ..ai.ml.trainer import list_checkpoints
            checkpoints = list_checkpoints()
            
            # Clear and repopulate
            self.checkpoint_combo.clear()
            self.checkpoint_combo.addItem("Start Fresh", None)
            
            for cp in reversed(checkpoints):  # Most recent first
                label = f"Step {cp['step']:,} ({cp['name']})"
                self.checkpoint_combo.addItem(label, cp['path'])
            
            # Add option for latest.pt
            latest_path = Path('models/latest.pt')
            if latest_path.exists():
                self.checkpoint_combo.insertItem(1, "Latest (models/latest.pt)", str(latest_path))
            
        except Exception as e:
            self._log(f"Error loading checkpoints: {e}")

    def _start_training(self):
        """Start the training process."""
        # Get resume checkpoint
        resume_path = self.checkpoint_combo.currentData()
        
        if resume_path:
            self._log(f"Resuming from: {resume_path}")
        else:
            self._log("Starting fresh training...")

        # Prepare arguments
        args = {
            'device': self.device_combo.currentText(),
            'amp': True,
            'cpu_workers': self.workers_spin.value(),
            'batch_size': self.batch_spin.value(),
            'train_steps': self.steps_spin.value(),
            'selfplay_games': self.games_spin.value(),
            'checkpoint_every': 1000,
            'resume': resume_path,
            # Model testing settings
            'test_vs_algo': self.test_enabled_check.isChecked(),
            'test_every': self.test_every_spin.value(),
            'test_games': self.test_games_spin.value(),
            'test_difficulty': self.test_difficulty_combo.currentText(),
        }
        
        # Clear live loss data
        self._live_loss_data = []
        self._stop_requested = False

        # Create queues
        self._control_queue = mp.Queue()
        self._status_queue = mp.Queue()

        # Start process
        self._process = mp.Process(
            target=_trainer_process,
            args=(self._control_queue, self._status_queue, args)
        )
        self._process.start()

        # Start worker thread to monitor status
        self._worker = TrainingWorker(self._status_queue, self._process)
        self._worker.status_update.connect(self._on_status_update)
        self._worker.training_complete.connect(self._on_training_complete)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

        # Update UI
        self._set_training_state(True)
        self.state_label.setText("State: Running")
        self.progress_bar.setRange(0, self.steps_spin.value())

    def _pause_training(self):
        """Pause training."""
        if self._control_queue:
            self._control_queue.put({'type': MSG_PAUSE})
            # Transitional label; flips to "Paused" when the trainer's
            # STATUS_REPLY confirms in _on_status_update.
            self.state_label.setText("State: Pausing...")
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)
            self._log("Pause requested")

    def _resume_training(self):
        """Resume training."""
        if self._control_queue:
            self._control_queue.put({'type': MSG_RESUME})
            self.state_label.setText("State: Resuming...")
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)
            self._log("Resume requested")

    def _stop_training(self):
        """Stop training."""
        if self._control_queue:
            self._control_queue.put({'type': MSG_STOP})
            self._stop_requested = True
            self.state_label.setText("State: Stopping...")
            # Pause/Resume make no sense once a stop is in flight
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self._log("Stopping training (will save a final checkpoint)...")

    def _on_status_update(self, status: dict):
        """Handle status update from trainer."""
        # Checkpoint notifications carry only step + path; handle them
        # separately so they don't clobber the richer heartbeat label.
        if status.get('type') == MSG_CHECKPOINT:
            path = status.get('checkpoint_path')
            if path:
                self._log(f"Checkpoint saved: {path}")
                # Refresh stats panel when checkpoint is saved
                self.stats_panel.load_stats()
            return

        step = status.get('step', 0)
        gpu_mem = status.get('gpu_mem_mb', 0)
        epoch = status.get('epoch', 0)
        recent_loss = status.get('recent_loss')
        paused = status.get('paused', False)

        # Reflect the trainer's actual pause state (the buttons set the
        # label optimistically; this confirms once the trainer reacts).
        if ('paused' in status and not self._stop_requested
                and not status.get('complete') and not status.get('stopped')):
            self.state_label.setText("State: Paused" if paused else "State: Running")

        # Update step label with more info
        step_text = f"Step: {step}"
        if epoch > 0:
            step_text += f" | Epoch: {epoch}"
        if recent_loss is not None:
            step_text += f" | Loss: {recent_loss:.4f}"
        self.step_label.setText(step_text)

        self.progress_bar.setValue(step)

        if gpu_mem > 0:
            self.gpu_label.setText(f"GPU: {gpu_mem:.0f} MB")

    def _on_training_complete(self, stopped: bool = False):
        """Handle training completion (natural or user-stopped)."""
        self._log("Training stopped." if stopped else "Training complete!")
        self._cleanup()
        self.state_label.setText("State: Stopped" if stopped else "State: Complete")
        # Refresh stats
        self.stats_panel.load_stats()
        # Refresh checkpoints
        self._refresh_checkpoints()
        if stopped:
            QMessageBox.information(self, "Training Stopped",
                                   "Training was stopped.\n\n"
                                   "A final checkpoint was saved at models/latest.pt\n"
                                   "View the Statistics tab for training metrics.")
        else:
            QMessageBox.information(self, "Training Complete",
                                   "ML model training has completed successfully.\n\n"
                                   "The trained model is saved at models/latest.pt\n"
                                   "View the Statistics tab for training metrics.")

    def _on_error(self, message: str):
        """Handle training error."""
        self._log(f"ERROR: {message}")
        self._cleanup()
        self.state_label.setText("State: Error")
        QMessageBox.warning(self, "Training Error", message)

    def _set_training_state(self, running: bool):
        """Update UI for training state."""
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(running)

        # Disable config during training
        self.device_combo.setEnabled(not running)
        self.workers_spin.setEnabled(not running)
        self.batch_spin.setEnabled(not running)
        self.steps_spin.setEnabled(not running)
        self.games_spin.setEnabled(not running)
        self.checkpoint_combo.setEnabled(not running)

    def _cleanup(self):
        """Clean up after training stops."""
        if self._worker:
            self._worker.stop()
            self._worker.wait()
            self._worker = None

        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                # terminate() (SIGTERM) did not work; force-kill so the
                # trainer cannot outlive the GUI.
                self._process.kill()
                self._process.join(timeout=2)

        self._process = None
        self._control_queue = None
        self._status_queue = None

        self._set_training_state(False)

    def shutdown(self):
        """Stop all child processes and worker threads (called on app close)."""
        self._cleanup()
        self.test_panel._cleanup_test()

    def _reload_model(self):
        """Reload the ML model for inference."""
        try:
            from ..ai.ml.inference import clear_model_cache
            clear_model_cache()
            self._log("ML model cache cleared. New model will be loaded on next AI move.")
            QMessageBox.information(self, "Model Reloaded",
                                   "The ML model cache has been cleared.\n"
                                   "The latest model will be loaded on the next AI move.")
        except Exception as e:
            self._log(f"Failed to reload model: {e}")

    def closeEvent(self, event):
        """Handle panel close."""
        self.shutdown()
        event.accept()
