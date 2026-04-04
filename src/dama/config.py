"""Configuration management for Filipino Dama."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict


def get_config_dir() -> Path:
    """Get the configuration directory."""
    # Use XDG on Linux/WSL, or fallback
    if os.name == 'nt':
        config_base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:
        config_base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))

    config_dir = config_base / 'dama'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file() -> Path:
    """Get the configuration file path."""
    return get_config_dir() / 'settings.yaml'


@dataclass
class BoardColors:
    """Board color settings."""
    light_color: str = "#F0D9B5"
    dark_color: str = "#B58863"
    highlight_color: str = "#7FFF00"


@dataclass
class PieceColors:
    """Piece color settings."""
    p1_color: str = "#FFFFFF"
    p2_color: str = "#000000"
    style: str = "flat"  # flat, outlined, sprite


@dataclass
class UISettings:
    """UI-related settings."""
    board: BoardColors = field(default_factory=BoardColors)
    pieces: PieceColors = field(default_factory=PieceColors)


@dataclass
class RuleSettings:
    """Game rule settings."""
    forced_capture: bool = True
    multi_jump: bool = True
    backward_capture: bool = False  # Allow regular pieces to capture backwards
    king_flying_capture: bool = True  # Kings can capture pieces at any distance along diagonal


@dataclass
class GameSettings:
    """Game-related settings."""
    rules: RuleSettings = field(default_factory=RuleSettings)


@dataclass
class AlgorithmicAISettings:
    """Algorithmic AI settings."""
    difficulty: str = "medium"  # easy, medium, hard, super_hard, custom
    # Custom parameters (used when difficulty is "custom")
    time_budget: float = 1.0  # seconds
    max_depth: int = 6
    # Multithreading
    use_parallel: bool = True
    num_threads: int = 0  # 0 = auto (use all CPU cores)
    # Evaluation weights
    weight_man: int = 100
    weight_king: int = 200
    weight_mobility: int = 5
    weight_advancement: int = 2
    weight_center_control: int = 3
    weight_back_rank: int = 10


@dataclass
class MLAISettings:
    """ML AI settings."""
    model_path: str = "models/latest.pt"
    available_models: str = ""  # Comma-separated list of discovered model paths


@dataclass
class AISettings:
    """AI-related settings."""
    algorithmic: AlgorithmicAISettings = field(default_factory=AlgorithmicAISettings)
    ml: MLAISettings = field(default_factory=MLAISettings)


@dataclass
class PlayerSettings:
    """Player configuration."""
    p1_type: str = "human"  # human, algorithmic, ml
    p2_type: str = "human"


@dataclass
class Config:
    """Main configuration class."""
    ui: UISettings = field(default_factory=UISettings)
    game: GameSettings = field(default_factory=GameSettings)
    ai: AISettings = field(default_factory=AISettings)
    players: PlayerSettings = field(default_factory=PlayerSettings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'ui': {
                'board': asdict(self.ui.board),
                'pieces': asdict(self.ui.pieces),
            },
            'game': {
                'rules': asdict(self.game.rules),
            },
            'ai': {
                'algorithmic': asdict(self.ai.algorithmic),
                'ml': asdict(self.ai.ml),
            },
            'players': asdict(self.players),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()

        if 'ui' in data:
            ui_data = data['ui']
            if 'board' in ui_data:
                config.ui.board = BoardColors(**ui_data['board'])
            if 'pieces' in ui_data:
                config.ui.pieces = PieceColors(**ui_data['pieces'])

        if 'game' in data:
            game_data = data['game']
            if 'rules' in game_data:
                config.game.rules = RuleSettings(**game_data['rules'])

        if 'ai' in data:
            ai_data = data['ai']
            if 'algorithmic' in ai_data:
                config.ai.algorithmic = AlgorithmicAISettings(**ai_data['algorithmic'])
            if 'ml' in ai_data:
                config.ai.ml = MLAISettings(**ai_data['ml'])

        if 'players' in data:
            config.players = PlayerSettings(**data['players'])

        return config

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = get_config_file()

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from file."""
        if path is None:
            path = get_config_file()

        if not path.exists():
            return cls()

        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                if data is None:
                    return cls()
                return cls.from_dict(data)
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
            return cls()


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def save_config() -> None:
    """Save the global configuration."""
    global _config
    if _config is not None:
        _config.save()


def reset_config() -> Config:
    """Reset configuration to defaults."""
    global _config
    _config = Config()
    _config.save()
    return _config
