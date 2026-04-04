"""
Pytest configuration and fixtures.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def src_dir(project_root):
    """Return the src directory."""
    return project_root / "src"


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def initial_game_state():
    """Create an initial game state."""
    from dama.game_state import GameState
    return GameState.initial()


@pytest.fixture
def sample_board():
    """Create an initial board."""
    from dama.board import Board
    return Board.initial()
