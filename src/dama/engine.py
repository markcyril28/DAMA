"""Game engine - orchestrates game play."""

from enum import Enum
from typing import Optional, Callable, List
from dataclasses import dataclass

from .types import Move, Player
from .game_state import GameState
from .config import get_config


class PlayerType(Enum):
    """Type of player."""
    HUMAN = "human"
    ALGORITHMIC = "algorithmic"
    ML = "ml"


@dataclass
class GameResult:
    """Result of a completed game."""
    winner: Optional[Player]
    total_moves: int
    final_state: GameState


class Engine:
    """
    Game engine that orchestrates turns and dispatches to appropriate handlers.

    For Human players, the engine waits for move input.
    For AI players, the engine requests moves from the AI modules.
    """

    def __init__(self):
        self.state: GameState = GameState.initial()
        self.move_history: List[Move] = []
        self.state_history: List[GameState] = []

        # Player types (default from config)
        config = get_config()
        self.player_types = {
            Player.ONE: PlayerType(config.players.p1_type),
            Player.TWO: PlayerType(config.players.p2_type),
        }

        # Callbacks
        self.on_state_changed: Optional[Callable[[GameState], None]] = None
        self.on_game_over: Optional[Callable[[GameResult], None]] = None
        self.on_move_request: Optional[Callable[[Player, PlayerType], None]] = None

        # AI modules (lazy loaded)
        self._algorithmic_ai = None
        self._ml_ai = None

    def new_game(self) -> None:
        """Start a new game."""
        self.state = GameState.initial()
        self.move_history = []
        self.state_history = [self.state]
        self._notify_state_changed()
        self._request_move_if_ai()

    def set_player_type(self, player: Player, player_type: PlayerType) -> None:
        """Set the type of a player."""
        self.player_types[player] = player_type

    def get_player_type(self, player: Player) -> PlayerType:
        """Get the type of a player."""
        return self.player_types[player]

    def get_current_player_type(self) -> PlayerType:
        """Get the type of the current player."""
        return self.player_types[self.state.current_player]

    def legal_moves(self) -> List[Move]:
        """Get legal moves for the current player."""
        return self.state.legal_moves()

    def make_move(self, move: Move) -> bool:
        """
        Make a move in the game.

        Returns True if the move was valid and applied.
        """
        legal = self.state.legal_moves()
        if move not in legal:
            # Try to find a matching move (path comparison)
            matching = [m for m in legal if m.path == move.path]
            if matching:
                move = matching[0]
            else:
                return False

        # Apply the move
        self.state_history.append(self.state)
        self.move_history.append(move)
        self.state = self.state.apply_move(move)

        self._notify_state_changed()

        # Check for game over
        if self.state.is_terminal():
            result = GameResult(
                winner=self.state.winner(),
                total_moves=len(self.move_history),
                final_state=self.state,
            )
            if self.on_game_over:
                self.on_game_over(result)
        else:
            # Request next move if AI
            self._request_move_if_ai()

        return True

    def undo(self) -> bool:
        """Undo the last move. Returns True if successful."""
        if not self.state_history:
            return False

        self.state = self.state_history.pop()
        if self.move_history:
            self.move_history.pop()

        self._notify_state_changed()
        return True

    def get_ai_move(self, player_type: PlayerType) -> Optional[Move]:
        """Get a move from the appropriate AI."""
        if player_type == PlayerType.ALGORITHMIC:
            return self._get_algorithmic_move()
        elif player_type == PlayerType.ML:
            return self._get_ml_move()
        return None

    def _get_algorithmic_move(self) -> Optional[Move]:
        """Get a move from the algorithmic AI."""
        try:
            from .ai.algorithmic.search import get_best_move
            config = get_config()
            algo_config = config.ai.algorithmic
            difficulty = algo_config.difficulty
            
            # Get parallel search settings
            use_parallel = algo_config.use_parallel
            num_threads = algo_config.num_threads if algo_config.num_threads > 0 else None
            
            return get_best_move(
                self.state, 
                difficulty, 
                use_parallel=use_parallel,
                num_threads=num_threads
            )
        except ImportError as e:
            print(f"Algorithmic AI not available: {e}")
            # Fallback to random legal move
            moves = self.legal_moves()
            if moves:
                import random
                return random.choice(moves)
        return None

    def _get_ml_move(self) -> Optional[Move]:
        """Get a move from the ML AI."""
        try:
            from .ai.ml.inference import get_ml_move
            config = get_config()
            model_path = config.ai.ml.model_path
            return get_ml_move(self.state, model_path)
        except ImportError as e:
            print(f"ML AI not available: {e}")
            # Fallback to algorithmic
            return self._get_algorithmic_move()
        except FileNotFoundError:
            print("ML model not found, falling back to algorithmic AI")
            return self._get_algorithmic_move()

    def _notify_state_changed(self) -> None:
        """Notify listeners of state change."""
        if self.on_state_changed:
            self.on_state_changed(self.state)

    def _request_move_if_ai(self) -> None:
        """Request a move if the current player is AI."""
        player_type = self.get_current_player_type()
        if player_type != PlayerType.HUMAN:
            if self.on_move_request:
                self.on_move_request(self.state.current_player, player_type)
