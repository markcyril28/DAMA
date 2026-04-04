"""Main entry point for Filipino Dama."""

import sys
import multiprocessing as mp

# Set spawn method for CUDA compatibility before any other multiprocessing imports
# This is required for GPU training to work with multiprocessing
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # Already set


def print_initial_state():
    """Print the initial board state and legal moves (Milestone A acceptance test)."""
    from .game_state import GameState

    state = GameState.initial()

    print("=" * 40)
    print("Filipino Dama - Initial State")
    print("=" * 40)
    print()
    print(state)
    print()

    moves = state.legal_moves()
    print(f"Legal moves for Player {state.current_player}: {len(moves)}")
    print()
    for i, move in enumerate(moves, 1):
        print(f"  {i}. {move}")
    print()


def main():
    """Main entry point."""
    # Check if we should just print the board (Milestone A test mode)
    if "--test" in sys.argv:
        print_initial_state()
        return

    # Try to import PyQt6 and run the GUI
    try:
        from .ui.main_window import run_gui
        run_gui()
    except ImportError as e:
        print(f"GUI not available: {e}")
        print("Running in test mode instead...")
        print()
        print_initial_state()


if __name__ == "__main__":
    main()
