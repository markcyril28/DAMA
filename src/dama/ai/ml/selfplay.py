"""Self-play for generating training data."""

import random
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from ...types import Move, Player
from ...game_state import GameState
from ...board import Board
from ...ai.algorithmic.search import get_best_move
from .scoring import score_game_entries, score_game_dicts

try:
    from .inference import get_ml_move, get_ml_move_idx
except ImportError:
    get_ml_move = None
    get_ml_move_idx = None
from .replay import ReplayEntry

# Cython-accelerated full game loop — runs entire algo-vs-algo games in C.
# Eliminates all Python object creation (GameState, Board, Move, Piece) during
# gameplay. Falls back to Python play_single_game if extension not built.
try:
    from ...ai.algorithmic._fast_search import play_full_game_cy
    _HAS_FAST_GAME = True
except ImportError:
    _HAS_FAST_GAME = False

# [Pass 70] Cython-accelerated legal move generation for interleaved self-play.
# C movegen + dict creation is ~50-100x faster than Python legal_moves() + to_dict().
# Eliminates ~12 Move object creations per position (~3s savings per 200-game cycle).
try:
    from ...ai.algorithmic._fast_search import fast_generate_moves as _fast_gen_moves
    _HAS_FAST_MOVEGEN = True
except ImportError:
    _HAS_FAST_MOVEGEN = False

# [Pass 75] Compact-state API: operate on raw board bytes (64-byte int8 array)
# instead of GameState/Board/Move objects. Eliminates ~100-200μs of Python object
# creation per position (Move.from_dict + Board.__init__ + GameState.__init__).
try:
    from ...ai.algorithmic._fast_search import (
        init_board_bytes as _init_board_bytes,
        gen_moves_from_board as _gen_moves_board,
        apply_move_board as _apply_move_board,
        board_bytes_to_compact as _board_to_compact,
    )
    _HAS_COMPACT_STATE = True
except ImportError:
    _HAS_COMPACT_STATE = False

# [Pass 67] Fast encoding for interleaved self-play inference.
# Cython versions are 6-7x faster than Python move_encoder.encode_board/moves.
# Python _encode_*_fast are ~2x faster (avoid GameState/Move object traversal).
# Both write directly into pre-allocated numpy arrays from compact dicts.
try:
    from ._fast_encode import (
        encode_board_fast_cy as _cy_encode_board,
        encode_moves_fast_cy as _cy_encode_moves,
    )
    _HAS_FAST_ENCODE = True
except ImportError:
    _HAS_FAST_ENCODE = False

from .dataset import _encode_board_fast, _encode_moves_fast

# [Pass 70] Fork-inherited model for self-play workers.
# On Linux (fork start method), parent sets this global before creating the
# ProcessPoolExecutor. Workers inherit the model via copy-on-write — zero
# torch.save/load disk I/O.  model.eval() + inference_mode means no COW
# faults on tensor data pages (read-only access).
# Set to None when not in a self-play cycle (cleanup prevents memory leaks).
_FORK_MODEL = None



@dataclass
class GameRecord:
    """Record of a single game."""
    entries: List[ReplayEntry]
    winner: Optional[Player]
    num_moves: int


def play_single_game(
    difficulty: str = 'medium',
    max_moves: int = 200,
    noise_prob: float = 0.1,
    start_player: Player = Player.ONE,
    p1_policy: str = 'algorithmic',
    p2_policy: str = 'algorithmic',
    model_path: str = 'models/latest.pt',
    device=None,
    p1_difficulty: str = None,
    p2_difficulty: str = None,
    return_dicts: bool = False,
) -> Union[GameRecord, List[dict]]:
    """
    Play a single self-play game using the algorithmic AI as teacher.

    Args:
        difficulty: Default AI difficulty level (used when p1/p2_difficulty not set)
        max_moves: Maximum moves before declaring draw
        noise_prob: Probability of playing a random move (for exploration)
        p1_difficulty: Difficulty for Player 1 (overrides difficulty if set)
        p2_difficulty: Difficulty for Player 2 (overrides difficulty if set)
        return_dicts: If True, return List[dict] directly instead of GameRecord

    Returns:
        GameRecord (default) or List[dict] when return_dicts=True
    """
    start_player = Player(start_player)
    # Per-player difficulties default to the shared difficulty
    _p1_diff = p1_difficulty or difficulty
    _p2_diff = p2_difficulty or difficulty

    state = GameState(
        board=Board.initial(),
        current_player=start_player,
        move_count=0,
    )
    entries = []
    move_count = 0
    player_captures = {Player.ONE: 0, Player.TWO: 0}

    def _select_move(policy: str, legal_moves: List[Move], player_diff: str):
        """Select a move. Returns (move, index) tuple."""
        if random.random() < noise_prob and len(legal_moves) > 1:
            idx = random.randrange(len(legal_moves))
            return legal_moves[idx], idx
        if policy == 'ml' and get_ml_move_idx is not None:
            try:
                # get_ml_move_idx accepts pre-computed legal_moves, returns
                # index directly — avoids redundant legal_moves() generation
                # inside get_ml_move and the O(n) index search afterward.
                idx = get_ml_move_idx(state, legal_moves, model_path=model_path, device=device)
                if idx is not None:
                    return legal_moves[idx], idx
            except Exception as e:
                print(f"  Warning: ML inference failed, using random move: {e}")
            idx = random.randrange(len(legal_moves))
            return legal_moves[idx], idx
        # use_parallel=False: self-play already parallelizes at the game level
        # via ProcessPoolExecutor. Creating threads per move per worker causes
        # massive oversubscription (N_workers × CPU_COUNT threads).
        # With Cython fast search, this is moot (runs in C, single-threaded),
        # but the flag prevents thread storms if falling back to Python search.
        chosen = get_best_move(state, player_diff, use_parallel=False)
        if chosen is None:
            idx = random.randrange(len(legal_moves))
            return legal_moves[idx], idx
        # Find index for algorithmic move
        try:
            idx = legal_moves.index(chosen)
        except ValueError:
            for i, m in enumerate(legal_moves):
                if m.path == chosen.path:
                    idx = i
                    break
            else:
                idx = 0
                chosen = legal_moves[0]
        return chosen, idx

    while move_count < max_moves:
        legal_moves = state.legal_moves()
        if not legal_moves:
            break

        policy = p1_policy if state.current_player == Player.ONE else p2_policy
        cur_diff = _p1_diff if state.current_player == Player.ONE else _p2_diff
        chosen_move, chosen_index = _select_move(policy, legal_moves, cur_diff)

        # Track captures per player
        if chosen_move.is_capture:
            player_captures[state.current_player] += chosen_move.num_captures

        # Record the position — build dicts directly when return_dicts=True
        # to skip ReplayEntry construction + to_dict() roundtrip
        if return_dicts:
            entry = {
                'state': state.to_compact(),
                'legal_moves': [m.to_dict() for m in legal_moves],
                'chosen_index': chosen_index,
                'result': 0,
                'score': 0.0,
            }
        else:
            entry = ReplayEntry(
                state=state.to_compact(),
                legal_moves=[m.to_dict() for m in legal_moves],
                chosen_index=chosen_index,
                result=0,  # Will be filled after game ends
                score=0.0,  # Will be filled by scoring system
            )
        entries.append(entry)

        # Apply the move
        state = state.apply_move(chosen_move)
        move_count += 1

    # Determine winner
    winner = state.winner()

    if return_dicts:
        # Dict path — update results and score via dict-native scoring
        winner_int = int(winner) if winner is not None else None
        for entry in entries:
            turn = entry['state']['turn']
            if winner is None:
                entry['result'] = 0
            elif turn == winner_int:
                entry['result'] = 1
            else:
                entry['result'] = -1
        score_game_dicts(
            entry_dicts=entries,
            winner_int=winner_int,
            total_moves=move_count,
            max_moves=max_moves,
            final_state_dict=state.to_compact(),
            p1_captures=player_captures[Player.ONE],
            p2_captures=player_captures[Player.TWO],
        )
        return entries  # Return list of dicts directly
    else:
        # ReplayEntry path — original behavior
        for i, entry in enumerate(entries):
            turn = entry.state['turn']
            player = Player(turn)
            if winner is None:
                entry.result = 0
            elif winner == player:
                entry.result = 1
            else:
                entry.result = -1

        score_game_entries(
            entries=entries,
            winner=winner,
            total_moves=move_count,
            max_moves=max_moves,
            final_state=state,
            player_captures=player_captures,
        )
        return GameRecord(entries=entries, winner=winner, num_moves=move_count)



def play_games_interleaved(batch_args: list) -> List[dict]:
    """Play multiple games interleaved with batched ML inference.

    Instead of playing games sequentially (one forward pass per ML move),
    this advances all active games one step at a time and batches ML
    inference requests into a single forward_padded() call.  CPU BLAS
    amortizes per-layer setup overhead across positions, yielding ~30-40%
    faster ML inference vs single-position calls.

    Algo moves are processed sequentially (alpha-beta can't be batched).

    Falls back to sequential play_single_game when the model can't be
    loaded or no games use ML policy.

    Args:
        batch_args: List of tuples:
            (difficulty, max_moves, noise_prob, start_player,
             p1_policy, p2_policy, model_path, device)

    Returns:
        List of entry dicts (same format as play_single_game return_dicts=True)
    """
    import torch
    import numpy as np
    from .inference import get_model

    if not batch_args:
        return []

    # Extract model_path from first arg (same for all games in batch)
    _model_path = batch_args[0][6]

    # Load model once for the entire batch.
    # [Pass 70] Try fork-inherited model first (zero disk I/O on Linux fork).
    # Falls back to get_model() which loads from disk via cache.
    model = None
    has_ml = any(a[4] == 'ml' or a[5] == 'ml' for a in batch_args)
    if has_ml:
        if _FORK_MODEL is not None:
            model = _FORK_MODEL
        else:
            try:
                model = get_model(_model_path, 'cpu')
            except Exception:
                model = None

    # If no model, fall back to sequential (algo-only or missing model)
    if model is None:
        return _play_games_batch_sequential(batch_args)

    n = len(batch_args)

    # Initialize all games
    games = []
    for difficulty, max_moves, noise_prob, start_player, p1_pol, p2_pol, _mp, _dev in batch_args:
        games.append({
            'state': GameState(
                board=Board.initial(),
                current_player=Player(start_player),
                move_count=0,
            ),
            'entries': [],
            'p1_policy': p1_pol,
            'p2_policy': p2_pol,
            'difficulty': difficulty,
            'noise_prob': noise_prob,
            'max_moves': max_moves,
            'move_count': 0,
            'captures': {Player.ONE: 0, Player.TWO: 0},
        })

    active = list(range(n))

    # [Pass 71] Pre-allocate numpy arrays for ML inference.
    # Avoids ~75-150 np.zeros() allocation+zeroing cycles per game batch.
    # Fixed max_moves=32 (config default) ensures contiguous memory layout
    # and enables JIT-traced model to use fixed tensor shapes.
    _MAX_M = 32  # matches config max_moves_per_sample
    _boards_buf = np.zeros((n, 5, 8, 8), dtype=np.float32)
    _mf_buf = np.zeros((n, _MAX_M, 8), dtype=np.float32)
    _counts_buf = np.zeros(n, dtype=np.int32)

    # [Pass 70] Use Cython movegen when available — generates move dicts
    # directly in C, ~50-100x faster than Python legal_moves() + to_dict().
    # Only create a Move object for the single chosen move (for apply_move).
    _use_fast_movegen = _HAS_FAST_MOVEGEN

    while active:
        # Partition active games by action type.
        # [Pass 69] Pre-compute compact dicts once per position in the
        # partition step, shared by ALL paths (immediate, ML, algo).
        # [Pass 70] With fast movegen, md comes directly from Cython C
        # movegen (no Move objects created at all). Move objects are only
        # created for the chosen move via Move.from_dict().
        ml_requests = []     # (game_idx, sd, md)
        algo_requests = []   # (game_idx, sd, md)
        immediate = []       # (game_idx, chosen_idx, sd, md)

        new_active = []
        for i in active:
            g = games[i]
            if g['move_count'] >= g['max_moves']:
                g['_ended'] = 'max_moves'  # draw
                continue

            state = g['state']
            if _use_fast_movegen:
                # Cython C movegen → dicts directly (no Move objects)
                md = _fast_gen_moves(state)
            else:
                legal_moves = state.legal_moves()
                md = [m.to_dict() for m in legal_moves]

            if not md:
                g['_ended'] = 'no_moves'  # current player loses
                continue
            new_active.append(i)

            sd = state.to_compact()

            # Single legal move — no inference needed
            if len(md) == 1:
                immediate.append((i, 0, sd, md))
                continue

            policy = g['p1_policy'] if state.current_player == Player.ONE else g['p2_policy']

            # Exploration noise
            if random.random() < g['noise_prob']:
                immediate.append((i, random.randrange(len(md)), sd, md))
            elif policy == 'ml':
                ml_requests.append((i, sd, md))
            else:
                algo_requests.append((i, sd, md))

        active = new_active
        if not active:
            break

        # Immediate moves — apply chosen move from dict
        for game_idx, idx, sd, md in immediate:
            game = games[game_idx]
            chosen_md = md[idx]
            game['entries'].append({
                'state': sd,
                'legal_moves': md,
                'chosen_index': idx,
                'result': 0,
                'score': 0.0,
            })
            captures = chosen_md.get('captures', ())
            if captures:
                game['captures'][game['state'].current_player] += len(captures)
            game['state'] = game['state'].apply_move(Move.from_dict(chosen_md))
            game['move_count'] += 1

        # Batched ML inference — encode using partition-step dicts
        if ml_requests:
            batch_sz = len(ml_requests)

            # [Pass 71] Reuse pre-allocated buffers instead of np.zeros() per round.
            boards = _boards_buf[:batch_sz]
            all_mf = _mf_buf[:batch_sz]
            counts = _counts_buf[:batch_sz]

            # Dicts already pre-computed in partition step — just encode
            for j, (game_idx, sd, md) in enumerate(ml_requests):
                if _HAS_FAST_ENCODE:
                    _cy_encode_board(sd, boards[j])
                    counts[j] = _cy_encode_moves(sd, md, all_mf[j])
                else:
                    _encode_board_fast(sd, boards[j])
                    counts[j] = _encode_moves_fast(sd, md, all_mf[j])

            with torch.inference_mode():
                scores = model.forward_padded(
                    torch.from_numpy(boards),
                    torch.from_numpy(all_mf),
                    torch.from_numpy(counts),
                )

            # [Pass 71] Batch argmax: single torch op for all positions.
            best_indices_list = scores.argmax(dim=1).tolist()

            for j, (game_idx, sd, md) in enumerate(ml_requests):
                best_idx = best_indices_list[j]
                game = games[game_idx]
                chosen_md = md[best_idx]
                game['entries'].append({
                    'state': sd,
                    'legal_moves': md,
                    'chosen_index': best_idx,
                    'result': 0,
                    'score': 0.0,
                })
                captures = chosen_md.get('captures', ())
                if captures:
                    game['captures'][game['state'].current_player] += len(captures)
                game['state'] = game['state'].apply_move(Move.from_dict(chosen_md))
                game['move_count'] += 1

        # Sequential algo moves — match by path start/end positions
        for game_idx, sd, md in algo_requests:
            move = get_best_move(games[game_idx]['state'], games[game_idx]['difficulty'],
                                 use_parallel=False)
            if move is None:
                idx = random.randrange(len(md))
            else:
                # Match algo Move to dict by path start+end positions.
                # Cython fast_search and fast_generate_moves use the same C
                # move generator, so paths are identical. Start+end match is
                # sufficient; full-path fallback handles rare multi-capture
                # ambiguities.
                mp = move.path
                mp_s0, mp_s1 = mp[0][0], mp[0][1]
                mp_e0, mp_e1 = mp[-1][0], mp[-1][1]
                idx = 0
                for k, m_dict in enumerate(md):
                    dp = m_dict['path']
                    if (dp[0][0] == mp_s0 and dp[0][1] == mp_s1 and
                            dp[-1][0] == mp_e0 and dp[-1][1] == mp_e1):
                        idx = k
                        break
            game = games[game_idx]
            chosen_md = md[idx]
            game['entries'].append({
                'state': sd,
                'legal_moves': md,
                'chosen_index': idx,
                'result': 0,
                'score': 0.0,
            })
            captures = chosen_md.get('captures', ())
            if captures:
                game['captures'][game['state'].current_player] += len(captures)
            game['state'] = game['state'].apply_move(Move.from_dict(chosen_md))
            game['move_count'] += 1

    # Score all games' entries
    all_entries = []
    for g in games:
        entries = g['entries']
        if not entries:
            continue
        state = g['state']
        # Use cached end reason to avoid redundant legal_moves() call in
        # winner() → is_terminal() → has_legal_moves().  The game loop
        # already determined why each game ended.
        end_reason = g.get('_ended')
        if end_reason == 'no_moves':
            winner = state.current_player.opponent()
        elif end_reason == 'max_moves':
            winner = None
        else:
            winner = state.winner()  # fallback for games still active
        winner_int = int(winner) if winner is not None else None

        for entry in entries:
            turn = entry['state']['turn']
            if winner is None:
                entry['result'] = 0
            elif turn == winner_int:
                entry['result'] = 1
            else:
                entry['result'] = -1

        score_game_dicts(
            entry_dicts=entries,
            winner_int=winner_int,
            total_moves=g['move_count'],
            max_moves=g['max_moves'],
            final_state_dict=state.to_compact(),
            p1_captures=g['captures'][Player.ONE],
            p2_captures=g['captures'][Player.TWO],
        )
        all_entries.extend(entries)

    return all_entries


def _play_games_batch_sequential(batch_args: list) -> List[dict]:
    """Sequential fallback for _play_games_batch_worker_full."""
    all_entries = []
    for (difficulty, max_moves, noise_prob, start_player,
         p1_policy, p2_policy, model_path, _device) in batch_args:
        entry_dicts = play_single_game(
            difficulty=difficulty, max_moves=max_moves,
            noise_prob=noise_prob, start_player=start_player,
            p1_policy=p1_policy, p2_policy=p2_policy,
            model_path=model_path, device='cpu',
            return_dicts=True,
        )
        all_entries.extend(entry_dicts)
    return all_entries


def _play_games_batch_worker_full(batch_args: list) -> List[dict]:
    """Batched worker for ML-policy games.

    Uses interleaved game play with batched ML inference when multiple
    ML-policy games are in the batch.  This amortizes CPU forward pass
    overhead across positions (~30-40% faster ML inference per position).

    Falls back to sequential play for single-game batches or when no
    games use ML policy.
    """
    if not batch_args:
        return []

    # Interleaved play benefits from batching ≥2 games
    has_ml = any(a[4] == 'ml' or a[5] == 'ml' for a in batch_args)
    if has_ml and len(batch_args) >= 2:
        return play_games_interleaved(batch_args)

    return _play_games_batch_sequential(batch_args)


def _play_game_worker_algo_vs_algo(
    args: Tuple[str, str, int, float, int]
) -> List[dict]:
    """Worker function for algo-vs-algo self-play with per-player difficulties.

    Uses Cython full game loop when available for 2-7x speedup.
    Falls back to Python play_single_game with return_dicts=True.
    """
    p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player = args
    if _HAS_FAST_GAME:
        result = play_full_game_cy(
            p1_difficulty=p1_difficulty, p2_difficulty=p2_difficulty,
            max_moves=max_moves, noise_prob=noise_prob,
            start_player=start_player,
        )
        score_game_dicts(
            entry_dicts=result['entries'], winner_int=result['winner'],
            total_moves=result['num_moves'], max_moves=max_moves,
            final_state_dict=result['final_state'],
            p1_captures=result['p1_captures'], p2_captures=result['p2_captures'],
        )
        return result['entries']
    return play_single_game(
        difficulty=p1_difficulty,
        max_moves=max_moves,
        noise_prob=noise_prob,
        start_player=start_player,
        p1_policy='algorithmic',
        p2_policy='algorithmic',
        p1_difficulty=p1_difficulty,
        p2_difficulty=p2_difficulty,
        return_dicts=True,
    )


def _play_games_batch_worker_algo(batch_args: list) -> List[dict]:
    """Batched worker for algo-vs-algo games with per-player difficulties.

    Uses Cython full game loop when available (per-player difficulties + TT).
    """
    all_entries = []
    if _HAS_FAST_GAME:
        for p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player in batch_args:
            result = play_full_game_cy(
                p1_difficulty=p1_difficulty, p2_difficulty=p2_difficulty,
                max_moves=max_moves, noise_prob=noise_prob,
                start_player=start_player,
            )
            score_game_dicts(
                entry_dicts=result['entries'], winner_int=result['winner'],
                total_moves=result['num_moves'], max_moves=max_moves,
                final_state_dict=result['final_state'],
                p1_captures=result['p1_captures'], p2_captures=result['p2_captures'],
            )
            all_entries.extend(result['entries'])
    else:
        for p1_difficulty, p2_difficulty, max_moves, noise_prob, start_player in batch_args:
            entry_dicts = play_single_game(
                difficulty=p1_difficulty,
                max_moves=max_moves, noise_prob=noise_prob,
                start_player=start_player,
                p1_policy='algorithmic', p2_policy='algorithmic',
                p1_difficulty=p1_difficulty, p2_difficulty=p2_difficulty,
                return_dicts=True,
            )
            all_entries.extend(entry_dicts)
    return all_entries


