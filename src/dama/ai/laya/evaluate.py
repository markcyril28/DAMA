"""Measure the Laya player: frozen teacher-suite agreement and balanced games.

Run under the dama env (from the project root, with src on PYTHONPATH):

    python -m dama.ai.laya.evaluate agreement --count 500 --seed 7 [--cnn CKPT]
    python -m dama.ai.laya.evaluate games --opponent random --games 100

Laya runs in its own environment behind the stdio bridge. This process imports
torch only for the optional ``--cnn`` comparison, which runs on the CPU. Every
record goes to one JSONL file, by default models/laya_eval/<cmd>_<UTC stamp>.jsonl,
never under models/test_stats (that tree feeds the training dashboard).
"""

import argparse
import dataclasses
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...config import get_config
from ...game_state import GameState
from ...types import Move, Player
from ..ml import model_vs_algo as mva
from ..ml.acceptance import wilson_score_interval_95
from . import policy
from .errors import LayaBudgetError, LayaError

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # laya -> ai -> dama -> src -> root
DEFAULT_SUITE = "data/validation_policy_distillation/frozen_hard_5000.jsonl"
SUITE_STATE_COUNT = 5000
EVAL_DIR = Path("models") / "laya_eval"
FORBIDDEN_OUTPUT_DIR = Path("models") / "test_stats"
OPPONENTS = ("random", "easy", "medium", "hard")
DEFAULT_OPENING_PLIES = (2, 4, 6, 8)
DEFAULT_OPENING_SEED = 20260819
DEFAULT_MAX_MOVES = 200
# Same salt as model_vs_algo._play_single_test_game, so a random opponent plays
# the same moves it would against the CNN from the same opening.
OPPONENT_RNG_SALT = 0x4D595DF4D0F33173
WILSON_METHOD = "wilson_score_95"
TRAINER_MODULE = "dama.ai.ml.trainer"
TRAINER_TITLES = ("micro-trainer", "micro_trainer")
RUN_STATUS_FILE = "run_status.json"  # dama.ai.ml.run_status.RUN_STATUS_FILENAME
PROGRESS_EVERY = 25
PIN_DIR_PREFIX = ".pinned_"  # <output dir>/.pinned_<pid>: the CNN checkpoint scored by that run

# chooser(state, legal_moves) -> LayaDecision; the CLI wraps policy.choose_laya_move.
Chooser = Callable[[GameState, Sequence[Move]], Optional[policy.LayaDecision]]
CnnEvaluator = Callable[[str, Sequence[Any]], Dict[str, Any]]


# --- agreement --------------------------------------------------------------

def decision_indices(entries: Sequence[Any]) -> List[int]:
    """Suite indices of decision states (more than one stored legal move)."""
    return [i for i, entry in enumerate(entries) if len(entry.legal_moves) > 1]


def forced_fraction(entries: Sequence[Any]) -> float:
    """Share of suite states with a single legal move (always agree by construction)."""
    if not entries:
        return 0.0
    return sum(1 for entry in entries if len(entry.legal_moves) <= 1) / len(entries)


def sample_decision_states(entries: Sequence[Any], count: int, seed: int) -> List[int]:
    """Suite indices of count decision states, drawn with random.Random(seed).sample.

    Draw order is kept (not sorted), so any prefix of an interrupted run is
    itself a uniform sample. The selected set equals Random(seed).sample(dec, count).
    """
    decision = decision_indices(entries)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(decision):
        raise ValueError(f"--count must be between 1 and {len(decision)} "
                         f"(decision states in the suite), got {count!r}")
    picked = random.Random(seed).sample(range(len(decision)), count)
    return [decision[j] for j in picked]


def random_baseline(legal_counts: Sequence[int]) -> float:
    """Expected top-1 agreement of a uniform random chooser: mean of 1/n."""
    if not legal_counts:
        return 0.0
    return sum(1.0 / n for n in legal_counts) / len(legal_counts)


def wilson_interval(hits: int, total: int) -> Dict[str, Any]:
    """95% Wilson score interval of hits/total."""
    lower, upper = wilson_score_interval_95(hits, 0, total - hits)
    return {"lower": lower, "upper": upper, "method": WILSON_METHOD}


def all_state_equivalent(decision_accuracy: float, forced: float) -> float:
    """All-state agreement implied by a decision-state accuracy (forced states always agree)."""
    return forced + (1.0 - forced) * decision_accuracy


def score_state(entry: Any, choose: Chooser) -> Dict[str, Any]:
    """Ask the chooser about one suite state and score it against the stored label.

    The legal list is rebuilt in the STORED order, because chosen_index indexes
    it; live move generation orders some positions differently.
    """
    state = GameState.from_compact(entry.state)
    legal = [Move.from_dict(m) for m in entry.legal_moves]
    teacher = int(entry.chosen_index)
    row: Dict[str, Any] = {"legal_count": len(legal), "teacher_index": teacher,
                           "side_to_move": int(state.current_player)}
    try:
        decision = choose(state, legal)
    except LayaBudgetError as exc:
        row.update(hit=False, laya_index=None, error=f"budget: {exc}")
        return row
    if decision is None or not 0 <= decision.index < len(legal) or decision.move != legal[decision.index]:
        raise RuntimeError("Laya chooser returned a move outside the stored legal list")
    labels = decision.labels
    row.update(
        hit=decision.index == teacher,
        laya_index=decision.index,
        teacher_label=labels[teacher] if teacher < len(labels) else None,
        laya_label=labels[decision.index] if decision.index < len(labels) else None,
        teacher_probability=(float(decision.probabilities[teacher])
                             if teacher < len(decision.probabilities) else None),
        probabilities=[float(p) for p in decision.probabilities],
        confidence=float(decision.confidence),
        device=decision.device,
        elapsed_ms=float(decision.elapsed_ms),
        label_style=decision.label_style,
        head_max_len=int(decision.head_max_len),
        error=None,
    )
    return row


def run_agreement(
    entries: Sequence[Any],
    indices: Sequence[int],
    choose: Chooser,
    *,
    on_row: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Score choose on entries[i] for i in indices; a Laya failure ends the run as aborted."""
    rows: List[Dict[str, Any]] = []
    status, error = "complete", None
    try:
        for i in indices:
            row = {"suite_index": int(i), **score_state(entries[i], choose)}
            rows.append(row)
            if on_row is not None:
                on_row(row)
    except LayaError as exc:  # budget errors are scored per state inside score_state
        status, error = "aborted", str(exc)
    except KeyboardInterrupt:
        status, error = "interrupted", "KeyboardInterrupt"
    return {"status": status, "error": error, "rows": rows}


def summarize_agreement(rows: Sequence[Dict[str, Any]], forced: float) -> Dict[str, Any]:
    """Decision-state accuracy, Wilson interval, random baseline and all-state equivalents."""
    scored = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    answered = [r for r in rows if r.get("error") is None]
    elapsed = [r["elapsed_ms"] for r in answered if r.get("elapsed_ms") is not None]
    teacher_p = [r["teacher_probability"] for r in answered if r.get("teacher_probability") is not None]
    if scored:
        accuracy = hits / scored
        ci = wilson_interval(hits, scored)
        baseline = random_baseline([r["legal_count"] for r in rows])
        figures = {
            "decision_accuracy": accuracy,
            "decision_accuracy_ci95": ci,
            "random_baseline_decision_accuracy": baseline,
            "lift_over_random": accuracy - baseline,
            "all_state_equivalent": all_state_equivalent(accuracy, forced),
            "all_state_equivalent_ci95": {
                "lower": all_state_equivalent(ci["lower"], forced),
                "upper": all_state_equivalent(ci["upper"], forced),
                "method": WILSON_METHOD,
            },
            "random_baseline_all_state_equivalent": all_state_equivalent(baseline, forced),
        }
    else:
        # Nothing was measured; a 0.0 accuracy or a 0.2296 all-state figure would read as real
        figures = dict.fromkeys((
            "decision_accuracy", "decision_accuracy_ci95", "random_baseline_decision_accuracy",
            "lift_over_random", "all_state_equivalent", "all_state_equivalent_ci95",
            "random_baseline_all_state_equivalent"))
    return {
        "scored": scored,
        "hits": hits,
        "budget_errors": scored - len(answered),
        "decision_accuracy": figures["decision_accuracy"],
        "decision_accuracy_ci95": figures["decision_accuracy_ci95"],
        "random_baseline_decision_accuracy": figures["random_baseline_decision_accuracy"],
        "lift_over_random": figures["lift_over_random"],
        "forced_fraction": forced,
        "sample": "uniform random decision states",
        "all_state_equivalent": figures["all_state_equivalent"],
        "all_state_equivalent_ci95": figures["all_state_equivalent_ci95"],
        "random_baseline_all_state_equivalent": figures["random_baseline_all_state_equivalent"],
        "mean_teacher_probability": sum(teacher_p) / len(teacher_p) if teacher_p else None,
        "mean_elapsed_ms": sum(elapsed) / len(elapsed) if elapsed else None,
        "compact_label_retries": sum(1 for r in answered if r.get("label_style") == "compact"),
        "max_head_max_len": max((r.get("head_max_len") or 0 for r in answered), default=0),
        "devices": dict(Counter(r.get("device") for r in answered)),
    }


def evaluate_cnn(checkpoint: str, entries: Sequence[Any]) -> Dict[str, Any]:
    """Top-1 agreement of a CNN checkpoint on entries, on the CPU."""
    from ..ml.inference import get_model
    from ..ml.teacher_validation import evaluate_teacher_agreement

    return evaluate_teacher_agreement(get_model(checkpoint, "cpu"), list(entries))


def _cnn_summary(checkpoint: str, sha256: str, result: Dict[str, Any], forced: float) -> Dict[str, Any]:
    """CNN block of the summary; every sampled state is a decision state."""
    total = int(result["decision_states"])
    hits = int(result["decision_correct_states"])
    accuracy = hits / total if total else 0.0
    return {
        "checkpoint": checkpoint,
        "sha256": sha256,
        "device": "cpu",
        "scored": total,
        "hits": hits,
        "decision_accuracy": accuracy,
        "decision_accuracy_ci95": wilson_interval(hits, total),
        "all_state_equivalent": all_state_equivalent(accuracy, forced),
        "evaluate_teacher_agreement": result,
    }


def _file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input_path(value: str) -> Path:
    """A user path: absolute, else relative to the cwd if it exists there, else the project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if (Path.cwd() / path).exists():
        return Path.cwd() / path
    return PROJECT_ROOT / path


def _pin_checkpoint(checkpoint: Path, pin_dir: Path) -> Path:
    """A private hard link (a copy across filesystems) of checkpoint.

    Promotion and retention replace or delete aliases such as promoted_<ns>.pt;
    the pin keeps one inode, so the hash and every CNN score refer to one file.
    """
    pin_dir.mkdir(parents=True, exist_ok=True)
    pinned = pin_dir / checkpoint.name
    try:
        os.link(checkpoint, pinned)
    except OSError:
        shutil.copy2(checkpoint, pinned)
    return pinned


def _drop_stale_pins(parent: Path) -> None:
    """Remove pin directories whose run is gone; a killed run cannot clean up."""
    for pin_dir in parent.glob(PIN_DIR_PREFIX + "*"):
        try:
            pid = int(pin_dir.name[len(PIN_DIR_PREFIX):])
            os.kill(pid, 0)
        except ValueError:
            continue
        except ProcessLookupError:
            shutil.rmtree(pin_dir, ignore_errors=True)
        except OSError:
            continue


def _load_suite(path: str) -> Tuple[List[Any], dict]:
    """Load the frozen suite, verifying its SHA-256 and exact state count."""
    from ..ml.teacher_validation import load_frozen_teacher_suite

    return load_frozen_teacher_suite(path, expected_count=SUITE_STATE_COUNT)


# --- games ------------------------------------------------------------------

def opponent_setup(opponent: str) -> Tuple[str, str]:
    """(difficulty, opponent_type) for model_vs_algo; random stamps easy, as acceptance does."""
    if opponent == "random":
        return "easy", "random"
    if opponent in OPPONENTS:
        return opponent, "algorithm"
    raise ValueError(f"Unknown opponent {opponent!r}; expected one of {', '.join(OPPONENTS)}")


def build_game_specs(
    player_label: str,
    opponent: str,
    num_games: int,
    max_moves: int,
    opening_plies: Sequence[int],
    opening_seed: int,
) -> Tuple[List[tuple], int, str]:
    """Balanced paired-opening specs, identical to a CNN run with the same suite settings."""
    difficulty, opponent_type = opponent_setup(opponent)
    if isinstance(max_moves, bool) or not isinstance(max_moves, int) or max_moves <= 0:
        raise ValueError("max_moves must be a positive integer")
    return mva._build_balanced_game_specs(
        player_label, difficulty, opponent_type, num_games, max_moves,
        opening_plies, opening_seed, 1)


def play_game(
    spec: tuple,
    choose: Chooser,
    *,
    opponent_move: Callable[..., Optional[Move]] = mva._choose_opponent_move,
) -> Dict[str, Any]:
    """Play one spec to the end; returns a GameTestRecord dict plus Laya fields.

    A LayaBudgetError on Laya's turn forfeits the game (counted as a loss), so a
    position Laya cannot read never turns into a free move.
    """
    _, difficulty, opponent_type, player_value, max_moves, opening, depth = spec
    laya_player = Player(player_value)
    state = mva._apply_random_opening(GameState.initial(), opening)
    opening_plies, game_seed = opening or (0, 0)
    opponent_rng = random.Random(game_seed ^ OPPONENT_RNG_SALT)
    moves = laya_moves = opponent_moves = forced = retries = 0
    laya_ms = 0.0
    devices: Counter = Counter()
    error = None
    started = time.perf_counter()
    while moves < max_moves:
        legal = state.legal_moves()
        if not legal:
            break
        if state.current_player == laya_player:
            try:
                decision = choose(state, legal)
            except LayaBudgetError as exc:
                error = f"budget: {exc}"
                break
            if decision is None or decision.move not in legal:
                raise RuntimeError("Laya chooser returned a move outside the legal set")
            move = decision.move
            laya_moves += 1
            if decision.forced:
                forced += 1
            else:
                laya_ms += float(decision.elapsed_ms)
                devices[decision.device] += 1
                retries += decision.label_style == "compact"
        else:
            move = opponent_move(state, legal, difficulty, opponent_type, opponent_rng)
            if move is None or move not in legal:
                raise RuntimeError(f"{opponent_type} opponent returned a move outside the legal set")
            opponent_moves += 1
        state = state.apply_move(move)
        moves += 1
    game_ms = (time.perf_counter() - started) * 1000.0
    winner = laya_player.opponent() if error is not None else state.winner()
    if winner is None:
        result = mva.TestResult.DRAW
    elif winner == laya_player:
        result = mva.TestResult.ML_WIN
    else:
        result = mva.TestResult.ALGO_WIN
    record = mva.GameTestRecord(
        result=result, ml_player=laya_player, winner=winner, num_moves=moves,
        ml_moves=laya_moves, algo_moves=opponent_moves, game_time_ms=game_ms,
        opponent_type=opponent_type, opening_plies=opening_plies, opening_seed=game_seed,
        ml_inference_depth=depth,
    ).to_dict()
    record.update(
        laya_decisions=laya_moves - forced,
        laya_forced=forced,
        laya_elapsed_ms=laya_ms,
        laya_devices=dict(devices),
        compact_label_retries=retries,
        forfeit=error is not None,
        error=error,
    )
    return record


def _ingest(stats: "mva.TestStatistics", record: Dict[str, Any]) -> None:
    """Add one game to stats exactly as ModelVsAlgoTester.run_tests does."""
    game = mva.GameTestRecord.from_dict(record)
    stats.total_games += 1
    side = "p1" if game.ml_player == Player.ONE else "p2"
    if game.result == mva.TestResult.ML_WIN:
        stats.ml_wins += 1
        key = f"ml_as_{side}_wins"
    elif game.result == mva.TestResult.ALGO_WIN:
        stats.algo_wins += 1
        key = f"ml_as_{side}_losses"
    else:
        stats.draws += 1
        key = f"ml_as_{side}_draws"
    setattr(stats, key, getattr(stats, key) + 1)
    stats.games.append(record)
    stats.avg_game_length = sum(g["num_moves"] for g in stats.games) / stats.total_games
    stats.avg_game_time_ms = sum(g["game_time_ms"] for g in stats.games) / stats.total_games


def laya_game_totals(games: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Laya-side counters summed over game records."""
    decisions = sum(g.get("laya_decisions", 0) for g in games)
    elapsed = sum(g.get("laya_elapsed_ms", 0.0) for g in games)
    devices: Counter = Counter()
    for g in games:
        devices.update(g.get("laya_devices") or {})
    return {
        "decisions": decisions,
        "forced": sum(g.get("laya_forced", 0) for g in games),
        "forfeits": sum(1 for g in games if g.get("forfeit")),
        "compact_label_retries": sum(g.get("compact_label_retries", 0) for g in games),
        "mean_elapsed_ms": elapsed / decisions if decisions else None,
        "devices": dict(devices),
    }


def play_games(
    choose: Chooser,
    specs: Sequence[tuple],
    base_seed: int,
    suite_id: str,
    *,
    player_label: str,
    opening_plies: Sequence[int],
    on_game: Optional[Callable[[Dict[str, Any]], None]] = None,
    opponent_move: Callable[..., Optional[Move]] = mva._choose_opponent_move,
) -> Dict[str, Any]:
    """Play every spec in order; returns status, error, TestStatistics and Laya totals."""
    if not specs:
        raise ValueError("No game specs to play")
    plies = [int(v) for v in opening_plies] or [0]  # normalized as _build_balanced_game_specs does
    stats = mva.TestStatistics(
        model_path=player_label, algo_difficulty=specs[0][1], opponent_type=specs[0][2],
        opening_seed=base_seed, opening_plies=plies, opening_suite_id=suite_id,
        opening_suite_size=len(specs) // 2, ml_inference_depth=specs[0][6],
        start_time=datetime.now().isoformat())
    status, error = "complete", None
    try:
        for spec in specs:
            record = play_game(spec, choose, opponent_move=opponent_move)
            _ingest(stats, record)
            if on_game is not None:
                on_game(record)
    except LayaError as exc:  # budget errors forfeit a game inside play_game
        status, error = "aborted", str(exc)
    except KeyboardInterrupt:
        status, error = "interrupted", "KeyboardInterrupt"
    stats.end_time = datetime.now().isoformat()
    if status == "complete":
        half = len(specs) // 2
        if stats.total_games != len(specs) or stats.ml_as_p1_games != half or stats.ml_as_p2_games != half:
            raise RuntimeError("Laya evaluation did not preserve the balanced game suite")
    return {"status": status, "error": error, "stats": stats, "laya": laya_game_totals(stats.games)}


# --- live trainer detection -------------------------------------------------

def _is_trainer_argv(argv: Sequence[str]) -> bool:
    """Whole-token match of a dama trainer command line.

    Substring matching is wrong here: agent CLIs carry these names inside their
    prompt arguments, and tools such as pgrep take the module name as an argument.
    """
    if not argv:
        return False
    if argv[0].startswith(TRAINER_TITLES):  # setproctitle: "micro-trainer | step=..."
        return True
    for i, token in enumerate(argv):
        if token == "-m" + TRAINER_MODULE:
            return True
        if token == TRAINER_MODULE and i > 0 and argv[i - 1] == "-m":
            return True
    return False


def _is_alive(proc_dir: str) -> bool:
    """True unless the process is gone or a zombie."""
    try:
        with open(os.path.join(proc_dir, "stat"), "r", encoding="utf-8", errors="replace") as fh:
            stat = fh.read()
    except OSError:
        return False
    fields = stat.rpartition(")")[2].split()
    return bool(fields) and fields[0] not in ("Z", "X")


def _read_argv(proc_dir: str) -> Optional[List[str]]:
    """argv of a process from its cmdline file, or None when it cannot be read."""
    try:
        with open(os.path.join(proc_dir, "cmdline"), "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _running_marker_pids(logs_root: str) -> List[int]:
    """pids of ``running`` run_status.json markers in logs_root and its direct subdirectories."""
    paths = [os.path.join(logs_root, RUN_STATUS_FILE)]
    try:
        with os.scandir(logs_root) as entries:
            paths += [os.path.join(e.path, RUN_STATUS_FILE) for e in entries if e.is_dir()]
    except OSError:
        return []
    pids = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                marker = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(marker, dict) or marker.get("status") != "running":
            continue
        pid = marker.get("pid")
        if type(pid) is int and pid > 0:
            pids.append(pid)
    return pids


def find_live_trainers(proc_root: str = "/proc",
                       exclude_pids: Optional[Sequence[int]] = None,
                       logs_root: Optional[str] = None) -> List[Tuple[int, str]]:
    """(pid, command line) of live dama trainer processes, excluding this process.

    A trainer started from the GUI training panel is a multiprocessing spawn
    child whose command line names no trainer, so the pid in a ``running``
    run_status.json marker also counts when it is a live Python process.
    """
    exclude = {os.getpid()} if exclude_pids is None else set(exclude_pids)
    found = {}
    try:
        names = os.listdir(proc_root)
    except OSError:
        names = []
    for name in names:
        if not name.isdigit() or int(name) in exclude:
            continue
        proc_dir = os.path.join(proc_root, name)
        argv = _read_argv(proc_dir)
        if argv and _is_trainer_argv(argv) and _is_alive(proc_dir):
            found[int(name)] = " ".join(argv)[:200]
    root = str(PROJECT_ROOT / "logs") if logs_root is None else logs_root
    for pid in _running_marker_pids(root):
        if pid in exclude or pid in found:
            continue
        proc_dir = os.path.join(proc_root, str(pid))
        argv = _read_argv(proc_dir)
        # A stale marker's pid may have been reused by an unrelated process.
        if argv and os.path.basename(argv[0]).startswith("python") and _is_alive(proc_dir):
            found[pid] = " ".join(argv)[:200]
    return sorted(found.items())


# --- CLI ----------------------------------------------------------------------

def _opening_plies(text: str) -> Tuple[int, ...]:
    """argparse type: comma-separated non-negative ints."""
    try:
        values = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if not values or any(v < 0 for v in values):
        raise argparse.ArgumentTypeError(f"expected non-negative integers, got {text!r}")
    return values


def _positive_int(text: str) -> int:
    """argparse type: an integer > 0."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected an integer > 0, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for both subcommands."""
    common = argparse.ArgumentParser(add_help=False)
    laya = common.add_argument_group("Laya settings (default: Settings > Laya AI in the user config)")
    laya.add_argument("--python", help="interpreter with laya installed")
    laya.add_argument("--conda-env", help="conda env searched when --python is not set")
    laya.add_argument("--hf-home", help="Hugging Face cache directory")
    laya.add_argument("--device", choices=policy.DEVICES)
    laya.add_argument("--state-format", choices=policy.STATE_FORMATS)
    laya.add_argument("--perspective", choices=policy.PERSPECTIVES)
    laya.add_argument("--no-shuffle", action="store_true", help="offer options in canonical label order")
    laya.add_argument("--model", help="hub repo id or local checkpoint directory")
    laya.add_argument("--subfolder", help='"" English root, "multilingual" or "typed-decisions"')
    common.add_argument("--out", help="JSONL output (default models/laya_eval/<cmd>_<UTC stamp>.jsonl)")
    common.add_argument("--allow-shared-gpu", action="store_true",
                        help="run on cuda/auto even while a dama trainer is running")

    parser = argparse.ArgumentParser(
        prog="python -m dama.ai.laya.evaluate",
        description="Measure the Laya player on the frozen teacher suite or in balanced games.")
    sub = parser.add_subparsers(dest="command", required=True)

    agreement = sub.add_parser(
        "agreement", parents=[common],
        help="top-1 agreement with the frozen hard-teacher suite on sampled decision states")
    agreement.add_argument("--count", type=_positive_int, required=True,
                           help="decision states to sample")
    agreement.add_argument("--seed", type=int, required=True, help="sampling seed")
    agreement.add_argument("--cnn", help="CNN checkpoint scored on the identical subset (CPU)")
    agreement.add_argument("--suite", default=DEFAULT_SUITE, help="frozen suite JSONL (default %(default)s)")

    games = sub.add_parser("games", parents=[common],
                           help="balanced games against a random or algorithmic opponent")
    games.add_argument("--opponent", choices=OPPONENTS, required=True)
    games.add_argument("--games", type=_positive_int, required=True,
                       help="total games (even: half as each side)")
    games.add_argument("--opening-plies", type=_opening_plies, default=DEFAULT_OPENING_PLIES,
                       help="random opening lengths cycled over games (default 2,4,6,8)")
    games.add_argument("--opening-seed", type=int, default=DEFAULT_OPENING_SEED,
                       help="opening suite seed (default %(default)s)")
    games.add_argument("--max-moves", type=_positive_int, default=DEFAULT_MAX_MOVES,
                       help="plies before a game is a draw (default %(default)s)")
    return parser


def effective_settings(base: Any, args: argparse.Namespace) -> Any:
    """Copy of the configured Laya settings with the CLI overrides applied."""
    overrides = {name: getattr(args, name) for name in (
        "python", "conda_env", "hf_home", "device", "state_format", "perspective",
        "model", "subfolder") if getattr(args, name) is not None}
    if args.no_shuffle:
        overrides["shuffle_options"] = False
    return dataclasses.replace(base, **overrides)


def default_output_path(command: str, now: datetime, root: Path = PROJECT_ROOT) -> Path:
    """models/laya_eval/<command>_<UTC stamp>.jsonl under root, with a suffix if taken."""
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / EVAL_DIR / f"{command}_{stamp}.jsonl"
    n = 2
    while path.exists():
        path = root / EVAL_DIR / f"{command}_{stamp}_{n}.jsonl"
        n += 1
    return path


def resolve_output_path(out: Optional[str], command: str, now: datetime,
                        root: Path = PROJECT_ROOT) -> Path:
    """Output path; refuses anything under the project's models/test_stats."""
    if out:
        path = Path(out).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = default_output_path(command, now, root)
    forbidden = (root / FORBIDDEN_OUTPUT_DIR).resolve()
    resolved = path.resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise ValueError(f"Refusing to write Laya results under {forbidden}: "
                         "that tree feeds the training dashboard; use models/laya_eval/")
    return path


def _default_bridge_factory(spec: Any) -> Any:
    """A dedicated bridge for this run (not the GUI's process-wide one)."""
    from .client import LayaBridge

    return LayaBridge(spec)


def _player_label(settings: Any) -> str:
    """Opaque model_path stamp for game records."""
    label = f"laya:{settings.model}"
    return f"{label}/{settings.subfolder}" if settings.subfolder else label


class _Recorder:
    """Appends JSONL records stamped with the run context and the current bridge info."""

    def __init__(self, path: Path, context: Dict[str, Any], bridge: Any, ready: Dict[str, Any]) -> None:
        self.path = path
        self._context = context
        self._bridge = bridge
        self._ready = ready
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def bridge_info(self) -> Dict[str, Any]:
        """Ready info of the live worker (it can respawn), else the one from start()."""
        info = getattr(self._bridge, "ready_info", None) or self._ready
        return {k: v for k, v in dict(info).items() if k != "event"}

    def write(self, kind: str, payload: Dict[str, Any]) -> None:
        """Write one record and flush it."""
        record = {"record": kind, **self._context, "bridge": self.bridge_info(), **payload}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the file."""
        self._fh.close()


def _status_exit_code(status: str) -> int:
    """Exit code for a finished run."""
    return {"complete": 0, "interrupted": 130}.get(status, 1)


def _prepare_agreement(args: argparse.Namespace, suite_loader: Callable[[str], Tuple[List[Any], dict]],
                       cnn_evaluator: CnnEvaluator, pin_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the suite, draw the subset and score the CNN before Laya starts.

    The CNN is scored from a pinned copy (job["pin_dir"]), which the caller removes.
    """
    suite_path = _resolve_input_path(args.suite)
    entries, manifest = suite_loader(str(suite_path))
    indices = sample_decision_states(entries, args.count, args.seed)
    job: Dict[str, Any] = {
        "entries": entries,
        "indices": indices,
        "forced": forced_fraction(entries),
        "params": {
            "suite_path": str(suite_path),
            "suite_sha256": manifest.get("suite_sha256"),
            "suite_states": len(entries),
            "decision_states": len(decision_indices(entries)),
            "count": args.count,
            "seed": args.seed,
            "cnn": None,
        },
        "cnn": None,
        "pin_dir": None,
    }
    if args.cnn:
        checkpoint = _resolve_input_path(args.cnn)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint}")
        if pin_dir is None:
            pin_dir = Path(tempfile.mkdtemp(prefix="laya_eval_pin_"))
        job["pin_dir"] = pin_dir
        try:
            pinned = _pin_checkpoint(checkpoint, pin_dir)
            sha = _file_sha256(pinned)
            print(f"Scoring CNN {checkpoint} on the subset (CPU)...", file=sys.stderr)
            result = cnn_evaluator(str(pinned), [entries[i] for i in indices])
        except BaseException:
            shutil.rmtree(pin_dir, ignore_errors=True)
            raise
        job["cnn"] = {"checkpoint": str(checkpoint), "pinned": str(pinned), "sha256": sha, "full": result}
        job["params"]["cnn"] = str(checkpoint)
    return job


def _prepare_games(args: argparse.Namespace, settings: Any) -> Dict[str, Any]:
    """Build the balanced specs before Laya starts, so bad arguments fail fast."""
    label = _player_label(settings)
    specs, base_seed, suite_id = build_game_specs(
        label, args.opponent, args.games, args.max_moves, args.opening_plies, args.opening_seed)
    difficulty, opponent_type = opponent_setup(args.opponent)
    return {
        "specs": specs,
        "base_seed": base_seed,
        "suite_id": suite_id,
        "label": label,
        "params": {
            "opponent": args.opponent,
            "difficulty": difficulty,
            "opponent_type": opponent_type,
            "games": args.games,
            "opening_plies": list(args.opening_plies),
            "opening_seed": base_seed,
            "opening_suite_id": suite_id,
            "max_moves": args.max_moves,
            "player_label": label,
            # Live move generation follows the user's rule settings.
            "rules": dataclasses.asdict(get_config().game.rules),
        },
    }


def _run_agreement(job: Dict[str, Any], choose: Chooser, recorder: _Recorder,
                   cnn_evaluator: CnnEvaluator) -> int:
    """Score Laya on the subset, then write the summary."""
    entries, indices = job["entries"], job["indices"]
    total = len(indices)
    progress: List[bool] = []

    def on_row(row: Dict[str, Any]) -> None:
        recorder.write("state", row)
        progress.append(row["hit"])
        if len(progress) % PROGRESS_EVERY == 0 or len(progress) == total:
            print(f"  {len(progress)}/{total} states, {sum(progress)} agree", file=sys.stderr)

    outcome = run_agreement(entries, indices, choose, on_row=on_row)
    rows = outcome["rows"]
    summary = summarize_agreement(rows, job["forced"])
    cnn = None
    if job["cnn"] is not None and rows:
        result = job["cnn"]["full"]
        if len(rows) != total:  # keep the CNN on exactly the states Laya answered
            result = cnn_evaluator(job["cnn"]["pinned"], [entries[r["suite_index"]] for r in rows])
        cnn = _cnn_summary(job["cnn"]["checkpoint"], job["cnn"]["sha256"], result, job["forced"])
    recorder.write("summary", {
        "status": outcome["status"], "error": outcome["error"], "requested": total,
        "suite_indices": [r["suite_index"] for r in rows], "laya": summary, "cnn": cnn,
    })
    _print_agreement(outcome, summary, cnn, total)
    return _status_exit_code(outcome["status"])


def _print_agreement(outcome: Dict[str, Any], s: Dict[str, Any], cnn: Optional[Dict[str, Any]],
                     total: int) -> None:
    """Human-readable agreement summary."""
    if not s["scored"]:
        print(f"Laya agreement ({outcome['status']}): no states scored of {total}")
        if outcome["error"]:
            print(f"  stopped early: {outcome['error']}")
        return
    ci = s["decision_accuracy_ci95"]
    print(f"Laya agreement ({outcome['status']}): {s['hits']}/{s['scored']} of {total} decision states = "
          f"{s['decision_accuracy']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    print(f"  random baseline {s['random_baseline_decision_accuracy']:.4f}; all-state equivalent "
          f"{s['all_state_equivalent']:.4f} (random {s['random_baseline_all_state_equivalent']:.4f})")
    print(f"  budget errors {s['budget_errors']}, compact retries {s['compact_label_retries']}, "
          f"devices {s['devices']}")
    if cnn is not None:
        print(f"  CNN {cnn['checkpoint']}: {cnn['decision_accuracy']:.4f}, "
              f"all-state equivalent {cnn['all_state_equivalent']:.4f}")
    if outcome["error"]:
        print(f"  stopped early: {outcome['error']}")


def _run_games(job: Dict[str, Any], choose: Chooser, recorder: _Recorder) -> int:
    """Play the balanced suite, then write the summary."""
    total = len(job["specs"])
    count = [0]

    def on_game(record: Dict[str, Any]) -> None:
        count[0] += 1
        recorder.write("game", {"game_index": count[0], **record})
        print(f"  game {count[0]}/{total}: {record['result']} as P{record['ml_player']} "
              f"in {record['num_moves']} plies", file=sys.stderr)

    outcome = play_games(choose, job["specs"], job["base_seed"], job["suite_id"],
                         player_label=job["label"], opening_plies=job["params"]["opening_plies"],
                         on_game=on_game)
    stats = outcome["stats"].to_dict()
    recorder.write("summary", {"status": outcome["status"], "error": outcome["error"],
                               "requested": total, "stats": stats, "laya": outcome["laya"]})
    ci = stats["match_score_ci_95"]
    print(f"Laya vs {job['params']['opponent']} ({outcome['status']}): {stats['total_games']} games, "
          f"W-D-L {stats['ml_wins']}-{stats['draws']}-{stats['algo_wins']}, match score "
          f"{stats['match_score']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]")
    print(f"  as P1 {stats['ml_as_p1_match_score']:.3f}, as P2 {stats['ml_as_p2_match_score']:.3f}; "
          f"forfeits {outcome['laya']['forfeits']}, devices {outcome['laya']['devices']}")
    if outcome["error"]:
        print(f"  stopped early: {outcome['error']}")
    return _status_exit_code(outcome["status"])


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    bridge_factory: Optional[Callable[[Any], Any]] = None,
    suite_loader: Optional[Callable[[str], Tuple[List[Any], dict]]] = None,
    cnn_evaluator: Optional[CnnEvaluator] = None,
    proc_root: str = "/proc",
    logs_root: Optional[str] = None,
    now: Optional[datetime] = None,
) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    started = now or datetime.now(timezone.utc)
    cnn_evaluator = cnn_evaluator or evaluate_cnn
    try:
        settings = effective_settings(get_config().ai.laya, args)
        spec = policy.spec_from_settings(settings)
        out_path = resolve_output_path(args.out, args.command, started)
        trainers = find_live_trainers(proc_root, logs_root=logs_root)
        if trainers:
            listing = "; ".join(f"pid {pid}: {cmd}" for pid, cmd in trainers)
            if spec.device in ("auto", "cuda") and not args.allow_shared_gpu:
                print(f"error: a dama trainer is running ({listing}). Laya on device {spec.device!r} "
                      "would share its GPU; use --device cpu, or --allow-shared-gpu to run anyway.",
                      file=sys.stderr)
                return 2
            print(f"warning: a dama trainer is running ({listing}); timings and time-budgeted "
                  "search may be affected.", file=sys.stderr)
        if args.command == "agreement":
            pin_dir = None
            if args.cnn:
                _drop_stale_pins(out_path.parent)
                pin_dir = out_path.parent / f"{PIN_DIR_PREFIX}{os.getpid()}"
            job = _prepare_agreement(args, suite_loader or _load_suite, cnn_evaluator, pin_dir)
        else:
            job = _prepare_games(args, settings)
    except (LayaError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    bridge = (bridge_factory or _default_bridge_factory)(spec)
    recorder = None
    try:
        print(f"Starting Laya ({spec.model}{'/' + spec.subfolder if spec.subfolder else ''}, "
              f"device {spec.device}) with {spec.python}...", file=sys.stderr)
        try:
            ready = bridge.start()
        except LayaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:  # during the model load: nothing was written yet
            print("interrupted while starting Laya", file=sys.stderr)
            return 130
        context = {
            "run_id": f"{args.command}_{started.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "cmd": args.command,
            "started_utc": started.astimezone(timezone.utc).isoformat(),
            "settings": dataclasses.asdict(settings),
            "python": spec.python,
            "live_trainers": [pid for pid, _ in trainers],
            "params": job["params"],
        }
        recorder = _Recorder(out_path, context, bridge, ready)

        def choose(state: GameState, legal: Sequence[Move]) -> Optional[policy.LayaDecision]:
            return policy.choose_laya_move(state, legal, settings, bridge=bridge)

        if args.command == "agreement":
            code = _run_agreement(job, choose, recorder, cnn_evaluator)
        else:
            code = _run_games(job, choose, recorder)
        print(f"Results: {out_path}")
        return code
    finally:
        try:
            if recorder is not None:
                recorder.close()  # can raise again, e.g. ENOSPC flushing buffered bytes
        finally:
            try:
                bridge.close()
            finally:
                if job.get("pin_dir") is not None:
                    shutil.rmtree(job["pin_dir"], ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
