"""Laya move policy: asks the Laya decision model to pick one of Dama's legal moves.

The position is shown from the mover's side by default (a 180-degree turn for
Player 2), so Laya always sees itself on ranks 1-3 moving toward rank 8. Board
text and move labels come from the same square naming, and the answer is mapped
back by index into the exact legal list the caller passed in.
"""

import copy
import hashlib
import json
import random
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ...game_state import GameState
from ...types import Move, PieceType, Player, Position
from .errors import LayaBudgetError, LayaUnavailableError
from .spec import BridgeSpec

FILES = "abcdefgh"
LABEL_STYLES = ("path", "compact")
DEVICES = ("auto", "cuda", "cpu")
STATE_FORMATS = ("ascii", "json")
PERSPECTIVES = ("side_to_move", "absolute")

# Forced capture is a user-toggleable rule, so the prompt must not claim it.
INSTRUCTIONS_TEMPLATE = (
    "Filipino Dama (checkers). You are {sym}, moving {direction}. Which move is best?"
)


@dataclass(frozen=True)
class LayaDecision:
    """One Laya move choice, aligned to the legal list that was passed in."""
    move: Move  # the identical object from the legal list
    index: int  # index into the legal list
    probabilities: Tuple[float, ...]  # aligned to the legal list
    confidence: float
    device: str  # "" when forced
    elapsed_ms: float
    labels: Tuple[str, ...]  # labels that were sent, aligned to the legal list
    label_style: str  # "path" or "compact"
    forced: bool  # single legal move: the bridge was not asked
    head_max_len: int  # 0 when forced


def _check_choice(name: str, value: Any, allowed: Sequence[str]) -> str:
    """Return value if it is one of allowed, else raise LayaUnavailableError."""
    if value not in allowed:
        raise LayaUnavailableError(
            f"Invalid Laya setting {name}={value!r}; expected one of {', '.join(allowed)}")
    return value


def _flip_for(state: GameState, settings: Any) -> bool:
    """Whether the board is turned 180 degrees for this state under the settings."""
    perspective = _check_choice("perspective", settings.perspective, PERSPECTIVES)
    return perspective == "side_to_move" and state.current_player == Player.TWO


def _display(pos: Position, flip: bool) -> Tuple[int, int]:
    """Displayed (row, col) of a Dama square after the optional flip."""
    r, c = pos
    return (7 - r, 7 - c) if flip else (r, c)


def square_name(pos: Position, flip: bool) -> str:
    """Displayed name of a square: file a-h from the column, rank 1-8 from the row."""
    r, c = _display(pos, flip)
    return f"{FILES[c]}{r + 1}"


def move_label(move: Move, flip: bool, style: str = "path") -> str:
    """Label of one move: every landing square ("path") or only first and last ("compact")."""
    sep = "x" if move.captures else "-"
    if style == "path":
        return sep.join(square_name(p, flip) for p in move.path)
    if style == "compact":
        return square_name(move.start, flip) + sep + square_name(move.end, flip)
    raise ValueError(f"Unknown label style {style!r}; expected one of {', '.join(LABEL_STYLES)}")


def _symbols(state: GameState, flip: bool) -> Tuple[Player, str, str]:
    """Return (the player drawn as w, the mover's symbol, the direction text)."""
    mover = state.current_player
    w_player = mover if flip else Player.ONE
    sym = "w" if mover == w_player else "b"
    toward_eight = (not flip and mover == Player.ONE) or flip
    return w_player, sym, "toward rank 8" if toward_eight else "toward rank 1"


def render_state(state: GameState, flip: bool, fmt: str = "ascii") -> Union[str, Dict[str, Any]]:
    """Render the board as the ASCII grid or the JSON dict Laya reads as its state."""
    w_player, sym, direction = _symbols(state, flip)
    if fmt == "json":
        groups: Dict[str, List[str]] = {"w_men": [], "w_kings": [], "b_men": [], "b_kings": []}
        for pos, piece in state.board.get_pieces():
            side = "w" if piece.player == w_player else "b"
            kind = "kings" if piece.piece_type is PieceType.KING else "men"
            groups[f"{side}_{kind}"].append(square_name(pos, flip))
        payload: Dict[str, Any] = {"to_move": sym, "direction": direction}
        for key, names in groups.items():
            payload[key] = sorted(names)
        return payload
    if fmt != "ascii":
        raise LayaUnavailableError(
            f"Invalid Laya setting state_format={fmt!r}; expected one of {', '.join(STATE_FORMATS)}")
    cells: Dict[Tuple[int, int], str] = {}
    for pos, piece in state.board.get_pieces():
        letter = "w" if piece.player == w_player else "b"
        cells[_display(pos, flip)] = letter.upper() if piece.piece_type is PieceType.KING else letter
    opp = "b" if sym == "w" else "w"
    lines = [f"{sym}/{sym.upper()} = your men/kings, {opp}/{opp.upper()} = opponent men/kings, "
             ". = empty dark square, - = light square"]
    for r in range(7, -1, -1):
        row = [cells.get((r, c), ".") if (r + c) % 2 == 1 else "-" for c in range(8)]
        lines.append(f"{r + 1} " + " ".join(row))
    lines.append("  " + " ".join(FILES))
    return "\n".join(lines)


def instructions_for(state: GameState, flip: bool) -> str:
    """Question text naming the mover's symbol and direction."""
    _, sym, direction = _symbols(state, flip)
    return INSTRUCTIONS_TEMPLATE.format(sym=sym, direction=direction)


def _dedupe(labels: Sequence[str], order: Sequence[int]) -> List[str]:
    """Append #2, #3, ... to later repeats of a label, visiting indices in order."""
    out = list(labels)
    seen: Dict[str, int] = {}
    for i in order:
        count = seen.get(labels[i], 0) + 1
        seen[labels[i]] = count
        if count > 1:
            out[i] = f"{labels[i]}#{count}"
    return out


def _labels(legal_moves: Sequence[Move], flip: bool) -> Tuple[List[str], List[str]]:
    """Unique (path, compact) labels aligned to legal_moves, independent of list order."""
    base_path = [move_label(m, flip, "path") for m in legal_moves]
    # Repeated path labels are ordered by move content before list index, so
    # the "#k" suffixes do not depend on the order the legal list arrived in.
    order = sorted(range(len(legal_moves)), key=lambda i: (
        base_path[i],
        tuple(square_name(p, flip) for p in legal_moves[i].captures),
        bool(legal_moves[i].promotion),
        i,
    ))
    path = _dedupe(base_path, order)
    compact = _dedupe([move_label(m, flip, "compact") for m in legal_moves], order)
    return path, compact


def _state_seed_payload(state: GameState) -> Dict[str, Any]:
    """Compact state without move_count, with each piece list sorted."""
    data = state.to_compact()
    data.pop("move_count", None)
    # to_compact follows board insertion order, which differs between a live
    # game and the same position rebuilt by from_compact.
    for key in ("p1_men", "p1_kings", "p2_men", "p2_kings"):
        data[key] = sorted([int(p[0]), int(p[1])] for p in data.get(key, []))
    return data


def option_permutation(state: GameState, labels: Sequence[str], shuffle: bool = True) -> List[int]:
    """Legal indices in the order they are offered: perm[j] is the index behind option j.

    ``labels`` are the unique path labels aligned to the legal list. The order is
    a pure function of the position and the set of labels.
    """
    canonical = sorted(range(len(labels)), key=lambda i: (labels[i], i))
    if not shuffle:
        return canonical
    text = (json.dumps(_state_seed_payload(state), sort_keys=True)
            + "|" + "|".join(labels[i] for i in canonical))
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    perm = list(canonical)
    random.Random(seed).shuffle(perm)
    return perm


def build_request(
    state: GameState,
    legal_moves: Sequence[Move],
    settings: Any,
    label_style: str = "path",
) -> Tuple[Union[str, Dict[str, Any]], str, List[str], List[int], List[str]]:
    """Return (state payload, instructions, options, perm, labels aligned to legal_moves)."""
    if label_style not in LABEL_STYLES:
        raise ValueError(f"Unknown label style {label_style!r}; expected one of {', '.join(LABEL_STYLES)}")
    flip = _flip_for(state, settings)
    fmt = _check_choice("state_format", settings.state_format, STATE_FORMATS)
    path, compact = _labels(legal_moves, flip)
    labels = path if label_style == "path" else compact
    # The permutation always comes from path labels, so a compact retry only
    # changes the label text, never which move sits in which slot.
    perm = option_permutation(state, path, shuffle=bool(settings.shuffle_options))
    options = [labels[i] for i in perm]
    return render_state(state, flip, fmt), instructions_for(state, flip), options, perm, labels


def _positive(name: str, value: Any, allow_zero: bool = False) -> float:
    """Return value as a finite float > 0 (or >= 0), else raise LayaUnavailableError."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if not (number >= 0.0 if allow_zero else number > 0.0) or number == float("inf"):
        bound = ">= 0" if allow_zero else "> 0"
        raise LayaUnavailableError(f"Invalid Laya setting {name}={value!r}; expected a number {bound}")
    return number


def spec_from_settings(settings: Any) -> BridgeSpec:
    """Validate LayaAISettings-like settings and build the worker spec."""
    device = _check_choice("device", settings.device, DEVICES)
    _check_choice("state_format", settings.state_format, STATE_FORMATS)
    _check_choice("perspective", settings.perspective, PERSPECTIVES)
    model = str(settings.model or "").strip()
    if not model:
        raise LayaUnavailableError("Invalid Laya setting model='': a hub repo id or checkpoint directory is required")
    startup = _positive("startup_timeout_sec", settings.startup_timeout_sec)
    request = _positive("request_timeout_sec", settings.request_timeout_sec)
    idle = _positive("idle_shutdown_sec", settings.idle_shutdown_sec, allow_zero=True)
    from .client import resolve_python
    # A hand-edited YAML number or bool must fail as a setting, not as a TypeError
    # that would skip the configured fallback.
    python = resolve_python(str(settings.python or "").strip(), str(settings.conda_env or "").strip())
    return BridgeSpec(
        python=python,
        model=model,
        subfolder=str(settings.subfolder or "").strip(),
        device=device,
        hf_home=str(settings.hf_home or "").strip(),
        offline=bool(settings.offline),
        startup_timeout_sec=startup,
        request_timeout_sec=request,
        idle_shutdown_sec=idle,
    )


def _ask(bridge: Any, state: GameState, legal_moves: Sequence[Move], settings: Any,
         label_style: str) -> LayaDecision:
    """Send one choose request and map the answer back onto legal_moves."""
    payload, instructions, options, perm, labels = build_request(state, legal_moves, settings, label_style)
    choice = bridge.choose(payload, instructions, options)
    n = len(options)
    if len(choice.probabilities) != n:
        raise LayaUnavailableError(
            f"Laya returned {len(choice.probabilities)} probabilities for {n} options")
    j = choice.choice_index
    if isinstance(j, bool) or not isinstance(j, int) or not 0 <= j < n:
        raise LayaUnavailableError(f"Laya returned choice index {j!r} for {n} options")
    probabilities = [0.0] * n
    for slot, p in enumerate(choice.probabilities):
        probabilities[perm[slot]] = float(p)
    index = perm[j]
    return LayaDecision(
        move=legal_moves[index],
        index=index,
        probabilities=tuple(probabilities),
        confidence=float(choice.confidence),
        device=str(choice.device),
        elapsed_ms=float(choice.elapsed_ms),
        labels=tuple(labels),
        label_style=label_style,
        forced=False,
        head_max_len=int(choice.head_max_len),
    )


def choose_laya_move(
    state: GameState,
    legal_moves: Sequence[Move],
    settings: Any,
    bridge: Any = None,
) -> Optional[LayaDecision]:
    """Ask Laya to choose among legal_moves; never regenerates the legal list."""
    if not legal_moves:
        return None
    if len(legal_moves) == 1:
        label = move_label(legal_moves[0], _flip_for(state, settings), "path")
        return LayaDecision(
            move=legal_moves[0], index=0, probabilities=(1.0,), confidence=1.0,
            device="", elapsed_ms=0.0, labels=(label,), label_style="path",
            forced=True, head_max_len=0,
        )
    if bridge is None:
        from .client import get_bridge
        bridge = get_bridge(spec_from_settings(settings))
    try:
        return _ask(bridge, state, legal_moves, settings, "path")
    except LayaBudgetError:
        return _ask(bridge, state, legal_moves, settings, "compact")


def prewarm_bridge(settings: Any) -> None:
    """Start the worker in a daemon thread; errors surface on the first real move."""
    snapshot = copy.copy(settings)

    def _run() -> None:
        try:
            from .client import get_bridge
            get_bridge(spec_from_settings(snapshot)).start()
        except Exception as exc:
            print(f"Laya prewarm failed: {exc}", file=sys.stderr)

    threading.Thread(target=_run, name="laya-prewarm", daemon=True).start()
