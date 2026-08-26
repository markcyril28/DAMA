"""Versioned replay-corpus snapshots for policy-distillation recovery.

The snapshot manager keeps training data immutable for a training window,
holds validation out by whole replay file, and admits a new snapshot only when
at least the configured fraction of its canonical states are new relative to
the previous snapshot.

Canonicalization matches the model's side-to-move perspective. ``move_count``
is intentionally ignored because it is not part of the policy input.
"""

from __future__ import annotations

from collections import Counter, defaultdict, OrderedDict
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import threading
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .move_encoder import ENCODING_VERSION
from .replay import ReplayEntry
from . import run_status


SNAPSHOT_SCHEMA_VERSION = 1
# Upper bound on hold-out shards as a multiple of the configured target.
# The hold-out never releases a shard, so a quota measured against the
# still-present count would otherwise grow it once per corpus rotation.
HOLDOUT_FILE_CEILING = 3

# The hold-out artifact is versioned independently of the snapshot schema so a
# contaminated split can be *replaced* rather than repaired.  Version 1 keeps
# the historical ``validation/`` directory; every later version gets its own
# ``validation_v<N>/`` directory, so the superseded split survives untouched as
# evidence and can never be silently reused by a rebuilt lineage.
VALIDATION_SPLIT_VERSION_DEFAULT = 1

# Append-only record of every replay shard and canonical state this namespace
# has ever served as *training* data.  Snapshot retention prunes old snapshot
# directories, so manifests alone are only a lower bound on the all-time
# trained set; the ledger is what makes "never hold out something the model has
# already fit" checkable across pruning, renaming, and process restarts.
TRAINED_LEDGER_SCHEMA_VERSION = 1


def _posix_relpath(target: Path, start: Path) -> str:
    """Store relative paths with POSIX separators regardless of host.

    ``os.path.relpath``/``str(Path(...))`` emit the *host* separator, so a
    snapshot written by the native-Windows launcher records
    ``..\\validation\\manifest.json``.  Read back on WSL that path does not
    resolve, ``current_manifest_path()`` returns ``None``, and
    ``consider_snapshot`` takes the no-previous-corpus branch -- which is how
    snapshot_v000012 was admitted with the >=50% freshness floor skipped.
    """
    return Path(os.path.relpath(target, start)).as_posix()


def _read_relpath(value: str) -> str:
    """Accept either separator when reading a stored relative path."""
    return str(value).replace("\\", "/")
POLICY_REPLAY_CONTRACT_VERSION = 1
CANONICAL_RULES_ID = "filipino-dama-default-v1"


# Corpus gating repeatedly inspects the same immutable replay files (contract
# audit, byte hash, and exact diversity analysis).  Keep this cache strictly
# process-local: replay files are the source of truth and no cache state is
# persisted alongside a corpus or snapshot.  The identity includes both the
# pathname and the filesystem identity/metadata so replacing, truncating, or
# appending to a replay file naturally evicts the old entry.
_REPLAY_FILE_CACHE_MAX = 64
_REPLAY_CACHE_LOCK = threading.RLock()
_REPLAY_ANALYSIS_CACHE: "OrderedDict[tuple, _ReplayFileAnalysis]" = OrderedDict()
_REPLAY_HASH_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_REPLAY_AUDIT_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_REPLAY_LATEST_IDENTITY: Dict[str, tuple] = {}


@dataclass(frozen=True)
class _ReplayFileIdentity:
    """A stat-based identity for one path during this process."""

    resolved_path: str
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int

    def as_key(self) -> tuple:
        return (
            self.resolved_path,
            self.st_dev,
            self.st_ino,
            self.st_size,
            self.st_mtime_ns,
        )


@dataclass(frozen=True)
class _ReplayFileAnalysis:
    """Exact per-file facts needed to reproduce ``analyze_replay_files``."""

    identity: tuple
    sha256: str
    records: int
    malformed_records: int
    forced_move_count: int
    state_counts: Mapping[str, int]
    state_cycles: Mapping[str, frozenset[str]]
    source_counts: Mapping[str, int]
    game_sources: Mapping[str, str]


def _replay_file_identity(path: Path) -> _ReplayFileIdentity:
    """Return an identity that changes for normal in-place/replacement edits."""

    path = Path(path)
    resolved = str(path.resolve())
    stat_result = path.stat()
    return _ReplayFileIdentity(
        resolved_path=resolved,
        st_dev=int(stat_result.st_dev),
        st_ino=int(stat_result.st_ino),
        st_size=int(stat_result.st_size),
        st_mtime_ns=int(stat_result.st_mtime_ns),
    )


def _cache_touch(
    cache: "OrderedDict[tuple, Any]", key: tuple, value: Any,
) -> None:
    """Insert an item and enforce the process-local LRU bound."""

    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _REPLAY_FILE_CACHE_MAX:
        cache.popitem(last=False)


def _cache_prepare_identity(identity: _ReplayFileIdentity) -> tuple:
    """Drop an older generation of the same pathname before cache lookup."""

    identity_key = identity.as_key()
    path_key = identity.resolved_path
    with _REPLAY_CACHE_LOCK:
        previous = _REPLAY_LATEST_IDENTITY.get(path_key)
        if previous is not None and previous != identity_key:
            _REPLAY_ANALYSIS_CACHE.pop(previous, None)
            _REPLAY_HASH_CACHE.pop(previous, None)
            for audit_key in tuple(_REPLAY_AUDIT_CACHE):
                if audit_key[0] == previous:
                    _REPLAY_AUDIT_CACHE.pop(audit_key, None)
        _REPLAY_LATEST_IDENTITY[path_key] = identity_key
        if len(_REPLAY_LATEST_IDENTITY) > _REPLAY_FILE_CACHE_MAX:
            active = set(_REPLAY_ANALYSIS_CACHE)
            active.update(_REPLAY_HASH_CACHE)
            active.update(key[0] for key in _REPLAY_AUDIT_CACHE)
            for stale_path, stale_identity in tuple(_REPLAY_LATEST_IDENTITY.items()):
                if len(_REPLAY_LATEST_IDENTITY) <= _REPLAY_FILE_CACHE_MAX:
                    break
                if stale_path != path_key and stale_identity not in active:
                    _REPLAY_LATEST_IDENTITY.pop(stale_path, None)
    return identity_key


def _clear_replay_file_cache() -> None:
    """Clear process-local replay caches (used by focused tests)."""

    with _REPLAY_CACHE_LOCK:
        _REPLAY_ANALYSIS_CACHE.clear()
        _REPLAY_HASH_CACHE.clear()
        _REPLAY_AUDIT_CACHE.clear()
        _REPLAY_LATEST_IDENTITY.clear()


def _rotate(position: Sequence[int]) -> Tuple[int, int]:
    return 7 - int(position[0]), 7 - int(position[1])


def _bitboard(positions: Iterable[Sequence[int]]) -> int:
    value = 0
    for row, col in positions:
        value |= 1 << (int(row) * 8 + int(col))
    return value


def canonical_state_payload(state: Mapping[str, Any]) -> bytes:
    """Return a stable side-to-move representation of a compact state.

    Complexity is O(p), where ``p`` is the number of pieces and is bounded by
    24 for a legal Dama position. Space usage is O(p) for Player 2 rotation.
    """

    turn = int(state.get("turn", 1))
    if turn == 1:
        own_men = state.get("p1_men", ())
        own_kings = state.get("p1_kings", ())
        opp_men = state.get("p2_men", ())
        opp_kings = state.get("p2_kings", ())
    elif turn == 2:
        own_men = [_rotate(p) for p in state.get("p2_men", ())]
        own_kings = [_rotate(p) for p in state.get("p2_kings", ())]
        opp_men = [_rotate(p) for p in state.get("p1_men", ())]
        opp_kings = [_rotate(p) for p in state.get("p1_kings", ())]
    else:
        raise ValueError(f"Invalid compact-state turn: {turn}")

    values = (
        _bitboard(own_men),
        _bitboard(own_kings),
        _bitboard(opp_men),
        _bitboard(opp_kings),
    )
    header = f"{CANONICAL_RULES_ID}|encoding={ENCODING_VERSION}|".encode("ascii")
    return header + b"".join(v.to_bytes(8, "big", signed=False) for v in values)


def canonical_state_key(state: Mapping[str, Any]) -> str:
    """Return the SHA-256 key for a canonical compact state."""

    return hashlib.sha256(canonical_state_payload(state)).hexdigest()


def _sha256_file_uncached(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_file_sha256(path: Path) -> str:
    """Return a replay-file digest, reusing only a still-identical file."""

    path = Path(path)
    identity = _replay_file_identity(path)
    identity_key = _cache_prepare_identity(identity)
    with _REPLAY_CACHE_LOCK:
        cached = _REPLAY_HASH_CACHE.get(identity_key)
        if cached is not None:
            _REPLAY_HASH_CACHE.move_to_end(identity_key)
            return cached

    digest = _sha256_file_uncached(path)
    try:
        unchanged = _replay_file_identity(path).as_key() == identity_key
    except OSError:
        unchanged = False
    if unchanged:
        with _REPLAY_CACHE_LOCK:
            _cache_touch(_REPLAY_HASH_CACHE, identity_key, digest)
    return digest


def _audit_policy_replay_file_uncached(
    path: Path,
    allowed_opening_plies: Sequence[int],
) -> Dict[str, Any]:
    """Fail closed on replay files that predate the repaired sample contract.

    A contract violation is definitive for admission, so there is no value in
    scanning the remaining records of a rejected legacy file.  Candidate-valid
    files are still consumed completely; this keeps all diversity/provenance
    checks intact for files that can actually enter a snapshot, and lets the
    whole-file 70/30 trajectory split be checked once the scan finishes.
    """
    allowed = {int(value) for value in allowed_opening_plies}
    errors: Counter[str] = Counter()
    game_sources: Dict[str, str] = {}
    records = 0

    def _result(complete: bool = False) -> Dict[str, Any]:
        source_games = Counter(game_sources.values())
        if complete and not errors:
            # A completed self-play cycle writes exactly one replay file whose
            # trajectories are an exact 70/30 algorithm/current-model split.  A
            # fully-parsed file that misses the ratio is a partial cycle: the
            # process died mid-generation, so the trainer's in-process
            # quarantine (discard_current_file) never ran.  Admitting it drags
            # the whole corpus off contract and, because the file is never
            # rewritten, wedges every later run behind the same aggregate
            # error.  Reject the one bad file instead.
            algorithm = source_games.get("algorithm", 0)
            model = source_games.get("current_model", 0)
            if algorithm + model == 0 or algorithm * 3 != model * 7:
                errors["unbalanced_policy_trajectory_split"] += 1
        return {
            "contract_version": POLICY_REPLAY_CONTRACT_VERSION,
            "valid": records > 0 and not errors,
            "records": records,
            "game_count": len(game_sources),
            "source_game_counts": dict(sorted(source_games.items())),
            "errors": dict(sorted(errors.items())),
        }

    for entry in _iter_entry_dicts(path):
        records += 1
        legal_moves = entry.get("legal_moves")
        if not isinstance(legal_moves, list) or not legal_moves:
            errors["missing_legal_moves"] += 1
        else:
            for key in (
                "chosen_index",
                "played_index",
                "trajectory_source",
                "was_exploration",
                "teacher_difficulty",
                "opening_plies",
                "game_id",
            ):
                if key not in entry:
                    errors[f"missing_{key}"] += 1
            chosen = entry.get("chosen_index")
            played = entry.get("played_index")
            if not isinstance(chosen, int) or not 0 <= chosen < len(legal_moves):
                errors["invalid_teacher_index"] += 1
            if not isinstance(played, int) or not 0 <= played < len(legal_moves):
                errors["invalid_played_index"] += 1
            if entry.get("teacher_difficulty") != "hard":
                errors["non_hard_teacher"] += 1
            source = entry.get("trajectory_source")
            if source not in {"algorithm", "current_model"}:
                errors["invalid_trajectory_source"] += 1
            opening = entry.get("opening_plies")
            if allowed and opening not in allowed:
                errors["opening_outside_configured_suite"] += 1
            game_id = entry.get("game_id")
            if not isinstance(game_id, str) or not game_id:
                errors["invalid_game_id"] += 1
            elif source in {"algorithm", "current_model"}:
                previous = game_sources.setdefault(game_id, source)
                if previous != source:
                    errors["game_has_multiple_sources"] += 1
        # No rejected file can become admissible by scanning more records.
        # Return after the first observed contract error, including the useful
        # count of records consumed and the exact error categories found.
        if errors:
            return _result()

    return _result(complete=True)


def audit_policy_replay_file(
    path: Path,
    allowed_opening_plies: Sequence[int],
) -> Dict[str, Any]:
    """Audit a replay file, caching only a still-identical complete result."""

    path = Path(path)
    try:
        identity_key = _cache_prepare_identity(_replay_file_identity(path))
    except OSError:
        # Preserve the underlying iterator/open error for missing paths and
        # test doubles that intentionally do not exist on disk.
        return _audit_policy_replay_file_uncached(path, allowed_opening_plies)
    allowed_values = tuple(int(value) for value in allowed_opening_plies)
    allowed_key = tuple(sorted(set(allowed_values)))
    cache_key = (identity_key, allowed_key)
    with _REPLAY_CACHE_LOCK:
        cached = _REPLAY_AUDIT_CACHE.get(cache_key)
        if cached is not None:
            _REPLAY_AUDIT_CACHE.move_to_end(cache_key)
            return copy.deepcopy(cached)

    # JSON/iterator errors deliberately bypass the cache.  A contract-invalid
    # result is safe to cache as a negative result, but malformed input never
    # becomes a reusable valid audit entry.
    result = _audit_policy_replay_file_uncached(path, allowed_values)
    try:
        unchanged = _replay_file_identity(path).as_key() == identity_key
    except OSError:
        unchanged = False
    if unchanged:
        with _REPLAY_CACHE_LOCK:
            _cache_touch(_REPLAY_AUDIT_CACHE, cache_key, copy.deepcopy(result))
    return result


def _iter_entry_dicts(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid replay JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Replay entry is not an object at {path}:{line_number}")
            yield value


def _read_replay_file_analysis(
    path: Path, identity_key: tuple,
) -> _ReplayFileAnalysis:
    """Build exact per-file facts without publishing a partial cache entry."""

    path = Path(path)
    state_counts: Counter[str] = Counter()
    state_cycles: Dict[str, Set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    game_sources: Dict[str, str] = {}
    forced = 0
    malformed = 0
    total = 0

    # Hash and semantic parsing are intentionally done as one uncached build;
    # the result is published only after both complete and the file identity
    # is checked again by the caller.
    file_hash = _sha256_file_uncached(path)
    for entry in _iter_entry_dicts(path):
        try:
            key = canonical_state_key(entry["state"])
            legal_moves = entry["legal_moves"]
        except (KeyError, TypeError, ValueError):
            malformed += 1
            continue
        total += 1
        state_counts[key] += 1
        game_id = entry.get("game_id")
        cycle_match = re.match(r"^cycle-([^ -]+)-", str(game_id or ""))
        if cycle_match:
            state_cycles[key].add(cycle_match.group(1))
        if len(legal_moves) == 1:
            forced += 1
        source_counts[str(entry.get("trajectory_source", "legacy"))] += 1
        source = entry.get("trajectory_source")
        if isinstance(game_id, str) and game_id and isinstance(source, str):
            game_sources[game_id] = source

    return _ReplayFileAnalysis(
        identity=identity_key,
        sha256=file_hash,
        records=total,
        malformed_records=malformed,
        forced_move_count=forced,
        state_counts=dict(state_counts),
        state_cycles={key: frozenset(value) for key, value in state_cycles.items()},
        source_counts=dict(source_counts),
        game_sources=dict(game_sources),
    )


def _cached_replay_file_analysis(path: Path) -> _ReplayFileAnalysis:
    """Load exact file facts from the bounded cache or build them once."""

    path = Path(path)
    identity_key = _cache_prepare_identity(_replay_file_identity(path))
    with _REPLAY_CACHE_LOCK:
        cached = _REPLAY_ANALYSIS_CACHE.get(identity_key)
        if cached is not None:
            _REPLAY_ANALYSIS_CACHE.move_to_end(identity_key)
            return cached

    analysis = _read_replay_file_analysis(path, identity_key)
    try:
        unchanged = _replay_file_identity(path).as_key() == identity_key
    except OSError:
        unchanged = False
    # A file with malformed semantic records is intentionally not cached.  It
    # remains measurable with the legacy semantics, but cannot leave a valid
    # reusable analysis result behind.  Invalid JSON likewise never reaches
    # this publication point because _iter_entry_dicts raises.
    if unchanged and analysis.malformed_records == 0:
        with _REPLAY_CACHE_LOCK:
            _cache_touch(_REPLAY_ANALYSIS_CACHE, identity_key, analysis)
            _cache_touch(_REPLAY_HASH_CACHE, identity_key, analysis.sha256)
    return analysis


def _state_set_digest(state_keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_keys):
        digest.update(key.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def analyze_replay_files(
    files: Sequence[Path],
    previous_state_keys: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Any], Set[str]]:
    """Measure exact replay diversity and freshness across a file set.

    Time complexity is O(r log u) because the final fingerprint sorts ``u``
    unique keys. Space complexity is O(u + r_source), where ``r_source`` is
    the bounded set of source labels.
    """

    previous = previous_state_keys or set()
    unique_keys: Set[str] = set()
    state_counts: Counter[str] = Counter()
    state_files: Dict[str, Set[str]] = defaultdict(set)
    state_cycles: Dict[str, Set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    game_sources: Dict[str, str] = {}
    forced = 0
    malformed = 0
    total = 0

    for path in files:
        path = Path(path)
        analysis = _cached_replay_file_analysis(path)
        total += analysis.records
        malformed += analysis.malformed_records
        forced += analysis.forced_move_count
        source_counts.update(analysis.source_counts)
        # Updating in caller-provided file order preserves the original
        # last-write-wins behavior when a game ID appears in multiple files.
        game_sources.update(analysis.game_sources)
        for key, count in analysis.state_counts.items():
            state_counts[key] += count
            unique_keys.add(key)
            state_files[key].add(path.name)
        for key, cycles in analysis.state_cycles.items():
            state_cycles[key].update(cycles)

    new_unique = unique_keys.difference(previous)
    fresh_records = sum(
        count for key, count in state_counts.items() if key not in previous
    )
    cross_file_states = sum(1 for names in state_files.values() if len(names) > 1)
    cross_file_unique_states = sum(1 for names in state_files.values() if len(names) == 1)
    cross_file_duplicate_records = sum(
        max(0, state_counts[state_key] - 1)
        for state_key, names in state_files.items() if len(names) > 1
    )
    cycle_observed_states = {
        key: cycles for key, cycles in state_cycles.items() if cycles
    }
    cross_cycle_states = sum(
        1 for cycles in cycle_observed_states.values() if len(cycles) > 1
    )
    cross_cycle_unique_states = sum(
        1 for cycles in cycle_observed_states.values() if len(cycles) == 1
    )
    metrics = {
        "records": total,
        "malformed_records": malformed,
        "unique_state_count": len(unique_keys),
        "unique_state_rate": (len(unique_keys) / total) if total else 0.0,
        "forced_move_count": forced,
        "forced_move_rate": (forced / total) if total else 0.0,
        "cross_file_repeated_state_count": cross_file_states,
        "cross_file_unique_state_count": cross_file_unique_states,
        "cross_file_duplicate_record_count": cross_file_duplicate_records,
        # Rates are over unique canonical states, not raw records.  This makes
        # the metric interpretable when a trajectory visits a state repeatedly.
        "cross_file_unique_state_rate": (
            cross_file_unique_states / len(unique_keys) if unique_keys else 0.0
        ),
        "cross_cycle_observed_state_count": len(cycle_observed_states),
        "cross_cycle_repeated_state_count": cross_cycle_states,
        "cross_cycle_unique_state_count": cross_cycle_unique_states,
        "cross_cycle_unique_state_rate": (
            cross_cycle_unique_states / len(cycle_observed_states)
            if cycle_observed_states else 0.0
        ),
        "new_unique_state_count": len(new_unique),
        "fresh_unique_state_rate": (len(new_unique) / len(unique_keys)) if unique_keys else 0.0,
        "fresh_record_rate": (fresh_records / total) if total else 0.0,
        "source_counts": dict(sorted(source_counts.items())),
        "source_game_counts": dict(sorted(Counter(game_sources.values()).items())),
        "state_set_sha256": _state_set_digest(unique_keys),
    }
    return metrics, unique_keys


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    # Thin delegate: one atomic-JSON implementation project-wide (see
    # run_status._write_json_atomic for the temp+fsync+replace contract).
    run_status._write_json_atomic(path, value)


def _write_state_keys(path: Path, state_keys: Iterable[str]) -> None:
    # Proofread 2026-08-25 C2: this file is part of the committed snapshot --
    # ``manifest.json`` names it and is treated as "the single commit point",
    # so a truncated gz here is a permanent, unrecoverable split failure.
    # Follow the project-wide durability contract (see
    # run_status._write_json_atomic): temp file in the destination directory,
    # flush + fsync, then one atomic os.replace.  A mid-write failure leaves
    # the previous key file untouched instead of truncating it in place.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        # Wrap the mkstemp fd instead of letting gzip.open(path-name) open a
        # second handle: the reserved fd must be consumed here or it leaks
        # (one per growth/admission cycle until process exit).
        with os.fdopen(fd, "wb") as raw_handle:
            with gzip.open(raw_handle, "wt", encoding="ascii", newline="\n") as handle:
                for key in sorted(state_keys):
                    handle.write(key)
                    handle.write("\n")
                handle.flush()
            # The gzip CRC/size trailer is appended on close(), so fsync only
            # afterwards: the committed file must be durable in full.
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _read_state_keys(path: Path) -> Set[str]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        return {line.strip() for line in handle if line.strip()}


def _iter_state_keys(path: Path) -> Iterator[str]:
    """Stream a sorted key file without materialising the whole set."""
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            key = line.strip()
            if key:
                yield key


def _state_key_fingerprint(key: str) -> int:
    """64-bit membership fingerprint of a canonical state key.

    The all-time trained ledger reaches millions of keys, and holding them as
    64-character strings costs roughly 600 MB resident on a box with 24 GB
    total -- meaningful next to a trainer whose suspected silent deaths are
    OOM kills.  The keys are already SHA-256 digests, so their leading 64 bits
    are uniformly distributed: at ~4M keys the chance of any collision is about
    8e-7, and a collision can only *drop* one hold-out entry that was not in
    fact trained on.  That direction is conservative -- it can never admit a
    contaminated state into validation, only discard a clean one.
    """
    return int(key[:16], 16)


def _merge_state_keys_file(path: Path, new_keys: Iterable[str]) -> int:
    """Union ``new_keys`` into a sorted key file, streaming; return added count.

    Reading the existing file into a set to take the union would reintroduce
    the whole-set memory cost this representation exists to avoid, so the two
    sorted streams are merged directly into a replacement file.
    """
    additions = sorted(set(new_keys))
    if not additions:
        return 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    added = 0
    existing = _iter_state_keys(path) if path.is_file() else iter(())
    pending = next(existing, None)
    with gzip.open(temporary, "wt", encoding="ascii", newline="\n") as handle:
        index = 0
        while pending is not None or index < len(additions):
            if pending is not None and (
                index >= len(additions) or pending <= additions[index]
            ):
                handle.write(pending)
                handle.write("\n")
                if index < len(additions) and pending == additions[index]:
                    index += 1
                pending = next(existing, None)
            else:
                handle.write(additions[index])
                handle.write("\n")
                added += 1
                index += 1
    os.replace(temporary, path)
    return added


def _store_shard(
    source: Path,
    destination: Path,
    previous_files_dir: Optional[Path] = None,
) -> str:
    """Store one replay shard into a snapshot; return the storage mode used.

    ``previous_files_dir`` enables shard reuse: when the previous snapshot
    already holds a byte-identical copy of ``source``, hardlink it instead of
    paying another full corpus copy per admission (~0.9 GB on a 60-file
    window, against a volume that runs 99% full).  This is safe because replay
    shards are write-once -- ReplayWriter opens a fresh timestamped file 'w'
    and never reopens it, and cleanup rotates files out by unlink -- so the
    predecessor's stored copy is the same immutable object.  Fail-closed in
    both directions: the candidate must match BOTH size and sha256 before the
    link is made, and load-time integrity verification re-hashes every stored
    shard afterwards, so a corrupted or mutated candidate degrades to a plain
    copy rather than ever admitting wrong bytes.  Any filesystem refusal
    (cross-device, permission, link exhaustion) also falls back to copy.
    """
    if previous_files_dir is not None:
        candidate = previous_files_dir / source.name
        try:
            if (
                candidate.is_file()
                and candidate.stat().st_size == source.stat().st_size
                and replay_file_sha256(candidate) == replay_file_sha256(source)
            ):
                os.link(str(candidate), str(destination))
                return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


@dataclass(frozen=True)
class SnapshotDecision:
    admitted: bool
    reason: str
    manifest_path: Optional[Path]
    metrics: Mapping[str, Any]


class CorpusSnapshotManager:
    """Create immutable rolling replay snapshots and one frozen validation set."""

    def __init__(
        self,
        replay_dir: str,
        snapshot_root: str,
        validation_fraction: float = 0.15,
        split_seed: int = 20260819,
        min_fresh_fraction: float = 0.50,
        enforce_policy_contract: bool = False,
        allowed_opening_plies: Sequence[int] = (),
        max_retained_snapshots: int = 0,
        grow_holdout: bool = True,
        validation_split_version: int = VALIDATION_SPLIT_VERSION_DEFAULT,
        lineage_base_manifest: Optional[str] = None,
        lineage_base_fingerprint: Optional[str] = None,
        lineage_excluded_fingerprints: Sequence[str] = (),
        trained_ledger_enabled: bool = False,
        trained_ledger_seed_roots: Sequence[str] = (),
        reuse_previous_shards: bool = True,
    ) -> None:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0.0 <= min_fresh_fraction <= 1.0:
            raise ValueError("min_fresh_fraction must be between 0 and 1")
        if int(max_retained_snapshots) < 0:
            raise ValueError("max_retained_snapshots must be zero or positive")
        if int(validation_split_version) < 1:
            raise ValueError("validation_split_version must be 1 or greater")
        if bool(lineage_base_manifest) != bool(lineage_base_fingerprint):
            raise ValueError(
                "lineage_base_manifest and lineage_base_fingerprint must be "
                "configured together"
            )
        self.replay_dir = Path(replay_dir)
        self.snapshot_root = Path(snapshot_root)
        self.validation_fraction = float(validation_fraction)
        self.split_seed = int(split_seed)
        self.min_fresh_fraction = float(min_fresh_fraction)
        # Snapshots store whole replay shards with storage="copy" (see
        # _link_or_copy), so each admission costs another full corpus on disk
        # and nothing ever reclaimed it.  At the steady-state corpus size a
        # multi-day run exhausts the volume long before its time limit.  Keep
        # the newest N admissions and drop older ones; 0 keeps every snapshot,
        # which is the historical behaviour.
        self.max_retained_snapshots = int(max_retained_snapshots)
        # Hardlink unchanged shards from the previous snapshot at admission
        # instead of copying the whole corpus again.  Digest-verified before
        # linking and re-verified by every load (see _store_shard); disable to
        # restore the copy-everything behaviour.
        self.reuse_previous_shards = bool(reuse_previous_shards)
        # Audit Suggestion 9: the most recent hold-out growth event, so the
        # freshness it cost this cycle is attributable from the artifacts
        # instead of by correlating two independent log lines.
        self._last_holdout_growth: Optional[Dict[str, Any]] = None
        # The hold-out is created once from whatever files existed then and,
        # historically, never grew -- so an approved 15% share decayed to 1.7%
        # as the rolling corpus expanded, and the manifest asserted a hold-out
        # it did not deliver.  Growth is append-only: a file that has ever been
        # held stays held, so no state ever moves from validation into train.
        self.grow_holdout = bool(grow_holdout)
        self.enforce_policy_contract = bool(enforce_policy_contract)
        self.allowed_opening_plies = tuple(int(value) for value in allowed_opening_plies)
        self.external_validation_state_keys: Set[str] = set()
        self.validation_split_version = int(validation_split_version)
        # An explicit external predecessor.  snapshot_v000012 was admitted with
        # a lost CURRENT pointer, so its recorded 100% freshness was an artifact
        # of an empty previous-key set and v13-v16 all descend from it.  Rather
        # than accept that chain, a rebuilt lineage names its last *valid*
        # predecessor here: the first admission in the new namespace is then
        # gated against that corpus instead of against nothing, and every
        # manifest records which base it descends from.
        self.lineage_base_manifest = (
            Path(lineage_base_manifest) if lineage_base_manifest else None)
        self.lineage_base_fingerprint = (
            str(lineage_base_fingerprint).lower() if lineage_base_fingerprint else None)
        self.lineage_excluded_fingerprints = frozenset(
            str(value).lower() for value in lineage_excluded_fingerprints if value)
        if (self.lineage_base_fingerprint
                and self.lineage_base_fingerprint in self.lineage_excluded_fingerprints):
            raise ValueError(
                "lineage_base_fingerprint cannot also be excluded from the lineage")
        self.trained_ledger_enabled = bool(trained_ledger_enabled)
        self.trained_ledger_seed_roots = tuple(
            Path(value) for value in trained_ledger_seed_roots)
        self._lineage_base_cache: Optional[Tuple[Path, dict, Set[str]]] = None
        self._trained_ledger_cache: Optional[Tuple[Set[str], Set[int]]] = None

    def set_external_validation_state_keys(self, state_keys: Iterable[str]) -> None:
        """Exclude a frozen external validation suite from every train snapshot."""
        self.external_validation_state_keys = {str(key) for key in state_keys}

    @property
    def current_pointer(self) -> Path:
        return self.snapshot_root / "CURRENT"

    @property
    def validation_dir_name(self) -> str:
        """Directory holding the active hold-out generation.

        Version 1 keeps the historical ``validation`` name so existing
        namespaces load unchanged; a rebuilt split lands beside it under
        ``validation_v<N>`` so the superseded artifact stays readable evidence
        and cannot be mistaken for the active one.
        """
        if self.validation_split_version <= 1:
            return "validation"
        return f"validation_v{self.validation_split_version}"

    @property
    def validation_manifest_path(self) -> Path:
        return self.snapshot_root / self.validation_dir_name / "manifest.json"

    @property
    def trained_ledger_dir(self) -> Path:
        return self.snapshot_root / "ledger"

    def replay_files(self) -> List[Path]:
        return sorted(
            (path for path in self.replay_dir.glob("replay_*.jsonl") if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )

    def eligible_replay_files(self) -> Tuple[List[Path], Dict[str, dict]]:
        """Return repaired-contract files and rejection diagnostics."""
        files = self.replay_files()
        if not self.enforce_policy_contract:
            return files, {}
        eligible = []
        rejected = {}
        for path in files:
            audit = audit_policy_replay_file(path, self.allowed_opening_plies)
            if audit["valid"]:
                eligible.append(path)
            else:
                rejected[path.name] = audit
        return eligible, rejected

    def current_manifest_path(self) -> Optional[Path]:
        if not self.current_pointer.exists():
            return None
        relative = self.current_pointer.read_text(encoding="utf-8").strip()
        if not relative:
            return None
        path = self.snapshot_root / _read_relpath(relative)
        return path if path.is_file() else None

    def _load_manifest(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def snapshot_matches_settings(
        self,
        manifest_path: Path,
        teacher_settings: Mapping[str, Any],
        noise_settings: Mapping[str, Any],
        generation_settings: Mapping[str, Any],
    ) -> bool:
        """Return whether a snapshot was built for the active data contract."""

        manifest = self._load_manifest(Path(manifest_path))
        return (
            dict(manifest.get("teacher_settings", {})) == dict(teacher_settings)
            and dict(manifest.get("noise_settings", {})) == dict(noise_settings)
            and dict(manifest.get("generation_settings", {}))
            == dict(generation_settings)
        )

    def _verify_manifest_integrity(
        self,
        manifest_path: Path,
        manifest: Mapping[str, Any],
        expected_kind: str,
    ) -> Set[str]:
        """Verify stored files and the canonical-state set before loading."""

        expected = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "kind": expected_kind,
            "encoding_version": ENCODING_VERSION,
            "rules_id": CANONICAL_RULES_ID,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"Corpus manifest uses {key}={manifest.get(key)!r}, "
                    f"expected {value!r}: {manifest_path}"
                )

        # ``record["path"]`` is read through ``_read_relpath`` for the same
        # reason the CURRENT pointer is: a snapshot written by the native-Windows
        # launcher stores ``files\\replay_*.jsonl``.  On WSL that is one filename
        # containing a backslash, so every stored shard "fails integrity
        # verification" while sitting untouched on disk right next to the manifest.
        for record in manifest.get("files", []):
            try:
                stored = manifest_path.parent / _read_relpath(record["path"])
                expected_size = int(record["size_bytes"])
                expected_sha256 = str(record["sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Corpus manifest has an invalid file record: {manifest_path}"
                ) from exc
            if (
                not stored.is_file()
                or stored.stat().st_size != expected_size
                or replay_file_sha256(stored) != expected_sha256
            ):
                raise RuntimeError(
                    f"Corpus snapshot file failed integrity verification: {stored}"
                )

        state_keys_name = manifest.get("state_keys_file")
        if not isinstance(state_keys_name, str) or not state_keys_name:
            raise RuntimeError(
                f"Corpus manifest has no canonical-state key file: {manifest_path}"
            )
        state_keys_path = manifest_path.parent / state_keys_name
        if not state_keys_path.is_file():
            raise RuntimeError(
                f"Corpus canonical-state key file is missing: {state_keys_path}"
            )
        state_keys = _read_state_keys(state_keys_path)
        expected_state_digest = manifest.get("metrics", {}).get(
            "state_set_sha256"
        )
        if expected_state_digest != _state_set_digest(state_keys):
            raise RuntimeError(
                f"Corpus canonical-state fingerprint is invalid: {manifest_path}"
            )
        return state_keys

    # ------------------------------------------------------------------
    # Rebuilt lineage: an explicit, verified external predecessor
    # ------------------------------------------------------------------

    def _lineage_base(self) -> Optional[Tuple[Path, dict, Set[str]]]:
        """Load and verify the pinned external predecessor snapshot.

        Returns ``None`` when no lineage base is configured, which is the
        historical behaviour.  When one *is* configured it is treated as a
        read-only input: its integrity is verified and its recorded
        fingerprint must equal the configured pin, so a rebuilt lineage cannot
        silently branch from a different (or defective) corpus.
        """
        if self.lineage_base_manifest is None:
            return None
        if self._lineage_base_cache is not None:
            return self._lineage_base_cache
        manifest_path = self.lineage_base_manifest
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Configured corpus lineage base manifest is missing: {manifest_path}"
            )
        manifest = self._load_manifest(manifest_path)
        state_keys = self._verify_manifest_integrity(
            manifest_path, manifest, "training_snapshot"
        )
        recorded = str(manifest.get("fingerprint", "")).lower()
        if recorded != self.lineage_base_fingerprint:
            raise RuntimeError(
                "Corpus lineage base fingerprint mismatch at "
                f"{manifest_path}: expected {self.lineage_base_fingerprint}, "
                f"got {recorded or None}"
            )
        self._lineage_base_cache = (manifest_path, manifest, state_keys)
        return self._lineage_base_cache

    def _lineage_base_record(self) -> Optional[dict]:
        """Provenance block stamped into every snapshot of a rebuilt lineage.

        Recorded on *each* manifest rather than only the first, because
        retention prunes the oldest directories: after the first admission is
        pruned, a chain walk can no longer reach the base, and this record is
        what keeps the ancestry claim verifiable for the whole namespace.
        """
        base = self._lineage_base()
        if base is None:
            return None
        manifest_path, manifest, _keys = base
        return {
            "manifest": str(manifest_path.resolve()),
            "fingerprint": str(manifest.get("fingerprint", "")).lower(),
            "version": manifest.get("version"),
            "excluded_fingerprints": sorted(self.lineage_excluded_fingerprints),
        }

    def verify_lineage(self, manifest: Mapping[str, Any], manifest_path: Path) -> dict:
        """Fail closed unless this snapshot descends from the approved base.

        Three independent claims are checked, none of which depends on an
        unpruned chain:

        1. the snapshot records the configured base and excluded set;
        2. neither its own fingerprint nor its recorded predecessor is an
           excluded (defective or superseded) corpus;
        3. every retained ancestor link inside the namespace is contiguous and
           equally free of excluded fingerprints.

        Enforcement is opt-in per config, but opting *out* is not: a snapshot
        that records a lineage policy is refused by a run that declares none.
        """
        base = self._lineage_base()
        recorded = manifest.get("lineage_base")
        if base is None:
            # Opting out is not an escape hatch.  A snapshot that *records* a
            # lineage policy was admitted under one, and every refusal below --
            # including the relaxed-exclusion refusal, whose whole purpose is to
            # stop a weaker run from loading strictly-admitted data -- is only
            # reachable while a base is configured.  Without this branch a run
            # that dropped the entire ``lineage:`` block, rather than merely one
            # excluded fingerprint, would load that same data with no base
            # check, no exclusion check, and no chain check at all.  Legacy
            # namespaces stamp no such record and stay unenforced.
            if isinstance(recorded, Mapping):
                raise RuntimeError(
                    "Corpus snapshot was admitted under a lineage policy that "
                    f"this run does not declare: {manifest_path}"
                )
            return {"enforced": False}
        _base_path, base_manifest, _base_keys = base
        base_fingerprint = str(base_manifest.get("fingerprint", "")).lower()

        if not isinstance(recorded, Mapping):
            raise RuntimeError(
                "Corpus snapshot records no lineage base but one is required: "
                f"{manifest_path}"
            )
        if str(recorded.get("fingerprint", "")).lower() != base_fingerprint:
            raise RuntimeError(
                f"Corpus snapshot descends from an unapproved lineage base: "
                f"{manifest_path}"
            )
        chain: List[str] = []
        for value in (manifest.get("fingerprint"), manifest.get("previous_fingerprint")):
            if isinstance(value, str) and value:
                chain.append(value.lower())
        for _version, path in self._snapshot_dirs():
            candidate = path / "manifest.json"
            if not candidate.is_file():
                continue
            try:
                other = self._load_manifest(candidate)
            except (OSError, ValueError):
                continue
            for value in (other.get("fingerprint"), other.get("previous_fingerprint")):
                if isinstance(value, str) and value:
                    chain.append(value.lower())
        contaminated = sorted(set(chain) & set(self.lineage_excluded_fingerprints))
        if contaminated:
            raise RuntimeError(
                "Corpus lineage contains excluded snapshot fingerprint(s) "
                f"{contaminated}: {manifest_path}"
            )

        # The configured exclusions are enforced live by the chain check above,
        # so a *newly* discovered defective ancestor does not need to have been
        # known at admission time.  What must never happen is the reverse: a
        # snapshot admitted under a stricter policy being loaded by a run that
        # has since dropped one of those exclusions.
        recorded_excluded = {
            str(value).lower() for value in recorded.get("excluded_fingerprints", [])
        }
        relaxed = sorted(recorded_excluded - set(self.lineage_excluded_fingerprints))
        if relaxed:
            raise RuntimeError(
                "Corpus snapshot was admitted excluding fingerprint(s) "
                f"{relaxed}, which this run no longer excludes: {manifest_path}"
            )
        return {
            "enforced": True,
            "base_fingerprint": base_fingerprint,
            "base_version": base_manifest.get("version"),
            "excluded_fingerprints": sorted(self.lineage_excluded_fingerprints),
            "checked_fingerprints": len(set(chain)),
        }

    # ------------------------------------------------------------------
    # All-time trained ledger
    # ------------------------------------------------------------------

    @property
    def _ledger_shards_path(self) -> Path:
        return self.trained_ledger_dir / "trained_shards.jsonl"

    @property
    def _ledger_state_keys_path(self) -> Path:
        return self.trained_ledger_dir / "trained_state_keys.txt.gz"

    @property
    def _ledger_seed_path(self) -> Path:
        return self.trained_ledger_dir / "seed.json"

    def _seed_trained_ledger(self) -> None:
        """Recover the all-time trained set from preserved historical roots.

        Reads only; every write lands under this namespace's ledger directory.
        Retention has already pruned some historical snapshots, so the result
        is a *lower bound* on the all-time trained set -- which is recorded
        explicitly in ``seed.json`` rather than being implied to be complete.
        Both prior training snapshots and prior hold-out generations are
        seeded: a shard that sat in a contaminated hold-out was demonstrably
        trained on, so re-holding it would reproduce the original defect.
        """
        shard_records: Dict[str, dict] = {}
        state_keys: Set[str] = set()
        sources: List[dict] = []
        for root in self.trained_ledger_seed_roots:
            if not root.is_dir():
                sources.append({"root": str(root), "status": "missing"})
                continue
            manifests = sorted(root.glob("snapshot_v*/manifest.json"))
            manifests.extend(sorted(root.glob("validation*/manifest.json")))
            seeded = 0
            for manifest_path in manifests:
                try:
                    manifest = self._load_manifest(manifest_path)
                except (OSError, ValueError):
                    continue
                for record in manifest.get("files", []):
                    name = record.get("name")
                    if not isinstance(name, str):
                        continue
                    shard_records.setdefault(name, {
                        "name": name,
                        "sha256": record.get("sha256"),
                        "origin": str(manifest_path),
                    })
                keys_file = manifest.get("state_keys_file")
                if isinstance(keys_file, str) and keys_file:
                    keys_path = manifest_path.parent / keys_file
                    if keys_path.is_file():
                        state_keys |= _read_state_keys(keys_path)
                seeded += 1
            sources.append({
                "root": str(root),
                "status": "read",
                "manifests": seeded,
            })

        self.trained_ledger_dir.mkdir(parents=True, exist_ok=True)
        with self._ledger_shards_path.open("w", encoding="utf-8", newline="\n") as handle:
            for name in sorted(shard_records):
                handle.write(json.dumps(
                    {**shard_records[name], "recorded_by": "seed"},
                    sort_keys=True, separators=(",", ":")) + "\n")
        self._ledger_state_keys_path.unlink(missing_ok=True)
        _merge_state_keys_file(self._ledger_state_keys_path, state_keys)
        _write_json_atomic(self._ledger_seed_path, {
            "schema_version": TRAINED_LEDGER_SCHEMA_VERSION,
            "seeded_at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "shard_count": len(shard_records),
            "state_key_count": len(state_keys),
            "completeness": (
                "lower_bound: snapshot retention may already have pruned older "
                "generations, so states trained before the oldest retained "
                "manifest cannot be recovered"
            ),
        })
        print(
            f"Trained-shard ledger seeded: {len(shard_records)} shard(s), "
            f"{len(state_keys)} canonical state(s) from "
            f"{len(self.trained_ledger_seed_roots)} historical root(s)"
        )

    def _ensure_trained_ledger(self) -> None:
        if not self.trained_ledger_enabled:
            return
        if self._ledger_seed_path.is_file():
            return
        self._seed_trained_ledger()

    def _load_trained_ledger(self) -> Tuple[Set[str], Set[int]]:
        """Return (shard names, state fingerprints) known to have been trained.

        States are held as 64-bit fingerprints rather than full keys; see
        :func:`_state_key_fingerprint` for why, and for why the failure
        direction is safe.
        """
        if not self.trained_ledger_enabled:
            return set(), set()
        if self._trained_ledger_cache is not None:
            return self._trained_ledger_cache
        self._ensure_trained_ledger()
        names: Set[str] = set()
        if self._ledger_shards_path.is_file():
            with self._ledger_shards_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = record.get("name")
                    if isinstance(name, str):
                        names.add(name)
        fingerprints: Set[int] = set()
        if self._ledger_state_keys_path.is_file():
            for key in _iter_state_keys(self._ledger_state_keys_path):
                fingerprints.add(_state_key_fingerprint(key))
        self._trained_ledger_cache = (names, fingerprints)
        return self._trained_ledger_cache

    def trained_ledger_shard_names(self) -> Set[str]:
        return set(self._load_trained_ledger()[0])

    def trained_ledger_state_fingerprints(self) -> Set[int]:
        """Membership set used to reject historically trained hold-out states."""
        return set(self._load_trained_ledger()[1])

    def trained_ledger_state_keys(self) -> Set[str]:
        """Full canonical keys, read from disk for auditing.

        Deliberately uncached: the whole point of the fingerprint set is not to
        keep this in memory for the life of a run.
        """
        if not self.trained_ledger_enabled:
            return set()
        self._ensure_trained_ledger()
        if not self._ledger_state_keys_path.is_file():
            return set()
        return _read_state_keys(self._ledger_state_keys_path)

    def _record_trained_ledger(
        self,
        *,
        version: int,
        file_records: Sequence[Mapping[str, Any]],
        state_keys: Set[str],
    ) -> None:
        """Append one admission to the append-only all-time trained ledger.

        Called only after the snapshot is durable, so the ledger never claims a
        shard that no admission used.  Shard rows are appended; the state set is
        rewritten as the union, which is the only representation that stays
        answerable in one read after arbitrary pruning.
        """
        if not self.trained_ledger_enabled:
            return
        self._ensure_trained_ledger()
        known_names, known_fingerprints = self._load_trained_ledger()
        self.trained_ledger_dir.mkdir(parents=True, exist_ok=True)
        recorded_at = datetime.now(timezone.utc).isoformat()
        new_rows = []
        for record in file_records:
            name = record.get("name")
            if not isinstance(name, str) or name in known_names:
                continue
            known_names.add(name)
            new_rows.append({
                "name": name,
                "sha256": record.get("sha256"),
                "origin": f"snapshot_v{int(version):06d}",
                "recorded_at": recorded_at,
                "recorded_by": "admission",
            })
        if new_rows:
            with self._ledger_shards_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                for row in new_rows:
                    handle.write(json.dumps(
                        row, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        added_states = _merge_state_keys_file(
            self._ledger_state_keys_path, state_keys)
        known_fingerprints |= {
            _state_key_fingerprint(key) for key in state_keys}
        self._trained_ledger_cache = (known_names, known_fingerprints)
        if new_rows or added_states:
            print(
                f"  Trained ledger: +{len(new_rows)} shard(s), "
                f"+{added_states} canonical state(s) "
                f"({len(known_names)} shard(s) all-time)"
            )

    @staticmethod
    def _realized_split_share(
        *, validation_files: int, total_files: int, present_files: int = -1,
    ) -> dict:
        """Describe the share the validation split actually holds.

        ``split.fraction`` records the *requested* hold-out.  Reporting only
        that value lets the manifest assert a hold-out it does not deliver,
        which is exactly how an approved 15% decayed to 1.7%: the split was
        created once and never grew while the rolling corpus expanded.
        ``_grow_validation`` now tracks the requested share append-only, and
        these additive fields keep the realized share checkable either way --
        including under ``grow_holdout=False``, where the split stays frozen
        at creation size and the realized share still decays.
        """
        total = max(0, int(total_files))
        held = max(0, int(validation_files))
        # ``held`` counts every shard the hold-out has ever absorbed, including
        # ones the rolling replay window has since dropped.  Reporting only that
        # ratio lets a decayed hold-out keep advertising its target share, which
        # is the same class of lie the fields were added to prevent -- so the
        # still-in-corpus count is reported alongside it.  Both are clamped:
        # held shards outliving the window can otherwise exceed 100%.
        present = held if present_files < 0 else max(0, min(int(present_files), held))
        return {
            "validation_file_count": held,
            "source_file_count": total,
            "realized_file_fraction": min(1.0, held / total) if total else 0.0,
            "validation_files_in_corpus": present,
            "realized_in_corpus_fraction": (
                min(1.0, present / total) if total else 0.0),
        }

    def _validation_rank_key(self, name: str) -> str:
        """Deterministic hold-out ordering, identical to the creation-time rank."""
        return hashlib.sha256(
            f"{self.split_seed}:{name}".encode("utf-8")
        ).hexdigest()

    def _validation_growth_quota(
        self, *, total_files: int, held: int, candidates: int,
        all_time_held: int = 0,
    ) -> int:
        """How many unheld files the hold-out may absorb on this pass."""
        if total_files <= 0 or candidates <= 0:
            return 0
        # Target the share of *today's* corpus, which is itself capped by
        # replay_max_files.  Sizing against an all-time file count instead
        # would grow the hold-out without bound as shards rotate, eventually
        # starving training; this settles at the configured share and stops.
        target = max(1, int(round(total_files * self.validation_fraction)))
        need = target - held
        # ``held`` is now the still-present count, so shards rotating out of the
        # replay window reopen the quota.  Without a ceiling that is unbounded
        # over a long run: the manifest never releases a shard, so it would
        # accumulate one per rotation.  Cap the manifest at HOLDOUT_FILE_CEILING
        # times the target -- enough headroom to track the live corpus, bounded
        # like the snapshot root now is.
        ceiling = HOLDOUT_FILE_CEILING * target
        if all_time_held >= ceiling:
            return 0
        need = min(need, ceiling - all_time_held)
        if need <= 0:
            return 0
        # consider_snapshot fails closed when the split leaves no training
        # file, so growth must always leave at least one behind.
        return max(0, min(need, candidates - 1))

    def _trained_shard_names(self) -> Set[str]:
        """Names of replay shards any retained snapshot has served as training data.

        Promoting such a shard into the hold-out produces a set the model has
        already fit, so measurements on it read as memorisation rather than
        generalisation -- the exact failure this document's F3 fix must not
        reintroduce.  Retention can prune older snapshots, so this is a lower
        bound on the all-time trained set; it always includes the active
        corpus, which is what matters for the run in progress.

        When the append-only trained ledger is enabled it is unioned in, which
        is what closes that gap: the ledger outlives the directories retention
        deletes, so a shard trained on months ago is still refused here.
        """
        trained: Set[str] = set(self.trained_ledger_shard_names())
        for _version, path in self._snapshot_dirs():
            manifest_path = path / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = self._load_manifest(manifest_path)
            except (OSError, ValueError):
                continue
            for record in manifest.get("files", []):
                name = record.get("name")
                if isinstance(name, str):
                    trained.add(name)
        return trained

    def _grow_validation(
        self, manifest: dict, files: Sequence[Path],
    ) -> Optional[Tuple[dict, Set[str]]]:
        """Extend the frozen hold-out toward its configured share, append-only.

        Held files are never dropped or reordered, so a canonical state can
        only ever move from train into validation -- never the reverse -- and
        the leakage-resistant whole-file property is preserved.  The manifest
        write is the single commit point: the regenerated key set is written
        under a new generation-scoped name first, so a crash before the commit
        leaves the previous generation intact and independently verifiable.
        """
        if not self.grow_holdout:
            return None
        manifest_path = self.validation_manifest_path
        validation_dir = manifest_path.parent
        held_names = {str(record["name"]) for record in manifest.get("files", [])}
        trained_names = self._trained_shard_names()
        candidates = sorted(
            (path for path in files
             if path.name not in held_names and path.name not in trained_names),
            key=lambda path: self._validation_rank_key(path.name),
        )
        # The quota is measured against still-present held shards, so a hold-out
        # whose files have rotated out of the replay window is not counted as if
        # it still covered the live corpus.
        present_held = sum(1 for path in files if path.name in held_names)
        add_count = self._validation_growth_quota(
            total_files=len(files),
            held=present_held,
            candidates=len(candidates),
            all_time_held=len(held_names),
        )
        if add_count <= 0:
            return None

        files_dir = validation_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        added_records = []
        for source in candidates[:add_count]:
            destination = files_dir / source.name
            # The hold-out is append-only and never re-stores a held shard, so
            # it has no predecessor to reuse from; always copy here.
            link_mode = _store_shard(source, destination)
            added_records.append({
                "name": source.name,
                "path": (Path("files") / source.name).as_posix(),
                "sha256": replay_file_sha256(source),
                "size_bytes": source.stat().st_size,
                "storage": link_mode,
            })

        # Audit Suggestion 9: the canonical states this growth event moved from
        # training into the hold-out.  Analysis is per-file cached, so this is
        # a dictionary lookup for shards the caller has already measured.
        _added_metrics, added_state_keys = analyze_replay_files(
            [files_dir / record["name"] for record in added_records]
        )

        file_records = list(manifest.get("files", [])) + added_records
        metrics, state_keys = analyze_replay_files(
            [validation_dir / _read_relpath(record["path"])
             for record in file_records]
        )

        history = list(manifest.get("growth_history", []))
        generation = len(history) + 1
        state_keys_file = f"canonical_state_keys_g{generation:03d}.txt.gz"
        _write_state_keys(validation_dir / state_keys_file, state_keys)
        history.append({
            "generation": generation,
            "grown_at": datetime.now(timezone.utc).isoformat(),
            "added_files": [record["name"] for record in added_records],
            "validation_file_count": len(file_records),
            "source_file_count": len(files),
        })

        previous_keys_file = manifest.get("state_keys_file")
        grown = dict(manifest)
        grown["files"] = file_records
        grown["state_keys_file"] = state_keys_file
        grown["metrics"] = metrics
        grown["growth_history"] = history
        grown_names = {str(record["name"]) for record in file_records}
        grown["split"] = dict(
            manifest.get("split", {}),
            **self._realized_split_share(
                validation_files=len(file_records), total_files=len(files),
                present_files=sum(1 for p in files if p.name in grown_names),
            ),
        )
        _write_json_atomic(manifest_path, grown)

        # Unreferenced once the manifest commit lands; snapshots point at the
        # manifest, never at a key file directly.
        if (isinstance(previous_keys_file, str)
                and previous_keys_file != state_keys_file):
            try:
                (validation_dir / previous_keys_file).unlink(missing_ok=True)
            except OSError:
                pass

        self._last_holdout_growth = {
            "added_files": [record["name"] for record in added_records],
            "added_state_keys": added_state_keys,
        }

        print(
            f"Validation hold-out grew by {len(added_records)} file(s): "
            f"{len(file_records)} of {len(files)} = "
            f"{grown['split']['realized_file_fraction'] * 100.0:.2f}% realized "
            f"(target {self.validation_fraction * 100.0:.2f}%)"
        )
        return grown, state_keys

    def _ensure_validation(self, files: Sequence[Path]) -> Tuple[dict, Set[str]]:
        # Audit Suggestion 9: one cycle, one growth event.  Reset before the
        # split is resolved so an unchanged hold-out reports no cost rather
        # than repeating the previous cycle's.
        self._last_holdout_growth = None
        manifest_path = self.validation_manifest_path
        if manifest_path.exists():
            manifest = self._load_manifest(manifest_path)
            split = manifest.get("split", {})
            state_keys = self._verify_manifest_integrity(
                manifest_path, manifest, "immutable_validation"
            )
            if (split.get("unit") != "replay_file"
                    or abs(float(split.get("fraction", -1.0)) - self.validation_fraction)
                    > 1.0e-12
                    or int(split.get("seed", -1)) != self.split_seed):
                raise RuntimeError(
                    "Frozen validation manifest does not match the configured "
                    "whole-file split fraction and seed"
                )
            # A rebuilt split must never be satisfied by the artifact it
            # replaces.  Version 1 manifests predate the field, so an absent
            # version reads as 1 rather than as "unknown, accept anything".
            stored_version = int(split.get("version", VALIDATION_SPLIT_VERSION_DEFAULT))
            if stored_version != self.validation_split_version:
                raise RuntimeError(
                    f"Frozen validation manifest is generation {stored_version}, "
                    f"but generation {self.validation_split_version} is configured: "
                    f"{manifest_path}"
                )
            grown = self._grow_validation(manifest, files)
            if grown is not None:
                manifest, state_keys = grown
                split = manifest.get("split", {})
            held_now = {str(r["name"]) for r in manifest.get("files", [])}
            realized = self._realized_split_share(
                validation_files=len(held_now),
                total_files=len(files),
                present_files=sum(1 for p in files if p.name in held_now),
            )
            manifest["split"] = dict(split, **realized)
            # Persist the recalculation, not just print it.  A decay-only pass
            # (no growth) used to leave the manifest advertising the share it
            # had at its last *growth*, so an on-disk audit read 9/58 while the
            # live run was at 3/60.  The manifest must state what it is worth
            # today even when nothing was added.
            if any(split.get(key) != value for key, value in realized.items()):
                _write_json_atomic(manifest_path, manifest)
            # Report what the hold-out is actually worth against today's
            # corpus, not only the fraction the manifest requests.
            _stale = (realized['validation_file_count']
                      - realized['validation_files_in_corpus'])
            print(
                f"Validation hold-out: "
                f"{realized['validation_file_count']} of "
                f"{realized['source_file_count']} replay file(s) = "
                f"{realized['realized_file_fraction'] * 100.0:.2f}% realized "
                f"(target {self.validation_fraction * 100.0:.2f}%)"
                + (f"; {realized['validation_files_in_corpus']} still in corpus "
                   f"= {realized['realized_in_corpus_fraction'] * 100.0:.2f}% "
                   f"({_stale} rotated out)" if _stale else "")
            )
            return manifest, state_keys

        if len(files) < 2:
            raise RuntimeError("At least two replay files are required for a whole-file validation split")

        desired = max(1, int(round(len(files) * self.validation_fraction)))
        desired = min(desired, len(files) - 1)
        # Growth already refuses shards any snapshot has trained on; creation
        # must apply the same rule or a rebuilt split starts contaminated on
        # its very first generation.
        trained_names = self._trained_shard_names()
        eligible = [path for path in files if path.name not in trained_names]
        if not eligible:
            raise RuntimeError(
                "Every replay file has already been used as training data, so "
                "no leakage-resistant validation split can be created under "
                f"{self.snapshot_root}"
            )
        if len(eligible) < desired:
            print(
                f"[warn] Validation split reduced from {desired} to "
                f"{len(eligible)} file(s): the remainder were already trained on"
            )
            desired = len(eligible)
        ranked = sorted(
            eligible,
            key=lambda path: hashlib.sha256(
                f"{self.split_seed}:{path.name}".encode("utf-8")
            ).hexdigest(),
        )
        selected = ranked[:desired]
        validation_dir = manifest_path.parent
        files_dir = validation_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=False)

        file_records = []
        for source in selected:
            destination = files_dir / source.name
            link_mode = _store_shard(source, destination)
            file_records.append({
                "name": source.name,
                "path": (Path("files") / source.name).as_posix(),
                "sha256": replay_file_sha256(source),
                "size_bytes": source.stat().st_size,
                "storage": link_mode,
            })

        metrics, state_keys = analyze_replay_files(selected)
        state_keys_file = "canonical_state_keys.txt.gz"
        _write_state_keys(validation_dir / state_keys_file, state_keys)
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "kind": "immutable_validation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": {
                "unit": "replay_file",
                "version": self.validation_split_version,
                "fraction": self.validation_fraction,
                "seed": self.split_seed,
                **self._realized_split_share(
                    validation_files=len(file_records),
                    total_files=len(files),
                    present_files=len(file_records),
                ),
            },
            "encoding_version": ENCODING_VERSION,
            "rules_id": CANONICAL_RULES_ID,
            "files": file_records,
            "state_keys_file": state_keys_file,
            "metrics": metrics,
        }
        _write_json_atomic(manifest_path, manifest)
        return manifest, state_keys

    def _snapshot_dirs(self) -> List[Tuple[int, Path]]:
        """Return admitted snapshot directories as (version, path), oldest first."""
        found: List[Tuple[int, Path]] = []
        if not self.snapshot_root.exists():
            return found
        for path in self.snapshot_root.glob("snapshot_v*"):
            if not path.is_dir():
                continue
            try:
                found.append((int(path.name.removeprefix("snapshot_v")), path))
            except ValueError:
                continue
        found.sort()
        return found

    def _prune_old_snapshots(self, keep_dir: Path) -> List[str]:
        """Drop all but the newest ``max_retained_snapshots`` admissions.

        Only ``current.json``/``CURRENT`` are ever read back, ``_next_version``
        needs only the highest directory name, and the manifest chain links by
        fingerprint rather than by path -- so older directories are audit
        history, not live inputs.  Deleting is best-effort: a shard held open
        by another process (common on drvfs) must never abort an admission,
        and the next admission retries the same directory.
        """
        if self.max_retained_snapshots <= 0:
            return []
        snapshots = self._snapshot_dirs()
        if len(snapshots) <= self.max_retained_snapshots:
            return []
        keep_resolved = keep_dir.resolve()
        stale = snapshots[: len(snapshots) - self.max_retained_snapshots]
        removed: List[str] = []
        for _version, path in stale:
            if path.resolve() == keep_resolved:
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                print(f"[warn] Could not prune corpus snapshot {path.name}: {exc}")
                continue
            removed.append(path.name)
        return removed

    def _next_version(self) -> int:
        versions = []
        if self.snapshot_root.exists():
            for path in self.snapshot_root.glob("snapshot_v*"):
                try:
                    versions.append(int(path.name.removeprefix("snapshot_v")))
                except ValueError:
                    continue
        return max(versions, default=0) + 1

    def consider_snapshot(
        self,
        teacher_settings: Mapping[str, Any],
        noise_settings: Mapping[str, Any],
        generation_settings: Mapping[str, Any],
    ) -> SnapshotDecision:
        """Admit a new immutable snapshot if its fresh-state gate passes."""

        files, rejected_files = self.eligible_replay_files()
        if not files:
            raise RuntimeError(
                "No replay files satisfy the repaired policy-distillation contract"
            )
        validation_manifest, validation_keys = self._ensure_validation(files)
        validation_keys = validation_keys.union(self.external_validation_state_keys)
        validation_hashes = {record["sha256"] for record in validation_manifest["files"]}

        file_records = []
        train_files = []
        for path in files:
            file_hash = replay_file_sha256(path)
            if file_hash in validation_hashes:
                continue
            train_files.append(path)
            file_records.append({
                "name": path.name,
                "sha256": file_hash,
                "size_bytes": path.stat().st_size,
            })
        if not train_files:
            raise RuntimeError("No replay files remain after the immutable validation split")

        current_path = self.current_manifest_path()
        previous_keys: Set[str] = set()
        previous_fingerprint = None
        previous_source = None
        if current_path is not None:
            current = self._load_manifest(current_path)
            previous_keys = _read_state_keys(current_path.parent / current["state_keys_file"])
            previous_fingerprint = current.get("fingerprint")
            previous_source = "current_snapshot"
        else:
            # First admission of a rebuilt lineage.  Without a base this branch
            # is a genuine cold start and the freshness gate cannot apply; with
            # one, the approved external predecessor supplies the previous
            # corpus, so the very first snapshot is gated exactly like every
            # later one instead of being admitted at a meaningless 100%.
            base = self._lineage_base()
            if base is not None:
                _base_path, base_manifest, base_keys = base
                previous_keys = base_keys
                previous_fingerprint = base_manifest.get("fingerprint")
                previous_source = "lineage_base"

        metrics, all_train_keys = analyze_replay_files(train_files, previous_keys)
        leakage_keys = all_train_keys.intersection(validation_keys)
        train_keys = all_train_keys.difference(validation_keys)
        new_keys = train_keys.difference(previous_keys)
        fresh_rate = (len(new_keys) / len(train_keys)) if train_keys else 0.0
        # Audit Suggestion 9.  A growth event moves a whole *fresh* shard out of
        # training -- _grow_validation can only choose never-trained shards, by
        # design, because a trained one would measure memorisation.  The
        # consequence is that the cycle's entire freshness gain is cancelled,
        # which looked from the logs exactly like generator saturation.  Price
        # it here, where the previous corpus is in hand, and record it in the
        # manifest so a stalled admission is explainable from artifacts alone.
        growth = self._last_holdout_growth or {}
        withheld = set(growth.get("added_state_keys", ())).difference(train_keys)
        withheld_fresh = withheld.difference(previous_keys)
        counterfactual_denominator = len(train_keys) + len(withheld)
        counterfactual_fresh_rate = (
            (len(new_keys) + len(withheld_fresh)) / counterfactual_denominator
            if counterfactual_denominator else 0.0
        )

        metrics.update({
            "holdout_growth_files": list(growth.get("added_files", ())),
            "states_transferred_to_holdout": len(withheld),
            "fresh_states_transferred_to_holdout": len(withheld_fresh),
            "fresh_unique_state_rate_without_holdout_growth": (
                counterfactual_fresh_rate),
            "validation_overlap_state_count_removed": len(leakage_keys),
            "external_validation_state_count": len(
                self.external_validation_state_keys),
            "post_dedup_unique_state_count": len(train_keys),
            "fresh_unique_state_count": len(new_keys),
            "fresh_unique_state_rate": fresh_rate,
            "state_set_sha256": _state_set_digest(train_keys),
            "rejected_replay_files": rejected_files,
        })
        source_games = metrics.get("source_game_counts", {})
        algorithm_games = int(source_games.get("algorithm", 0))
        model_games = int(source_games.get("current_model", 0))
        if self.enforce_policy_contract and (
            algorithm_games * 3 != model_games * 7
            or algorithm_games + model_games == 0
        ):
            # Per-file admission already rejects every off-ratio file, so this
            # is a backstop.  Name the offenders anyway: an aggregate count on
            # its own is not actionable.
            offenders = []
            for path in train_files:
                counts = Counter(
                    _cached_replay_file_analysis(path).game_sources.values())
                algorithm = counts.get("algorithm", 0)
                model = counts.get("current_model", 0)
                if algorithm * 3 != model * 7:
                    offenders.append(f"{path.name} ({algorithm}/{model})")
            detail = (
                f"; off-ratio file(s): {', '.join(offenders)}"
                if offenders else ""
            )
            raise RuntimeError(
                "Eligible replay games do not satisfy the exact 70/30 "
                f"trajectory contract: {algorithm_games}/{model_games}{detail}"
            )

        fingerprint_payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "encoding_version": ENCODING_VERSION,
            "rules_id": CANONICAL_RULES_ID,
            "files": file_records,
            "state_set_sha256": metrics["state_set_sha256"],
            "teacher_settings": dict(teacher_settings),
            "noise_settings": dict(noise_settings),
            "generation_settings": dict(generation_settings),
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        if previous_fingerprint == fingerprint:
            return SnapshotDecision(False, "unchanged", current_path, metrics)
        # A missing CURRENT pointer used to disable the freshness gate silently:
        # the guard below reads ``current_path is not None``, so a lost pointer
        # made every candidate admissible and reported a 100% fresh rate that is
        # only an artifact of an empty previous-key set.  snapshot_v000012 in the
        # policy-distillation namespace was admitted exactly that way.  A genuine
        # first admission has no snapshots at all and must still be allowed;
        # a pointer that vanished while snapshots exist is corruption, and this
        # gate fails closed like every other integrity check in this module.
        if current_path is None and previous_source is None and self._snapshot_dirs():
            raise RuntimeError(
                "Corpus snapshot pointer is missing while "
                f"{len(self._snapshot_dirs())} snapshot(s) exist under "
                f"{self.snapshot_root}. The minimum-fresh-state gate cannot be "
                "evaluated without the previous corpus, so no admission is "
                "possible until CURRENT/current.json is restored to the "
                "intended snapshot."
            )
        if withheld_fresh:
            print(
                "  Hold-out growth cost this cycle: "
                f"{len(withheld_fresh)} fresh state(s) of "
                f"{len(withheld)} moved into validation "
                f"({', '.join(growth.get('added_files', ())) or 'unknown file'}). "
                f"Freshness reads {fresh_rate:.2%}; without the transfer it "
                f"would read {counterfactual_fresh_rate:.2%}."
            )
        if previous_source is not None and fresh_rate < self.min_fresh_fraction:
            return SnapshotDecision(
                False,
                f"fresh_unique_state_rate {fresh_rate:.6f} is below {self.min_fresh_fraction:.6f}",
                current_path,
                metrics,
            )

        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        version = self._next_version()
        final_dir = self.snapshot_root / f"snapshot_v{version:06d}"
        if final_dir.exists():
            # Proofread 2026-08-25 C1.  os.replace(staging, final_dir) has no
            # atomic guard against an existing target: a half-written earlier
            # snapshot_vNNNNNN fails admission with errno 39, and an *empty*
            # leftover is silently adopted under a lineage that belongs to
            # nothing.  Fail closed like every other integrity check here;
            # the leftover is preserved as evidence for manual inspection.
            raise RuntimeError(
                f"Corpus snapshot directory {final_dir.name} already exists "
                f"under {self.snapshot_root}; a previous admission may have "
                "crashed or a concurrent writer holds this version number. "
                "Refusing to overwrite; inspect and remove the leftover "
                "snapshot directory manually."
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.", dir=self.snapshot_root))
        try:
            files_dir = staging / "files"
            files_dir.mkdir()
            # Shard reuse: shards unchanged since the previous snapshot are
            # hardlinked from it instead of copied, cutting the per-admission
            # cost from another full corpus to only the new/rotated shards.
            # See _store_shard for why this is fail-closed (digest match before
            # linking; load-time integrity re-verification afterwards).
            previous_files_dir: Optional[Path] = None
            if self.reuse_previous_shards and current_path is not None:
                candidate_root = current_path.parent / "files"
                if candidate_root.is_dir():
                    previous_files_dir = candidate_root
                    print(
                        f"  Corpus: reusing unchanged shard(s) from "
                        f"{current_path.parent.name}/files when possible"
                    )
            stored_files = []
            reused_count = 0
            reused_bytes = 0
            for source, record in zip(train_files, file_records):
                destination = files_dir / source.name
                link_mode = _store_shard(source, destination, previous_files_dir)
                if link_mode == "hardlink":
                    reused_count += 1
                    reused_bytes += int(record["size_bytes"])
                stored_files.append({
                    **record,
                    "path": (Path("files") / source.name).as_posix(),
                    "storage": link_mode,
                })

            state_keys_file = "canonical_state_keys.txt.gz"
            _write_state_keys(staging / state_keys_file, train_keys)
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "kind": "training_snapshot",
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "previous_fingerprint": previous_fingerprint,
                "encoding_version": ENCODING_VERSION,
                "rules_id": CANONICAL_RULES_ID,
                "files": stored_files,
                "state_keys_file": state_keys_file,
                "validation_manifest": _posix_relpath(
                    self.validation_manifest_path, staging
                ),
                "teacher_settings": dict(teacher_settings),
                "noise_settings": dict(noise_settings),
                "generation_settings": dict(generation_settings),
                "admission": {
                    "minimum_fresh_unique_state_rate": self.min_fresh_fraction,
                    "observed_fresh_unique_state_rate": fresh_rate,
                    "passed": True,
                    "previous_corpus_source": previous_source,
                    # Shard-reuse accounting: how much of this admission's
                    # storage came from hardlinks into the previous snapshot
                    # instead of fresh copies.  The bytes are also shared with
                    # the predecessor, so they cost no additional disk.
                    "reused_shard_count": reused_count,
                    "reused_shard_bytes": reused_bytes,
                    "copied_shard_count": len(stored_files) - reused_count,
                },
                "metrics": metrics,
            }
            lineage_base_record = self._lineage_base_record()
            if lineage_base_record is not None:
                manifest["lineage_base"] = lineage_base_record
            _write_json_atomic(staging / "manifest.json", manifest)
            os.replace(staging, final_dir)
            _write_json_atomic(
                self.snapshot_root / "current.json",
                {"manifest": (Path(final_dir.name) / "manifest.json").as_posix(), "fingerprint": fingerprint},
            )
            pointer_temp = self.snapshot_root / "CURRENT.tmp"
            pointer_temp.write_text((Path(final_dir.name) / "manifest.json").as_posix() + "\n", encoding="utf-8")
            os.replace(pointer_temp, self.current_pointer)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

        # Only now that the snapshot is durable and current may the all-time
        # ledger claim these shards and states as trained.
        self._record_trained_ledger(
            version=version, file_records=stored_files, state_keys=train_keys)

        # Reclaim disk only after the new snapshot is durable and current.
        pruned = self._prune_old_snapshots(final_dir)
        if pruned:
            print(
                f"  Corpus: pruned {len(pruned)} old snapshot(s) "
                f"(keeping newest {self.max_retained_snapshots})"
            )

        return SnapshotDecision(True, "admitted", final_dir / "manifest.json", metrics)

    def load_split(
        self,
        manifest_path: Optional[Path] = None,
        max_train_entries: int = 0,
    ) -> Tuple[List[ReplayEntry], List[ReplayEntry], dict]:
        """Load a frozen train/validation split with cross-split deduplication."""

        path = manifest_path or self.current_manifest_path()
        if path is None:
            raise RuntimeError("No corpus snapshot is active")
        manifest = self._load_manifest(path)
        self._verify_manifest_integrity(path, manifest, "training_snapshot")
        manifest["lineage_verification"] = self.verify_lineage(manifest, path)
        manifest["manifest_path"] = str(path.resolve())
        validation_path = (
            path.parent / _read_relpath(manifest["validation_manifest"])).resolve()
        if validation_path != self.validation_manifest_path.resolve():
            raise RuntimeError(
                "Training snapshot references an unexpected validation manifest"
            )
        validation_manifest = self._load_manifest(validation_path)
        stored_validation_keys = self._verify_manifest_integrity(
            validation_path, validation_manifest, "immutable_validation"
        )
        validation_split = validation_manifest.get("split", {})
        stored_split_version = int(
            validation_split.get("version", VALIDATION_SPLIT_VERSION_DEFAULT))
        if stored_split_version != self.validation_split_version:
            raise RuntimeError(
                f"Training snapshot references validation generation "
                f"{stored_split_version}, but generation "
                f"{self.validation_split_version} is configured"
            )
        if (
            validation_split.get("unit") != "replay_file"
            or abs(
                float(validation_split.get("fraction", -1.0))
                - self.validation_fraction
            )
            > 1.0e-12
            or int(validation_split.get("seed", -1)) != self.split_seed
        ):
            raise RuntimeError(
                "Frozen validation manifest does not match the configured "
                "whole-file split fraction and seed"
            )
        manifest["validation_manifest_path"] = str(validation_path)

        # Whole-file hold-out is necessary but not sufficient: an individual
        # state can recur across shards, so a held shard can still contain
        # states an earlier snapshot trained on.  Measuring generalisation on
        # those reads as memorisation, which is the exact defect F3 records.
        # The all-time ledger is what makes the check answerable after
        # retention has pruned the snapshots that did the training.
        historically_trained = self.trained_ledger_state_fingerprints()

        validation_entries: List[ReplayEntry] = []
        validation_keys: Set[str] = set(stored_validation_keys)
        validation_keys.update(self.external_validation_state_keys)
        leaked_validation_states: Set[str] = set()
        leaked_validation_entries = 0
        for record in validation_manifest["files"]:
            file_path = validation_path.parent / _read_relpath(record["path"])
            for entry_dict in _iter_entry_dicts(file_path):
                key = canonical_state_key(entry_dict["state"])
                # The key stays in ``validation_keys`` either way, so a state
                # dropped here is never quietly handed back to training.
                validation_keys.add(key)
                if _state_key_fingerprint(key) in historically_trained:
                    leaked_validation_states.add(key)
                    leaked_validation_entries += 1
                    continue
                validation_entries.append(ReplayEntry.from_dict(entry_dict))
        manifest["validation_leakage"] = {
            "ledger_enabled": self.trained_ledger_enabled,
            "all_time_trained_state_count": len(historically_trained),
            "removed_validation_entry_count": leaked_validation_entries,
            "removed_validation_state_count": len(leaked_validation_states),
            "retained_validation_entry_count": len(validation_entries),
        }
        if leaked_validation_entries:
            print(
                f"Validation hold-out: removed {leaked_validation_entries} "
                f"entry/entries covering {len(leaked_validation_states)} "
                "canonical state(s) already present in the all-time trained "
                f"ledger; {len(validation_entries)} entry/entries remain"
            )

        train_entries: List[ReplayEntry] = []
        for record in manifest["files"]:
            file_path = path.parent / _read_relpath(record["path"])
            for entry_dict in _iter_entry_dicts(file_path):
                if canonical_state_key(entry_dict["state"]) in validation_keys:
                    continue
                train_entries.append(ReplayEntry.from_dict(entry_dict))

        if max_train_entries > 0 and len(train_entries) > max_train_entries:
            rng = random.Random(self.split_seed)
            indices = sorted(rng.sample(range(len(train_entries)), max_train_entries))
            train_entries = [train_entries[index] for index in indices]

        return train_entries, validation_entries, manifest


def split_replay_by_file(
    files: Sequence[Path],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[ReplayEntry], List[ReplayEntry]]:
    """Create a deterministic whole-file split with no canonical-state leakage."""

    if validation_fraction <= 0.0:
        entries = [ReplayEntry.from_dict(value) for path in files for value in _iter_entry_dicts(path)]
        return entries, []
    if len(files) < 2:
        raise RuntimeError("Whole-file validation requires at least two replay files")
    desired = max(1, min(len(files) - 1, int(round(len(files) * validation_fraction))))
    ranked = sorted(
        files,
        key=lambda path: hashlib.sha256(f"{seed}:{path.name}".encode("utf-8")).hexdigest(),
    )
    validation_files = set(ranked[:desired])
    validation_entries: List[ReplayEntry] = []
    validation_keys: Set[str] = set()
    for path in files:
        if path not in validation_files:
            continue
        for value in _iter_entry_dicts(path):
            validation_keys.add(canonical_state_key(value["state"]))
            validation_entries.append(ReplayEntry.from_dict(value))

    train_entries: List[ReplayEntry] = []
    for path in files:
        if path in validation_files:
            continue
        for value in _iter_entry_dicts(path):
            if canonical_state_key(value["state"]) not in validation_keys:
                train_entries.append(ReplayEntry.from_dict(value))
    return train_entries, validation_entries
