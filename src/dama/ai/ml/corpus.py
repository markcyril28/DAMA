"""Versioned replay-corpus snapshots for policy-distillation recovery.

The snapshot manager keeps training data immutable for a training window,
holds validation out by whole replay file, and admits a new snapshot only when
at least the configured fraction of its canonical states are new relative to
the previous snapshot.

Canonicalization matches the model's side-to-move perspective. ``move_count``
is intentionally ignored because it is not part of the policy input.
"""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
import stat as stat_module
import sys
import tempfile
import threading
from time import perf_counter_ns
from typing import (
    AbstractSet,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .move_encoder import ENCODING_VERSION
from .replay import ReplayEntry, _json_loads as _replay_json_loads
from . import run_status

try:
    from ._fast_stat import stat_paths as _fast_stat_paths
    _HAS_FAST_STAT = True
except ImportError:  # pragma: no cover - exercised on unbuilt/native Windows trees
    _fast_stat_paths = None
    _HAS_FAST_STAT = False


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

# The canonical trained-state ledger is a gzip stream of 64-character hashes.
# Loading millions of them just to retain their leading 64-bit membership
# fingerprints is expensive on drvfs, so keep an optional, derived binary
# representation beside it.  A hit still hashes the full canonical source and
# validates the binary payload before it is trusted.
_LEDGER_FINGERPRINT_SIDECAR_MAGIC = b"DAMA_LEDGER_FINGERPRINTS_V1\n"
_LEDGER_FINGERPRINT_SIDECAR_VERSION = 1
_LEDGER_FINGERPRINT_SIDECAR_HEADER_LIMIT = 4096
_LEDGER_FINGERPRINT_BYTES = 8

# SHA-256 over immutable snapshot shards is I/O-bound on drvfs.  A small,
# bounded thread pool overlaps independent reads without multiplying the
# canonical-state sets that dominate this process's memory footprint.  Keep
# tiny manifests synchronous so tests and small corpora do not pay pool setup.
_MANIFEST_HASH_WORKERS = 8
_PARALLEL_MANIFEST_HASH_MIN_BYTES = 64 * 1024 * 1024
# One live manager repeatedly verifies the immutable hold-out and active
# snapshot. Retain only their fully digest-verified shard identities so an
# unchanged manifest can prove integrity with one final metadata transaction.
_MANIFEST_INTEGRITY_IDENTITY_CACHE_MAX = 2

# drvfs serves independent stat requests with high latency.  Admission reads
# 61 replay and 27 hold-out identities several times to preserve its fail-closed
# transaction boundary, so overlap those metadata requests without changing
# which fields are checked.  Small directories stay synchronous to avoid pool
# setup overhead in tests and non-production corpora.
_METADATA_STAT_WORKERS = 8
# Native C11 threads carry no Future objects and all join before return, so the
# slow drvfs batch benefits from a wider fan-out than the Python pool. Direct
# 61 plus 27 shard controls place the knee at 32 threads; the Python fallback
# remains at the measured eight-worker optimum.
_NATIVE_METADATA_STAT_WORKERS = 32
_PARALLEL_METADATA_STAT_MIN_FILES = 8
_METADATA_STAT_PROBE_FILES = 4
_PARALLEL_METADATA_STAT_MIN_PROBE_NS = 1_000_000
_METADATA_STAT_STRATEGY_CACHE_MAX = 64
_METADATA_STAT_STRATEGY_LOCK = threading.Lock()
_METADATA_STAT_PARALLEL_BY_PARENT: "OrderedDict[str, bool]" = OrderedDict()
# Probe-only switch used by mirrored performance controls. Production keeps
# the compiled batch enabled whenever it imported successfully.
_FAST_METADATA_STAT_ENABLED = True

# ``gzip.GzipFile`` text iteration pays substantial per-line overhead for the
# fixed-width canonical-key files on drvfs.  Small key files can instead take
# gzip's whole-member C path, while the cap keeps the much larger all-time
# ledger and any unexpectedly large artifact on the streaming path.
_STATE_KEYS_BULK_READ_MAX_BYTES = 64 * 1024 * 1024
# A live snapshot manager repeatedly needs exactly two immutable key sets:
# the active training snapshot and the append-only validation manifest.  Keep
# only those two decompressed members process-local.  A larger bound spends
# scarce trainer RAM on historical snapshots that admission never revisits.
_STATE_KEY_FILE_CACHE_MAX = 2


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
# Two bounds, because the entries differ by four orders of magnitude.  A
# per-file analysis retains every canonical state key of its shard (about
# 6 MB retained for a 14.9 MB, 14K-record shard), so it stays tightly bounded.  A digest or a
# contract audit is a few hundred bytes, and one admission check touches the
# whole replay window plus every held-out shard (61 + 27 = 88 identities on
# the c174k window): under the shared 64-entry bound that cyclic access
# pattern evicted every entry before its next use, so every self-play cycle
# re-hashed all 88 shards (1.33 GB through drvfs, 12.3 s of a 12.9 s
# admission check; Journal Pass 182).  Size the small caches for the window,
# the hold-out, and a snapshot verification together, with headroom.
_REPLAY_FILE_CACHE_MAX = 64
_REPLAY_DIGEST_CACHE_MAX = 1024
_REPLAY_IDENTITY_MAP_MAX = max(_REPLAY_FILE_CACHE_MAX, _REPLAY_DIGEST_CACHE_MAX)
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


@dataclass(frozen=True, slots=True)
class _FastStatResult:
    """Subset of ``os.stat_result`` consumed by corpus identity checks."""

    st_mode: int
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int


@dataclass(frozen=True)
class _ReplayFileAnalysis:
    """Exact per-file facts needed to reproduce ``analyze_replay_files``."""

    identity: tuple
    sha256: str
    records: int
    malformed_records: int
    forced_move_count: int
    state_counts: Mapping[str, int]
    # Empty for a proven uniform-cycle shard: ``uniform_generation_cycle``
    # then applies to every key in ``state_counts``.  Mixed and legacy shards
    # retain the exact per-state mapping.
    state_cycles: Mapping[str, frozenset[str]]
    uniform_generation_cycle: Optional[str]
    source_counts: Mapping[str, int]
    game_sources: Mapping[str, str]


@dataclass
class _ReplayWindowAnalysis:
    """Exact rolling aggregate for the production replay-window shape.

    ReplayWriter closes one uniquely named shard per generation cycle.  When
    every shard proves that shape, a new self-play cycle normally replaces one
    file in the rolling window.  Retaining the two aggregate counters lets the
    next admission update only those changed shards instead of merging every
    canonical state in the other 55 unchanged files again.
    """

    ordered_identities: Tuple[tuple, ...]
    analyses: Dict[tuple, _ReplayFileAnalysis]
    state_counts: Counter[str]
    state_file_counts: Counter[str]
    source_counts: Counter[str]
    records: int
    malformed_records: int
    forced_move_count: int
    cross_file_repeated_state_count: int
    cross_file_duplicate_record_count: int
    # The active snapshot key set is a cached frozenset and normally remains
    # unchanged across many rejected self-play cycles.  Retain freshness
    # against that exact immutable object so a one-shard replay rotation only
    # updates keys from the removed and added shards.  Mutable predecessor sets
    # never populate this reference and therefore keep the unrestricted full
    # difference on every call.
    freshness_reference: Optional[AbstractSet[str]]
    fresh_state_keys: Set[str]
    fresh_record_count: int
    # Validation and frozen-suite exclusions are immutable for ordinary
    # production cycles. Retain only their two exact cardinalities and update
    # them inside the existing one-shard rotation walk. Mutable references or a
    # changed predecessor deliberately invalidate this acceleration.
    exclusion_validation_reference: Optional[AbstractSet[str]]
    exclusion_external_reference: Optional[AbstractSet[str]]
    exclusion_freshness_reference: Optional[AbstractSet[str]]
    excluded_state_count: int
    fresh_excluded_state_count: int


@dataclass
class _SnapshotSplitContext:
    """Verified immutable inputs needed to materialize one train/validation split.

    Keeping validation preparation separate from training-entry materialization
    lets a caller reuse a tensor cache that is cryptographically keyed to the
    verified snapshot.  The context is intentionally process-local: the
    manifests and their files remain the durable source of truth.
    """

    manifest_path: Path
    manifest: dict
    validation_path: Path
    validation_manifest: dict
    validation_keys: Set[str]
    historically_trained: Set[int]
    max_train_entries: int


def _replay_file_identity_from_stat(
    path: Path, stat_result: os.stat_result,
) -> _ReplayFileIdentity:
    """Build the canonical cache identity from one already-completed stat."""

    resolved = os.path.abspath(str(Path(path)))
    return _ReplayFileIdentity(
        resolved_path=resolved,
        st_dev=int(stat_result.st_dev),
        st_ino=int(stat_result.st_ino),
        st_size=int(stat_result.st_size),
        st_mtime_ns=int(stat_result.st_mtime_ns),
    )


def _replay_file_identity(path: Path) -> _ReplayFileIdentity:
    """Return an identity that changes for normal in-place/replacement edits.

    The pathname component is the absolute, normalised spelling the caller
    used, not a symlink-resolved one: ``Path.resolve()`` lstat()s every path
    component and cost 1.3-4.5 ms per call on drvfs, about 200 calls per
    self-play cycle (Journal Pass 182), while the device, inode, size, and
    mtime fields are what actually guard against a stale entry.  Two
    spellings of one file would simply hold two self-consistent entries.
    """

    path = Path(path)
    return _replay_file_identity_from_stat(path, path.stat())


def _directory_entry_stat(
    entry: Any,
) -> Tuple[Any, Optional[Any], Optional[OSError]]:
    """Return one directory entry's metadata without losing its exception."""

    try:
        return entry, entry.stat(), None
    except OSError as exc:
        return entry, None, exc


def _native_directory_entry_stats(
    entries: Sequence[Any], workers: int,
) -> Optional[List[Tuple[Any, Optional[Any], Optional[OSError]]]]:
    """Run one exact native stat batch, or decline to the Python fallback."""

    if (
        not _FAST_METADATA_STAT_ENABLED
        or _fast_stat_paths is None
        or not all(isinstance(entry, os.DirEntry) for entry in entries)
    ):
        return None
    try:
        raw_results = _fast_stat_paths(
            [entry.path for entry in entries],
            min(_NATIVE_METADATA_STAT_WORKERS, len(entries)),
        )
    except Exception:
        # This extension is an acceleration only. A missing or broken build
        # must preserve the existing exact Python path rather than weakening
        # corpus verification or preventing a launch.
        return None
    if len(raw_results) != len(entries):
        return None
    results = []
    for entry, (fields, error_number) in zip(entries, raw_results):
        if error_number:
            error = OSError(
                int(error_number), os.strerror(int(error_number)), entry.path)
            results.append((entry, None, error))
        elif fields is None or len(fields) != 5:
            return None
        else:
            results.append((entry, _FastStatResult(*map(int, fields)), None))
    return results


def _parallel_directory_entry_stats(
    entries: Sequence[Any], workers: int,
) -> List[Tuple[Any, Optional[Any], Optional[OSError]]]:
    """Overlap a proven-slow metadata batch through native or Python threads."""

    native = _native_directory_entry_stats(entries, workers)
    if native is not None:
        return native
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="corpus-metadata",
    ) as pool:
        return list(pool.map(_directory_entry_stat, entries))


def _directory_entry_stats(
    entries: Sequence[Any],
) -> List[Tuple[Any, Optional[os.stat_result], Optional[OSError]]]:
    """Read independent directory-entry identities with bounded concurrency."""

    entries = list(entries)
    if len(entries) < _PARALLEL_METADATA_STAT_MIN_FILES:
        return [_directory_entry_stat(entry) for entry in entries]
    parent_key: Optional[str] = None
    try:
        parents = {
            os.path.abspath(os.path.dirname(os.fspath(entry.path)))
            for entry in entries
        }
        if len(parents) == 1:
            parent_key = parents.pop()
    except (AttributeError, TypeError):
        pass
    parallel: Optional[bool] = None
    if parent_key is not None:
        with _METADATA_STAT_STRATEGY_LOCK:
            parallel = _METADATA_STAT_PARALLEL_BY_PARENT.get(parent_key)
            if parallel is not None:
                _METADATA_STAT_PARALLEL_BY_PARENT.move_to_end(parent_key)
    if parallel is False:
        return [_directory_entry_stat(entry) for entry in entries]
    if parallel is True:
        workers = min(_METADATA_STAT_WORKERS, len(entries))
        return _parallel_directory_entry_stats(entries, workers)
    # A thread pool is a large regression on a low-latency local filesystem
    # (the server profile normally uses one), while drvfs metadata is slow
    # enough to benefit by an order of magnitude.  Time a small prefix from
    # this exact directory instead of hardcoding a platform or mount name.
    probe_count = min(_METADATA_STAT_PROBE_FILES, len(entries))
    started_ns = perf_counter_ns()
    prefix = [
        _directory_entry_stat(entry) for entry in entries[:probe_count]
    ]
    parallel = (
        perf_counter_ns() - started_ns
        >= _PARALLEL_METADATA_STAT_MIN_PROBE_NS
    )
    if parent_key is not None:
        with _METADATA_STAT_STRATEGY_LOCK:
            _METADATA_STAT_PARALLEL_BY_PARENT[parent_key] = parallel
            _METADATA_STAT_PARALLEL_BY_PARENT.move_to_end(parent_key)
            while (len(_METADATA_STAT_PARALLEL_BY_PARENT)
                   > _METADATA_STAT_STRATEGY_CACHE_MAX):
                _METADATA_STAT_PARALLEL_BY_PARENT.popitem(last=False)
    if not parallel:
        return prefix + [
            _directory_entry_stat(entry) for entry in entries[probe_count:]
        ]
    workers = min(_METADATA_STAT_WORKERS, len(entries) - probe_count)
    remainder = _parallel_directory_entry_stats(
        entries[probe_count:], workers)
    return prefix + remainder


def _cache_touch(
    cache: "OrderedDict[tuple, Any]", key: tuple, value: Any,
    bound: int = _REPLAY_FILE_CACHE_MAX,
) -> None:
    """Insert an item and enforce the process-local LRU bound."""

    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > bound:
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
        if len(_REPLAY_LATEST_IDENTITY) > _REPLAY_IDENTITY_MAP_MAX:
            active = set(_REPLAY_ANALYSIS_CACHE)
            active.update(_REPLAY_HASH_CACHE)
            active.update(key[0] for key in _REPLAY_AUDIT_CACHE)
            for stale_path, stale_identity in tuple(_REPLAY_LATEST_IDENTITY.items()):
                if len(_REPLAY_LATEST_IDENTITY) <= _REPLAY_IDENTITY_MAP_MAX:
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
    with _METADATA_STAT_STRATEGY_LOCK:
        _METADATA_STAT_PARALLEL_BY_PARENT.clear()


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


def _exclude_validation_state_keys(
    state_keys: AbstractSet[str],
    validation_keys: AbstractSet[str],
    external_validation_keys: AbstractSet[str],
) -> Set[str]:
    """Return trainable keys without materializing the full exclusion union.

    The immutable hold-out contains hundreds of thousands of keys while the
    frozen external suite contains 5,000.  Building their union duplicates the
    large set on every admission check.  Subtract the hold-out once, then remove
    the small external suite in place.  This is exactly ``S - (V | E)``.
    """

    remaining = state_keys - validation_keys
    remaining.difference_update(external_validation_keys)
    return remaining


def _validation_overlap_state_count(
    state_keys: AbstractSet[str],
    validation_keys: AbstractSet[str],
    external_validation_keys: AbstractSet[str],
) -> int:
    """Count excluded keys without copying the full trainable complement.

    Most generated candidates are below the freshness gate and never become a
    snapshot. Their exact post-exclusion cardinalities are still required for
    the decision metrics, but their hundreds-of-thousands-entry training set is
    not. Intersect the two validation sources with the candidate and merge the
    much smaller overlap so keys present in both sources are counted once.
    """

    validation_overlap = validation_keys.intersection(state_keys)
    external_only = external_validation_keys.difference(validation_keys)
    return (
        len(validation_overlap)
        + len(external_only.intersection(state_keys))
    )


def _sha256_file_uncached(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_file_sha256_for_identity(
    path: Path, identity: _ReplayFileIdentity,
) -> str:
    """Return a digest using an identity already verified by the caller."""

    path = Path(path)
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
            _cache_touch(
                _REPLAY_HASH_CACHE, identity_key, digest,
                _REPLAY_DIGEST_CACHE_MAX)
    return digest


def replay_file_sha256(path: Path) -> str:
    """Return a replay-file digest, reusing only a still-identical file."""

    path = Path(path)
    return _replay_file_sha256_for_identity(
        path, _replay_file_identity(path))


def _policy_replay_audit_result(
    records: int,
    game_sources: Mapping[str, str],
    errors: Counter[str],
    *,
    complete: bool = False,
) -> Dict[str, Any]:
    """Finalize the repaired replay-contract verdict for one file."""

    source_games = Counter(game_sources.values())
    if complete and not errors:
        # A completed self-play cycle writes exactly one replay file whose
        # trajectories are an exact 70/30 algorithm/current-model split.  A
        # fully-parsed file that misses the ratio is a partial cycle: the
        # process died mid-generation, so the trainer's in-process quarantine
        # never ran.  Reject that one incomplete file.
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


def _audit_policy_replay_entry(
    entry: Mapping[str, Any],
    allowed: Set[int],
    game_sources: Dict[str, str],
    errors: Counter[str],
) -> None:
    """Accumulate the exact repaired replay-contract checks for one row."""

    legal_moves = entry.get("legal_moves")
    if not isinstance(legal_moves, list) or not legal_moves:
        errors["missing_legal_moves"] += 1
        return
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

    for entry in _iter_entry_dicts(path):
        records += 1
        _audit_policy_replay_entry(entry, allowed, game_sources, errors)
        # No rejected file can become admissible by scanning more records.
        # Return after the first observed contract error, including the useful
        # count of records consumed and the exact error categories found.
        if errors:
            return _policy_replay_audit_result(
                records, game_sources, errors)

    return _policy_replay_audit_result(
        records, game_sources, errors, complete=True)


def _audit_policy_replay_file_for_identity(
    path: Path,
    allowed_opening_plies: Sequence[int],
    identity: _ReplayFileIdentity,
) -> Dict[str, Any]:
    """Audit a replay file using an identity already verified by the caller."""

    path = Path(path)
    identity_key = _cache_prepare_identity(identity)
    allowed_values = tuple(int(value) for value in allowed_opening_plies)
    allowed_key = tuple(sorted(set(allowed_values)))
    cache_key = (identity_key, allowed_key)
    with _REPLAY_CACHE_LOCK:
        cached = _REPLAY_AUDIT_CACHE.get(cache_key)
        if cached is not None:
            _REPLAY_AUDIT_CACHE.move_to_end(cache_key)
            return copy.deepcopy(cached)

    # The active admission path needs both the contract verdict and the exact
    # diversity analysis for every valid shard.  Build both during this one
    # JSON decode instead of parsing every cold shard again in
    # analyze_replay_files().  A rejected shard still stops at its first
    # contract error and publishes no partial analysis.
    analysis, result = _read_replay_file_analysis_and_audit(
        path, identity_key, allowed_values)
    try:
        unchanged = _replay_file_identity(path).as_key() == identity_key
    except OSError:
        unchanged = False
    if unchanged:
        with _REPLAY_CACHE_LOCK:
            _cache_touch(
                _REPLAY_AUDIT_CACHE, cache_key, copy.deepcopy(result),
                _REPLAY_DIGEST_CACHE_MAX)
            if analysis is not None and analysis.malformed_records == 0:
                _cache_touch(_REPLAY_ANALYSIS_CACHE, identity_key, analysis)
                _cache_touch(
                    _REPLAY_HASH_CACHE, identity_key, analysis.sha256,
                    _REPLAY_DIGEST_CACHE_MAX)
    return result


def audit_policy_replay_file(
    path: Path,
    allowed_opening_plies: Sequence[int],
) -> Dict[str, Any]:
    """Audit a replay file, caching only a still-identical complete result."""

    path = Path(path)
    try:
        identity = _replay_file_identity(path)
    except OSError:
        # Preserve the underlying iterator/open error for missing paths and
        # test doubles that intentionally do not exist on disk.
        return _audit_policy_replay_file_uncached(path, allowed_opening_plies)
    return _audit_policy_replay_file_for_identity(
        path, allowed_opening_plies, identity)


def _iter_entry_dicts(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = _replay_json_loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid replay JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Replay entry is not an object at {path}:{line_number}")
            yield value


def _iter_entry_dicts_with_digest(
    path: Path, digest: Any,
) -> Iterator[dict]:
    """Yield replay objects while hashing the exact bytes already being read.

    The ordinary iterator stays text based for its many legacy callers.  The
    cold analysis path also needs the shard SHA-256, so its binary iterator
    updates the digest from each large input block before decoding its complete
    lines.  Fully consuming this iterator therefore hashes every byte, including
    blank lines, original newline bytes, and an unterminated final line, without
    a second file read.
    """

    def decode_line(raw_line: bytes, line_number: int) -> Optional[dict]:
        if not raw_line.strip():
            return None
        try:
            value = _replay_json_loads(raw_line)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid replay JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Replay entry is not an object at {path}:{line_number}")
        return value

    # BufferedReader's per-line binary iterator is exceptionally slow on
    # drvfs.  Read and hash large blocks, carrying only the unfinished final
    # line into the next block.  Splitting on LF preserves normal JSONL and
    # CRLF JSONL semantics; the retained CR is valid trailing JSON whitespace.
    path = Path(path)
    chunk_bytes = 1024 * 1024
    pending = b""
    line_number = 0
    with path.open("rb", buffering=chunk_bytes) as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            lines = (pending + chunk).split(b"\n")
            pending = lines.pop()
            for raw_line in lines:
                line_number += 1
                value = decode_line(raw_line, line_number)
                if value is not None:
                    yield value
        if pending:
            line_number += 1
            value = decode_line(pending, line_number)
            if value is not None:
                yield value


def _scan_replay_file_analysis(
    path: Path,
    identity_key: tuple,
    audit_opening_plies: Optional[Sequence[int]] = None,
) -> Tuple[Optional[_ReplayFileAnalysis], Optional[Dict[str, Any]]]:
    """Build analysis and, when requested, the policy audit in one decode.

    Contract-invalid files preserve the audit's fail-fast behavior and return
    no analysis, so a prefix can never enter the reusable analysis cache.
    """

    path = Path(path)
    state_counts: Counter[str] = Counter()
    # ReplayWriter normally closes one shard per cycle.  Keep that common case
    # compact while scanning: ``None`` means every valid record seen so far had
    # the same cycle id, so ``state_counts`` itself proves which states observed
    # it.  The first missing provenance creates only a set of observed keys;
    # the full per-key map is materialized only if a second cycle actually
    # appears in the same shard.
    state_cycles: Optional[Dict[str, Set[str]]] = None
    single_cycle_id: Optional[str] = None
    single_cycle_state_keys: Optional[Set[str]] = None
    source_counts: Counter[str] = Counter()
    game_sources: Dict[str, str] = {}
    forced = 0
    malformed = 0
    total = 0
    audit_enabled = audit_opening_plies is not None
    audit_allowed = {
        int(value) for value in (audit_opening_plies or ())}
    audit_errors: Counter[str] = Counter()
    audit_game_sources: Dict[str, str] = {}
    audit_records = 0

    # Hash the exact bytes consumed by semantic parsing.  The result is
    # published only after both complete and the file identity is checked again
    # by the caller.  Contract-invalid files return before a complete digest is
    # available and publish neither a hash nor a partial analysis.
    file_digest = hashlib.sha256()
    for entry in _iter_entry_dicts_with_digest(path, file_digest):
        if audit_enabled:
            audit_records += 1
            _audit_policy_replay_entry(
                entry, audit_allowed, audit_game_sources, audit_errors)
            if audit_errors:
                return None, _policy_replay_audit_result(
                    audit_records, audit_game_sources, audit_errors)
        try:
            key = canonical_state_key(entry["state"])
            legal_moves = entry["legal_moves"]
        except (KeyError, TypeError, ValueError):
            malformed += 1
            continue
        total += 1
        game_id = entry.get("game_id")
        cycle_match = re.match(r"^cycle-([^ -]+)-", str(game_id or ""))
        if cycle_match:
            cycle_id = cycle_match.group(1)
            if state_cycles is not None:
                state_cycles[key].add(cycle_id)
            elif single_cycle_id is None:
                single_cycle_id = cycle_id
                if single_cycle_state_keys is not None:
                    single_cycle_state_keys.add(key)
            elif cycle_id == single_cycle_id:
                if single_cycle_state_keys is not None:
                    single_cycle_state_keys.add(key)
            else:
                state_cycles = defaultdict(set)
                previously_observed = (
                    state_counts.keys()
                    if single_cycle_state_keys is None
                    else single_cycle_state_keys
                )
                for observed_key in previously_observed:
                    state_cycles[observed_key].add(single_cycle_id)
                state_cycles[key].add(cycle_id)
                single_cycle_state_keys = None
        elif state_cycles is None and single_cycle_state_keys is None:
            # All earlier valid records carried ``single_cycle_id`` (or this is
            # the first valid record).  Snapshot their keys only when missing
            # provenance makes that proof insufficient.
            single_cycle_state_keys = set(state_counts)
        state_counts[key] += 1
        if len(legal_moves) == 1:
            forced += 1
        source_counts[str(entry.get("trajectory_source", "legacy"))] += 1
        source = entry.get("trajectory_source")
        if isinstance(game_id, str) and game_id and isinstance(source, str):
            game_sources[game_id] = source

    file_hash = file_digest.hexdigest()
    uniform_generation_cycle = None
    if state_cycles is not None:
        frozen_state_cycles = {
            key: frozenset(value) for key, value in state_cycles.items()}
    elif (
        single_cycle_id is not None
        and (
            single_cycle_state_keys is None
            or len(single_cycle_state_keys) == len(state_counts)
        )
    ):
        # Every canonical state observed the shard's one cycle.  The analyzer
        # expands this single id only if an ambiguous multi-file fallback needs
        # it, avoiding one set plus one frozenset per state in the live cache.
        uniform_generation_cycle = single_cycle_id
        frozen_state_cycles = {}
    elif single_cycle_id is not None:
        shared_cycle = frozenset((single_cycle_id,))
        frozen_state_cycles = {
            key: shared_cycle for key in (single_cycle_state_keys or ())}
    else:
        frozen_state_cycles = {}
    analysis = _ReplayFileAnalysis(
        identity=identity_key,
        sha256=file_hash,
        records=total,
        malformed_records=malformed,
        forced_move_count=forced,
        state_counts=dict(state_counts),
        state_cycles=frozen_state_cycles,
        # ReplayWriter closes one shard per completed generation cycle.  Record
        # that fact only when the contents prove it: every valid canonical state
        # must carry the same one cycle id.  Mixed or legacy shards store None
        # and retain the general merge path.
        uniform_generation_cycle=uniform_generation_cycle,
        source_counts=dict(source_counts),
        game_sources=dict(game_sources),
    )
    audit = None
    if audit_enabled:
        audit = _policy_replay_audit_result(
            audit_records, audit_game_sources, audit_errors, complete=True)
        if not audit["valid"]:
            return None, audit
    return analysis, audit


def _read_replay_file_analysis(
    path: Path, identity_key: tuple,
) -> _ReplayFileAnalysis:
    """Build exact per-file facts without publishing a partial cache entry."""

    analysis, _audit = _scan_replay_file_analysis(path, identity_key)
    assert analysis is not None
    return analysis


def _read_replay_file_analysis_and_audit(
    path: Path,
    identity_key: tuple,
    allowed_opening_plies: Sequence[int],
) -> Tuple[Optional[_ReplayFileAnalysis], Dict[str, Any]]:
    """Build a fail-closed policy audit and complete analysis together."""

    analysis, audit = _scan_replay_file_analysis(
        path, identity_key, allowed_opening_plies)
    assert audit is not None
    return analysis, audit


def _cached_replay_file_analysis_for_identity(
    path: Path, identity: _ReplayFileIdentity,
) -> _ReplayFileAnalysis:
    """Load exact file facts using an identity already verified by the caller."""

    path = Path(path)
    identity_key = _cache_prepare_identity(identity)
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
            _cache_touch(
                _REPLAY_HASH_CACHE, identity_key, analysis.sha256,
                _REPLAY_DIGEST_CACHE_MAX)
    return analysis


def _cached_replay_file_analysis(path: Path) -> _ReplayFileAnalysis:
    """Load exact file facts from the bounded cache or build them once."""

    path = Path(path)
    return _cached_replay_file_analysis_for_identity(
        path, _replay_file_identity(path))


def _state_set_digest(state_keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_keys):
        digest.update(key.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def analyze_replay_files(
    files: Sequence[Path],
    previous_state_keys: Optional[Set[str]] = None,
    *,
    include_state_digest: bool = True,
    _file_identities: Optional[Mapping[Path, _ReplayFileIdentity]] = None,
) -> Tuple[Dict[str, Any], Set[str]]:
    """Measure exact replay diversity and freshness across a file set.

    Time complexity is O(r log u) because the final fingerprint sorts ``u``
    unique keys. Space complexity is O(u + r_source), where ``r_source`` is
    the bounded set of source labels.

    ``include_state_digest=False`` reports ``state_set_sha256`` as ``None``.
    It exists for the one caller that fingerprints a different key set (the
    post-deduplication training set) and overwrites the field anyway:
    sorting and hashing the ~636K pre-dedup keys of the c174k window costs
    about 0.4 s on every self-play cycle for a value nobody reads
    (Journal Pass 182).  Every other caller keeps the digest.  The private
    identity map lets one fail-closed admission transaction reuse its initial
    metadata snapshot; ordinary callers still stat each source themselves.
    """

    previous = previous_state_keys or set()
    state_counts: Counter[str] = Counter()
    unique_keys: Set[str] = set()
    # Production replay basenames are unique.  In that common case, set
    # intersection finds the relatively small cross-file overlap in C and a
    # repeated-key set replaces a 636K-entry per-state file counter.  The
    # general path retains exact historical semantics for callers that supply
    # two different paths with the same basename: those paths count as one
    # logical file for the cross-file metrics.
    unique_file_names = len({Path(path).name for path in files}) == len(files)
    repeated_file_keys: Set[str] = set()
    state_file_counts: Counter[str] = Counter()
    name_states: Dict[str, List[Mapping[str, int]]] = {}
    cycle_analyses: List[_ReplayFileAnalysis] = []
    uniform_cycle_ids: List[str] = []
    all_uniform_cycles = True
    source_counts: Counter[str] = Counter()
    game_sources: Dict[str, str] = {}
    forced = 0
    malformed = 0
    total = 0

    for path in files:
        path = Path(path)
        identity = (
            _file_identities.get(path)
            if _file_identities is not None else None)
        analysis = (
            _cached_replay_file_analysis_for_identity(path, identity)
            if identity is not None
            else _cached_replay_file_analysis(path))
        total += analysis.records
        malformed += analysis.malformed_records
        forced += analysis.forced_move_count
        source_counts.update(analysis.source_counts)
        # Updating in caller-provided file order preserves the original
        # last-write-wins behavior when a game ID appears in multiple files.
        game_sources.update(analysis.game_sources)
        file_states = analysis.state_counts
        state_counts.update(file_states)
        if unique_file_names:
            repeated_file_keys.update(unique_keys.intersection(file_states))
            unique_keys.update(file_states)
        else:
            unique_keys.update(file_states)
            counted = name_states.get(path.name)
            if counted is None:
                name_states[path.name] = [file_states]
                state_file_counts.update(file_states.keys())
            else:
                state_file_counts.update(
                    key for key in file_states
                    if not any(key in earlier for earlier in counted)
                )
                counted.append(file_states)
        cycle_analyses.append(analysis)
        if analysis.uniform_generation_cycle is None:
            all_uniform_cycles = False
        else:
            uniform_cycle_ids.append(analysis.uniform_generation_cycle)

    new_unique = unique_keys.difference(previous)
    # ``new_unique`` already embodies the membership test against ``previous``;
    # iterating only that normally-small set avoids re-testing every standing
    # state to calculate the fresh-record rate.
    fresh_records = sum(state_counts[key] for key in new_unique)
    if unique_file_names:
        cross_file_states = len(repeated_file_keys)
        cross_file_duplicate_records = sum(
            state_counts[state_key] - 1 for state_key in repeated_file_keys)
    else:
        cross_file_states = sum(
            1 for names in state_file_counts.values() if names > 1)
        cross_file_duplicate_records = sum(
            state_counts[state_key] - 1
            for state_key, names in state_file_counts.items() if names > 1)
    cross_file_unique_states = len(unique_keys) - cross_file_states

    if (unique_file_names
            and all_uniform_cycles
            and len(set(uniform_cycle_ids)) == len(uniform_cycle_ids)):
        # Each state is observed in its shard's sole cycle, and no two shards
        # share a cycle.  A state is therefore a cross-cycle repeat exactly when
        # it is already known to repeat across files.  This is the live corpus
        # shape and avoids rebuilding a 636K-entry cycle dictionary per check.
        cycle_observed_state_count = len(unique_keys)
        cross_cycle_states = len(repeated_file_keys)
    else:
        state_cycles: Dict[str, frozenset[str]] = {}
        for analysis in cycle_analyses:
            if analysis.uniform_generation_cycle is not None:
                shared_cycle = frozenset((analysis.uniform_generation_cycle,))
                file_cycles = (
                    (key, shared_cycle) for key in analysis.state_counts)
            else:
                file_cycles = analysis.state_cycles.items()
            for key, cycles in file_cycles:
                known = state_cycles.get(key)
                if known is None:
                    state_cycles[key] = cycles
                elif known is not cycles and not cycles <= known:
                    state_cycles[key] = known | cycles
        cycle_observed_state_count = 0
        cross_cycle_states = 0
        for cycles in state_cycles.values():
            if cycles:
                cycle_observed_state_count += 1
                if len(cycles) > 1:
                    cross_cycle_states += 1
    cross_cycle_unique_states = cycle_observed_state_count - cross_cycle_states
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
        "cross_cycle_observed_state_count": cycle_observed_state_count,
        "cross_cycle_repeated_state_count": cross_cycle_states,
        "cross_cycle_unique_state_count": cross_cycle_unique_states,
        "cross_cycle_unique_state_rate": (
            cross_cycle_unique_states / cycle_observed_state_count
            if cycle_observed_state_count else 0.0
        ),
        "new_unique_state_count": len(new_unique),
        "fresh_unique_state_rate": (len(new_unique) / len(unique_keys)) if unique_keys else 0.0,
        "fresh_record_rate": (fresh_records / total) if total else 0.0,
        "source_counts": dict(sorted(source_counts.items())),
        "source_game_counts": dict(sorted(Counter(game_sources.values()).items())),
        "state_set_sha256": (
            _state_set_digest(unique_keys) if include_state_digest else None),
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
    # Snapshot key files are immutable single-member gzip streams written by
    # ``_write_state_keys``.  Reading the compressed member at once lets zlib
    # decompress it without hundreds of thousands of TextIO iteration calls.
    # Check the opened file's size and cap the read itself so a concurrent
    # in-place growth cannot silently bypass the memory guard.
    with path.open("rb") as raw_handle:
        if os.fstat(raw_handle.fileno()).st_size <= _STATE_KEYS_BULK_READ_MAX_BYTES:
            compressed = raw_handle.read(_STATE_KEYS_BULK_READ_MAX_BYTES + 1)
            if len(compressed) <= _STATE_KEYS_BULK_READ_MAX_BYTES:
                text = gzip.decompress(compressed).decode("ascii")
                return {
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                }

    # Preserve bounded-memory behavior for the multi-million-key trained
    # ledger and for artifacts outside the normal snapshot-size envelope.
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
        # Snapshot retention predates shard reuse, so it also bounds legacy
        # copy-only snapshots.  At steady state, an unbounded history can
        # exhaust the volume even when new admissions hardlink unchanged
        # shards.  Keep the newest N admissions; 0 preserves every snapshot.
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
        self.external_validation_state_keys: AbstractSet[str] = frozenset()
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
        # Immutable manifest key files otherwise get decompressed, parsed and
        # fingerprinted again after every self-play cycle.  Entries are keyed
        # by the same stat identity as the replay caches, so a replacement or
        # ordinary in-place edit misses naturally.  The retained frozenset
        # cannot be corrupted by a caller.
        self._state_key_file_cache: (
            "OrderedDict[tuple, Tuple[frozenset[str], str]]"
        ) = OrderedDict()
        # A cache entry is published only after every declared shard digest and
        # the closing identity transaction both pass. On the next load, one
        # batched identity comparison proves that the same immutable bytes are
        # still present; any mismatch falls back to the complete verifier.
        self._manifest_integrity_identity_cache: (
            "OrderedDict[tuple, Dict[Path, _ReplayFileIdentity]]"
        ) = OrderedDict()
        # One manager follows one live rolling window.  Under ReplayWriter's
        # proven one-file-per-cycle shape, retain its exact aggregate so the
        # next admission merges only added/removed shards.  Any mixed-cycle,
        # duplicate-name, or duplicate-cycle window bypasses and clears it.
        self._replay_window_analysis: Optional[_ReplayWindowAnalysis] = None
        # Set only after the canonical gzip ledger has been read and verified.
        # Consumers use this as a source identity for derived caches, never as
        # a substitute for the ledger's own verification.
        self._trained_ledger_source_sha256: Optional[str] = None

    def set_external_validation_state_keys(self, state_keys: Iterable[str]) -> None:
        """Exclude a frozen external validation suite from every train snapshot."""
        self.external_validation_state_keys = frozenset(
            str(key) for key in state_keys)

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

    def _replay_files_with_identities(
        self,
    ) -> Tuple[List[Path], Dict[Path, _ReplayFileIdentity]]:
        """Scan the replay window once and retain each file's exact identity.

        ``Path.glob`` followed by ``is_file`` and a sorting ``stat`` performed
        two metadata round trips per shard on drvfs.  ``DirEntry`` reuses the
        directory scan's metadata, and the returned identities can serve the
        audit, digest, and analysis phases of one admission transaction.  The
        caller must recheck them before publishing any decision.
        """

        records: List[Tuple[int, str, Path, _ReplayFileIdentity]] = []
        try:
            entries = os.scandir(self.replay_dir)
        except FileNotFoundError:
            return [], {}
        candidates = []
        with entries:
            for entry in entries:
                name = entry.name
                if not name.startswith("replay_") or not name.endswith(".jsonl"):
                    continue
                candidates.append(entry)
        for entry, stat_result, error in _directory_entry_stats(candidates):
            if (error is not None or stat_result is None
                    or not stat_module.S_ISREG(stat_result.st_mode)):
                continue
            name = entry.name
            path = Path(entry.path)
            identity = _replay_file_identity_from_stat(path, stat_result)
            records.append((identity.st_mtime_ns, name, path, identity))
        records.sort(key=lambda record: (record[0], record[1]))
        files = [record[2] for record in records]
        return files, {record[2]: record[3] for record in records}

    def replay_files(self) -> List[Path]:
        files, _identities = self._replay_files_with_identities()
        return files

    def _eligible_replay_files_with_identities(
        self,
    ) -> Tuple[List[Path], Dict[str, dict], Dict[Path, _ReplayFileIdentity]]:
        """Return contract-valid files plus their initial stat identities."""

        files, identities = self._replay_files_with_identities()
        if not self.enforce_policy_contract:
            return files, {}, identities
        eligible = []
        rejected = {}
        for path in files:
            audit = _audit_policy_replay_file_for_identity(
                path, self.allowed_opening_plies, identities[path])
            if audit["valid"]:
                eligible.append(path)
            else:
                rejected[path.name] = audit
        return eligible, rejected, identities

    def eligible_replay_files(self) -> Tuple[List[Path], Dict[str, dict]]:
        """Return repaired-contract files and rejection diagnostics."""
        eligible, rejected, _identities = (
            self._eligible_replay_files_with_identities())
        return eligible, rejected

    @staticmethod
    def _verify_replay_file_identities(
        identities: Mapping[Path, _ReplayFileIdentity],
    ) -> None:
        """Fail closed if any shard changed during an admission transaction.

        Replay shards normally share one directory.  A separate ``Path.stat``
        for each file turns the mandatory final transaction check into dozens
        of drvfs metadata round trips.  Scan each parent once and reuse the
        ``DirEntry`` metadata, while retaining the exact path, device, inode,
        size, and nanosecond-mtime comparison for every observed shard.
        """

        by_parent: Dict[Path, Dict[str, Tuple[Path, _ReplayFileIdentity]]] = {}
        for observed_path, expected in identities.items():
            path = Path(observed_path)
            by_parent.setdefault(path.parent, {})[path.name] = (path, expected)

        for parent, expected_by_name in by_parent.items():
            try:
                entries = os.scandir(parent)
            except OSError as exc:
                path = next(iter(expected_by_name.values()))[0]
                raise RuntimeError(
                    f"Replay file disappeared during corpus analysis: {path}"
                ) from exc
            remaining = dict(expected_by_name)
            matched = []
            with entries:
                for entry in entries:
                    observed = remaining.pop(entry.name, None)
                    if observed is None:
                        continue
                    path, expected = observed
                    matched.append((entry, path, expected))
            if remaining:
                path = next(iter(remaining.values()))[0]
                raise RuntimeError(
                    f"Replay file disappeared during corpus analysis: {path}"
                )
            stats = _directory_entry_stats(
                [entry for entry, _path, _expected in matched])
            for (_entry, path, expected), (_, stat_result, error) in zip(
                matched, stats,
            ):
                if error is not None or stat_result is None:
                    raise RuntimeError(
                        "Replay file disappeared during corpus analysis: "
                        f"{path}"
                    ) from error
                actual = _replay_file_identity_from_stat(path, stat_result)
                if actual != expected:
                    raise RuntimeError(
                        f"Replay file changed during corpus analysis: {path}"
                    )

    @staticmethod
    def _build_replay_window_analysis(
        analyses: Sequence[_ReplayFileAnalysis],
    ) -> _ReplayWindowAnalysis:
        """Build the exact aggregate once using Counter's bulk update path."""

        state_counts: Counter[str] = Counter()
        state_file_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        records = 0
        malformed = 0
        forced = 0
        for analysis in analyses:
            state_counts.update(analysis.state_counts)
            state_file_counts.update(analysis.state_counts.keys())
            source_counts.update(analysis.source_counts)
            records += analysis.records
            malformed += analysis.malformed_records
            forced += analysis.forced_move_count
        repeated = {
            key for key, file_count in state_file_counts.items()
            if file_count > 1
        }
        return _ReplayWindowAnalysis(
            ordered_identities=tuple(analysis.identity for analysis in analyses),
            analyses={analysis.identity: analysis for analysis in analyses},
            state_counts=state_counts,
            state_file_counts=state_file_counts,
            source_counts=source_counts,
            records=records,
            malformed_records=malformed,
            forced_move_count=forced,
            cross_file_repeated_state_count=len(repeated),
            cross_file_duplicate_record_count=sum(
                state_counts[key] - 1 for key in repeated),
            freshness_reference=None,
            fresh_state_keys=set(),
            fresh_record_count=0,
            exclusion_validation_reference=None,
            exclusion_external_reference=None,
            exclusion_freshness_reference=None,
            excluded_state_count=0,
            fresh_excluded_state_count=0,
        )

    @staticmethod
    def _update_replay_window_analysis(
        window: _ReplayWindowAnalysis,
        analysis: _ReplayFileAnalysis,
        direction: int,
    ) -> None:
        """Add or remove one shard while retaining exact cross-file metrics."""

        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        for key, file_records in analysis.state_counts.items():
            old_files = window.state_file_counts.get(key, 0)
            old_records = window.state_counts.get(key, 0)
            new_files = old_files + direction
            new_records = old_records + direction * file_records
            if new_files < 0 or new_records < 0:
                raise RuntimeError("Rolling replay aggregate underflow")

            freshness_reference = window.freshness_reference
            if freshness_reference is not None and key not in freshness_reference:
                window.fresh_record_count += direction * file_records
                if window.fresh_record_count < 0:
                    raise RuntimeError(
                        "Rolling replay fresh-record aggregate underflow")
                if old_files == 0 and new_files > 0:
                    window.fresh_state_keys.add(key)
                elif old_files > 0 and new_files == 0:
                    window.fresh_state_keys.discard(key)

            validation_reference = window.exclusion_validation_reference
            external_reference = window.exclusion_external_reference
            if (
                validation_reference is not None
                and external_reference is not None
                and (key in validation_reference or key in external_reference)
            ):
                unique_delta = 0
                if old_files == 0 and new_files > 0:
                    unique_delta = 1
                elif old_files > 0 and new_files == 0:
                    unique_delta = -1
                window.excluded_state_count += unique_delta
                if (
                    unique_delta
                    and freshness_reference is not None
                    and window.exclusion_freshness_reference
                    is freshness_reference
                    and key not in freshness_reference
                ):
                    window.fresh_excluded_state_count += unique_delta
                if (
                    window.excluded_state_count < 0
                    or window.fresh_excluded_state_count < 0
                ):
                    raise RuntimeError(
                        "Rolling replay validation-overlap aggregate underflow")

            old_duplicate_records = (
                old_records - 1 if old_files > 1 else 0)
            new_duplicate_records = (
                new_records - 1 if new_files > 1 else 0)
            window.cross_file_duplicate_record_count += (
                new_duplicate_records - old_duplicate_records)
            if old_files > 1:
                window.cross_file_repeated_state_count -= 1
            if new_files > 1:
                window.cross_file_repeated_state_count += 1

            if new_files:
                window.state_file_counts[key] = new_files
                window.state_counts[key] = new_records
            else:
                if new_records:
                    raise RuntimeError(
                        "Rolling replay aggregate lost its final file before "
                        "its final record")
                window.state_file_counts.pop(key, None)
                window.state_counts.pop(key, None)

        for source, count in analysis.source_counts.items():
            updated = window.source_counts.get(source, 0) + direction * count
            if updated < 0:
                raise RuntimeError("Rolling replay source aggregate underflow")
            if updated:
                window.source_counts[source] = updated
            else:
                window.source_counts.pop(source, None)
        window.records += direction * analysis.records
        window.malformed_records += direction * analysis.malformed_records
        window.forced_move_count += direction * analysis.forced_move_count

    @staticmethod
    def _refresh_replay_window_freshness(
        window: _ReplayWindowAnalysis,
        previous_state_keys: AbstractSet[str],
    ) -> None:
        """Rebuild exact freshness when the immutable predecessor changes.

        Production snapshot keys come from ``_cached_state_key_file`` as a
        frozenset, so object identity is a safe O(1) cache key.  A mutable set
        could change without its identity changing; deliberately leave the
        reference unset for that input and rebuild on every call.
        """

        fresh_state_keys = window.state_counts.keys() - previous_state_keys
        window.fresh_state_keys = fresh_state_keys
        window.fresh_record_count = sum(
            window.state_counts[key] for key in fresh_state_keys)
        window.freshness_reference = (
            previous_state_keys
            if isinstance(previous_state_keys, frozenset)
            else None
        )
        # The fresh exclusion count was derived against the old predecessor.
        # The next overlap request rebuilds it once against this exact set.
        window.exclusion_freshness_reference = None

    @staticmethod
    def _refresh_replay_window_exclusion_counts(
        window: _ReplayWindowAnalysis,
        validation_keys: AbstractSet[str],
        external_validation_keys: AbstractSet[str],
    ) -> Tuple[int, int]:
        """Rebuild exact excluded-state counts and cache only immutable inputs."""

        excluded = _validation_overlap_state_count(
            window.state_counts.keys(),
            validation_keys,
            external_validation_keys,
        )
        fresh_excluded = _validation_overlap_state_count(
            window.fresh_state_keys,
            validation_keys,
            external_validation_keys,
        )
        window.excluded_state_count = excluded
        window.fresh_excluded_state_count = fresh_excluded
        if (
            isinstance(validation_keys, frozenset)
            and isinstance(external_validation_keys, frozenset)
            and window.freshness_reference is not None
        ):
            window.exclusion_validation_reference = validation_keys
            window.exclusion_external_reference = external_validation_keys
            window.exclusion_freshness_reference = window.freshness_reference
        else:
            # A mutable set can change without changing identity. It receives
            # exact counts for this call but can never drive incremental reuse.
            window.exclusion_validation_reference = None
            window.exclusion_external_reference = None
            window.exclusion_freshness_reference = None
        return excluded, fresh_excluded

    def _rolling_validation_overlap_counts(
        self,
        validation_keys: AbstractSet[str],
        external_validation_keys: AbstractSet[str],
    ) -> Optional[Tuple[int, int]]:
        """Return overlap counts maintained by the proven rolling-window path."""

        window = self._replay_window_analysis
        if window is None:
            return None
        if (
            window.exclusion_validation_reference is validation_keys
            and window.exclusion_external_reference is external_validation_keys
            and window.exclusion_freshness_reference
            is window.freshness_reference
            and window.freshness_reference is not None
        ):
            return (
                window.excluded_state_count,
                window.fresh_excluded_state_count,
            )
        return self._refresh_replay_window_exclusion_counts(
            window, validation_keys, external_validation_keys)

    def _analyze_replay_window(
        self,
        files: Sequence[Path],
        previous_state_keys: AbstractSet[str],
        identities: Mapping[Path, _ReplayFileIdentity],
    ) -> Tuple[Dict[str, Any], AbstractSet[str], Set[str]]:
        """Incrementally analyze ReplayWriter's proven rolling-window shape.

        The general analyzer remains the oracle and fallback.  This path is
        entered only when basenames are unique, every shard proves one cycle,
        and cycle ids are unique across the window.  Those facts make
        cross-cycle repetition identical to cross-file repetition.  Any
        ambiguous or legacy input clears the rolling aggregate and runs the
        unrestricted implementation.  The third result is the exact
        pre-validation fresh-state set already needed for replay metrics, so
        the admission caller can exclude held-out states from that small set
        instead of repeating a full difference over the training set.
        """

        paths = [Path(path) for path in files]
        unique_names = len({path.name for path in paths}) == len(paths)
        analyses = [
            _cached_replay_file_analysis_for_identity(path, identities[path])
            for path in paths
        ]
        cycle_ids = [analysis.uniform_generation_cycle for analysis in analyses]
        exact_rolling_shape = (
            bool(analyses)
            and unique_names
            and all(cycle_id is not None for cycle_id in cycle_ids)
            and len(set(cycle_ids)) == len(cycle_ids)
        )
        if not exact_rolling_shape:
            self._replay_window_analysis = None
            metrics, unique_keys = analyze_replay_files(
                paths,
                previous_state_keys,
                include_state_digest=False,
                _file_identities=identities,
            )
            return (
                metrics,
                unique_keys,
                unique_keys.difference(previous_state_keys),
            )

        ordered_identities = tuple(analysis.identity for analysis in analyses)
        current = {analysis.identity: analysis for analysis in analyses}
        window = self._replay_window_analysis
        if window is None:
            window = self._build_replay_window_analysis(analyses)
            self._replay_window_analysis = window
        else:
            # Invalidate before applying a rotation if the predecessor changed.
            # The post-update rebuild below then derives freshness from the
            # completed window rather than incrementally mutating stale facts.
            if window.freshness_reference is not previous_state_keys:
                window.freshness_reference = None
                window.exclusion_freshness_reference = None
            old_ids = set(window.analyses)
            new_ids = set(current)
            removed_ids = old_ids.difference(new_ids)
            added_ids = new_ids.difference(old_ids)
            # Bulk Counter updates are faster for a wholesale replacement;
            # the incremental path is for the ordinary one-shard rotation.
            if len(removed_ids) + len(added_ids) > max(8, len(analyses) // 2):
                window = self._build_replay_window_analysis(analyses)
                self._replay_window_analysis = window
            else:
                for identity in removed_ids:
                    self._update_replay_window_analysis(
                        window, window.analyses[identity], -1)
                for identity in added_ids:
                    self._update_replay_window_analysis(
                        window, current[identity], 1)
                window.ordered_identities = ordered_identities
                window.analyses = current

        # The Counter's key view is set-like and remains valid until the next
        # call.  The admission caller immediately derives its own train set, so
        # copying all ~636K keys here would add allocation without ownership.
        unique_keys = window.state_counts.keys()
        if window.freshness_reference is not previous_state_keys:
            self._refresh_replay_window_freshness(
                window, previous_state_keys)
        new_unique = window.fresh_state_keys
        fresh_records = window.fresh_record_count
        unique_count = len(unique_keys)
        cross_file_states = window.cross_file_repeated_state_count
        cross_file_unique_states = unique_count - cross_file_states
        game_sources: Dict[str, str] = {}
        for identity in ordered_identities:
            game_sources.update(current[identity].game_sources)
        metrics = {
            "records": window.records,
            "malformed_records": window.malformed_records,
            "unique_state_count": unique_count,
            "unique_state_rate": (
                unique_count / window.records if window.records else 0.0),
            "forced_move_count": window.forced_move_count,
            "forced_move_rate": (
                window.forced_move_count / window.records
                if window.records else 0.0),
            "cross_file_repeated_state_count": cross_file_states,
            "cross_file_unique_state_count": cross_file_unique_states,
            "cross_file_duplicate_record_count": (
                window.cross_file_duplicate_record_count),
            "cross_file_unique_state_rate": (
                cross_file_unique_states / unique_count if unique_count else 0.0),
            "cross_cycle_observed_state_count": unique_count,
            "cross_cycle_repeated_state_count": cross_file_states,
            "cross_cycle_unique_state_count": cross_file_unique_states,
            "cross_cycle_unique_state_rate": (
                cross_file_unique_states / unique_count if unique_count else 0.0),
            "new_unique_state_count": len(new_unique),
            "fresh_unique_state_rate": (
                len(new_unique) / unique_count if unique_count else 0.0),
            "fresh_record_rate": (
                fresh_records / window.records if window.records else 0.0),
            "source_counts": dict(sorted(window.source_counts.items())),
            "source_game_counts": dict(sorted(
                Counter(game_sources.values()).items())),
            "state_set_sha256": None,
        }
        return metrics, unique_keys, new_unique

    def current_manifest_path(self) -> Optional[Path]:
        try:
            relative = self.current_pointer.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not relative:
            return None
        path = self.snapshot_root / _read_relpath(relative)
        return path if path.is_file() else None

    def _load_manifest(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_current_manifest(self) -> Tuple[Optional[Path], Optional[dict]]:
        """Resolve CURRENT and load its regular-file target in one open.

        ``current_manifest_path()`` must remain a path-only public lookup, so it
        uses ``is_file()`` before returning. Admission immediately opened that
        path again to parse it, paying a redundant metadata round trip on
        drvfs. Opening the target and checking the opened descriptor preserves
        the missing and non-regular target semantics without a separate path
        stat or a path-check/open race.
        """

        try:
            relative = self.current_pointer.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None, None
        if not relative:
            return None, None
        path = self.snapshot_root / _read_relpath(relative)
        try:
            with path.open("r", encoding="utf-8") as handle:
                if not stat_module.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    return None, None
                return path, json.load(handle)
        except (FileNotFoundError, IsADirectoryError):
            return None, None

    def _cached_state_key_file(
        self, path: Path,
    ) -> Tuple[frozenset[str], str]:
        """Return immutable canonical keys and their digest for one file.

        The file remains the source of truth.  Publication waits for an
        unchanged post-read identity, while the two-entry LRU merely avoids
        rebuilding the same verified snapshot and hold-out sets on every
        admission check.
        """

        path = Path(path)
        identity_key = _replay_file_identity(path).as_key()
        cached = self._state_key_file_cache.get(identity_key)
        if cached is not None:
            self._state_key_file_cache.move_to_end(identity_key)
            return cached

        keys = frozenset(_read_state_keys(path))
        result = (keys, _state_set_digest(keys))
        try:
            unchanged = _replay_file_identity(path).as_key() == identity_key
        except OSError:
            unchanged = False
        if unchanged:
            # Remove an older generation of this pathname immediately instead
            # of retaining it until the global size bound happens to evict it.
            resolved_path = identity_key[0]
            for previous in tuple(self._state_key_file_cache):
                if previous[0] == resolved_path and previous != identity_key:
                    self._state_key_file_cache.pop(previous, None)
            _cache_touch(
                self._state_key_file_cache,
                identity_key,
                result,
                _STATE_KEY_FILE_CACHE_MAX,
            )
        return result

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

    def _verify_manifest_state_keys(
        self, manifest_path: Path, manifest: Mapping[str, Any],
    ) -> Set[str]:
        """Verify and return one manifest's canonical-state key member."""

        state_keys_name = manifest.get("state_keys_file")
        if not isinstance(state_keys_name, str) or not state_keys_name:
            raise RuntimeError(
                f"Corpus manifest has no canonical-state key file: {manifest_path}"
            )
        state_keys_path = manifest_path.parent / state_keys_name
        try:
            state_keys, actual_state_digest = self._cached_state_key_file(
                state_keys_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Corpus canonical-state key file is missing: {state_keys_path}"
            ) from exc
        expected_state_digest = manifest.get("metrics", {}).get(
            "state_set_sha256"
        )
        if expected_state_digest != actual_state_digest:
            raise RuntimeError(
                f"Corpus canonical-state fingerprint is invalid: {manifest_path}"
            )
        return state_keys

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

        # The manifest is immutable JSON. A semantic content digest plus its
        # path identifies the exact declaration that earned the cached shard
        # identities. Recheck those identities once; a changed file takes the
        # full digest path below, while an unchanged file cannot need a second
        # scan around cache-only digest lookups.
        manifest_cache_key = (
            os.path.abspath(str(manifest_path)),
            expected_kind,
            hashlib.sha256(json.dumps(
                manifest, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).digest(),
        )
        cached_identities = self._manifest_integrity_identity_cache.get(
            manifest_cache_key)
        if cached_identities is not None:
            try:
                self._verify_replay_file_identities(cached_identities)
            except RuntimeError:
                self._manifest_integrity_identity_cache.pop(
                    manifest_cache_key, None)
            else:
                self._manifest_integrity_identity_cache.move_to_end(
                    manifest_cache_key)
                return self._verify_manifest_state_keys(
                    manifest_path, manifest)

        # ``record["path"]`` is read through ``_read_relpath`` for the same
        # reason the CURRENT pointer is: a snapshot written by the native-Windows
        # launcher stores ``files\\replay_*.jsonl``.  On WSL that is one filename
        # containing a backslash, so every stored shard "fails integrity
        # verification" while sitting untouched on disk right next to the manifest.
        parsed_file_checks: List[Tuple[Path, int, str]] = []
        paths_by_parent: Dict[Path, Dict[str, Path]] = {}
        total_size = 0
        for record in manifest.get("files", []):
            try:
                stored = manifest_path.parent / _read_relpath(record["path"])
                expected_size = int(record["size_bytes"])
                expected_sha256 = str(record["sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Corpus manifest has an invalid file record: {manifest_path}"
                ) from exc
            parsed_file_checks.append((stored, expected_size, expected_sha256))
            paths_by_parent.setdefault(stored.parent, {})[
                os.path.normcase(stored.name)
            ] = stored
            total_size += expected_size

        # A manifest normally stores every shard below one ``files``
        # directory.  Independent ``Path.is_file()``, ``Path.stat()``, and
        # digest-cache identity calls turned each warm verification into three
        # metadata round trips per shard on drvfs.  Enumerate each parent once,
        # retain the exact stat identity, and pass it into the existing digest
        # cache.  Missing, non-file, replaced, resized, or normally edited
        # shards still fail closed through the same identity fields.
        identities: Dict[Path, _ReplayFileIdentity] = {}
        for parent, expected_by_name in paths_by_parent.items():
            try:
                entries = os.scandir(parent)
            except OSError as exc:
                stored = next(iter(expected_by_name.values()))
                raise RuntimeError(
                    f"Corpus snapshot file failed integrity verification: {stored}"
                ) from exc
            remaining = dict(expected_by_name)
            matched = []
            with entries:
                for entry in entries:
                    stored = remaining.pop(os.path.normcase(entry.name), None)
                    if stored is None:
                        continue
                    matched.append((entry, stored))
            if remaining:
                stored = next(iter(remaining.values()))
                raise RuntimeError(
                    f"Corpus snapshot file failed integrity verification: {stored}"
                )
            stats = _directory_entry_stats(
                [entry for entry, _stored in matched])
            for (_entry, stored), (_, stat_result, error) in zip(matched, stats):
                if (error is not None or stat_result is None
                        or not stat_module.S_ISREG(stat_result.st_mode)):
                    raise RuntimeError(
                        "Corpus snapshot file failed integrity "
                        f"verification: {stored}"
                    ) from error
                identities[stored] = _replay_file_identity_from_stat(
                    stored, stat_result)

        file_checks: List[Tuple[Path, str, _ReplayFileIdentity]] = []
        for stored, expected_size, expected_sha256 in parsed_file_checks:
            identity = identities.get(stored)
            if identity is None or identity.st_size != expected_size:
                raise RuntimeError(
                    f"Corpus snapshot file failed integrity verification: {stored}"
                )
            file_checks.append((stored, expected_sha256, identity))

        def _verify_digest(
            check: Tuple[Path, str, _ReplayFileIdentity],
        ) -> Tuple[Path, bool]:
            stored, expected_sha256, identity = check
            return (
                stored,
                _replay_file_sha256_for_identity(stored, identity)
                == expected_sha256,
            )

        if (
            len(file_checks) > 1
            and total_size >= _PARALLEL_MANIFEST_HASH_MIN_BYTES
        ):
            workers = min(_MANIFEST_HASH_WORKERS, len(file_checks))
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="corpus-integrity",
            ) as pool:
                digest_results = pool.map(_verify_digest, file_checks)
                for stored, matches in digest_results:
                    if not matches:
                        raise RuntimeError(
                            "Corpus snapshot file failed integrity "
                            f"verification: {stored}"
                        )
        else:
            for check in file_checks:
                stored, matches = _verify_digest(check)
                if not matches:
                    raise RuntimeError(
                        "Corpus snapshot file failed integrity "
                        f"verification: {stored}"
                    )

        # A cache hit does not reopen the shard, so close the interval between
        # the directory snapshot and the digest verdict with one more batched
        # identity check.  This preserves fail-closed behavior if a shard is
        # replaced or edited while verification is in progress without
        # restoring one independent stat call per path.
        try:
            self._verify_replay_file_identities(identities)
        except RuntimeError as exc:
            raise RuntimeError(
                "Corpus snapshot file failed integrity verification: "
                f"{manifest_path}"
            ) from exc
        state_keys = self._verify_manifest_state_keys(manifest_path, manifest)
        _cache_touch(
            self._manifest_integrity_identity_cache,
            manifest_cache_key,
            dict(identities),
            _MANIFEST_INTEGRITY_IDENTITY_CACHE_MAX,
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
    def _ledger_fingerprints_path(self) -> Path:
        """Path for the optional, source-verified fingerprint sidecar."""

        return self.trained_ledger_dir / "trained_state_fingerprints.v1.bin"

    @property
    def _ledger_seed_path(self) -> Path:
        return self.trained_ledger_dir / "seed.json"

    @staticmethod
    def _ledger_source_identity(identity: _ReplayFileIdentity) -> dict:
        """Serialize the stat fields that invalidate a ledger sidecar."""

        return {
            "st_dev": identity.st_dev,
            "st_ino": identity.st_ino,
            "st_size": identity.st_size,
            "st_mtime_ns": identity.st_mtime_ns,
        }

    def _load_ledger_fingerprint_sidecar(self) -> Optional[Set[int]]:
        """Load a verified binary ledger index, or return ``None`` on a miss.

        The sidecar is strictly an acceleration of the canonical gzip ledger,
        never an independent source of training history.  Its source stat
        identity cheaply rejects normal replacements, its recorded SHA-256
        catches an in-place alteration, and its own payload digest catches a
        torn or corrupt sidecar.  Every failure deliberately falls through to
        the established gzip parser.
        """

        source_path = self._ledger_state_keys_path
        sidecar_path = self._ledger_fingerprints_path
        if not source_path.is_file() or not sidecar_path.is_file():
            return None
        try:
            source_identity = _replay_file_identity(source_path)
            expected_identity = self._ledger_source_identity(source_identity)
            with sidecar_path.open("rb") as handle:
                if handle.readline(len(_LEDGER_FINGERPRINT_SIDECAR_MAGIC) + 1) != (
                    _LEDGER_FINGERPRINT_SIDECAR_MAGIC
                ):
                    return None
                header_line = handle.readline(
                    _LEDGER_FINGERPRINT_SIDECAR_HEADER_LIMIT + 1)
                if (
                    not header_line.endswith(b"\n")
                    or len(header_line) > _LEDGER_FINGERPRINT_SIDECAR_HEADER_LIMIT
                ):
                    return None
                header = json.loads(header_line)
                if not isinstance(header, dict):
                    return None
                count = header.get("fingerprint_count")
                if (
                    header.get("version") != _LEDGER_FINGERPRINT_SIDECAR_VERSION
                    or header.get("byteorder") != "little"
                    or header.get("source_identity") != expected_identity
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                ):
                    return None
                source_sha256 = header.get("source_sha256")
                payload_sha256 = header.get("payload_sha256")
                if (
                    not isinstance(source_sha256, str)
                    or len(source_sha256) != 64
                    or not isinstance(payload_sha256, str)
                    or len(payload_sha256) != 64
                ):
                    return None
                payload_offset = handle.tell()
                expected_size = payload_offset + count * _LEDGER_FINGERPRINT_BYTES
                if sidecar_path.stat().st_size != expected_size:
                    return None

                # Re-hash the canonical source before using a persisted index.
                # An atomic source rewrite changes the stat identity above; the
                # digest also detects the rare in-place modification case.
                if _sha256_file_uncached(source_path) != source_sha256:
                    return None
                if _replay_file_identity(source_path) != source_identity:
                    return None

                payload = handle.read()
            if hashlib.sha256(payload).hexdigest() != payload_sha256:
                return None
            if len(payload) != count * _LEDGER_FINGERPRINT_BYTES:
                return None
            values = array("Q")
            if values.itemsize != _LEDGER_FINGERPRINT_BYTES:
                return None
            values.frombytes(payload)
            if sys.byteorder != "little":
                values.byteswap()
            fingerprints = set(values)
            # The source-key representation is a set, so duplicate binary
            # values mean a malformed sidecar rather than a valid cache hit.
            if len(fingerprints) != count:
                return None
            self._trained_ledger_source_sha256 = source_sha256
            return fingerprints
        except (
            OSError,
            ValueError,
            TypeError,
            UnicodeDecodeError,
            OverflowError,
        ):
            return None

    def _write_ledger_fingerprint_sidecar(self, fingerprints: Set[int]) -> None:
        """Best-effort atomically persist a verified compact ledger index."""

        source_path = self._ledger_state_keys_path
        sidecar_path = self._ledger_fingerprints_path
        if not source_path.is_file():
            return
        temp_name: Optional[str] = None
        try:
            source_identity = _replay_file_identity(source_path)
            values = array("Q", fingerprints)
            if values.itemsize != _LEDGER_FINGERPRINT_BYTES:
                raise RuntimeError("unexpected unsigned-long-long item size")
            if sys.byteorder != "little":
                values.byteswap()
            header = {
                "version": _LEDGER_FINGERPRINT_SIDECAR_VERSION,
                "byteorder": "little",
                "source_identity": self._ledger_source_identity(source_identity),
                "source_sha256": _sha256_file_uncached(source_path),
                "fingerprint_count": len(values),
                "payload_sha256": hashlib.sha256(values).hexdigest(),
            }
            if _replay_file_identity(source_path) != source_identity:
                return
            # The source digest was calculated from an unchanged canonical
            # ledger.  Preserve it for any derived cache assembled later in
            # this manager's verified split preparation.
            self._trained_ledger_source_sha256 = header["source_sha256"]
            header_bytes = json.dumps(
                header, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
            fd, temp_name = tempfile.mkstemp(
                prefix=sidecar_path.name + ".",
                suffix=".tmp",
                dir=sidecar_path.parent,
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(_LEDGER_FINGERPRINT_SIDECAR_MAGIC)
                handle.write(header_bytes)
                handle.write(b"\n")
                values.tofile(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, sidecar_path)
            temp_name = None
        except (OSError, ValueError, TypeError, OverflowError, RuntimeError) as exc:
            # A sidecar must never make the authoritative ledger unavailable.
            # Leave a previous sidecar in place, where its source identity will
            # reject it after a successful canonical-ledger rewrite.
            print(f"[warn] Could not cache trained-ledger fingerprints: {exc}")
        finally:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass

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
        self._write_ledger_fingerprint_sidecar({
            _state_key_fingerprint(key) for key in state_keys
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
        fingerprints = self._load_ledger_fingerprint_sidecar()
        if fingerprints is None:
            fingerprints = set()
            if self._ledger_state_keys_path.is_file():
                for key in _iter_state_keys(self._ledger_state_keys_path):
                    fingerprints.add(_state_key_fingerprint(key))
                self._write_ledger_fingerprint_sidecar(fingerprints)
        else:
            print(
                "Loaded trained-ledger fingerprint sidecar "
                f"({len(fingerprints):,} states)."
            )
        self._trained_ledger_cache = (names, fingerprints)
        return self._trained_ledger_cache

    def trained_ledger_shard_names(self) -> Set[str]:
        return set(self._load_trained_ledger()[0])

    def trained_ledger_state_fingerprints(self) -> Set[int]:
        """Membership set used to reject historically trained hold-out states."""
        return set(self._load_trained_ledger()[1])

    def trained_ledger_source_sha256(self) -> Optional[str]:
        """Return the digest of the canonical ledger verified for this split.

        A caller must first obtain the membership set through the ordinary
        ledger load path.  This method deliberately triggers that same path,
        so an unavailable or unverifiable ledger returns ``None`` instead of
        letting a derived cache claim a source identity it did not verify.
        """

        self._load_trained_ledger()
        return self._trained_ledger_source_sha256

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
        if added_states:
            self._write_ledger_fingerprint_sidecar(known_fingerprints)
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
        # Check the quota against the optimistic candidate set before scanning
        # the all-time trained ledger and retained snapshot manifests.  If no
        # growth is possible even when every unheld live shard is eligible,
        # the leakage filter cannot change that verdict.  This is the steady
        # state once the append-only hold-out reaches its file ceiling.
        present_held = sum(1 for path in files if path.name in held_names)
        unheld_files = sum(1 for path in files if path.name not in held_names)
        if self._validation_growth_quota(
            total_files=len(files),
            held=present_held,
            candidates=unheld_files,
            all_time_held=len(held_names),
        ) <= 0:
            return None

        trained_names = self._trained_shard_names()
        candidates = sorted(
            (path for path in files
             if path.name not in held_names and path.name not in trained_names),
            key=lambda path: self._validation_rank_key(path.name),
        )
        # The quota is measured against still-present held shards, so a hold-out
        # whose files have rotated out of the replay window is not counted as if
        # it still covered the live corpus.
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
        try:
            manifest = self._load_manifest(manifest_path)
        except FileNotFoundError:
            manifest = None
        if manifest is not None:
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

        files, rejected_files, replay_identities = (
            self._eligible_replay_files_with_identities())
        if not files:
            raise RuntimeError(
                "No replay files satisfy the repaired policy-distillation contract"
            )
        validation_manifest, validation_keys = self._ensure_validation(files)
        validation_hashes = {record["sha256"] for record in validation_manifest["files"]}

        file_records = []
        train_files = []
        for path in files:
            identity = replay_identities[path]
            file_hash = _replay_file_sha256_for_identity(path, identity)
            if file_hash in validation_hashes:
                continue
            train_files.append(path)
            file_records.append({
                "name": path.name,
                "sha256": file_hash,
                "size_bytes": identity.st_size,
            })
        if not train_files:
            raise RuntimeError("No replay files remain after the immutable validation split")

        current_path, current = self._load_current_manifest()
        previous_keys: Set[str] = set()
        previous_keys_digest: Optional[str] = None
        previous_fingerprint = None
        previous_source = None
        if current_path is not None:
            assert current is not None
            previous_keys, previous_keys_digest = self._cached_state_key_file(
                current_path.parent / current["state_keys_file"])
            recorded_previous_digest = current.get("metrics", {}).get(
                "state_set_sha256")
            if (recorded_previous_digest is not None
                    and recorded_previous_digest != previous_keys_digest):
                raise RuntimeError(
                    "Corpus canonical-state fingerprint is invalid: "
                    f"{current_path}"
                )
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
                previous_keys_digest = str(
                    base_manifest["metrics"]["state_set_sha256"])
                previous_fingerprint = base_manifest.get("fingerprint")
                previous_source = "lineage_base"

        # The digest of the pre-deduplication key set is never read: the
        # manifest fingerprints ``train_keys`` (post-dedup) below.
        metrics, all_train_keys, pre_validation_new_keys = (
            self._analyze_replay_window(
                train_files,
                previous_keys,
                replay_identities,
            )
        )
        # A rejected candidate needs exact post-validation cardinalities, not
        # a materialized training-key complement. Count the much smaller
        # validation overlap first and defer construction of ``train_keys``
        # until a candidate can reach a durable digest or snapshot write.
        rolling_overlap_counts = self._rolling_validation_overlap_counts(
            validation_keys, self.external_validation_state_keys)
        if rolling_overlap_counts is None:
            validation_overlap_count = _validation_overlap_state_count(
                all_train_keys,
                validation_keys,
                self.external_validation_state_keys,
            )
            fresh_validation_overlap_count = _validation_overlap_state_count(
                pre_validation_new_keys,
                validation_keys,
                self.external_validation_state_keys,
            )
        else:
            (
                validation_overlap_count,
                fresh_validation_overlap_count,
            ) = rolling_overlap_counts
        train_key_count = len(all_train_keys) - validation_overlap_count
        fresh_key_count = (
            len(pre_validation_new_keys) - fresh_validation_overlap_count)
        fresh_rate = (
            fresh_key_count / train_key_count if train_key_count else 0.0)
        train_keys: Optional[Set[str]] = None
        # Audit Suggestion 9.  A growth event moves a whole *fresh* shard out of
        # training -- _grow_validation can only choose never-trained shards, by
        # design, because a trained one would measure memorisation.  The
        # consequence is that the cycle's entire freshness gain is cancelled,
        # which looked from the logs exactly like generator saturation.  Price
        # it here, where the previous corpus is in hand, and record it in the
        # manifest so a stalled admission is explainable from artifacts alone.
        growth = self._last_holdout_growth or {}
        if growth:
            # Hold-out growth is rare and needs the exact transferred set for
            # its counterfactual. Preserve the original full-set calculation
            # on that path; the ceiling-bound recurring path stays cardinality
            # only until its freshness verdict.
            train_keys = _exclude_validation_state_keys(
                all_train_keys,
                validation_keys,
                self.external_validation_state_keys,
            )
            withheld = set(growth.get("added_state_keys", ())).difference(
                train_keys)
        else:
            withheld = set()
        withheld_fresh = withheld.difference(previous_keys)
        counterfactual_denominator = train_key_count + len(withheld)
        counterfactual_fresh_rate = (
            (fresh_key_count + len(withheld_fresh))
            / counterfactual_denominator
            if counterfactual_denominator else 0.0
        )

        metrics.update({
            "holdout_growth_files": list(growth.get("added_files", ())),
            "states_transferred_to_holdout": len(withheld),
            "fresh_states_transferred_to_holdout": len(withheld_fresh),
            "fresh_unique_state_rate_without_holdout_growth": (
                counterfactual_fresh_rate),
            "validation_overlap_state_count_removed": validation_overlap_count,
            "external_validation_state_count": len(
                self.external_validation_state_keys),
            "post_dedup_unique_state_count": train_key_count,
            "fresh_unique_state_count": fresh_key_count,
            "fresh_unique_state_rate": fresh_rate,
            # A rejected candidate is not a durable corpus artifact.  Defer its
            # O(u log u) sorted digest until the gate can admit it.  An exact
            # unchanged candidate reuses the already-verified predecessor
            # digest below, while every admitted snapshot still derives and
            # persists its digest from the post-deduplication training keys.
            "state_set_sha256": None,
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

        # Audit, digest, and diversity work above intentionally shares one
        # metadata snapshot.  Recheck every observed shard before using those
        # results for any decision, so an append, replacement, or deletion
        # during the transaction fails closed instead of publishing mixed data.
        self._verify_replay_file_identities(replay_identities)

        # A lost pointer must fail before any early freshness return.  Otherwise
        # a stale candidate could hide the lineage corruption merely by missing
        # the admission threshold.
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

        candidate_matches_previous_keys = (
            previous_source is not None
            and train_key_count == len(previous_keys)
            and fresh_key_count == 0
        )
        if (previous_source is not None
                and fresh_rate < self.min_fresh_fraction
                and not candidate_matches_previous_keys):
            return SnapshotDecision(
                False,
                f"fresh_unique_state_rate {fresh_rate:.6f} is below {self.min_fresh_fraction:.6f}",
                current_path,
                metrics,
            )

        if candidate_matches_previous_keys and previous_keys_digest is not None:
            metrics["state_set_sha256"] = previous_keys_digest
        else:
            if train_keys is None:
                train_keys = _exclude_validation_state_keys(
                    all_train_keys,
                    validation_keys,
                    self.external_validation_state_keys,
                )
            metrics["state_set_sha256"] = _state_set_digest(train_keys)

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
        if previous_source is not None and fresh_rate < self.min_fresh_fraction:
            return SnapshotDecision(
                False,
                f"fresh_unique_state_rate {fresh_rate:.6f} is below {self.min_fresh_fraction:.6f}",
                current_path,
                metrics,
            )

        if train_keys is None:
            train_keys = _exclude_validation_state_keys(
                all_train_keys,
                validation_keys,
                self.external_validation_state_keys,
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

    def prepare_split(
        self,
        manifest_path: Optional[Path] = None,
        max_train_entries: int = 0,
    ) -> _SnapshotSplitContext:
        """Verify frozen split inputs before materializing their entry lists.

        This deliberately retains every integrity, lineage, validation-version,
        and all-time-ledger check from :meth:`load_split`.  Separating the
        expensive train-shard parse lets the trainer use a manifest-keyed tensor
        cache on a warm relaunch without trusting an unverified cache file.
        """

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
        validation_keys = set(stored_validation_keys)
        validation_keys.update(self.external_validation_state_keys)
        return _SnapshotSplitContext(
            manifest_path=path,
            manifest=manifest,
            validation_path=validation_path,
            validation_manifest=validation_manifest,
            validation_keys=validation_keys,
            historically_trained=historically_trained,
            max_train_entries=max(0, int(max_train_entries)),
        )

    def load_validation_entries(
        self, context: _SnapshotSplitContext,
    ) -> List[ReplayEntry]:
        """Materialize the leakage-filtered held-out entries from a context."""

        validation_entries: List[ReplayEntry] = []
        leaked_validation_states: Set[str] = set()
        leaked_validation_entries = 0
        for record in context.validation_manifest["files"]:
            file_path = context.validation_path.parent / _read_relpath(
                record["path"])
            for entry_dict in _iter_entry_dicts(file_path):
                key = canonical_state_key(entry_dict["state"])
                # The key stays in ``validation_keys`` either way, so a state
                # dropped here is never quietly handed back to training.
                context.validation_keys.add(key)
                if _state_key_fingerprint(key) in context.historically_trained:
                    leaked_validation_states.add(key)
                    leaked_validation_entries += 1
                    continue
                validation_entries.append(ReplayEntry.from_dict(entry_dict))
        context.manifest["validation_leakage"] = {
            "ledger_enabled": self.trained_ledger_enabled,
            "all_time_trained_state_count": len(context.historically_trained),
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
        return validation_entries

    def load_train_entries(
        self, context: _SnapshotSplitContext,
    ) -> List[ReplayEntry]:
        """Materialize the train entries after cross-split deduplication."""

        train_entries: List[ReplayEntry] = []
        for record in context.manifest["files"]:
            file_path = context.manifest_path.parent / _read_relpath(
                record["path"])
            for entry_dict in _iter_entry_dicts(file_path):
                if canonical_state_key(entry_dict["state"]) in context.validation_keys:
                    continue
                train_entries.append(ReplayEntry.from_dict(entry_dict))

        if (
            context.max_train_entries > 0
            and len(train_entries) > context.max_train_entries
        ):
            rng = random.Random(self.split_seed)
            indices = sorted(rng.sample(
                range(len(train_entries)), context.max_train_entries))
            train_entries = [train_entries[index] for index in indices]
        return train_entries

    def load_split(
        self,
        manifest_path: Optional[Path] = None,
        max_train_entries: int = 0,
    ) -> Tuple[List[ReplayEntry], List[ReplayEntry], dict]:
        """Load a frozen train/validation split with cross-split deduplication."""

        context = self.prepare_split(manifest_path, max_train_entries)
        validation_entries = self.load_validation_entries(context)
        train_entries = self.load_train_entries(context)
        return train_entries, validation_entries, context.manifest


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
