"""Versioned replay-corpus snapshots for policy-distillation recovery.

The snapshot manager keeps training data immutable for a training window,
holds validation out by whole replay file, and admits a new snapshot only when
at least the configured fraction of its canonical states are new relative to
the previous snapshot.

Canonicalization matches the model's side-to-move perspective. ``move_count``
is intentionally ignored because it is not part of the policy input.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .move_encoder import ENCODING_VERSION
from .replay import ReplayEntry


SNAPSHOT_SCHEMA_VERSION = 1
POLICY_REPLAY_CONTRACT_VERSION = 1
CANONICAL_RULES_ID = "filipino-dama-default-v1"


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


def replay_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_policy_replay_file(
    path: Path,
    allowed_opening_plies: Sequence[int],
) -> Dict[str, Any]:
    """Fail closed on replay files that predate the repaired sample contract."""
    allowed = {int(value) for value in allowed_opening_plies}
    errors: Counter[str] = Counter()
    game_sources: Dict[str, str] = {}
    records = 0
    for entry in _iter_entry_dicts(path):
        records += 1
        legal_moves = entry.get("legal_moves")
        if not isinstance(legal_moves, list) or not legal_moves:
            errors["missing_legal_moves"] += 1
            continue
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

    source_games = Counter(game_sources.values())
    return {
        "contract_version": POLICY_REPLAY_CONTRACT_VERSION,
        "valid": records > 0 and not errors,
        "records": records,
        "game_count": len(game_sources),
        "source_game_counts": dict(sorted(source_games.items())),
        "errors": dict(sorted(errors.items())),
    }


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
    record_keys: List[str] = []
    state_counts: Counter[str] = Counter()
    state_files: Dict[str, Set[str]] = defaultdict(set)
    state_cycles: Dict[str, Set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    game_sources: Dict[str, str] = {}
    forced = 0
    malformed = 0
    total = 0

    for path in files:
        for entry in _iter_entry_dicts(path):
            try:
                key = canonical_state_key(entry["state"])
                legal_moves = entry["legal_moves"]
            except (KeyError, TypeError, ValueError):
                malformed += 1
                continue
            total += 1
            record_keys.append(key)
            state_counts[key] += 1
            unique_keys.add(key)
            state_files[key].add(path.name)
            # Trainer-generated game IDs carry the generation cycle.  Keep
            # this separate from file identity so diversity can expose a
            # repeated state across cycles even after files are renamed or
            # copied into a snapshot.
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

    new_unique = unique_keys.difference(previous)
    fresh_records = sum(1 for key in record_keys if key not in previous)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _write_state_keys(path: Path, state_keys: Iterable[str]) -> None:
    with gzip.open(path, "wt", encoding="ascii", newline="\n") as handle:
        for key in sorted(state_keys):
            handle.write(key)
            handle.write("\n")


def _read_state_keys(path: Path) -> Set[str]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        return {line.strip() for line in handle if line.strip()}


def _link_or_copy(source: Path, destination: Path) -> str:
    # Snapshot shards must be independent immutable files.  Hardlinks make a
    # later replay append/truncate mutate an already-admitted snapshot in
    # place, defeating the integrity manifest and recovery contract.
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
    ) -> None:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0.0 <= min_fresh_fraction <= 1.0:
            raise ValueError("min_fresh_fraction must be between 0 and 1")
        self.replay_dir = Path(replay_dir)
        self.snapshot_root = Path(snapshot_root)
        self.validation_fraction = float(validation_fraction)
        self.split_seed = int(split_seed)
        self.min_fresh_fraction = float(min_fresh_fraction)
        self.enforce_policy_contract = bool(enforce_policy_contract)
        self.allowed_opening_plies = tuple(int(value) for value in allowed_opening_plies)
        self.external_validation_state_keys: Set[str] = set()

    def set_external_validation_state_keys(self, state_keys: Iterable[str]) -> None:
        """Exclude a frozen external validation suite from every train snapshot."""
        self.external_validation_state_keys = {str(key) for key in state_keys}

    @property
    def current_pointer(self) -> Path:
        return self.snapshot_root / "CURRENT"

    @property
    def validation_manifest_path(self) -> Path:
        return self.snapshot_root / "validation" / "manifest.json"

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
        path = self.snapshot_root / relative
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

        for record in manifest.get("files", []):
            try:
                stored = manifest_path.parent / record["path"]
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

    def _ensure_validation(self, files: Sequence[Path]) -> Tuple[dict, Set[str]]:
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
            return manifest, state_keys

        if len(files) < 2:
            raise RuntimeError("At least two replay files are required for a whole-file validation split")

        desired = max(1, int(round(len(files) * self.validation_fraction)))
        desired = min(desired, len(files) - 1)
        ranked = sorted(
            files,
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
            link_mode = _link_or_copy(source, destination)
            file_records.append({
                "name": source.name,
                "path": str(Path("files") / source.name),
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
                "fraction": self.validation_fraction,
                "seed": self.split_seed,
            },
            "encoding_version": ENCODING_VERSION,
            "rules_id": CANONICAL_RULES_ID,
            "files": file_records,
            "state_keys_file": state_keys_file,
            "metrics": metrics,
        }
        _write_json_atomic(manifest_path, manifest)
        return manifest, state_keys

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
        if current_path is not None:
            current = self._load_manifest(current_path)
            previous_keys = _read_state_keys(current_path.parent / current["state_keys_file"])
            previous_fingerprint = current.get("fingerprint")

        metrics, all_train_keys = analyze_replay_files(train_files, previous_keys)
        leakage_keys = all_train_keys.intersection(validation_keys)
        train_keys = all_train_keys.difference(validation_keys)
        new_keys = train_keys.difference(previous_keys)
        fresh_rate = (len(new_keys) / len(train_keys)) if train_keys else 0.0
        metrics.update({
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
            raise RuntimeError(
                "Eligible replay games do not satisfy the exact 70/30 "
                f"trajectory contract: {algorithm_games}/{model_games}"
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
        if current_path is not None and fresh_rate < self.min_fresh_fraction:
            return SnapshotDecision(
                False,
                f"fresh_unique_state_rate {fresh_rate:.6f} is below {self.min_fresh_fraction:.6f}",
                current_path,
                metrics,
            )

        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        version = self._next_version()
        final_dir = self.snapshot_root / f"snapshot_v{version:06d}"
        staging = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.", dir=self.snapshot_root))
        try:
            files_dir = staging / "files"
            files_dir.mkdir()
            stored_files = []
            for source, record in zip(train_files, file_records):
                destination = files_dir / source.name
                link_mode = _link_or_copy(source, destination)
                stored_files.append({
                    **record,
                    "path": str(Path("files") / source.name),
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
                "validation_manifest": os.path.relpath(
                    self.validation_manifest_path, staging
                ),
                "teacher_settings": dict(teacher_settings),
                "noise_settings": dict(noise_settings),
                "generation_settings": dict(generation_settings),
                "admission": {
                    "minimum_fresh_unique_state_rate": self.min_fresh_fraction,
                    "observed_fresh_unique_state_rate": fresh_rate,
                    "passed": True,
                },
                "metrics": metrics,
            }
            _write_json_atomic(staging / "manifest.json", manifest)
            os.replace(staging, final_dir)
            _write_json_atomic(
                self.snapshot_root / "current.json",
                {"manifest": str(Path(final_dir.name) / "manifest.json"), "fingerprint": fingerprint},
            )
            pointer_temp = self.snapshot_root / "CURRENT.tmp"
            pointer_temp.write_text(str(Path(final_dir.name) / "manifest.json") + "\n", encoding="utf-8")
            os.replace(pointer_temp, self.current_pointer)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

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
        manifest["manifest_path"] = str(path.resolve())
        validation_path = (path.parent / manifest["validation_manifest"]).resolve()
        if validation_path != self.validation_manifest_path.resolve():
            raise RuntimeError(
                "Training snapshot references an unexpected validation manifest"
            )
        validation_manifest = self._load_manifest(validation_path)
        stored_validation_keys = self._verify_manifest_integrity(
            validation_path, validation_manifest, "immutable_validation"
        )
        validation_split = validation_manifest.get("split", {})
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

        validation_entries: List[ReplayEntry] = []
        validation_keys: Set[str] = set(stored_validation_keys)
        validation_keys.update(self.external_validation_state_keys)
        for record in validation_manifest["files"]:
            file_path = validation_path.parent / record["path"]
            for entry_dict in _iter_entry_dicts(file_path):
                key = canonical_state_key(entry_dict["state"])
                validation_keys.add(key)
                validation_entries.append(ReplayEntry.from_dict(entry_dict))

        train_entries: List[ReplayEntry] = []
        for record in manifest["files"]:
            file_path = path.parent / record["path"]
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
