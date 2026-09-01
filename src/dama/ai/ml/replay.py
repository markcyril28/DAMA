"""Replay buffer management for training data."""

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Iterator, Optional, Dict, Any
from dataclasses import dataclass
import random

# Use orjson (Rust-backed, 3-5x faster) when available, fall back to stdlib json.
try:
    import orjson as _json_mod

    def _json_loads(s):
        return _json_mod.loads(s)

    def _json_dumps(obj) -> str:
        # orjson.dumps returns bytes; decode for JSONL text lines.
        return _json_mod.dumps(obj).decode('utf-8')
except ImportError:
    import json as _json_mod

    _json_loads = _json_mod.loads
    _json_dumps = _json_mod.dumps

# Errors raised when parsing a corrupt/truncated JSONL line: JSONDecodeError
# (stdlib and orjson) subclasses ValueError; KeyError/TypeError cover entry
# dicts with missing fields or malformed structures in ReplayEntry.from_dict.
_PARSE_ERRORS = (ValueError, KeyError, TypeError)

# [Pass 181] Durable per-shard physical line counts for count_entries().
# Replay shards are write-once, so a shard whose (size, mtime_ns) identity is
# unchanged has exactly the line count it had when it was last read.  The
# sidecar lives next to the shards, like generation_cycle_cache.json, and its
# name matches neither ``replay_*.jsonl`` nor ``*.jsonl`` so every corpus and
# replay glob keeps ignoring it.  Keys are shard basenames (no inode) so a
# corpus relocated across mounts or machines stays warm.
_ENTRY_COUNT_SIDECAR_NAME = 'entry_count_cache.json'
_ENTRY_COUNT_SIDECAR_SCHEMA = 1


def _count_replay_lines(path: Path) -> int:
    """Count the physical lines of one replay shard.

    Text-mode iteration, so the count matches the line enumeration used by
    ``sample_entries()``: an unterminated trailing line counts as a line.
    """
    with open(path, 'r') as fh:
        return sum(1 for _ in fh)


def _load_entry_count_sidecar(replay_dir: Path) -> Dict[str, tuple]:
    """Load persisted (size, mtime_ns, count) triples per shard basename.

    Best-effort by contract: any read or parse problem yields an empty
    mapping and count_entries() re-derives the counts from the shards.
    """
    sidecar = replay_dir / _ENTRY_COUNT_SIDECAR_NAME
    try:
        raw = json.loads(sidecar.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get('schema') != _ENTRY_COUNT_SIDECAR_SCHEMA:
        return {}
    entries = raw.get('entries')
    if not isinstance(entries, dict):
        return {}
    loaded: Dict[str, tuple] = {}
    for name, record in entries.items():
        if not isinstance(name, str) or not isinstance(record, list):
            continue
        if len(record) != 3:
            continue
        size, mtime_ns, count = record
        if not all(isinstance(v, int) and not isinstance(v, bool)
                   for v in (size, mtime_ns, count)):
            continue
        if size < 0 or count < 0:
            continue
        loaded[name] = (size, mtime_ns, count)
    return loaded


def _save_entry_count_sidecar(replay_dir: Path, entries: Dict[str, tuple]) -> bool:
    """Persist per-shard line counts atomically; report whether written.

    Always temp file + fsync + ``os.replace``: an in-place rewrite would go
    through a shared inode when the replay directory is a hardlink copy of
    another one (probes build such copies), and a torn write must never be
    observable.
    """
    payload = {
        'schema': _ENTRY_COUNT_SIDECAR_SCHEMA,
        'entries': {
            name: [size, mtime_ns, count]
            for name, (size, mtime_ns, count) in sorted(entries.items())
        },
    }
    sidecar = replay_dir / _ENTRY_COUNT_SIDECAR_NAME
    temp_name = ''
    try:
        fd, temp_name = tempfile.mkstemp(
            dir=str(replay_dir), prefix='.entry_count_cache.', suffix='.tmp')
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, sidecar)
        temp_name = ''
        return True
    except OSError:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        return False


@dataclass
class ReplayEntry:
    """A single training example from a game."""
    state: dict           # Compact state representation
    legal_moves: list     # List of move dicts
    chosen_index: int     # Hard teacher label index
    result: int           # Game result from this player's perspective (+1, -1, 0)
    score: float = 0.0    # Detailed shaped reward score (from scoring system)
    sample_weight: float = 1.0  # Extra multiplier for loss weighting
    # Policy-distillation audit metadata. Defaults keep old replay readable.
    played_index: Optional[int] = None
    trajectory_source: Optional[str] = None
    was_exploration: Optional[bool] = None
    teacher_difficulty: Optional[str] = None
    opening_plies: int = 0
    game_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            'state': self.state,
            'legal_moves': self.legal_moves,
            'chosen_index': self.chosen_index,
            'result': self.result,
        }
        # Only include score if non-zero (saves space for old-format entries)
        if self.score != 0.0:
            d['score'] = round(self.score, 4)
        if self.sample_weight != 1.0:
            d['sample_weight'] = round(float(self.sample_weight), 6)
        if self.played_index is not None:
            d['played_index'] = int(self.played_index)
        if self.trajectory_source is not None:
            d['trajectory_source'] = self.trajectory_source
        if self.was_exploration is not None:
            d['was_exploration'] = bool(self.was_exploration)
        if self.teacher_difficulty is not None:
            d['teacher_difficulty'] = self.teacher_difficulty
        if self.opening_plies:
            d['opening_plies'] = int(self.opening_plies)
        if self.game_id is not None:
            d['game_id'] = self.game_id
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ReplayEntry':
        legal_moves = data['legal_moves']
        chosen_index = data['chosen_index']
        if legal_moves and (chosen_index < 0 or chosen_index >= len(legal_moves)):
            raise ValueError(
                f"chosen_index {chosen_index} out of bounds for {len(legal_moves)} legal moves"
            )
        played_index = data.get('played_index')
        if (played_index is not None and legal_moves
                and (played_index < 0 or played_index >= len(legal_moves))):
            raise ValueError(
                f"played_index {played_index} out of bounds for {len(legal_moves)} legal moves"
            )
        return cls(
            state=data['state'],
            legal_moves=legal_moves,
            chosen_index=chosen_index,
            result=data.get('result', 0),
            score=data.get('score', 0.0),
            sample_weight=float(data.get('sample_weight', 1.0)),
            played_index=played_index,
            trajectory_source=data.get('trajectory_source'),
            was_exploration=data.get('was_exploration'),
            teacher_difficulty=data.get('teacher_difficulty'),
            opening_plies=int(data.get('opening_plies', 0)),
            game_id=(str(data['game_id']) if data.get('game_id') is not None else None),
        )


class ReplayBuffer:
    """
    Disk-backed replay buffer for training data.

    Stores replay data as JSONL files in the replay directory.
    """

    def __init__(
        self,
        replay_dir: str = "data/replay",
        max_files: int = 100,
        *,
        cache_written_entries: bool = True,
    ):
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        # Legacy training reads replay through load_all_entries(), where keeping
        # newly written entries avoids an immediate JSON parse.  Snapshot-mode
        # training reads the immutable shard through CorpusSnapshotManager and
        # never consumes this cache, so retaining each cycle's complete Python
        # object graph only spends session RAM.  See Journal Pass 183.
        self._cache_written_entries = bool(cache_written_entries)
        self._current_file = None
        self._current_writer = None
        # Incremental file cache: {path: (mtime, [ReplayEntry, ...])}
        # Avoids re-parsing unchanged JSONL files across epochs.
        self._file_cache: Dict[Path, tuple] = {}
        # In-memory entries written during the current session, keyed by file path.
        # Promoted to _file_cache on close() so the next load_all_entries() skips
        # re-parsing the file we just wrote.
        self._session_entries: Dict[Path, List[ReplayEntry]] = {}
        # Raw dicts from add_entry_dicts() — defers ReplayEntry creation to close().
        # Merged into _session_entries on _close_current() to avoid per-call overhead.
        self._session_dicts: Dict[Path, List[dict]] = {}
        # [Pass 181] Physical line count per shard basename, keyed by
        # (size, mtime_ns, count) and persisted to _ENTRY_COUNT_SIDECAR_NAME so
        # count_entries() never re-reads an unchanged write-once shard.  On
        # drvfs the 60-file window cost ~14 s of reads per self-play cycle
        # for a statistic.  The sidecar is loaded lazily on first use.
        self._entry_count_cache: Dict[str, tuple] = {}
        self._entry_count_sidecar_loaded = False
        self._entry_count_dirty = False

    def start_new_file(self) -> Path:
        """Start a new replay file."""
        self._close_current()

        # [Pass 109] The name has only second resolution, and the file is opened
        # 'w'. Two cycles finishing in the same second used to silently truncate
        # the first one's data. That was near-harmless while persistence only
        # ran on the first cycle; now that every cycle persists, disambiguate.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.replay_dir / f"replay_{timestamp}.jsonl"
        suffix = 0
        while filepath.exists():
            suffix += 1
            filepath = self.replay_dir / f"replay_{timestamp}_{suffix:02d}.jsonl"

        self._current_file = filepath
        self._current_writer = open(filepath, 'w')

        return filepath

    def add_entry(self, entry: ReplayEntry) -> None:
        """Add an entry to the current replay file."""
        if self._current_writer is None:
            self.start_new_file()

        line = _json_dumps(entry.to_dict())
        self._current_writer.write(line + '\n')
        self._session_entries.setdefault(self._current_file, []).append(entry)

    def add_entries(self, entries: List[ReplayEntry]) -> None:
        """Add multiple entries and flush once."""
        # Do not create an empty replay shard for a game batch that produced no
        # positions (for example a zero-move or immediately terminal batch).
        # Empty JSONL files are indistinguishable from interrupted writes to
        # corpus scanners and can poison a fail-closed contract audit.
        if not entries:
            return
        if self._current_writer is None:
            self.start_new_file()
        # Build all lines then write once — reduces syscall overhead.
        lines = [_json_dumps(entry.to_dict()) for entry in entries]
        self._current_writer.write('\n'.join(lines) + '\n')
        self._current_writer.flush()
        # Keep in memory so close() can promote to file cache without re-parsing.
        self._session_entries.setdefault(self._current_file, []).extend(entries)

    def add_entry_dicts(self, dicts: List[dict]) -> None:
        """Add entries from raw dicts — avoids dict→ReplayEntry→dict round-trip.

        Self-play workers already return dicts (serialized for IPC). Writing
        them directly to JSONL skips one to_dict() call per entry. ReplayEntry
        conversion is deferred to _close_current() to avoid per-call overhead.
        """
        if not dicts:
            return
        if self._current_writer is None:
            self.start_new_file()
        lines = [_json_dumps(d) for d in dicts]
        self._current_writer.write('\n'.join(lines) + '\n')
        self._current_writer.flush()
        # Store raw dicts — ReplayEntry creation deferred to _close_current()
        self._session_dicts.setdefault(self._current_file, []).extend(dicts)

    def _close_current(self) -> None:
        """Close the current file and promote session entries to file cache."""
        if self._current_writer is not None:
            try:
                self._current_writer.close()
            except OSError:
                pass
            finally:
                self._current_writer = None
            # Promote in-memory entries to file cache so load_all_entries()
            # skips re-parsing the file we just wrote.  Snapshot-mode callers
            # opt out: validate the same records, remember their exact physical
            # line count, then release the object graph after the durable shard
            # has closed.  CorpusSnapshotManager is that mode's reader.
            path = self._current_file
            if path is not None:
                deferred = self._session_dicts.pop(path, None)
                if self._cache_written_entries:
                    # Convert any deferred dicts to ReplayEntry now (bulk conversion)
                    if deferred:
                        entries = [ReplayEntry.from_dict(d) for d in deferred]
                        self._session_entries.setdefault(path, []).extend(entries)
                    if path in self._session_entries:
                        try:
                            st = path.stat()
                            entries = self._session_entries.pop(path)
                            self._file_cache[path] = (st.st_mtime, entries)
                            # Every add_* call wrote exactly one line per entry to
                            # a file opened 'w', so the promoted length is this
                            # shard's physical line count; make it durable.
                            self._remember_entry_count(path.name, st, len(entries))
                        except OSError:
                            self._session_entries.pop(path, None)
                else:
                    # Preserve ReplayEntry.from_dict's bounds/type validation
                    # without retaining the resulting wrappers.  Values remain
                    # referenced by ``deferred`` until the validation completes.
                    if deferred:
                        for record in deferred:
                            ReplayEntry.from_dict(record)
                    entries = self._session_entries.pop(path, None)
                    entry_count = len(deferred or ()) + len(entries or ())
                    if entry_count:
                        try:
                            st = path.stat()
                            self._remember_entry_count(
                                path.name, st, entry_count)
                        except OSError:
                            pass
            self._current_file = None

    def close(self) -> None:
        """Close the buffer."""
        self._close_current()

    def discard_current_file(self) -> Optional[Path]:
        """Close and remove the currently open replay file.

        Self-play writes a cycle to one file.  Callers can use this method
        when a cycle did not complete, ensuring partial samples cannot be
        discovered by a later corpus scan.  Only the exact current file is
        targeted; older replay files and their caches are left untouched.
        """
        # Close the writer directly instead of via _close_current(): the file
        # is about to be deleted, so promoting its entries to the read cache is
        # wasted work, and ReplayEntry.from_dict() on a half-written cycle can
        # raise — which would leave the partial file on disk, the exact outcome
        # this method exists to prevent.
        path = self._current_file
        if self._current_writer is not None:
            try:
                self._current_writer.close()
            except OSError:
                pass
            finally:
                self._current_writer = None
        self._current_file = None
        if path is None:
            return None
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Keep the cache consistent even when the filesystem refuses the
            # unlink.  The caller will reject the cycle, so it is safer to
            # leave a diagnostic orphan than to admit it as active data.
            return path
        self._file_cache.pop(path, None)
        self._session_entries.pop(path, None)
        self._session_dicts.pop(path, None)
        self._forget_entry_count(path.name)
        return path

    def get_replay_files(self) -> List[Path]:
        """Get all replay files, sorted by modification time (newest first)."""
        files = list(self.replay_dir.glob("replay_*.jsonl"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def cleanup_old_files(self) -> int:
        """Remove old files beyond max_files limit. Returns number deleted.

        [Pass 109] Now actually called, once per persisted self-play cycle.
        Persisting every cycle writes roughly one 7MB file per minute, so
        without this the replay directory grows without bound over a 48h run.
        max_files <= 0 disables pruning.
        """
        if self.max_files <= 0:
            return 0
        files = self.get_replay_files()
        if len(files) <= self.max_files:
            return 0

        deleted = 0
        for f in files[self.max_files:]:
            if f == self._current_file:
                continue                     # never unlink the open writer
            try:
                f.unlink()
                deleted += 1
            except OSError:
                continue
            # Drop the parsed copy too, otherwise the cache keeps the entries
            # of a file that no longer exists alive for the whole session.
            self._file_cache.pop(f, None)
            self._session_entries.pop(f, None)
            self._session_dicts.pop(f, None)
            self._forget_entry_count(f.name)

        return deleted

    def clear_files(self) -> int:
        """Delete all replay files and clear the file cache. Returns number deleted.

        Call this after loading entries into memory to free disk space and
        prevent re-training on the same data.
        """
        self._close_current()
        files = self.get_replay_files()
        deleted = 0
        for f in files:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass
        self._file_cache.clear()
        self._session_entries.clear()
        self._session_dicts.clear()
        if self._entry_count_cache:
            self._entry_count_cache.clear()
            self._entry_count_dirty = True
        return deleted

    # ------------------------------------------------------------------
    # [Pass 181] Durable per-shard line counts
    # ------------------------------------------------------------------

    def _remember_entry_count(self, name: str, st: os.stat_result, count: int) -> None:
        """Record a shard's physical line count under its (size, mtime_ns)."""
        record = (int(st.st_size), int(st.st_mtime_ns), int(count))
        if self._entry_count_cache.get(name) != record:
            self._entry_count_cache[name] = record
            self._entry_count_dirty = True

    def _forget_entry_count(self, name: str) -> None:
        """Drop a deleted shard's record so the sidecar is pruned on next save."""
        if self._entry_count_cache.pop(name, None) is not None:
            self._entry_count_dirty = True

    def _ensure_entry_count_sidecar_loaded(self) -> None:
        """Merge the persisted sidecar under any records this session made."""
        if self._entry_count_sidecar_loaded:
            return
        self._entry_count_sidecar_loaded = True
        loaded = _load_entry_count_sidecar(self.replay_dir)
        if not loaded:
            self._entry_count_dirty = self._entry_count_dirty or bool(self._entry_count_cache)
            return
        merged = dict(loaded)
        merged.update(self._entry_count_cache)
        if merged != loaded:
            self._entry_count_dirty = True
        self._entry_count_cache = merged

    def _persist_entry_counts(self, live_names) -> None:
        """Write the sidecar when it may differ from memory; prune dead shards."""
        if not self._entry_count_dirty:
            return
        live = {
            name: record for name, record in self._entry_count_cache.items()
            if name in live_names
        }
        if _save_entry_count_sidecar(self.replay_dir, live):
            self._entry_count_cache = live
            self._entry_count_dirty = False

    def count_entries(self) -> int:
        """Count total entries across all files.

        Uses cached entry counts where available (session cache, file cache,
        then the durable per-shard line-count cache) and reads only shards
        whose (size, mtime_ns) identity has never been counted.  A shard is
        therefore read at most once in its lifetime, instead of once per
        self-play cycle for every file not written by this session.
        """
        files = self.get_replay_files()
        if not files:
            return 0

        self._ensure_entry_count_sidecar_loaded()
        open_file = self._current_file if self._current_writer is not None else None

        total = 0
        live_names = set()
        uncached_files = []  # (path, size, mtime_ns)
        for f in files:
            live_names.add(f.name)
            # Check session entries first (not yet promoted to file cache).
            # The open writer's shard is counted here; its identity is still
            # changing, so it is never persisted until _close_current().
            session_count = len(self._session_entries.get(f, ()))
            session_count += len(self._session_dicts.get(f, ()))
            if session_count > 0:
                total += session_count
                continue
            try:
                st = f.stat()
            except OSError:
                # Rotated away between the glob and the stat: not live.
                live_names.discard(f.name)
                continue
            # Check file cache (promoted after close)
            cached = self._file_cache.get(f)
            if cached is not None and st.st_mtime == cached[0]:
                total += len(cached[1])
                continue
            if f == open_file:
                continue
            persisted = self._entry_count_cache.get(f.name)
            if (persisted is not None
                    and persisted[0] == st.st_size
                    and persisted[1] == st.st_mtime_ns):
                total += persisted[2]
                continue
            uncached_files.append((f, st.st_size, st.st_mtime_ns))

        if uncached_files:
            with ThreadPoolExecutor(max_workers=min(8, len(uncached_files))) as executor:
                futures = {
                    executor.submit(_count_replay_lines, p): (p, size, mtime_ns)
                    for p, size, mtime_ns in uncached_files
                }
                for future in as_completed(futures):
                    path, size, mtime_ns = futures[future]
                    try:
                        count = future.result()
                    except Exception as e:
                        print(f"  Warning: failed to count replay file {path}: {e}")
                        continue
                    total += count
                    # Persist only when the shard is provably the one that
                    # was read: a changed identity means the count is stale.
                    try:
                        st = path.stat()
                    except OSError:
                        continue
                    if st.st_size == size and st.st_mtime_ns == mtime_ns:
                        self._remember_entry_count(path.name, st, count)

        self._persist_entry_counts(live_names)
        return total

    def iterate_entries(self, shuffle_files: bool = True) -> Iterator[ReplayEntry]:
        """Iterate over all entries in all files."""
        files = self.get_replay_files()

        if shuffle_files:
            random.shuffle(files)

        for filepath in files:
            skipped = 0
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = _json_loads(line)
                            entry = ReplayEntry.from_dict(data)
                        except _PARSE_ERRORS:
                            skipped += 1
                            continue
                        yield entry
            if skipped:
                print(f"  Warning: skipped {skipped} corrupt line(s) in replay file {filepath}")

    def _load_file_cached(self, path: Path) -> List[ReplayEntry]:
        """Load entries from a single file, using mtime cache to skip unchanged files."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return []

        cached = self._file_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        # Cache miss — parse from disk.
        entries = []
        skipped = 0
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(ReplayEntry.from_dict(_json_loads(line)))
                    except _PARSE_ERRORS:
                        skipped += 1
        if skipped:
            print(f"  Warning: skipped {skipped} corrupt line(s) in replay file {path}")
        self._file_cache[path] = (mtime, entries)
        return entries

    def load_all_entries(self) -> List[ReplayEntry]:
        """Load all entries from all files in parallel (single-pass).

        Uses an mtime-based cache: unchanged files are returned from memory
        instantly, only new/modified files are re-parsed from disk.
        """
        files = self.get_replay_files()
        if not files:
            return []

        # Prune cache: remove entries for files that no longer exist.
        live_set = set(files)
        for stale in list(self._file_cache.keys()):
            if stale not in live_set:
                del self._file_cache[stale]

        # Separate cached (instant) from uncached (need I/O).
        uncached_files = []
        all_entries: List[ReplayEntry] = []
        for f in files:
            cached = self._file_cache.get(f)
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            if cached is not None and cached[0] == mtime:
                all_entries.extend(cached[1])
            else:
                uncached_files.append(f)

        if uncached_files:
            # Parallel load only the files that changed.
            num_workers = min(16, max(1, len(uncached_files)))
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self._load_file_cached, p): p for p in uncached_files}
                for future in as_completed(futures):
                    try:
                        all_entries.extend(future.result())
                    except Exception as e:
                        print(f"  Warning: failed to load replay file {futures[future]}: {e}")

        return all_entries

    def sample_entries(self, n: int) -> List[ReplayEntry]:
        """Sample n random entries from the buffer.

        Uses single-pass bulk loading when n is large relative to total
        entries (avoids the separate counting pass). Falls back to
        index-based sampling for selective reads when n << total.
        """
        files = self.get_replay_files()
        if not files:
            return []

        # Estimate total entries from file sizes (~500 bytes per JSONL line).
        # This avoids a full file scan just for counting.
        estimated_total = 0
        file_sizes = []
        for f in files:
            try:
                sz = f.stat().st_size
                file_sizes.append((f, sz))
                estimated_total += sz
            except OSError:
                file_sizes.append((f, 0))
        avg_line_bytes = 500  # conservative estimate
        estimated_entries = max(1, estimated_total // avg_line_bytes)

        # If we need ≥40% of estimated entries, load all in one pass then subsample.
        # One pass (load all + random.sample) is faster than two passes (count + selective read)
        # because it avoids re-reading files and leverages parallel I/O.
        if n >= estimated_entries * 0.4:
            all_entries = self.load_all_entries()
            if len(all_entries) <= n:
                return all_entries
            return random.sample(all_entries, n)

        # For small n relative to total, use the two-pass approach:
        # count lines (fast, no JSON parsing) then selectively load sampled indices.
        def _count_file(path: Path) -> int:
            with open(path, 'r') as f:
                return sum(1 for _ in f)

        file_counts: dict = {}  # path → count, preserves file order below
        with ThreadPoolExecutor(max_workers=min(8, len(files))) as executor:
            futures = {executor.submit(_count_file, p): p for p in files}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    count = future.result()
                except Exception as e:
                    print(f"  Warning: failed to count entries in {path}: {e}")
                    count = 0
                file_counts[path] = count

        total = sum(file_counts.values())
        if total == 0:
            return []

        n = min(n, total)

        # Sample indices
        indices = set(random.sample(range(total), n))

        # Collect entries
        entries: List[ReplayEntry] = []
        current_idx = 0
        tasks = []

        def _load_entries(path: Path, indices_set: set) -> List[ReplayEntry]:
            if not indices_set:
                return []
            loaded = []
            skipped = 0
            with open(path, 'r') as f:
                # Enumerate every physical line (blank or corrupt included) so
                # indices stay aligned with _count_file's line counts.
                for i, line in enumerate(f):
                    if i in indices_set and line.strip():
                        try:
                            data = _json_loads(line)
                            loaded.append(ReplayEntry.from_dict(data))
                        except _PARSE_ERRORS:
                            skipped += 1
            if skipped:
                print(f"  Warning: skipped {skipped} corrupt line(s) in replay file {path}")
            return loaded

        # Iterate in original file order (sorted by mtime from get_replay_files)
        # so that index offsets are deterministic regardless of thread completion order.
        for filepath in files:
            count = file_counts[filepath]
            file_indices = set(
                i - current_idx for i in indices
                if current_idx <= i < current_idx + count
            )
            tasks.append((filepath, file_indices))
            current_idx += count

        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
            futures = {executor.submit(_load_entries, p, idxs): p for p, idxs in tasks}
            for future in as_completed(futures):
                try:
                    entries.extend(future.result())
                except Exception as e:
                    print(f"  Warning: failed to sample replay file {futures[future]}: {e}")

        return entries

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
