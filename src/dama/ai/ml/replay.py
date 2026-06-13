"""Replay buffer management for training data."""

import os
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


@dataclass
class ReplayEntry:
    """A single training example from a game."""
    state: dict           # Compact state representation
    legal_moves: list     # List of move dicts
    chosen_index: int     # Index of chosen move
    result: int           # Game result from this player's perspective (+1, -1, 0)
    score: float = 0.0    # Detailed shaped reward score (from scoring system)

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
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ReplayEntry':
        legal_moves = data['legal_moves']
        chosen_index = data['chosen_index']
        if legal_moves and (chosen_index < 0 or chosen_index >= len(legal_moves)):
            raise ValueError(
                f"chosen_index {chosen_index} out of bounds for {len(legal_moves)} legal moves"
            )
        return cls(
            state=data['state'],
            legal_moves=legal_moves,
            chosen_index=chosen_index,
            result=data.get('result', 0),
            score=data.get('score', 0.0),
        )


class ReplayBuffer:
    """
    Disk-backed replay buffer for training data.

    Stores replay data as JSONL files in the replay directory.
    """

    def __init__(self, replay_dir: str = "data/replay", max_files: int = 100):
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
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

    def start_new_file(self) -> Path:
        """Start a new replay file."""
        self._close_current()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"replay_{timestamp}.jsonl"
        filepath = self.replay_dir / filename

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
            # skips re-parsing the file we just wrote.
            path = self._current_file
            if path is not None:
                # Convert any deferred dicts to ReplayEntry now (bulk conversion)
                deferred = self._session_dicts.pop(path, None)
                if deferred:
                    entries = [ReplayEntry.from_dict(d) for d in deferred]
                    self._session_entries.setdefault(path, []).extend(entries)
                if path in self._session_entries:
                    try:
                        mtime = path.stat().st_mtime
                        self._file_cache[path] = (mtime, self._session_entries.pop(path))
                    except OSError:
                        self._session_entries.pop(path, None)
            self._current_file = None

    def close(self) -> None:
        """Close the buffer."""
        self._close_current()

    def get_replay_files(self) -> List[Path]:
        """Get all replay files, sorted by modification time (newest first)."""
        files = list(self.replay_dir.glob("replay_*.jsonl"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def cleanup_old_files(self) -> int:
        """Remove old files beyond max_files limit. Returns number deleted."""
        files = self.get_replay_files()
        if len(files) <= self.max_files:
            return 0

        to_delete = files[self.max_files:]
        for f in to_delete:
            f.unlink()

        return len(to_delete)

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
        return deleted

    def count_entries(self) -> int:
        """Count total entries across all files.

        Uses cached entry counts where available (file cache + session cache)
        and only reads uncached files from disk. ~500 bytes/entry estimate for
        files not yet loaded.
        """
        files = self.get_replay_files()
        if not files:
            return 0

        total = 0
        uncached_files = []
        for f in files:
            # Check session entries first (not yet promoted to file cache)
            session_count = len(self._session_entries.get(f, ()))
            session_count += len(self._session_dicts.get(f, ()))
            if session_count > 0:
                total += session_count
                continue
            # Check file cache (promoted after close)
            cached = self._file_cache.get(f)
            if cached is not None:
                try:
                    if f.stat().st_mtime == cached[0]:
                        total += len(cached[1])
                        continue
                except OSError:
                    pass
            uncached_files.append(f)

        if uncached_files:
            def _count_file(path: Path) -> int:
                with open(path, 'r') as fh:
                    return sum(1 for _ in fh)

            with ThreadPoolExecutor(max_workers=min(8, len(uncached_files))) as executor:
                futures = {executor.submit(_count_file, p): p for p in uncached_files}
                for future in as_completed(futures):
                    try:
                        total += future.result()
                    except Exception as e:
                        print(f"  Warning: failed to count replay file {futures[future]}: {e}")
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
