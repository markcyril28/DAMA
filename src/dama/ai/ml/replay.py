"""Replay buffer management for training data."""

import errno
import json
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Iterator, Optional, Dict, Any

from .run_status import _fsync_directory

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
_ENTRY_COUNT_SIDECAR_LOCK_NAME = '.entry_count_cache.lock'
_ENTRY_COUNT_SIDECAR_LOCK_TIMEOUT_SECONDS = 0.25
_ENTRY_COUNT_SIDECAR_LOCK_POLL_SECONDS = 0.05

# Snapshot-mode self-play writes one complete cycle before the corpus manager
# can inspect it.  Buffer that cycle in bounded chunks instead of forcing each
# completed worker batch through drvfs immediately.  Legacy ReplayBuffer users
# retain the historical per-batch flush behavior.
_SNAPSHOT_WRITE_BUFFER_BYTES = 16 * 1024 * 1024


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
    """Persist per-shard line counts durably; report whether committed.

    Always temp file + file fsync + ``os.replace`` + directory fsync: an
    in-place rewrite would go through a shared inode when the replay directory
    is a hardlink copy of another one (probes build such copies), and neither a
    torn write nor a rename lost after sudden host failure may be acknowledged.
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
        _fsync_directory(replay_dir)
        return True
    except OSError:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        return False


@contextmanager
def _entry_count_sidecar_lock(replay_dir: Path):
    """Serialize sidecar merges, abandoning this optional cache if stalled."""
    lock_path = replay_dir / _ENTRY_COUNT_SIDECAR_LOCK_NAME
    with lock_path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'\0')
            stream.flush()

        deadline = (
            time.monotonic() + _ENTRY_COUNT_SIDECAR_LOCK_TIMEOUT_SECONDS
        )
        if os.name == 'nt':
            import msvcrt

            def acquire():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            retry_errnos = (errno.EACCES, errno.EAGAIN, errno.EDEADLK)
        else:
            import fcntl

            def acquire():
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            retry_errnos = (errno.EACCES, errno.EAGAIN)

        while True:
            try:
                acquire()
                break
            except OSError as exc:
                if exc.errno not in retry_errnos:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        errno.ETIMEDOUT,
                        f"timed out locking optional replay cache {lock_path}",
                    ) from exc
                time.sleep(min(
                    _ENTRY_COUNT_SIDECAR_LOCK_POLL_SECONDS,
                    remaining,
                ))

        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    # The cache remains fail-open even if a host invalidates
                    # the byte-range lock while unwinding an I/O failure.
                    pass
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


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
        # A shard close fsyncs this directory, but directory fsync is not
        # recursive: it cannot commit a newly created replay namespace's name
        # in its parent.  Commit that name before any writer can publish the
        # first cycle, and retry the boundary on later constructions in case a
        # prior parent sync failed after mkdir() became visible.  See Journal
        # Pass 426.
        _fsync_directory(self.replay_dir.parent)
        self.max_files = max_files
        # Legacy training reads replay through load_all_entries(), where keeping
        # newly written entries avoids an immediate JSON parse.  Snapshot-mode
        # training reads the immutable shard through CorpusSnapshotManager and
        # never consumes this cache, so retaining each cycle's complete Python
        # object graph only spends session RAM.  See Journal Pass 183.
        self._cache_written_entries = bool(cache_written_entries)
        # ``cache_written_entries=False`` is the snapshot-writer mode: the
        # closed shard is consumed by CorpusSnapshotManager rather than by a
        # concurrent ReplayBuffer reader.  Closing the cycle remains the flush
        # boundary, while a bounded buffer coalesces the batch writes on drvfs.
        self._buffer_snapshot_cycle = not self._cache_written_entries
        self._current_file = None
        # Snapshot-mode cycles stay under an ignored dotfile until close().
        # ``_current_file`` is the eventual immutable public name so every
        # cache and caller keeps using the established replay basename.
        self._current_staging_file = None
        self._current_writer = None
        # Incremental file cache: {path: (mtime, [ReplayEntry, ...])}
        # Avoids re-parsing unchanged JSONL files across epochs.
        self._file_cache: Dict[Path, tuple] = {}
        # In-memory entries written during the current session, keyed by file path.
        # Promoted to _file_cache on close() so the next load_all_entries() skips
        # re-parsing the file we just wrote.
        self._session_entries: Dict[Path, List[ReplayEntry]] = {}
        # Legacy raw dicts from add_entry_dicts() defer ReplayEntry creation to
        # close(), then merge into _session_entries for the parsed-file cache.
        self._session_dicts: Dict[Path, List[dict]] = {}
        # Snapshot writers validate each batch before writing and need only its
        # exact line count after that point. Retaining the complete nested dict
        # graph until cycle close spends scarce trainer RAM for no reader.
        self._session_entry_counts: Dict[Path, int] = {}
        # [Pass 181] Physical line count per shard basename, keyed by
        # (size, mtime_ns, count) and persisted to _ENTRY_COUNT_SIDECAR_NAME so
        # count_entries() never re-reads an unchanged write-once shard.  On
        # drvfs the 60-file window cost ~14 s of reads per self-play cycle
        # for a statistic.  The sidecar is loaded lazily on first use.
        self._entry_count_cache: Dict[str, tuple] = {}
        self._entry_count_sidecar_loaded = False
        self._entry_count_dirty = False
        # Buffer telemetry, replay rotation, and corpus admission run
        # back-to-back after every persisted snapshot cycle. Retain that one
        # point-in-time identity snapshot so both consumers can avoid
        # restatting the same immutable shards. The handoff is one-shot and is
        # accepted only while the directory identity and exact replay-name set
        # remain unchanged.
        self._cleanup_file_stats: Optional[tuple] = None

    def start_new_file(self) -> Path:
        """Start a new replay file."""
        self._cleanup_file_stats = None
        self._close_current()

        # [Pass 109] The name has only second resolution. Every cycle persists,
        # so claim the selected name with exclusive creation: an exists-then-
        # open('w') sequence lets concurrent writers both choose and truncate
        # the same shard. FileExistsError advances to the established suffix.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = 0
        while True:
            stem = f"replay_{timestamp}"
            if suffix:
                stem += f"_{suffix:02d}"
            filepath = self.replay_dir / f"{stem}.jsonl"
            staging_path = filepath.with_name(f".{filepath.name}.pending")
            try:
                if self._buffer_snapshot_cycle:
                    writer = open(
                        staging_path,
                        'x',
                        encoding='utf-8',
                        buffering=_SNAPSHOT_WRITE_BUFFER_BYTES,
                    )
                    # A complete shard with this name can coexist with an
                    # orphaned staging claim after a host failure. Never
                    # replace it: discard this claim and advance the suffix.
                    if filepath.exists():
                        writer.close()
                        writer = None
                        try:
                            staging_path.unlink()
                        except OSError:
                            pass
                        suffix += 1
                        continue
                else:
                    writer = open(filepath, 'x')
            except FileExistsError:
                suffix += 1
                continue
            break

        self._current_file = filepath
        self._current_staging_file = (
            staging_path if self._buffer_snapshot_cycle else None
        )
        self._current_writer = writer

        return filepath

    def add_entry(self, entry: ReplayEntry) -> None:
        """Add an entry to the current replay file."""
        if self._current_writer is None:
            self.start_new_file()

        line = _json_dumps(entry.to_dict())
        self._current_writer.write(line + '\n')
        if self._buffer_snapshot_cycle:
            self._session_entry_counts[self._current_file] = (
                self._session_entry_counts.get(self._current_file, 0) + 1
            )
        else:
            self._session_entries.setdefault(self._current_file, []).append(entry)

    def add_entries(self, entries: List[ReplayEntry]) -> None:
        """Add multiple entries, flushing immediately for legacy readers."""
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
        if not self._buffer_snapshot_cycle:
            self._current_writer.flush()
            # Legacy readers promote these objects to the parsed file cache.
            self._session_entries.setdefault(self._current_file, []).extend(entries)
        else:
            self._session_entry_counts[self._current_file] = (
                self._session_entry_counts.get(self._current_file, 0)
                + len(entries)
            )

    def add_entry_dicts(self, dicts: List[dict]) -> None:
        """Add entries from raw dicts — avoids dict→ReplayEntry→dict round-trip.

        Self-play workers already return dicts (serialized for IPC). Writing
        them directly to JSONL skips one to_dict() call per entry. Legacy mode
        defers ReplayEntry conversion until close. Snapshot mode validates each
        batch before writing, retains only its count, and defers the buffered
        flush until the cycle closes.
        """
        if not dicts:
            return
        if self._buffer_snapshot_cycle:
            # Validate the exact objects about to be serialized, before any
            # bytes from this batch can enter the durable shard. Constructed
            # wrappers are released immediately instead of keeping the whole
            # cycle's nested dictionary graph alive until close().
            for record in dicts:
                ReplayEntry.from_dict(record)
        if self._current_writer is None:
            self.start_new_file()
        lines = [_json_dumps(d) for d in dicts]
        self._current_writer.write('\n'.join(lines) + '\n')
        if not self._buffer_snapshot_cycle:
            self._current_writer.flush()
            # Store raw dicts — ReplayEntry creation deferred to _close_current().
            self._session_dicts.setdefault(self._current_file, []).extend(dicts)
        else:
            self._session_entry_counts[self._current_file] = (
                self._session_entry_counts.get(self._current_file, 0)
                + len(dicts)
            )

    def _close_current(self) -> None:
        """Close the current file and promote session entries to file cache."""
        if self._current_writer is not None:
            close_error = None
            publication_error = None
            writer = self._current_writer
            try:
                if self._buffer_snapshot_cycle:
                    # The hidden shard is the only complete copy of this
                    # generation cycle.  Make its bytes durable before the
                    # atomic public hardlink can expose it to corpus scans.
                    writer.flush()
                    os.fsync(writer.fileno())
            except OSError as exc:
                close_error = exc
            try:
                writer.close()
            except OSError as exc:
                if close_error is None:
                    close_error = exc
            finally:
                self._current_writer = None
            if close_error is not None and self._buffer_snapshot_cycle:
                # Deferred snapshot writes can surface an I/O failure only at
                # the cycle boundary. Fail closed and quarantine that exact
                # shard instead of publishing its expected entry count for a
                # short or malformed file. Legacy callers retain their prior
                # best-effort close behavior.
                # Reuse the incomplete-cycle removal path so an immediate
                # unlink refusal moves the shard out of the active replay
                # namespace instead of leaving it visible to corpus scans.
                self.discard_current_file()
                raise close_error
            if self._buffer_snapshot_cycle:
                # The closed staging file is complete, but it is still absent
                # from every replay/corpus glob. Publish with link(2), whose
                # destination creation is atomic and refuses to overwrite an
                # unexpectedly colliding shard. Unlinking the hidden name then
                # leaves one immutable public inode without a visibility gap.
                staging_path = self._current_staging_file
                final_path = self._current_file
                if staging_path is None or final_path is None:
                    raise RuntimeError(
                        "snapshot replay writer lost its publication paths"
                    )
                try:
                    os.link(staging_path, final_path)
                except OSError:
                    self.discard_current_file()
                    raise
                try:
                    staging_path.unlink()
                except OSError:
                    # The public shard is already complete and immutable. A
                    # hidden extra hardlink consumes no additional data blocks
                    # and remains ignored by all replay readers.
                    pass
                self._current_staging_file = None
                try:
                    # Commit both the public link and best-effort removal of
                    # the hidden staging name.  On native Windows the shared
                    # helper retains the established atomic-link fallback.
                    _fsync_directory(self.replay_dir)
                except OSError as exc:
                    # The complete public shard may already be visible after
                    # a directory-sync failure. Finish the in-memory close so
                    # a retry cannot overwrite its bookkeeping, then report
                    # the durability failure after cleanup below.
                    publication_error = exc
            # Promote in-memory entries to file cache so load_all_entries()
            # skips re-parsing the file we just wrote.  Snapshot-mode callers
            # opt out: their batches were already validated before writing, so
            # remember only the exact physical line count after the durable
            # shard closes. CorpusSnapshotManager is that mode's reader.
            path = self._current_file
            if path is not None:
                tracked_count = self._session_entry_counts.pop(path, 0)
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
                    entry_count = (
                        tracked_count
                        + len(deferred or ())
                        + len(entries or ())
                    )
                    if entry_count:
                        try:
                            st = path.stat()
                            self._remember_entry_count(
                                path.name, st, entry_count)
                        except OSError:
                            pass
            self._current_file = None
            if publication_error is not None:
                raise publication_error

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
        self._cleanup_file_stats = None
        path = self._current_file
        staging_path = self._current_staging_file
        if self._current_writer is not None:
            try:
                self._current_writer.close()
            except OSError:
                pass
            finally:
                self._current_writer = None
        self._current_file = None
        self._current_staging_file = None
        if path is None:
            return None
        removal_path = staging_path or path
        try:
            removal_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Some hosts can refuse an unlink immediately after the buffered
            # writer closes.  Preserve the bytes for diagnosis, but move them
            # out of the replay_*.jsonl namespace so buffer statistics and
            # corpus scans cannot treat an incomplete cycle as active data.
            quarantine = path.with_name(f".{path.name}.incomplete")
            suffix = 0
            while quarantine.exists():
                suffix += 1
                quarantine = path.with_name(
                    f".{path.name}.incomplete.{suffix}"
                )
            try:
                removal_path.rename(quarantine)
            except OSError:
                if staging_path is None:
                    # A legacy writer's file is still in the public namespace.
                    # Retain its bookkeeping so count_entries() reports the
                    # physical diagnostic shard rather than silently
                    # disagreeing with the filesystem.
                    return path
                # A snapshot staging file remains hidden even when both
                # cleanup operations fail. Drop its integer bookkeeping: no
                # replay reader can observe or count that diagnostic dotfile.
        self._file_cache.pop(path, None)
        self._session_entries.pop(path, None)
        self._session_dicts.pop(path, None)
        self._session_entry_counts.pop(path, None)
        self._forget_entry_count(path.name)
        return path

    def get_replay_files(self) -> List[Path]:
        """Get all replay files, sorted by modification time (newest first)."""
        # Reuse the DirEntry metadata captured by the shared one-scan helper.
        # Path.glob() followed by Path.stat() makes every shard pay a separate
        # pathname lookup on DrvFS, even though scandir already owns the entry.
        file_stats = self._replay_file_stats(strict=True)
        file_stats.sort(key=lambda item: item[1].st_mtime, reverse=True)
        return [path for path, _stat in file_stats]

    def cleanup_old_files(self) -> int:
        """Remove old files beyond max_files limit. Returns number deleted.

        [Pass 109] Now actually called, once per persisted self-play cycle.
        Persisting every cycle writes roughly one 7MB file per minute, so
        without this the replay directory grows without bound over a 48h run.
        max_files <= 0 disables pruning.
        """
        if self.max_files <= 0:
            self._cleanup_file_stats = None
            return 0
        file_stats = self._take_cleanup_file_stats()
        if file_stats is None:
            file_stats = self._replay_file_stats(strict=True)
        file_stats.sort(key=lambda item: item[1].st_mtime, reverse=True)
        if len(file_stats) <= self.max_files:
            # Corpus admission immediately follows cleanup in snapshot mode.
            # Restage the unchanged identities so that transaction can avoid
            # another complete DrvFS stat pass.
            self._stage_cleanup_file_stats(file_stats)
            return 0

        deleted = 0
        retained_stats = list(file_stats[:self.max_files])
        for f, stat_result in file_stats[self.max_files:]:
            if f == self._current_file:
                retained_stats.append((f, stat_result))
                continue                     # never unlink the open writer
            try:
                f.unlink()
                deleted += 1
            except OSError:
                retained_stats.append((f, stat_result))
                continue
            # Drop the parsed copy too, otherwise the cache keeps the entries
            # of a file that no longer exists alive for the whole session.
            self._file_cache.pop(f, None)
            self._session_entries.pop(f, None)
            self._session_dicts.pop(f, None)
            self._session_entry_counts.pop(f, None)
            self._forget_entry_count(f.name)

        if deleted:
            # Snapshot admission consumes the reduced replay window immediately
            # after rotation.  Commit the batched unlink operations before
            # handing that view to the corpus manager, otherwise an abrupt host
            # loss can resurrect shards that cleanup reported as pruned.  One
            # directory sync covers every successful deletion in this batch.
            # See Journal Pass 433.
            _fsync_directory(self.replay_dir)

        # Capture the post-prune directory identity. Corpus validates it and
        # the exact replay-name set before accepting these immutable stats;
        # any failed deletion or concurrent publisher therefore falls back to
        # its ordinary full scan.
        self._stage_cleanup_file_stats(retained_stats)
        return deleted

    def clear_files(self) -> int:
        """Delete all replay files and clear the file cache. Returns number deleted.

        Call this after loading entries into memory to free disk space and
        prevent re-training on the same data.
        """
        self._cleanup_file_stats = None
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
        self._session_entry_counts.clear()
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
        """Merge and write the sidecar; prune dead shards without lost updates."""
        if not self._entry_count_dirty:
            return
        # Atomic replace prevents torn readers, but it is not a compare-and-swap:
        # two ReplayBuffer instances can otherwise each replace the complete map
        # with only their own newly closed shard. Merge while holding a narrow
        # process lock so concurrent standalone generators retain both records.
        try:
            with _entry_count_sidecar_lock(self.replay_dir):
                # The caller's point-in-time scan may predate a competing
                # writer's new shard. Refresh names under the publication lock
                # so merging its durable record does not immediately prune it.
                try:
                    with os.scandir(self.replay_dir) as directory:
                        current_live_names = {
                            entry.name
                            for entry in directory
                            if entry.name.startswith('replay_')
                            and entry.name.endswith('.jsonl')
                        }
                except OSError:
                    current_live_names = live_names
                live = {
                    name: record
                    for name, record in _load_entry_count_sidecar(
                        self.replay_dir).items()
                    if name in current_live_names
                }
                live.update({
                    name: record
                    for name, record in self._entry_count_cache.items()
                    if name in current_live_names
                })
                if _save_entry_count_sidecar(self.replay_dir, live):
                    self._entry_count_cache = live
                    self._entry_count_dirty = False
        except OSError:
            # The sidecar is a performance cache. Read-only directories and
            # unavailable host locking must never turn an exact replay count
            # into a training-session failure.
            return

    def _replay_file_stats(self, *, strict: bool = False) -> List[tuple]:
        """Capture each replay shard and its identity in one directory scan.

        Buffer telemetry remains best-effort, while public replay listing uses
        strict mode to preserve its fail-closed behavior for inaccessible
        directories or shard identities.
        """
        records = []
        try:
            directory = os.scandir(self.replay_dir)
        except FileNotFoundError:
            return []
        except OSError:
            if strict:
                raise
            return []
        with directory:
            for entry in directory:
                name = entry.name
                if not (name.startswith('replay_') and name.endswith('.jsonl')):
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    if strict:
                        raise
                    # A shard rotated between enumeration and stat is not
                    # part of this point-in-time telemetry view.
                    continue
                records.append((Path(entry.path), stat))
        return records

    @staticmethod
    def _directory_identity(path: Path) -> tuple:
        """Return the fields that change when a directory entry changes."""
        stat = os.stat(path)
        return (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )

    def _stage_cleanup_file_stats(self, file_stats: List[tuple]) -> None:
        """Offer one immutable telemetry scan to the next cleanup call."""
        self._cleanup_file_stats = None
        if not self._buffer_snapshot_cycle or self._current_writer is not None:
            # Only snapshot-mode public shards carry the write-once contract.
            # Legacy callers can modify visible shards outside this instance,
            # so they retain the ordinary strict cleanup identity scan.
            return
        try:
            identity = self._directory_identity(self.replay_dir)
        except OSError:
            return
        self._cleanup_file_stats = (identity, list(file_stats))

    def _take_cleanup_file_stats(self) -> Optional[List[tuple]]:
        """Consume a still-current telemetry scan or request strict fallback.

        Public replay shards are write-once after close.  A stable replay
        directory identity plus the exact same shard-name set therefore proves
        that their captured modification times still define the cleanup order.
        Concurrent publication, rotation, a partial best-effort telemetry scan,
        or any directory race rejects the handoff and makes the caller perform
        the established strict identity scan.
        """
        cached = self._cleanup_file_stats
        self._cleanup_file_stats = None
        if cached is None:
            return None
        expected_identity, file_stats = cached
        try:
            before = self._directory_identity(self.replay_dir)
            with os.scandir(self.replay_dir) as directory:
                live_names = {
                    entry.name
                    for entry in directory
                    if entry.name.startswith('replay_')
                    and entry.name.endswith('.jsonl')
                }
            after = self._directory_identity(self.replay_dir)
        except FileNotFoundError:
            return None
        expected_names = {path.name for path, _stat in file_stats}
        if (
            before != expected_identity
            or after != before
            or live_names != expected_names
        ):
            return None
        return file_stats

    def take_replay_file_stats_handoff(self) -> Optional[tuple]:
        """Consume the post-cleanup identity snapshot for corpus admission.

        The corpus manager, not this producer, validates the recorded
        directory identity and complete replay-name set. Keeping that proof at
        the consumer makes a stale or malformed handoff an ordinary cache miss
        while avoiding a second validation scan here.
        """

        cached = self._cleanup_file_stats
        self._cleanup_file_stats = None
        return cached

    def _count_entries_from_stats(self, file_stats: List[tuple]) -> int:
        """Count entries using an already captured shard identity snapshot.

        Uses cached entry counts where available (session cache, file cache,
        then the durable per-shard line-count cache) and reads only shards
        whose (size, mtime_ns) identity has never been counted.  A shard is
        therefore read at most once in its lifetime, instead of once per
        self-play cycle for every file not written by this session.
        """
        if not file_stats:
            return 0

        self._ensure_entry_count_sidecar_loaded()
        open_file = self._current_file if self._current_writer is not None else None

        total = 0
        live_names = set()
        uncached_files = []  # (path, size, mtime_ns)
        for f, st in file_stats:
            live_names.add(f.name)
            # Check session entries first (not yet promoted to file cache).
            # The open writer's shard is counted here; its identity is still
            # changing, so it is never persisted until _close_current().
            session_count = len(self._session_entries.get(f, ()))
            session_count += len(self._session_dicts.get(f, ()))
            session_count += self._session_entry_counts.get(f, 0)
            if session_count > 0:
                total += session_count
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

    def get_buffer_state(self) -> tuple[int, int, int]:
        """Return exact ``(entries, files, bytes)`` from one metadata scan.

        The self-play statistics path needs all three values after every
        completed cycle.  Capturing shard identities once avoids sorting and
        then restatting the same rolling window for each aggregate.
        """
        file_stats = self._replay_file_stats()
        # Snapshot staging is deliberately invisible to get_replay_files(),
        # corpus admission, and other processes. Preserve this instance's
        # historical open-cycle statistics by representing its staging
        # identity under the eventual public path for the local count pass.
        if (self._current_writer is not None
                and self._current_file is not None
                and self._current_staging_file is not None):
            try:
                staging_stat = self._current_staging_file.stat()
            except OSError:
                pass
            else:
                file_stats.append((self._current_file, staging_stat))
        if not file_stats:
            self._stage_cleanup_file_stats(file_stats)
            return 0, 0, 0
        total_entries = self._count_entries_from_stats(file_stats)
        total_bytes = sum(stat.st_size for _, stat in file_stats)
        # Stage only after count-sidecar publication, which can itself replace
        # a name in this directory.  Cleanup validates this final identity and
        # the replay-only name set before reusing any captured shard metadata.
        self._stage_cleanup_file_stats(file_stats)
        return total_entries, len(file_stats), total_bytes

    def count_entries(self) -> int:
        """Count total entries across all replay shards."""
        return self.get_buffer_state()[0]

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
