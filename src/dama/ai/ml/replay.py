"""Replay buffer management for training data."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Iterator, Optional, Dict, Any
from dataclasses import dataclass
import random


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
        if legal_moves and chosen_index >= len(legal_moves):
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

        line = json.dumps(entry.to_dict())
        self._current_writer.write(line + '\n')

    def add_entries(self, entries: List[ReplayEntry]) -> None:
        """Add multiple entries and flush once."""
        if self._current_writer is None:
            self.start_new_file()
        for entry in entries:
            line = json.dumps(entry.to_dict())
            self._current_writer.write(line + '\n')
        self._current_writer.flush()

    def _close_current(self) -> None:
        """Close the current file."""
        if self._current_writer is not None:
            self._current_writer.close()
            self._current_writer = None
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

    def count_entries(self) -> int:
        """Count total entries across all files."""
        files = self.get_replay_files()
        if not files:
            return 0

        def _count_file(path: Path) -> int:
            with open(path, 'r') as f:
                return sum(1 for _ in f)

        total = 0
        with ThreadPoolExecutor(max_workers=min(8, len(files))) as executor:
            futures = [executor.submit(_count_file, p) for p in files]
            for future in as_completed(futures):
                try:
                    total += future.result()
                except Exception:
                    pass
        return total

    def iterate_entries(self, shuffle_files: bool = True) -> Iterator[ReplayEntry]:
        """Iterate over all entries in all files."""
        files = self.get_replay_files()

        if shuffle_files:
            random.shuffle(files)

        for filepath in files:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        yield ReplayEntry.from_dict(data)

    def sample_entries(self, n: int) -> List[ReplayEntry]:
        """Sample n random entries from the buffer."""
        # First, collect file sizes
        files = self.get_replay_files()
        if not files:
            return []

        def _count_file(path: Path) -> int:
            with open(path, 'r') as f:
                return sum(1 for _ in f)

        file_counts = []
        with ThreadPoolExecutor(max_workers=min(8, len(files))) as executor:
            futures = {executor.submit(_count_file, p): p for p in files}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    count = future.result()
                except Exception:
                    count = 0
                file_counts.append((path, count))

        total = sum(c for _, c in file_counts)
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
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i in indices_set and line.strip():
                        data = json.loads(line)
                        loaded.append(ReplayEntry.from_dict(data))
            return loaded

        for filepath, count in file_counts:
            file_indices = set(
                i - current_idx for i in indices
                if current_idx <= i < current_idx + count
            )
            tasks.append((filepath, file_indices))
            current_idx += count

        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
            futures = [executor.submit(_load_entries, p, idxs) for p, idxs in tasks]
            for future in as_completed(futures):
                try:
                    entries.extend(future.result())
                except Exception:
                    pass

        return entries

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
