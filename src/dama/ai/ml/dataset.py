"""Dataset for training the move scorer model."""

import os
import random
import psutil
from pathlib import Path
from typing import List, Tuple, Iterator, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset

from ...types import Move, Player
from ...game_state import GameState
from ...board import Board
from .replay import ReplayBuffer, ReplayEntry
from .move_encoder import encode_board, encode_moves, MOVE_FEATURE_SIZE, BOARD_PLANES
from .scoring import compute_reward_weight


def get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except Exception:
        return 0.0


def get_total_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        mem = psutil.virtual_memory()
        return mem.total / (1024 ** 3)
    except Exception:
        return 0.0


class DamaDataset(Dataset):
    """
    PyTorch dataset for training data.

    Each item returns:
    - board: (BOARD_PLANES, 8, 8) tensor
    - move_features: (num_moves, MOVE_FEATURE_SIZE) tensor
    - target: int (index of chosen move)
    - reward_weight: float (reward-based weight for loss)
    """

    def __init__(self, entries: List[ReplayEntry]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, float, float]:
        entry = self.entries[idx]

        # Reconstruct game state
        state = GameState.from_compact(entry.state)

        # Encode board
        board = encode_board(state)

        # Reconstruct moves and encode them
        moves = [Move.from_dict(m) for m in entry.legal_moves]
        move_features = encode_moves(state, moves)

        # Compute reward weight from score
        reward_weight = compute_reward_weight(entry.score)

        return (
            torch.from_numpy(board),
            torch.from_numpy(move_features),
            entry.chosen_index,
            reward_weight,
            float(entry.result),  # value target
        )


def preprocess_entries_to_tensors(
    entries: List[ReplayEntry],
    max_moves_per_sample: int = 64,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pre-process replay entries into pre-computed tensors for fast training.
    
    This eliminates the CPU bottleneck of parsing JSON and reconstructing
    GameState objects during training.
    
    Args:
        entries: List of replay entries to process
        max_moves_per_sample: Maximum number of moves to pad to (for fixed-size batching)
        show_progress: Whether to print progress updates
    
    Returns:
        Tuple of:
        - boards: (N, BOARD_PLANES, 8, 8) float32 tensor
        - move_features: (N, max_moves, MOVE_FEATURE_SIZE) float32 tensor (padded)
        - move_counts: (N,) int64 tensor (actual number of moves per sample)
        - targets: (N,) int64 tensor (chosen move index)
        - reward_weights: (N,) float32 tensor (reward-based weights)
        - value_targets: (N,) float32 tensor (game result: +1, -1, 0)
    """
    n = len(entries)
    if n == 0:
        return (
            torch.empty(0, BOARD_PLANES, 8, 8),
            torch.empty(0, max_moves_per_sample, MOVE_FEATURE_SIZE),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
        )
    
    # Pre-allocate arrays
    boards = np.zeros((n, BOARD_PLANES, 8, 8), dtype=np.float32)
    all_move_features = np.zeros((n, max_moves_per_sample, MOVE_FEATURE_SIZE), dtype=np.float32)
    move_counts = np.zeros(n, dtype=np.int64)
    targets = np.zeros(n, dtype=np.int64)
    reward_weights = np.zeros(n, dtype=np.float32)
    value_targets = np.zeros(n, dtype=np.float32)
    
    log_interval = max(1, n // 20)  # Log every 5%
    
    for i, entry in enumerate(entries):
        if show_progress and i > 0 and i % log_interval == 0:
            print(f"  Pre-processing: {i}/{n} ({100*i/n:.1f}%)")
        
        # Reconstruct game state
        state = GameState.from_compact(entry.state)
        
        # Encode board
        boards[i] = encode_board(state)
        
        # Reconstruct moves and encode them
        moves = [Move.from_dict(m) for m in entry.legal_moves]
        move_feats = encode_moves(state, moves)
        
        num_moves = min(move_feats.shape[0], max_moves_per_sample)
        all_move_features[i, :num_moves] = move_feats[:num_moves]
        move_counts[i] = num_moves
        if num_moves > 0:
            if entry.chosen_index >= num_moves:
                print(f"  Warning: chosen_index {entry.chosen_index} out of range for {num_moves} moves (clipped)")
            targets[i] = min(entry.chosen_index, num_moves - 1)
        else:
            print(f"  Warning: entry {i} has zero legal moves — skipping target (move_counts=0 will mask this entry)")
            targets[i] = 0  # masked by move_counts[i] == 0
        reward_weights[i] = compute_reward_weight(entry.score)
        value_targets[i] = float(entry.result)  # +1 win, -1 loss, 0 draw
    
    if show_progress:
        print(f"  Pre-processing complete: {n} entries")
    
    return (
        torch.from_numpy(boards),
        torch.from_numpy(all_move_features),
        torch.from_numpy(move_counts),
        torch.from_numpy(targets),
        torch.from_numpy(reward_weights),
        torch.from_numpy(value_targets),
    )


class CachedTensorDataset(Dataset):
    """
    Dataset backed by pre-computed tensors in RAM.
    
    This is the fastest dataset implementation when you have sufficient RAM
    to hold all training data. Eliminates all CPU preprocessing during training.
    """
    
    def __init__(
        self,
        boards: torch.Tensor,
        move_features: torch.Tensor,
        move_counts: torch.Tensor,
        targets: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
        value_targets: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            boards: (N, BOARD_PLANES, 8, 8) tensor
            move_features: (N, max_moves, MOVE_FEATURE_SIZE) tensor
            move_counts: (N,) tensor with actual move counts
            targets: (N,) tensor with target indices
            reward_weights: (N,) tensor with reward-based weights (optional)
            value_targets: (N,) tensor with game results for value head (optional)
        """
        self.boards = boards
        self.move_features = move_features
        self.move_counts = move_counts
        self.targets = targets
        # Default to uniform weights if not provided (backward compat)
        if reward_weights is not None:
            self.reward_weights = reward_weights
        else:
            self.reward_weights = torch.ones(len(boards), dtype=torch.float32)
        # Default to zero value targets if not provided (backward compat)
        if value_targets is not None:
            self.value_targets = value_targets
        else:
            self.value_targets = torch.zeros(len(boards), dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.boards)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, float, float]:
        """
        Returns:
            - board: (BOARD_PLANES, 8, 8) tensor
            - move_features: (max_moves, MOVE_FEATURE_SIZE) tensor
            - move_count: int (actual number of valid moves)
            - target: int (chosen move index)
            - reward_weight: float (reward-based weight for loss)
            - value_target: float (game result for value head)
        """
        return (
            self.boards[idx],
            self.move_features[idx],
            int(self.move_counts[idx]),
            int(self.targets[idx]),
            float(self.reward_weights[idx]),
            float(self.value_targets[idx]),
        )
    
    @classmethod
    def from_entries(
        cls,
        entries: List[ReplayEntry],
        max_moves_per_sample: int = 64,
        show_progress: bool = True,
    ) -> 'CachedTensorDataset':
        """Create a CachedTensorDataset from replay entries."""
        boards, move_features, move_counts, targets, reward_weights, value_targets = preprocess_entries_to_tensors(
            entries, max_moves_per_sample, show_progress
        )
        return cls(boards, move_features, move_counts, targets, reward_weights, value_targets)
    
    def save(self, path: str) -> None:
        """Save the cached tensors to a .pt file."""
        torch.save({
            'boards': self.boards,
            'move_features': self.move_features,
            'move_counts': self.move_counts,
            'targets': self.targets,
            'reward_weights': self.reward_weights,
            'value_targets': self.value_targets,
        }, path)
        print(f"Saved cached dataset to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CachedTensorDataset':
        """Load cached tensors from a .pt file."""
        data = torch.load(path, weights_only=True)
        return cls(
            data['boards'],
            data['move_features'],
            data['move_counts'],
            data['targets'],
            data.get('reward_weights', None),  # Backward compat with old caches
            data.get('value_targets', None),   # Backward compat with old caches
        )


def collate_cached_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int, float, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for CachedTensorDataset.
    
    Handles variable-length move features by using the pre-padded tensors.
    
    Returns:
    - boards: (batch_size, BOARD_PLANES, 8, 8)
    - all_move_features: (total_moves, MOVE_FEATURE_SIZE) - flattened
    - move_counts: (batch_size,) - number of moves per sample
    - targets: (batch_size,) - index of chosen move for each sample
    - reward_weights: (batch_size,) - reward-based weights for loss
    - value_targets: (batch_size,) - game result for value head
    """
    boards = []
    all_move_features = []
    move_counts = []
    targets = []
    reward_weights = []
    value_targets = []
    
    for board, move_feats, move_count, target, rw, vt in batch:
        boards.append(board)
        # Only take the valid moves (up to move_count)
        all_move_features.append(move_feats[:move_count])
        move_counts.append(move_count)
        targets.append(target)
        reward_weights.append(rw)
        value_targets.append(vt)
    
    return (
        torch.stack(boards),
        torch.cat(all_move_features, dim=0).contiguous(),
        torch.tensor(move_counts, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(reward_weights, dtype=torch.float32),
        torch.tensor(value_targets, dtype=torch.float32),
    )


def collate_padded_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int, float, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast collate for padded training path.

    Keeps move_features at (batch, max_moves, MOVE_FEATURE_SIZE) instead of
    flattening to (total_moves, MOVE_FEATURE_SIZE). Eliminates per-sample
    slicing and torch.cat, enabling forward_padded in the model.

    Returns:
    - boards: (batch_size, BOARD_PLANES, 8, 8)
    - move_features: (batch_size, max_moves, MOVE_FEATURE_SIZE) padded
    - move_counts: (batch_size,)
    - targets: (batch_size,)
    - reward_weights: (batch_size,)
    - value_targets: (batch_size,)
    """
    boards, move_feats, move_counts, targets, rws, vts = zip(*batch)
    return (
        torch.stack(boards),
        torch.stack(move_feats),
        torch.tensor(move_counts, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(rws, dtype=torch.float32),
        torch.tensor(vts, dtype=torch.float32),
    )


class CUDAPrefetcher:
    """Prefetches next batch to GPU in a separate CUDA stream.

    Overlaps H2D data transfer with GPU computation for the current batch,
    eliminating transfer latency from the critical path.
    """

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        self._iter = iter(self.dataloader)
        self._prefetch()
        return self

    def _prefetch(self):
        try:
            self._next_batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self._next_batch = tuple(
                t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in self._next_batch
            )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next_batch
        if batch is None:
            raise StopIteration
        for t in batch:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(torch.cuda.current_stream())
        self._prefetch()
        return batch

    def __len__(self):
        return len(self.dataloader)


class FastBatchIterator:
    """Direct tensor-indexing batch iterator for CachedTensorDataset.

    Bypasses DataLoader entirely — shuffles indices once per epoch and
    slices contiguous tensors with a single fancy-index per field.
    This eliminates per-sample __getitem__, collation, worker IPC, and
    Python type-conversion overhead that DataLoader imposes.

    Yields the same 6-tuple as collate_padded_batch so the training loop
    works unchanged:
      (boards, move_features, move_counts, targets, reward_weights, value_targets)

    Exposes a `.dataset` attribute for compatibility with trainer code that
    checks ``isinstance(loader.dataset, CachedTensorDataset)``.
    """

    def __init__(
        self,
        dataset: CachedTensorDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = len(dataset)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n)
        else:
            indices = torch.arange(self.n)

        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            if self.drop_last and (end - start) < self.batch_size:
                break
            idx = indices[start:end]
            yield (
                self.dataset.boards[idx],
                self.dataset.move_features[idx],
                self.dataset.move_counts[idx],
                self.dataset.targets[idx],
                self.dataset.reward_weights[idx],
                self.dataset.value_targets[idx],
            )


class StreamingDamaDataset(IterableDataset):
    """
    Streaming dataset that reads from replay files on-the-fly.

    More memory efficient for large datasets.
    """

    def __init__(self, replay_buffer: ReplayBuffer, shuffle: bool = True):
        self.replay_buffer = replay_buffer
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int, float, float]]:
        for entry in self.replay_buffer.iterate_entries(shuffle_files=self.shuffle):
            # Reconstruct game state
            state = GameState.from_compact(entry.state)

            # Encode board
            board = encode_board(state)

            # Reconstruct moves and encode them
            moves = [Move.from_dict(m) for m in entry.legal_moves]
            move_features = encode_moves(state, moves)

            # Compute reward weight
            reward_weight = compute_reward_weight(entry.score)

            yield (
                torch.from_numpy(board),
                torch.from_numpy(move_features),
                entry.chosen_index,
                reward_weight,
                float(entry.result),  # value target
            )


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, float, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.

    Returns:
    - boards: (batch_size, BOARD_PLANES, 8, 8)
    - all_move_features: (total_moves, MOVE_FEATURE_SIZE)
    - move_counts: (batch_size,) - number of moves per sample
    - targets: (batch_size,) - index of chosen move for each sample
    - reward_weights: (batch_size,) - reward-based weights for loss
    - value_targets: (batch_size,) - game result for value head
    """
    boards = []
    all_move_features = []
    move_counts = []
    targets = []
    reward_weights = []
    value_targets = []

    for board, move_features, target, rw, vt in batch:
        boards.append(board)
        all_move_features.append(move_features)
        move_counts.append(move_features.shape[0])
        targets.append(target)
        reward_weights.append(rw)
        value_targets.append(vt)

    return (
        torch.stack(boards),
        torch.cat(all_move_features, dim=0).contiguous(),
        torch.tensor(move_counts, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(reward_weights, dtype=torch.float32),
        torch.tensor(value_targets, dtype=torch.float32),
    )


def create_dataloader(
    entries: List[ReplayEntry],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_ram_cache: bool = True,
    ram_threshold_gb: float = 16.0,
) -> DataLoader:
    """
    Create a DataLoader from replay entries.
    
    Args:
        entries: List of replay entries
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        use_ram_cache: If True and sufficient RAM available, pre-process to tensors
        ram_threshold_gb: Minimum available RAM (GB) required for caching
    
    Returns:
        DataLoader instance
    """
    available_ram = get_available_ram_gb()
    total_ram = get_total_ram_gb()
    
    # Estimate memory needed for cached dataset
    # Each entry: ~5*8*8*4 (board) + 64*8*4 (moves) + 16 (counts/targets) ≈ 3.5 KB
    estimated_size_gb = len(entries) * 3.5 / (1024 ** 2)
    
    # Use RAM caching if:
    # 1. Explicitly enabled
    # 2. Sufficient RAM available (with safety margin)
    # 3. Total RAM is high (e.g., server with 1TB RAM)
    should_cache = (
        use_ram_cache 
        and available_ram > ram_threshold_gb 
        and (available_ram > estimated_size_gb * 2 or total_ram > 64)
    )
    
    if should_cache:
        print(f"RAM Caching enabled ({available_ram:.1f}GB available, {estimated_size_gb:.2f}GB needed)")
        print("Pre-processing entries to tensors...")

        cached_dataset = CachedTensorDataset.from_entries(
            entries,
            max_moves_per_sample=64,
            show_progress=True
        )

        # FastBatchIterator: bypass DataLoader entirely for pre-tensorized data.
        # Direct tensor[indices] is orders of magnitude faster than per-sample
        # __getitem__ + collation + worker IPC that DataLoader imposes.
        return FastBatchIterator(
            cached_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
    
    # Fall back to standard dataset
    print(f"Using standard dataset (available RAM: {available_ram:.1f}GB)")
    dataset = DamaDataset(entries)
    
    # persistent_workers can cause hangs - only use when we have enough batches
    # to make it worthwhile
    use_persistent = num_workers > 0 and len(entries) > batch_size * 100
    
    # Scale prefetch factor based on batch size for better GPU utilization
    # Larger batches benefit from more prefetching
    if num_workers > 0:
        if batch_size >= 4096:
            prefetch = 4  # More prefetching for large batches
        elif batch_size >= 1024:
            prefetch = 3
        else:
            prefetch = 2
    else:
        prefetch = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory and num_workers > 0,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
        drop_last=True,  # Drop incomplete last batch for consistent batch size
    )


def create_cached_dataloader(
    entries: List[ReplayEntry],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    cache_path: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader with forced RAM caching.
    
    This function always pre-processes data into tensors, regardless of
    available RAM. Use when you know you have sufficient memory.
    
    Args:
        entries: List of replay entries
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (reduced automatically)
        pin_memory: Whether to pin memory
        cache_path: Optional path to save/load cached tensors
    
    Returns:
        DataLoader instance
    """
    # Try to load from cache first
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached dataset from {cache_path}")
        cached_dataset = CachedTensorDataset.load(cache_path)
    else:
        print("Pre-processing entries to tensors (forced RAM caching)...")
        cached_dataset = CachedTensorDataset.from_entries(
            entries,
            max_moves_per_sample=64,
            show_progress=True,
        )
        
        # Save cache if path provided
        if cache_path:
            cached_dataset.save(cache_path)
    
    # Collate still does variable-length slicing — 4 workers keeps GPU fed
    effective_workers = min(num_workers, 4)

    return DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        collate_fn=collate_cached_batch,
        pin_memory=pin_memory and effective_workers > 0,
        persistent_workers=effective_workers > 0,
        prefetch_factor=3 if effective_workers > 0 else None,
        drop_last=True,
    )


def create_dataloader_from_dataset(
    dataset: CachedTensorDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> FastBatchIterator:
    """Create a fast batch iterator from a pre-built CachedTensorDataset.

    Uses direct tensor indexing instead of DataLoader for maximum throughput.
    Used when dataset has been pre-processed in a background thread
    to avoid blocking the training loop.
    """
    return FastBatchIterator(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )


def create_streaming_dataloader(
    replay_buffer: ReplayBuffer,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a streaming DataLoader from a replay buffer."""
    dataset = StreamingDamaDataset(replay_buffer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def prepare_training_data(
    replay_buffer: ReplayBuffer,
    max_entries: int = 100000,
    val_split: float = 0.1,
) -> Tuple[List[ReplayEntry], List[ReplayEntry]]:
    """
    Prepare training and validation data from replay buffer.

    Args:
        replay_buffer: Source of training data
        max_entries: Maximum entries to use
        val_split: Fraction for validation

    Returns:
        (train_entries, val_entries)
    """
    # Collect entries
    entries = replay_buffer.sample_entries(max_entries)

    if not entries:
        return [], []

    # Shuffle
    random.shuffle(entries)

    # Split
    val_size = int(len(entries) * val_split)
    val_entries = entries[:val_size]
    train_entries = entries[val_size:]

    return train_entries, val_entries
