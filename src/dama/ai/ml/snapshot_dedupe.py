"""Digest-verified hardlink deduplication for retained corpus snapshots.

New snapshot admissions reuse unchanged replay shards, but snapshots created
before that optimization still contain independent copies.  This maintenance
tool verifies every duplicate candidate against its immutable manifest before
atomically replacing redundant copies with hardlinks.  Snapshot paths and
manifests are not changed.

The default is a verified dry run.  Pass ``--apply`` to perform replacements.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Iterable
import uuid


_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_HASH_WORKERS = 8


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    allocated_bytes: int

    @property
    def inode_key(self) -> tuple[int, int]:
        return self.device, self.inode


@dataclass(frozen=True)
class _SnapshotShard:
    path: Path
    snapshot_version: int
    expected_size: int
    expected_sha256: str
    identity: _FileIdentity


@dataclass(frozen=True)
class DedupeReport:
    snapshot_count: int
    manifest_record_count: int
    distinct_payload_count: int
    duplicate_path_count: int
    already_hardlinked_path_count: int
    verified_inode_count: int
    reclaimable_file_bytes: int
    reclaimable_allocated_bytes: int
    linked_path_count: int
    remaining_reclaimable_file_bytes: int


def _identity(path: Path) -> _FileIdentity:
    result = path.lstat()
    if not stat.S_ISREG(result.st_mode) or path.is_symlink():
        raise RuntimeError(f"Snapshot shard is not a regular file: {path}")
    return _FileIdentity(
        device=int(result.st_dev),
        inode=int(result.st_ino),
        size=int(result.st_size),
        mtime_ns=int(result.st_mtime_ns),
        allocated_bytes=int(getattr(result, "st_blocks", 0)) * 512,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise RuntimeError(f"Invalid manifest SHA-256: {value!r}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"Invalid manifest SHA-256: {value!r}") from exc
    return value


def _load_snapshot_shards(snapshot_root: Path) -> tuple[int, list[_SnapshotShard]]:
    root = snapshot_root.resolve(strict=True)
    manifests = sorted(root.glob("snapshot_v*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"No admitted snapshot manifests found under {root}")

    shards: list[_SnapshotShard] = []
    seen_paths: set[Path] = set()
    digest_sizes: dict[str, int] = {}
    for manifest_path in manifests:
        snapshot_dir = manifest_path.parent.resolve(strict=True)
        try:
            version = int(snapshot_dir.name.removeprefix("snapshot_v"))
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid snapshot directory name: {snapshot_dir.name}"
            ) from exc
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "training_snapshot":
            raise RuntimeError(
                f"Unexpected manifest kind in {manifest_path}: "
                f"{manifest.get('kind')!r}"
            )
        records = manifest.get("files")
        if not isinstance(records, list):
            raise RuntimeError(f"Manifest files must be a list: {manifest_path}")

        for record in records:
            if not isinstance(record, dict):
                raise RuntimeError(f"Malformed file record in {manifest_path}")
            relative = record.get("path")
            if not isinstance(relative, str) or not relative:
                raise RuntimeError(f"Invalid shard path in {manifest_path}")
            # Native-Windows snapshots may carry backslashes even when this
            # maintenance command is later run under WSL.
            candidate = snapshot_dir / relative.replace("\\", "/")
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(snapshot_dir):
                raise RuntimeError(
                    f"Snapshot shard escapes its snapshot directory: {candidate}"
                )
            if candidate.is_symlink():
                raise RuntimeError(f"Snapshot shard must not be a symlink: {candidate}")
            if resolved in seen_paths:
                raise RuntimeError(f"Snapshot shard is recorded more than once: {resolved}")
            seen_paths.add(resolved)

            size_value = record.get("size_bytes")
            if (
                isinstance(size_value, bool)
                or not isinstance(size_value, int)
                or size_value < 0
            ):
                raise RuntimeError(f"Invalid shard size in {manifest_path}: {size_value!r}")
            digest = _valid_sha256(record.get("sha256"))
            prior_size = digest_sizes.setdefault(digest, size_value)
            if prior_size != size_value:
                raise RuntimeError(
                    f"One SHA-256 is recorded with conflicting sizes: {digest}"
                )
            identity = _identity(resolved)
            if identity.size != size_value:
                raise RuntimeError(
                    f"Snapshot shard size mismatch: {resolved} "
                    f"({identity.size} != {size_value})"
                )
            shards.append(
                _SnapshotShard(
                    path=resolved,
                    snapshot_version=version,
                    expected_size=size_value,
                    expected_sha256=digest,
                    identity=identity,
                )
            )
    return len(manifests), shards


def _payload_groups(
    shards: Iterable[_SnapshotShard],
) -> dict[tuple[int, str], list[_SnapshotShard]]:
    groups: dict[tuple[int, str], list[_SnapshotShard]] = {}
    for shard in shards:
        groups.setdefault(
            (shard.expected_size, shard.expected_sha256), []
        ).append(shard)
    return groups


def _verify_duplicate_inodes(
    groups: dict[tuple[int, str], list[_SnapshotShard]], workers: int
) -> int:
    expected_by_inode: dict[tuple[int, int], tuple[int, str]] = {}
    representatives: dict[tuple[int, int], _SnapshotShard] = {}
    for payload_key, items in groups.items():
        for item in items:
            existing = expected_by_inode.setdefault(
                item.identity.inode_key, payload_key
            )
            if existing != payload_key:
                raise RuntimeError(
                    "One snapshot inode is recorded with conflicting payload "
                    f"identities: {item.path}"
                )
        if len(items) < 2:
            continue
        for item in items:
            representatives.setdefault(item.identity.inode_key, item)

    def verify(item: _SnapshotShard) -> None:
        before = _identity(item.path)
        if before != item.identity:
            raise RuntimeError(f"Snapshot shard changed before verification: {item.path}")
        actual = _sha256_file(item.path)
        after = _identity(item.path)
        if after != before:
            raise RuntimeError(f"Snapshot shard changed during verification: {item.path}")
        if actual != item.expected_sha256:
            raise RuntimeError(
                f"Snapshot shard digest mismatch: {item.path} "
                f"({actual} != {item.expected_sha256})"
            )

    ordered = sorted(representatives.values(), key=lambda item: str(item.path))
    if ordered:
        with ThreadPoolExecutor(max_workers=min(workers, len(ordered))) as pool:
            list(pool.map(verify, ordered))
    for items in groups.values():
        for item in items:
            if _identity(item.path) != item.identity:
                raise RuntimeError(
                    f"Snapshot shard changed after verification: {item.path}"
                )
    return len(ordered)


def _dedupe_plan(
    groups: dict[tuple[int, str], list[_SnapshotShard]],
) -> tuple[list[tuple[_SnapshotShard, _SnapshotShard]], int, int, int, int]:
    replacements: list[tuple[_SnapshotShard, _SnapshotShard]] = []
    duplicate_paths = 0
    already_linked_paths = 0
    reclaimable_file_bytes = 0
    reclaimable_allocated_bytes = 0
    for items in groups.values():
        if len(items) < 2:
            continue
        duplicate_paths += len(items) - 1
        by_inode: dict[tuple[int, int], list[_SnapshotShard]] = {}
        for item in items:
            by_inode.setdefault(item.identity.inode_key, []).append(item)
        already_linked_paths += len(items) - len(by_inode)
        if len(by_inode) < 2:
            continue

        canonical = max(
            items, key=lambda item: (item.snapshot_version, str(item.path))
        )
        for inode_key, inode_items in by_inode.items():
            if inode_key == canonical.identity.inode_key:
                continue
            representative = inode_items[0]
            reclaimable_file_bytes += representative.identity.size
            reclaimable_allocated_bytes += representative.identity.allocated_bytes
            replacements.extend((canonical, target) for target in inode_items)
    return (
        replacements,
        duplicate_paths,
        already_linked_paths,
        reclaimable_file_bytes,
        reclaimable_allocated_bytes,
    )


def _replace_with_hardlink(canonical: _SnapshotShard, target: _SnapshotShard) -> None:
    if _identity(canonical.path) != canonical.identity:
        raise RuntimeError(f"Canonical shard changed before linking: {canonical.path}")
    if _identity(target.path) != target.identity:
        raise RuntimeError(f"Redundant shard changed before linking: {target.path}")
    temporary = target.path.parent / (
        f".{target.path.name}.dama-dedupe-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    )
    try:
        os.link(canonical.path, temporary)
        os.replace(temporary, target.path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if _identity(target.path).inode_key != canonical.identity.inode_key:
        raise RuntimeError(f"Hardlink replacement did not take effect: {target.path}")


def deduplicate_snapshot_shards(
    snapshot_root: Path | str,
    *,
    apply: bool = False,
    workers: int = 8,
) -> DedupeReport:
    """Verify duplicate snapshot shards and optionally hardlink their paths."""

    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= _MAX_HASH_WORKERS
    ):
        raise ValueError(f"workers must be between 1 and {_MAX_HASH_WORKERS}")
    root = Path(snapshot_root)
    snapshot_count, shards = _load_snapshot_shards(root)
    groups = _payload_groups(shards)
    verified_inode_count = _verify_duplicate_inodes(groups, workers)
    (
        replacements,
        duplicate_paths,
        already_linked_paths,
        reclaimable_file_bytes,
        reclaimable_allocated_bytes,
    ) = _dedupe_plan(groups)

    linked_path_count = 0
    if apply:
        for canonical, target in replacements:
            _replace_with_hardlink(canonical, target)
            linked_path_count += 1

    _, current_shards = _load_snapshot_shards(root)
    current_groups = _payload_groups(current_shards)
    _, _, _, remaining_reclaimable, _ = _dedupe_plan(current_groups)
    if apply and remaining_reclaimable:
        raise RuntimeError(
            f"Snapshot deduplication left {remaining_reclaimable} reclaimable bytes"
        )

    return DedupeReport(
        snapshot_count=snapshot_count,
        manifest_record_count=len(shards),
        distinct_payload_count=len(groups),
        duplicate_path_count=duplicate_paths,
        already_hardlinked_path_count=already_linked_paths,
        verified_inode_count=verified_inode_count,
        reclaimable_file_bytes=reclaimable_file_bytes,
        reclaimable_allocated_bytes=reclaimable_allocated_bytes,
        linked_path_count=linked_path_count,
        remaining_reclaimable_file_bytes=remaining_reclaimable,
    )


def _worker_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= _MAX_HASH_WORKERS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {_MAX_HASH_WORKERS}"
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically replace verified redundant copies with hardlinks",
    )
    parser.add_argument("--workers", type=_worker_count, default=8)
    args = parser.parse_args()
    report = deduplicate_snapshot_shards(
        args.snapshot_root, apply=args.apply, workers=args.workers
    )
    print(f"mode={'apply' if args.apply else 'verified-dry-run'}")
    for field, value in report.__dict__.items():
        print(f"{field}={value}")


if __name__ == "__main__":
    main()
