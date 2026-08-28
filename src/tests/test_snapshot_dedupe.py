import hashlib
import json
import os
from pathlib import Path

import pytest

from dama.ai.ml.snapshot_dedupe import deduplicate_snapshot_shards


def _write_snapshot(
    root: Path,
    version: int,
    payloads: dict[str, bytes],
    *,
    windows_paths: bool = False,
) -> None:
    snapshot = root / f"snapshot_v{version:06d}"
    files = snapshot / "files"
    files.mkdir(parents=True)
    records = []
    for name, payload in payloads.items():
        path = files / name
        path.write_bytes(payload)
        records.append(
            {
                "name": name,
                "path": (
                    f"files\\{name}" if windows_paths else f"files/{name}"
                ),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "storage": "copy",
            }
        )
    (snapshot / "manifest.json").write_text(
        json.dumps({"kind": "training_snapshot", "files": records}),
        encoding="utf-8",
    )


def test_snapshot_dedupe_verifies_then_atomically_hardlinks(tmp_path: Path) -> None:
    root = tmp_path / "snapshots"
    shared = b"shared immutable replay shard\n" * 20
    _write_snapshot(root, 1, {"shared.jsonl": shared, "old.jsonl": b"old\n"})
    _write_snapshot(
        root,
        2,
        {"shared.jsonl": shared, "new.jsonl": b"new\n"},
        windows_paths=True,
    )
    first = root / "snapshot_v000001/files/shared.jsonl"
    second = root / "snapshot_v000002/files/shared.jsonl"
    assert os.stat(first).st_ino != os.stat(second).st_ino

    dry_run = deduplicate_snapshot_shards(root, workers=2)
    assert dry_run.snapshot_count == 2
    assert dry_run.manifest_record_count == 4
    assert dry_run.distinct_payload_count == 3
    assert dry_run.duplicate_path_count == 1
    assert dry_run.reclaimable_file_bytes == len(shared)
    assert dry_run.linked_path_count == 0
    assert os.stat(first).st_ino != os.stat(second).st_ino

    applied = deduplicate_snapshot_shards(root, apply=True, workers=2)
    assert applied.linked_path_count == 1
    assert applied.remaining_reclaimable_file_bytes == 0
    assert os.stat(first).st_ino == os.stat(second).st_ino
    assert first.read_bytes() == second.read_bytes() == shared
    assert not list(root.rglob("*.dama-dedupe-*.tmp"))


def test_snapshot_dedupe_digest_mismatch_fails_before_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshots"
    payload = b"manifest-approved bytes"
    _write_snapshot(root, 1, {"shared.jsonl": payload})
    _write_snapshot(root, 2, {"shared.jsonl": payload})
    first = root / "snapshot_v000001/files/shared.jsonl"
    second = root / "snapshot_v000002/files/shared.jsonl"
    first.write_bytes(b"X" + payload[1:])
    before = (os.stat(first).st_ino, os.stat(second).st_ino)

    with pytest.raises(RuntimeError, match="digest mismatch"):
        deduplicate_snapshot_shards(root, apply=True, workers=2)

    assert (os.stat(first).st_ino, os.stat(second).st_ino) == before
    assert first.read_bytes() != second.read_bytes()


def test_snapshot_dedupe_rejects_conflicting_manifests_for_one_inode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshots"
    payload = b"shared inode"
    _write_snapshot(root, 1, {"shared.jsonl": payload})
    _write_snapshot(root, 2, {"shared.jsonl": payload})
    first = root / "snapshot_v000001/files/shared.jsonl"
    second = root / "snapshot_v000002/files/shared.jsonl"
    second.unlink()
    os.link(first, second)
    manifest_path = root / "snapshot_v000002/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="conflicting payload identities"):
        deduplicate_snapshot_shards(root, workers=2)
