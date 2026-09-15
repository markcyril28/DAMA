"""Verified ledger indexes release copied payloads before membership allocation."""

import hashlib
import json
import sys
import tracemalloc
from array import array
from pathlib import Path

import pytest

import dama.ai.ml.corpus as corpus


def _manager(tmp_path: Path) -> corpus.CorpusSnapshotManager:
    return corpus.CorpusSnapshotManager(
        str(tmp_path / "replay"),
        str(tmp_path / "snapshots"),
        trained_ledger_enabled=True,
    )


def _sidecar_fixture(tmp_path: Path, values) -> corpus.CorpusSnapshotManager:
    manager = _manager(tmp_path)
    manager.trained_ledger_dir.mkdir(parents=True)
    fingerprints = set(values)
    corpus._write_state_keys(
        manager._ledger_state_keys_path,
        (f"{value:016x}" + "0" * 48 for value in fingerprints),
    )
    manager._ledger_seed_path.write_text("{}", encoding="utf-8")
    manager._ledger_shards_path.write_text(
        json.dumps({"name": "historical.jsonl"}) + "\n", encoding="utf-8"
    )
    manager._write_ledger_fingerprint_sidecar(fingerprints)
    # Exercise the load boundary without the writer's cached source digest.
    return _manager(tmp_path)


def test_sidecar_does_not_retain_copied_payload_during_set_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the packed array, plus bounded metadata, should precede the set."""
    count = 262_144
    manager = _sidecar_fixture(tmp_path, range(count))
    observations = []
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]

        def observe_set(values):
            assert isinstance(values, array)
            observations.append((
                tracemalloc.get_traced_memory()[0] - before,
                sys.getsizeof(values),
            ))
            return set(values)

        # Observe actual live allocations at the membership constructor;
        # source text and frame-local names are deliberately not inspected.
        monkeypatch.setattr(corpus, "set", observe_set, raising=False)
        actual = manager._load_ledger_fingerprint_sidecar()
    finally:
        if started_tracing:
            tracemalloc.stop()

    assert actual == set(range(count))
    assert len(observations) == 1
    live_bytes, array_bytes = observations[0]
    # The 2 MiB payload must be gone. This allowance covers headers and Python
    # bookkeeping without depending on the array allocator's spare capacity.
    assert live_bytes <= array_bytes + 256 * 1024


@pytest.mark.parametrize(
    "values",
    [(), (0,), (0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1)],
    ids=["empty", "zero", "unsigned-boundaries"],
)
def test_sidecar_preserves_exact_values_and_verified_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values,
) -> None:
    manager = _sidecar_fixture(tmp_path, values)
    source_bytes = manager._ledger_state_keys_path.read_bytes()
    sidecar_bytes = manager._ledger_fingerprints_path.read_bytes()

    def unexpected_text_parse(_path):
        raise AssertionError("a verified sidecar must bypass canonical parsing")

    monkeypatch.setattr(corpus, "_iter_state_keys", unexpected_text_parse)
    names, fingerprints = manager._load_trained_ledger()

    assert names == {"historical.jsonl"}
    assert fingerprints == set(values)
    assert manager._trained_ledger_source_sha256 == hashlib.sha256(
        source_bytes).hexdigest()
    assert manager._ledger_state_keys_path.read_bytes() == source_bytes
    assert manager._ledger_fingerprints_path.read_bytes() == sidecar_bytes


@pytest.mark.parametrize("damage", ["duplicate", "corrupt_payload", "truncated"])
def test_invalid_sidecar_uses_canonical_fallback_and_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    expected = {0, 1 << 63, (1 << 64) - 1}
    manager = _sidecar_fixture(tmp_path, expected)
    source_bytes = manager._ledger_state_keys_path.read_bytes()
    path = manager._ledger_fingerprints_path
    with path.open("rb") as handle:
        magic = handle.readline()
        header = json.loads(handle.readline())
        payload = bytearray(handle.read())
    if damage == "duplicate":
        # A correct checksum cannot make duplicate membership rows valid.
        payload[8:16] = payload[:8]
        header["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    elif damage == "corrupt_payload":
        payload[-1] ^= 1
    else:
        del payload[-1]
    damaged_bytes = (
        magic
        + json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
        + b"\n"
        + bytes(payload)
    )
    path.write_bytes(damaged_bytes)

    assert manager._load_ledger_fingerprint_sidecar() is None
    assert manager._trained_ledger_source_sha256 is None
    assert path.read_bytes() == damaged_bytes
    parse_calls = []
    original_iter = corpus._iter_state_keys

    def record_text_parse(source):
        parse_calls.append(source)
        yield from original_iter(source)

    monkeypatch.setattr(corpus, "_iter_state_keys", record_text_parse)
    names, fingerprints = manager._load_trained_ledger()

    assert names == {"historical.jsonl"}
    assert fingerprints == expected
    assert parse_calls == [manager._ledger_state_keys_path]
    assert manager._trained_ledger_source_sha256 == hashlib.sha256(
        source_bytes).hexdigest()
    assert manager._ledger_state_keys_path.read_bytes() == source_bytes
    assert path.read_bytes() != damaged_bytes
    assert manager._load_ledger_fingerprint_sidecar() == expected
