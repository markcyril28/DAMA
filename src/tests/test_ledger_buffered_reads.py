"""Ledger read buffering preserves streaming, corruption checks and ownership."""

import gzip
import io
import math
from pathlib import Path

import pytest

import dama.ai.ml.corpus as corpus


def _track_read_handles(monkeypatch, path):
    real_open = Path.open
    handles = []

    def recording_open(candidate, *args, **kwargs):
        handle = real_open(candidate, *args, **kwargs)
        if candidate == path and args and args[0] == "rb":
            handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", recording_open)
    return handles


@pytest.mark.parametrize("members", [
    [b"\t00 \r\n\n 11\t\n \r\n22\n\t33 "],
    [b"00\n11", b"", b"\n22\n33"],
])
def test_buffered_stream_and_merge_preserve_logical_keys(tmp_path, members):
    path = tmp_path / "trained_state_keys.txt.gz"
    path.write_bytes(b"".join(gzip.compress(member) for member in members))

    assert list(corpus._iter_state_keys(path)) == ["00", "11", "22", "33"]
    assert corpus._merge_state_keys_file(path, ["44", "11", "44"]) == 1
    assert gzip.decompress(path.read_bytes()) == b"00\n11\n22\n33\n44\n"
    assert not path.with_suffix(".gz.tmp").exists()


def _damaged_payload(kind):
    # Keep corruption beyond the first text buffer: a merge has already
    # opened its replacement when this existing ledger stops being readable.
    data = b"".join(f"{index:064x}\n".encode("ascii") for index in range(3000))
    if kind == "ascii":
        return gzip.compress(data + b"\xff\n"), UnicodeDecodeError
    compressed = bytearray(gzip.compress(data))
    if kind == "crc":
        compressed[-8] ^= 1
        return bytes(compressed), gzip.BadGzipFile
    return bytes(compressed[:-4]), EOFError


@pytest.mark.parametrize("kind", ["ascii", "crc", "truncated"])
@pytest.mark.parametrize("merge", [False, True])
def test_corrupt_ledger_propagates_and_closes_without_publication(
    tmp_path, monkeypatch, kind, merge,
):
    path = tmp_path / "trained_state_keys.txt.gz"
    payload, error_type = _damaged_payload(kind)
    path.write_bytes(payload)
    handles = _track_read_handles(monkeypatch, path)

    with pytest.raises(error_type):
        if merge:
            corpus._merge_state_keys_file(path, ["f" * 64])
        else:
            list(corpus._iter_state_keys(path))

    # Retain the handles themselves so CPython finalization cannot hide a
    # missing close in the iterator's nested gzip/raw context managers.
    assert handles and all(handle.closed for handle in handles)
    assert path.read_bytes() == payload
    assert not path.with_suffix(".gz.tmp").exists()


def test_closing_partially_consumed_iterator_closes_raw_file(tmp_path, monkeypatch):
    path = tmp_path / "trained_state_keys.txt.gz"
    path.write_bytes(gzip.compress(b"00\n11\n22\n"))
    handles = _track_read_handles(monkeypatch, path)
    keys = corpus._iter_state_keys(path)

    assert next(keys) == "00"
    assert len(handles) == 1 and not handles[0].closed
    keys.close()
    assert handles[0].closed
    assert list(keys) == []


def test_gzip_initialization_failure_closes_raw_file(tmp_path, monkeypatch):
    path = tmp_path / "trained_state_keys.txt.gz"
    path.write_bytes(gzip.compress(b"00\n11\n"))
    handles = _track_read_handles(monkeypatch, path)
    failure = OSError("injected gzip initialization failure")

    def fail_open(*args, **kwargs):
        raise failure

    monkeypatch.setattr(corpus.gzip, "open", fail_open)
    with pytest.raises(OSError) as raised:
        next(corpus._iter_state_keys(path))
    assert raised.value is failure
    assert handles and all(handle.closed for handle in handles)


def test_read_failure_after_yield_closes_raw_file(tmp_path, monkeypatch):
    path = tmp_path / "trained_state_keys.txt.gz"
    path.write_bytes(gzip.compress(b"00\n" * 10000))
    handles = _track_read_handles(monkeypatch, path)
    keys = corpus._iter_state_keys(path)
    assert next(keys) == "00"
    failure = OSError("injected compressed-stream read failure")

    def fail_read(*args, **kwargs):
        raise failure

    # The first text buffer may still yield keys. Failure on its next refill
    # exercises cleanup after ownership has crossed a generator suspension.
    monkeypatch.setattr(gzip.GzipFile, "read1", fail_read)
    with pytest.raises(OSError) as raised:
        list(keys)
    assert raised.value is failure
    assert handles and all(handle.closed for handle in handles)


def test_stream_uses_bounded_raw_reads_without_materializing_keys(tmp_path, monkeypatch):
    path = tmp_path / "trained_state_keys.txt.gz"
    count = 140000
    # Stored DEFLATE blocks keep this deterministic input larger than two
    # read buffers without random data, large fixtures or expensive deflation.
    with gzip.open(path, "wb", compresslevel=0) as handle:
        for index in range(count):
            handle.write(f"{index:064x}\n".encode("ascii"))
    compressed_size = path.stat().st_size
    read_sizes = []
    raw_handles = []
    real_open = Path.open

    class CountingFileIO(io.FileIO):
        def readinto(self, buffer):
            read_sizes.append(len(buffer))
            return super().readinto(buffer)

    def counted_open(candidate, *args, **kwargs):
        if candidate != path or not args or args[0] != "rb":
            return real_open(candidate, *args, **kwargs)
        raw = CountingFileIO(candidate, "rb")
        raw_handles.append(raw)
        buffer_size = kwargs.get("buffering", io.DEFAULT_BUFFER_SIZE)
        return io.BufferedReader(raw, buffer_size=buffer_size)

    monkeypatch.setattr(Path, "open", counted_open)
    seen = 0
    for seen, key in enumerate(corpus._iter_state_keys(path), start=1):
        assert key == f"{seen - 1:064x}"

    assert seen == count
    assert raw_handles and all(handle.closed for handle in raw_handles)
    # This is a resource contract, not a wall-time assertion. Gzip's bounded
    # decompressor reads should coalesce into a handful of physical reads;
    # allow extra EOF probes without depending on one CPython patch version.
    bound = corpus._STATE_KEYS_READ_BUFFER_BYTES
    assert read_sizes and max(read_sizes) <= bound
    assert len(read_sizes) <= math.ceil(compressed_size / bound) + 4
