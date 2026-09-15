"""Buffered writer cleanup must not replace an interruption or publish output."""

import errno
import gzip
import os

import pytest

from dama.ai.ml import corpus, fork_writers, run_status


@pytest.mark.parametrize("writer", [
    "json", "text", "jsonl", "keys", "merge", "sidecar", "keys_gzip", "merge_gzip",
])
@pytest.mark.parametrize("interrupted", [True, False])
def test_atomic_writer_preserves_interruption_over_buffered_close_error(
    tmp_path, monkeypatch, capsys, writer, interrupted,
):
    fail_gzip = writer.endswith("_gzip")
    writer = writer.removesuffix("_gzip")
    manager = corpus.CorpusSnapshotManager(
        str(tmp_path / "replay"), str(tmp_path / "snapshots"),
        trained_ledger_enabled=True,
    )
    manager.trained_ledger_dir.mkdir(parents=True)
    keys = {"0" * 64, "1" * 64}
    corpus._write_state_keys(manager._ledger_state_keys_path, keys)
    target = (
        manager._ledger_fingerprints_path if writer == "sidecar"
        else tmp_path / "published"
    )
    prior = gzip.compress(b"00\n22\n") if writer == "merge" else b"prior output\n"
    target.write_bytes(prior)
    primary = KeyboardInterrupt("writer interrupted with buffered bytes")
    secondary = OSError("buffered close failed")
    streams, descriptors, raw_streams = [], [], []
    real_fdopen = os.fdopen
    real_gzip_open = gzip.open

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream
            self.close_calls = 0

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            self.close()

        def write(self, data):
            result = self.stream.write(data)
            if interrupted:
                raise primary
            return result

        def close(self):
            self.close_calls += 1
            self.stream.close()
            raise secondary

    def wrap(descriptor, *args, **kwargs):
        raw_stream = real_fdopen(descriptor, *args, **kwargs)
        raw_streams.append(raw_stream)
        descriptors.append(descriptor)
        if fail_gzip:
            return raw_stream
        stream = FailingStream(raw_stream)
        streams.append(stream)
        return stream

    def wrap_gzip(target, mode="rb", *args, **kwargs):
        stream = real_gzip_open(target, mode, *args, **kwargs)
        if mode == "wt":
            stream = FailingStream(stream)
            streams.append(stream)
        return stream

    def publish():
        if writer == "json":
            run_status._write_json_atomic(target, {"new": True})
        elif writer == "text":
            corpus._write_text_atomic(target, "new snapshot\n")
        elif writer == "jsonl":
            corpus._write_jsonl_atomic(target, [{"new": True}])
        elif writer == "keys":
            corpus._write_state_keys(target, keys)
        elif writer == "merge":
            corpus._merge_state_keys_file(target, ["11", "22"])
        else:
            manager._write_ledger_fingerprint_sidecar({
                corpus._state_key_fingerprint(key) for key in keys
            })

    with monkeypatch.context() as patch:
        patch.setattr(os, "fdopen", wrap)
        if fail_gzip:
            patch.setattr(gzip, "open", wrap_gzip)
        if writer == "sidecar" and not interrupted:
            # This derived cache keeps its existing fail-open error contract.
            publish()
            assert "buffered close failed" in capsys.readouterr().out
        else:
            with pytest.raises(KeyboardInterrupt if interrupted else OSError) as caught:
                publish()
            assert caught.value is (primary if interrupted else secondary)

    assert len(streams) == 1 and streams[0].stream.closed
    assert streams[0].close_calls == 1
    assert len(raw_streams) == 1 and raw_streams[0].closed
    with pytest.raises(OSError) as closed:
        os.fstat(descriptors[0])
    assert closed.value.errno == errno.EBADF
    assert not fork_writers._FORK_CHILD_DROPPED_FDS
    assert target.read_bytes() == prior
    assert not list(tmp_path.rglob("*.tmp"))
