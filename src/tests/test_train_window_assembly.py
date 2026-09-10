"""Snapshot assembly releases temporary sources without changing window ownership."""

import hashlib
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import torch

from dama.ai.ml.corpus import CorpusSnapshotManager, canonical_state_key
from dama.ai.ml.dataset import CachedTensorDataset
from dama.ai.ml.replay import ReplayEntry
from dama.ai.ml.trainer import Trainer
from dama.game_state import GameState


FIELDS = (
    "boards", "move_features", "move_counts", "targets",
    "reward_weights", "value_targets",
)


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=64 * 1024**3))
    state = GameState.initial()
    rows = []
    for index in range(12):
        moves = state.legal_moves()
        chosen = index % len(moves)
        rows.append(ReplayEntry(
            state=state.to_compact(), legal_moves=[move.to_dict() for move in moves],
            chosen_index=chosen, result=index % 3 - 1,
            sample_weight=1.0 + index / 4,
        ).to_dict())
        state = state.apply_move(moves[chosen])

    def record(name, selected):
        data = "".join(json.dumps(row) + "\n" for row in selected).encode()
        (tmp_path / name).write_bytes(data)
        return {"path": name, "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}

    records = [record("first.jsonl", rows[:5]), record("empty.jsonl", []),
               record("last.jsonl", rows[5:])]
    context = SimpleNamespace(
        manifest_path=tmp_path / "manifest.json", manifest={"files": records},
        validation_keys={canonical_state_key(rows[1]["state"])},
        max_train_entries=0,
    )
    manager = object.__new__(CorpusSnapshotManager)
    manager.split_seed = 20260819
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(policy_stage="policy_only", max_moves_per_sample=32)
    holder._snapshot_manager = manager
    return holder, context, record, rows


def _expected(holder, context):
    return CachedTensorDataset.from_entries(
        holder._snapshot_manager.load_train_entries(context),
        max_moves_per_sample=32, show_progress=False,
    )


def _assert_equal(actual, expected):
    assert len(actual) == len(expected)
    for field in FIELDS:
        assert torch.equal(getattr(actual, field), getattr(expected, field)), field


@pytest.mark.parametrize("cap", [0, 4])
@pytest.mark.parametrize("retain", [True, False])
def test_cold_window_retires_sources_after_each_copy(
    window, monkeypatch, cap, retain,
):
    holder, context, _, _ = window
    context.max_train_entries = cap
    expected = _expected(holder, context)
    if not retain:
        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    source_refs = []
    encode = CachedTensorDataset.from_entries
    original_cat = torch.cat
    copied = []

    def capture(*args, **kwargs):
        dataset = encode(*args, **kwargs)
        source_refs.extend(weakref.ref(getattr(dataset, field)) for field in FIELDS)
        return dataset

    def observe_copy(*args, **kwargs):
        index = len(copied)
        assert all(ref() is None for ref in source_refs[:index])
        assert all(ref() is not None for ref in source_refs[index:])
        result = original_cat(*args, **kwargs)
        copied.append(FIELDS[index])
        return result

    monkeypatch.setattr(CachedTensorDataset, "from_entries", capture)
    monkeypatch.setattr(torch, "cat", observe_copy)
    actual = holder._load_or_reuse_train_dataset(context)
    _assert_equal(actual, expected)
    assert copied == list(FIELDS)
    assert all(ref() is None for ref in source_refs)
    assert (holder._train_tensor_window is not None) == retain


@pytest.mark.parametrize("change", ["filter", "all_files", "empty_hit"])
def test_all_misses_after_cache_change_keep_old_window_independent(window, change):
    holder, context, record, rows = window
    previous = holder._load_or_reuse_train_dataset(context)
    saved = {field: getattr(previous, field).clone() for field in FIELDS}
    if change == "filter":
        context.validation_keys.add(canonical_state_key(rows[2]["state"]))
    else:
        files = [record("replacement.jsonl", list(reversed(rows)))]
        if change == "empty_hit":
            files.insert(0, context.manifest["files"][1])
        context.manifest = {"files": files}
    expected = _expected(holder, context)
    actual = holder._load_or_reuse_train_dataset(context)
    _assert_equal(actual, expected)
    for field in FIELDS:
        assert getattr(actual, field).data_ptr() != getattr(previous, field).data_ptr()
        getattr(actual, field).zero_()
        assert torch.equal(getattr(previous, field), saved[field])


@pytest.mark.parametrize("cap", [0, 4])
def test_mixed_window_copies_reused_rows_in_new_manifest_order(window, monkeypatch, cap):
    holder, context, record, rows = window
    previous = holder._load_or_reuse_train_dataset(context)
    saved = {field: getattr(previous, field).clone() for field in FIELDS}
    reused = context.manifest["files"][0]
    context.manifest = {"files": [record("new.jsonl", rows[8:]), reused]}
    context.max_train_entries = cap
    expected = _expected(holder, context)
    parsed = []
    parse = holder._snapshot_manager.load_train_file_entries

    def track(ctx, rec):
        parsed.append(rec["path"])
        return parse(ctx, rec)

    monkeypatch.setattr(holder._snapshot_manager, "load_train_file_entries", track)
    actual = holder._load_or_reuse_train_dataset(context)
    assert parsed == ["new.jsonl"]
    _assert_equal(actual, expected)
    for field in FIELDS:
        getattr(actual, field).zero_()
        assert torch.equal(getattr(previous, field), saved[field])


@pytest.mark.parametrize("abort_at", [1, 4, 5])
def test_aborted_new_window_keeps_previous_cache(window, abort_at):
    holder, context, _, rows = window
    previous = holder._load_or_reuse_train_dataset(context)
    cache = holder._train_tensor_window
    context.validation_keys.add(canonical_state_key(rows[2]["state"]))
    calls = 0

    def should_abort():
        nonlocal calls
        calls += 1
        return calls == abort_at

    assert holder._load_or_reuse_train_dataset(context, should_abort) is None
    assert holder._train_tensor_window is cache
    assert cache["dataset"] is previous


def test_empty_window_preserves_tensor_schema(window):
    holder, context, _, _ = window
    context.manifest = {"files": [context.manifest["files"][1]]}
    expected = _expected(holder, context)
    actual = holder._load_or_reuse_train_dataset(context)
    assert len(actual) == 0
    _assert_equal(actual, expected)


def test_failed_tensorization_does_not_publish_a_replacement(window, monkeypatch):
    holder, context, _, rows = window
    holder._load_or_reuse_train_dataset(context)
    cache = holder._train_tensor_window
    context.validation_keys.add(canonical_state_key(rows[2]["state"]))

    def fail(*args, **kwargs):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(CachedTensorDataset, "from_entries", fail)
    with pytest.raises(RuntimeError, match="encoding failed"):
        holder._load_or_reuse_train_dataset(context)
    assert holder._train_tensor_window is cache


@pytest.mark.parametrize("fail_at", [1, 3, 6])
def test_failed_assembly_keeps_previous_cache_and_can_retry(window, monkeypatch, fail_at):
    holder, context, record, rows = window
    previous = holder._load_or_reuse_train_dataset(context)
    cache = holder._train_tensor_window
    saved = {field: getattr(previous, field).clone() for field in FIELDS}
    context.manifest = {"files": [record("new.jsonl", rows[8:]),
                                  context.manifest["files"][0]]}
    expected = _expected(holder, context)
    cat = torch.cat
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("assembly failed")
        return cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", fail)
    with pytest.raises(RuntimeError, match="assembly failed"):
        holder._load_or_reuse_train_dataset(context)
    assert holder._train_tensor_window is cache
    for field in FIELDS:
        assert torch.equal(getattr(previous, field), saved[field])
    monkeypatch.setattr(torch, "cat", cat)
    _assert_equal(holder._load_or_reuse_train_dataset(context), expected)


def test_all_hits_preserve_previous_tensors_without_encoding(window, monkeypatch):
    holder, context, _, _ = window
    previous = holder._load_or_reuse_train_dataset(context)

    def unexpected(*args, **kwargs):
        raise AssertionError("Unchanged verified shards should not be parsed or encoded")

    monkeypatch.setattr(CachedTensorDataset, "from_entries", unexpected)
    monkeypatch.setattr(holder._snapshot_manager, "load_train_file_entries", unexpected)
    actual = holder._load_or_reuse_train_dataset(context)
    _assert_equal(actual, previous)
    for field in FIELDS:
        assert getattr(actual, field).data_ptr() != getattr(previous, field).data_ptr()


@pytest.mark.parametrize("cap", [0, 4])
@pytest.mark.parametrize("threshold", [1, 4, 5])
def test_chunked_window_releases_rows_before_reading_more_shards(
    window, monkeypatch, cap, threshold,
):
    holder, context, _, _ = window
    context.max_train_entries = cap
    expected = _expected(holder, context)
    monkeypatch.setattr(
        "dama.ai.ml.trainer._TRAIN_WINDOW_PARSE_CHUNK_ENTRIES", threshold)
    row_refs = []
    tensor_refs = []
    encoded_sizes = []
    parse = holder._snapshot_manager.load_train_file_entries
    encode = CachedTensorDataset.from_entries
    cat = torch.cat
    copies = []

    def observed_parse(*args):
        assert all(ref() is None for ref in row_refs)
        return parse(*args)

    def observed_encode(entries, **kwargs):
        encoded_sizes.append(len(entries))
        row_refs.extend(weakref.ref(entry) for entry in entries)
        result = encode(entries, **kwargs)
        tensor_refs.append([weakref.ref(getattr(result, field)) for field in FIELDS])
        return result

    def observed_cat(*args, **kwargs):
        field_index = len(copies)
        for refs in tensor_refs:
            assert all(ref() is None for ref in refs[:field_index])
            assert all(ref() is not None for ref in refs[field_index:])
        result = cat(*args, **kwargs)
        copies.append(FIELDS[field_index])
        return result

    monkeypatch.setattr(holder._snapshot_manager, "load_train_file_entries", observed_parse)
    monkeypatch.setattr(CachedTensorDataset, "from_entries", observed_encode)
    monkeypatch.setattr(torch, "cat", observed_cat)
    actual = holder._load_or_reuse_train_dataset(context)
    _assert_equal(actual, expected)
    assert encoded_sizes == ([4, 7] if threshold <= 4 else [11])
    assert copies == list(FIELDS)
    assert all(ref() is None for ref in row_refs)
    assert all(ref() is None for refs in tensor_refs for ref in refs)


@pytest.mark.parametrize("cap", [0, 5])
def test_chunked_misses_keep_interleaved_hits_and_weights_in_manifest_order(
    window, monkeypatch, cap,
):
    holder, context, record, rows = window
    previous = holder._load_or_reuse_train_dataset(context)
    old_cache = holder._train_tensor_window
    saved = {field: getattr(previous, field).clone() for field in FIELDS}
    new_rows = [dict(row, score=(index - 5) / 3,
                     sample_weight=1.234567 + index / 10)
                for index, row in enumerate(reversed(rows))]
    first, empty, last = context.manifest["files"]
    context.manifest = {"files": [
        record("new_first.jsonl", new_rows[:4]), first, empty,
        record("new_last.jsonl", new_rows[4:]), last,
        record("new_empty.jsonl", []),
    ]}
    context.max_train_entries = cap
    expected = _expected(holder, context)
    monkeypatch.setattr("dama.ai.ml.trainer._TRAIN_WINDOW_PARSE_CHUNK_ENTRIES", 3)
    parsed = []
    parse = holder._snapshot_manager.load_train_file_entries

    def track(ctx, rec):
        parsed.append(rec["path"])
        return parse(ctx, rec)

    monkeypatch.setattr(holder._snapshot_manager, "load_train_file_entries", track)
    actual = holder._load_or_reuse_train_dataset(context)
    assert parsed == ["new_first.jsonl", "new_last.jsonl", "new_empty.jsonl"]
    _assert_equal(actual, expected)
    assert holder._train_tensor_window is not old_cache
    for field in FIELDS:
        getattr(actual, field).zero_()
        assert torch.equal(getattr(previous, field), saved[field])


def test_later_chunk_encoding_failure_keeps_previous_cache_and_retries(window, monkeypatch):
    holder, context, _, rows = window
    holder._load_or_reuse_train_dataset(context)
    previous_cache = holder._train_tensor_window
    context.validation_keys.add(canonical_state_key(rows[2]["state"]))
    expected = _expected(holder, context)
    monkeypatch.setattr("dama.ai.ml.trainer._TRAIN_WINDOW_PARSE_CHUNK_ENTRIES", 3)
    encode = CachedTensorDataset.from_entries
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("later chunk failed")
        return encode(*args, **kwargs)

    monkeypatch.setattr(CachedTensorDataset, "from_entries", fail_second)
    with pytest.raises(RuntimeError, match="later chunk failed"):
        holder._load_or_reuse_train_dataset(context)
    assert holder._train_tensor_window is previous_cache
    monkeypatch.setattr(CachedTensorDataset, "from_entries", encode)
    _assert_equal(holder._load_or_reuse_train_dataset(context), expected)


@pytest.mark.parametrize("stop_during", ["parse", "encode"])
def test_chunk_boundary_abort_releases_temporary_rows_and_keeps_cache(
    window, monkeypatch, stop_during,
):
    holder, context, _, rows = window
    holder._load_or_reuse_train_dataset(context)
    previous_cache = holder._train_tensor_window
    context.validation_keys.add(canonical_state_key(rows[2]["state"]))
    monkeypatch.setattr("dama.ai.ml.trainer._TRAIN_WINDOW_PARSE_CHUNK_ENTRIES", 3)
    parse = holder._snapshot_manager.load_train_file_entries
    encode = CachedTensorDataset.from_entries
    stopped = False
    row_refs = []
    encoded_sizes = []
    parsed = []

    def stop_parse(ctx, rec):
        nonlocal stopped
        result = parse(ctx, rec)
        row_refs.extend(weakref.ref(entry) for entry in result)
        parsed.append(rec["path"])
        stopped = stop_during == "parse"
        return result

    def stop_encode(entries, **kwargs):
        nonlocal stopped
        result = encode(entries, **kwargs)
        encoded_sizes.append(len(entries))
        stopped = True
        return result

    monkeypatch.setattr(holder._snapshot_manager, "load_train_file_entries", stop_parse)
    monkeypatch.setattr(CachedTensorDataset, "from_entries", stop_encode)
    assert holder._load_or_reuse_train_dataset(context, lambda: stopped) is None
    assert holder._train_tensor_window is previous_cache
    assert parsed == ["first.jsonl"]
    assert encoded_sizes == ([] if stop_during == "parse" else [3])
    assert all(ref() is None for ref in row_refs)
