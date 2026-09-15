"""Retire only producer-owned ledger membership after validation is resolved."""

import json
import threading
import weakref
from types import SimpleNamespace

import pytest

import dama.ai.ml.corpus as corpus
import dama.ai.ml.trainer as trainer_module
from dama.ai.ml.corpus import CorpusSnapshotManager, _SnapshotSplitContext
from dama.ai.ml.trainer import Trainer, _VALIDATION_TENSORS_CURRENT


class _ManagerSubclass(CorpusSnapshotManager):
    pass


class _ContextSubclass(_SnapshotSplitContext):
    pass


def _entry(index):
    return {
        "state": {
            "p1_men": [[2, index * 2 + 1]],
            "p1_kings": [],
            "p2_men": [[5, 0]],
            "p2_kings": [],
            "turn": 1,
            "move_count": index,
        },
        "legal_moves": [
            {"path": [[0, 1], [1, 0]], "captures": [], "promotion": False},
            {"path": [[0, 1], [1, 2]], "captures": [], "promotion": False},
        ],
        "chosen_index": 0,
        "result": 0,
        "trajectory_source": "algorithm",
    }


class _ProducerHarness:
    def __init__(
        self, tmp_path, monkeypatch, route, *, reuse=False, keep_alias=False,
        manager_kind="exact", context_kind="exact", terminal=None,
    ):
        self.route = route
        self.reuse = reuse
        self.keep_alias = keep_alias
        self.terminal = terminal
        self.retirement_expected = manager_kind == context_kind == "exact"
        self.calls = []
        self.training_seen = threading.Event()
        self.validation_seen = threading.Event()
        self.errors = []
        self.dataset = [object(), object()]
        self.alias = None
        self.original_context_ref = None
        self.copied_membership_ref = None
        self.train_context = None
        self.validation_result = None
        self.validation_identity = None

        entries = [_entry(index) for index in range(3)]
        keys = [corpus.canonical_state_key(row["state"]) for row in entries]
        self.held_keys = frozenset(keys[:2])
        self.expected_train_key = keys[2]
        self.source_membership = {
            corpus._state_key_fingerprint(keys[0]),
            corpus._state_key_fingerprint(
                corpus.canonical_state_key(_entry(3)["state"])),
        }
        self.original_source = self.source_membership.copy()
        self.validation_keys = set(self.held_keys) | {"frozen-suite-exclusion"}
        self.validation_manifest = {"files": [{"path": "validation.jsonl"}]}
        self.manifest = {
            "version": 2,
            "metrics": {"fresh_unique_state_rate": 0.6},
            "files": [{"path": "train.jsonl"}],
        }
        (tmp_path / "validation.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in entries[:2]),
            encoding="utf-8",
        )
        (tmp_path / "train.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in entries[1:]),
            encoding="utf-8",
        )

        if manager_kind == "custom":
            manager = SimpleNamespace()
        else:
            manager_type = (
                CorpusSnapshotManager if manager_kind == "exact"
                else _ManagerSubclass
            )
            manager = object.__new__(manager_type)
        self.manager = manager
        manager.trained_ledger_enabled = True
        manager.split_seed = 20260819
        manager._trained_ledger_cache = ({"trained-shard"}, self.source_membership)
        manager.consider_snapshot = lambda **kwargs: SimpleNamespace(
            admitted=True, manifest_path=tmp_path / "manifest.json")
        if manager_kind == "custom":
            manager.validation_leak_fingerprints = (
                lambda context: CorpusSnapshotManager.validation_leak_fingerprints(
                    manager, context))
            manager.load_train_file_entries = (
                lambda context, record: CorpusSnapshotManager.load_train_file_entries(
                    manager, context, record))
            manager.train_cap_sample_indices = (
                lambda size, cap: CorpusSnapshotManager.train_cap_sample_indices(
                    manager, size, cap))

        def new_context():
            fields = dict(
                manifest_path=tmp_path / "manifest.json",
                manifest=self.manifest,
                validation_path=tmp_path / "validation_manifest.json",
                validation_manifest=self.validation_manifest,
                validation_keys=self.validation_keys,
                historically_trained=self.source_membership.copy(),
                max_train_entries=100,
                stored_validation_keys=self.held_keys,
            )
            if context_kind == "custom":
                return SimpleNamespace(**fields)
            context_type = (
                _SnapshotSplitContext if context_kind == "exact"
                else _ContextSubclass
            )
            return context_type(**fields)

        def prepare(path, max_train_entries=0):
            assert path == tmp_path / "manifest.json"
            assert max_train_entries == 100
            self.calls.append("prepare")
            context = new_context()
            self.original_context_id = id(context)
            if context_kind != "custom":
                self.original_context_ref = weakref.ref(context)
            self.copied_membership_ref = weakref.ref(context.historically_trained)
            if keep_alias:
                self.alias = context
            if terminal == "stop_prepare":
                holder._bg_selfplay_stop_event.set()
            return context

        manager.prepare_split = prepare
        manager.load_validation_entries = (
            lambda context: CorpusSnapshotManager.load_validation_entries(
                manager, context))
        # Save the concrete parser before disabling per-file dispatch in the
        # staged fallback cases. Both routes still exercise actual exclusions.
        parse_train_file = manager.load_train_file_entries
        cap_indices = manager.train_cap_sample_indices

        def parse_train(context):
            rows = [
                row for record in context.manifest["files"]
                for row in parse_train_file(context, record)
            ]
            indices = cap_indices(len(rows), context.max_train_entries)
            return rows if indices is None else [rows[index] for index in indices]

        holder = object.__new__(Trainer)
        self.holder = holder
        holder.config = SimpleNamespace(
            replay_max_files=60, replay_max_entries=100,
            max_moves_per_sample=32, policy_stage="policy_only",
        )
        holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
        holder._snapshot_manager = manager
        holder._bg_selfplay_thread = None
        holder._bg_selfplay_stop_event = threading.Event()
        holder._data_ready_event = threading.Event()
        holder._bg_selfplay_lock = threading.Lock()
        holder._stopped = holder._paused = False
        holder._corpus_settings = lambda **kwargs: ({}, {}, {})
        holder._validation_tensor_identity = None
        holder._bg_selfplay_dataset = self.old_dataset = object()
        generated = False

        def generate(*args, **kwargs):
            nonlocal generated
            if generated:
                holder._bg_selfplay_stop_event.wait(timeout=10)
                holder._bg_selfplay_stop_event.set()
            generated = True
            return 1, 23

        holder.run_selfplay = generate
        if reuse:
            identity_context = new_context()
            holder._validation_tensor_identity = {
                "identity": holder._validation_reuse_identity(identity_context),
                "leakage": {
                    "removed_validation_entry_count": 1,
                    "removed_validation_state_count": 1,
                    "retained_validation_entry_count": 1,
                },
            }
            del identity_context

        def validation(context):
            self.calls.append("validation")
            assert context.historically_trained == self.original_source
            assert context.historically_trained is not self.source_membership
            if terminal == "validation_error":
                holder._bg_selfplay_stop_event.set()
                self.validation_seen.set()
                raise OSError("controlled validation failure")
            result, identity = Trainer._load_or_reuse_validation_entries(
                holder, context)
            self.validation_result = result
            self.validation_identity = identity
            assert context.manifest["validation_leakage"][
                "all_time_trained_state_count"] == len(self.original_source)
            if terminal == "stop_validation":
                holder._bg_selfplay_stop_event.set()
            self.validation_seen.set()
            return result, identity

        holder._load_or_reuse_validation_entries = validation

        def train(context):
            self.calls.append("train")
            try:
                self.train_context = context
                assert self.source_membership == self.original_source
                assert manager._trained_ledger_cache[1] is self.source_membership
                if self.retirement_expected:
                    if keep_alias:
                        assert self.alias.historically_trained == self.original_source
                        assert self.copied_membership_ref() is self.alias.historically_trained
                    else:
                        assert self.copied_membership_ref() is None
                        assert self.original_context_ref() is None
                    assert context.historically_trained == set()
                else:
                    assert id(context) == self.original_context_id
                    assert context.historically_trained == self.original_source
                    assert self.copied_membership_ref() is context.historically_trained
                assert context.validation_keys is self.validation_keys
                assert context.manifest is self.manifest
                rows = parse_train(context)
                assert [corpus.canonical_state_key(row.state) for row in rows] == [
                    self.expected_train_key]
                return rows
            except BaseException as exc:
                self.errors.append(exc)
                holder._bg_selfplay_stop_event.set()
                raise
            finally:
                self.training_seen.set()

        manager.load_train_entries = train

        def per_file(context, should_abort=None):
            assert should_abort is not None and not should_abort()
            self.train_rows = train(context)
            return self.dataset

        holder._load_or_reuse_train_dataset = per_file
        if route == "fallback":
            manager.load_train_file_entries = None
            manager.train_cap_sample_indices = None

        def tensorize(rows, **kwargs):
            assert [corpus.canonical_state_key(row.state) for row in rows] == [
                self.expected_train_key]
            assert kwargs == {"max_moves_per_sample": 32, "show_progress": True}
            return self.dataset

        monkeypatch.setattr(
            trainer_module.CachedTensorDataset, "from_entries", staticmethod(tensorize))

    def start(self):
        self.holder._start_background_selfplay(72)

    def close(self):
        self.holder._bg_selfplay_stop_event.set()
        thread = self.holder._bg_selfplay_thread
        if thread is not None:
            thread.join(timeout=10)
            assert not thread.is_alive()

    def assert_publication(self):
        assert self.training_seen.wait(timeout=10)
        assert not self.errors, repr(self.errors)
        assert self.holder._data_ready_event.wait(timeout=10)
        holder = self.holder
        assert holder._bg_selfplay_dataset is self.dataset
        assert holder._bg_snapshot_manifest is self.manifest
        assert holder._bg_validation_entries is self.validation_result
        assert holder._bg_validation_identity is self.validation_identity
        if self.reuse:
            assert self.validation_result is _VALIDATION_TENSORS_CURRENT
        else:
            assert len(self.validation_result) == 1
        assert self.manifest["validation_leakage"] == {
            "ledger_enabled": True,
            "all_time_trained_state_count": len(self.original_source),
            "removed_validation_entry_count": 1,
            "removed_validation_state_count": 1,
            "retained_validation_entry_count": 1,
        }


@pytest.mark.parametrize("route", ["per_file", "fallback"])
@pytest.mark.parametrize("reuse", [False, True])
def test_producer_retires_private_ledger_only_after_validation(
    tmp_path, monkeypatch, route, reuse,
):
    harness = _ProducerHarness(tmp_path, monkeypatch, route, reuse=reuse)
    try:
        harness.start()
        harness.assert_publication()
    finally:
        harness.close()


@pytest.mark.parametrize("route", ["per_file", "fallback"])
def test_producer_does_not_mutate_an_external_context_alias(
    tmp_path, monkeypatch, route,
):
    harness = _ProducerHarness(tmp_path, monkeypatch, route, keep_alias=True)
    try:
        harness.start()
        harness.assert_publication()
    finally:
        harness.close()
    assert harness.alias.historically_trained == harness.original_source
    # The general context still performs its complete validation operation.
    assert len(harness.manager.load_validation_entries(harness.alias)) == 1


@pytest.mark.parametrize("route", ["per_file", "fallback"])
@pytest.mark.parametrize("manager_kind,context_kind", [
    ("subclass", "exact"), ("custom", "exact"),
    ("exact", "subclass"), ("exact", "custom"),
])
def test_producer_preserves_custom_manager_and_context_contracts(
    tmp_path, monkeypatch, route, manager_kind, context_kind,
):
    harness = _ProducerHarness(
        tmp_path, monkeypatch, route,
        manager_kind=manager_kind, context_kind=context_kind,
    )
    try:
        harness.start()
        harness.assert_publication()
    finally:
        harness.close()


@pytest.mark.parametrize("route", ["per_file", "fallback"])
@pytest.mark.parametrize("terminal", [
    "stop_prepare", "stop_validation", "validation_error",
])
def test_stop_or_validation_error_never_starts_train_materialization(
    tmp_path, monkeypatch, capsys, route, terminal,
):
    harness = _ProducerHarness(
        tmp_path, monkeypatch, route, terminal=terminal)
    try:
        harness.start()
        harness.holder._bg_selfplay_thread.join(timeout=10)
        assert not harness.holder._bg_selfplay_thread.is_alive()
        assert not harness.training_seen.is_set()
        assert harness.holder._bg_selfplay_dataset is harness.old_dataset
        assert not harness.holder._data_ready_event.is_set()
    finally:
        harness.close()
    assert harness.source_membership == harness.original_source
    assert harness.copied_membership_ref() is None
    captured = capsys.readouterr()
    if terminal == "validation_error":
        assert "controlled validation failure" in captured.out
    else:
        assert "Background snapshot self-play error" not in captured.out
