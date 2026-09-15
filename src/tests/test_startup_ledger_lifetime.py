"""Startup releases its copied ledger only after validation is fully resolved."""

import json
import weakref
from types import SimpleNamespace

import pytest

import dama.ai.ml.corpus as corpus
import dama.ai.ml.trainer as trainer_module
from dama.ai.ml.corpus import CorpusSnapshotManager, _SnapshotSplitContext
from dama.ai.ml.trainer import Trainer


class _ManagerSubclass(CorpusSnapshotManager):
    pass


class _ContextSubclass(_SnapshotSplitContext):
    pass


class _CachedRows(list):
    def __init__(self, leakage):
        super().__init__([object()])
        self.metadata = {"validation_leakage": leakage}


def _entry(index):
    return {
        "state": {
            "p1_men": [[2 + index // 4, 1 + 2 * (index % 4)]],
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


class _StartupHarness:
    def __init__(
        self, tmp_path, monkeypatch, *, train_cache="miss",
        validation_cache="miss", keep_alias=False, manager_kind="exact",
        context_kind="exact", failure=None, hidden_validation=False,
        staged=True,
    ):
        self.calls = []
        self.train_cache = train_cache
        self.validation_cache = validation_cache
        self.keep_alias = keep_alias
        self.staged = staged
        self.retirement_expected = staged and manager_kind == context_kind == "exact"
        self.alias = None
        self.original_context_ref = None
        self.failure = RuntimeError("controlled validation failure")
        entries = [_entry(index) for index in range(6)]
        keys = [corpus.canonical_state_key(row["state"]) for row in entries]
        self.expected_train_keys = keys[2:4]
        self.expected_validation_key = keys[1]
        self.stored_keys = frozenset(keys[:2])
        self.validation_keys = set(self.stored_keys) | {keys[4]}
        self.source_membership = {
            corpus._state_key_fingerprint(keys[index]) for index in (0, 3)
        }
        validation_rows = [entries[0], entries[0], entries[1]]
        train_rows = [entries[index] for index in (0, 1, 2, 4, 3)]
        if hidden_validation:
            self.source_membership.add(corpus._state_key_fingerprint(keys[5]))
            validation_rows.append(entries[5])
            train_rows.append(entries[5])
        self.original_source = self.source_membership.copy()
        self.leakage = {
            "ledger_enabled": True,
            "all_time_trained_state_count": len(self.original_source),
            "removed_validation_entry_count": 2 + int(hidden_validation),
            "removed_validation_state_count": 1 + int(hidden_validation),
            "retained_validation_entry_count": 1,
        }
        for name, rows in (("validation.jsonl", validation_rows), ("train.jsonl", train_rows)):
            (tmp_path / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.manifest = {
            "fingerprint": "d" * 64,
            "files": [{"path": "train.jsonl"}],
        }
        validation_manifest = {"files": [{"path": "validation.jsonl"}]}
        manifest_path = tmp_path / "manifest.json"
        if manager_kind == "custom":
            manager = SimpleNamespace()
            for name in (
                "validation_leak_fingerprints", "load_train_file_entries",
                "train_cap_sample_indices", "load_split",
            ):
                method = getattr(CorpusSnapshotManager, name)
                setattr(manager, name, method.__get__(manager))
        else:
            kind = CorpusSnapshotManager if manager_kind == "exact" else _ManagerSubclass
            manager = object.__new__(kind)
        self.manager = manager
        manager.trained_ledger_enabled = True
        manager.split_seed = 20260819
        manager.external_validation_state_keys = {keys[4]}
        manager._trained_ledger_cache = ({"trained-shard"}, self.source_membership)
        manager.trained_ledger_source_sha256 = lambda: "a" * 64
        manager.consider_snapshot = lambda **kwargs: SimpleNamespace(
            admitted=False, reason="unchanged", manifest_path=manifest_path, metrics={})
        manager.snapshot_matches_settings = lambda *args, **kwargs: True

        def prepare(path, max_train_entries=0):
            self.calls.append("prepare")
            assert path == manifest_path
            assert max_train_entries == 100
            fields = dict(
                manifest_path=manifest_path,
                manifest=self.manifest,
                validation_path=tmp_path / "validation_manifest.json",
                validation_manifest=validation_manifest,
                validation_keys=self.validation_keys,
                historically_trained=self.source_membership.copy(),
                max_train_entries=max_train_entries,
                stored_validation_keys=self.stored_keys,
            )
            if context_kind == "custom":
                context = SimpleNamespace(**fields)
            else:
                kind = _SnapshotSplitContext if context_kind == "exact" else _ContextSubclass
                context = kind(**fields)
                self.original_context_ref = weakref.ref(context)
            self.original_context_id = id(context)
            self.copied_membership_ref = weakref.ref(context.historically_trained)
            if keep_alias:
                self.alias = context
            return context

        manager.prepare_split = prepare

        def validation(context):
            self.calls.append("validation_parse")
            self.assert_full_ledger(context)
            if failure == "parse":
                raise self.failure
            return CorpusSnapshotManager.load_validation_entries(manager, context)

        manager.load_validation_entries = validation

        def train(context):
            self.calls.append("train_parse")
            self.assert_train_phase(context)
            return CorpusSnapshotManager.load_train_entries(manager, context)

        manager.load_train_entries = train
        holder = object.__new__(Trainer)
        self.holder = holder
        holder.config = SimpleNamespace(
            replay_max_entries=100, policy_stage="policy_only",
            max_moves_per_sample=8, ram_cache_enabled=train_cache != "disabled",
            ram_cache_file=str(tmp_path / "train.pt"),
            ram_cache_threshold_gb=16.0, ram_cache_compress=True,
            validation_tensor_cache_file=str(tmp_path / "validation.pt"),
        )
        holder.step = 7
        holder._snapshot_manager = manager
        holder._prelaunch_free_ram_gb = 20.0
        holder._corpus_settings = lambda **kwargs: ({}, {}, {})
        self.activated = None

        def activate(manifest):
            self.calls.append("activate")
            self.activated = manifest

        holder._activate_dataset_manifest = activate

        def reuse_identity(context):
            self.calls.append("reuse_identity")
            self.assert_full_ledger(context)
            identity = Trainer._validation_reuse_identity(holder, context)
            assert identity["leak_fingerprints"] == frozenset({
                corpus._state_key_fingerprint(keys[0])})
            return identity

        holder._validation_reuse_identity = reuse_identity

        def validation_metadata(context):
            self.calls.append("validation_metadata")
            self.assert_full_ledger(context)
            if failure == "metadata":
                raise self.failure
            return Trainer._validation_tensor_cache_metadata(holder, context)

        holder._validation_tensor_cache_metadata = validation_metadata

        def train_metadata(manifest, validation_keys):
            self.calls.append("train_metadata")
            self.assert_train_phase()
            assert manifest is self.manifest
            assert validation_keys is self.validation_keys
            return Trainer._snapshot_train_cache_metadata(holder, manifest, validation_keys)

        holder._snapshot_train_cache_metadata = train_metadata
        self.cached_train = [object(), object()]
        self.cached_validation = _CachedRows(dict(self.leakage))
        if validation_cache == "malformed":
            self.cached_validation.metadata["validation_leakage"] = {
                **self.leakage, "retained_validation_entry_count": 99}

        def load_cache(path, metadata, *, migrate_to_compressed=False):
            if path == holder.config.validation_tensor_cache_file:
                self.calls.append("validation_cache")
                assert self.copied_membership_ref() == self.original_source
                assert metadata["trained_ledger_source_sha256"] == "a" * 64
                if failure == "cache":
                    raise self.failure
                return None if validation_cache == "miss" else self.cached_validation
            self.calls.append("train_cache")
            self.assert_train_phase()
            assert path == holder.config.ram_cache_file
            assert migrate_to_compressed is True
            assert metadata["validation_exclusion_keys_sha256"] == (
                Trainer._snapshot_cache_key_digest(self.validation_keys))
            return self.cached_train if train_cache == "hit" else None

        monkeypatch.setattr(
            trainer_module, "load_matching_cached_tensor_dataset", load_cache)

    def assert_full_ledger(self, context):
        assert context.historically_trained == self.original_source
        assert context.historically_trained is not self.source_membership
        assert self.copied_membership_ref() is context.historically_trained

    def assert_train_phase(self, context=None):
        assert self.source_membership == self.original_source
        assert self.manager._trained_ledger_cache[1] is self.source_membership
        assert self.manifest["validation_leakage"] == self.leakage
        if self.retirement_expected:
            if self.keep_alias:
                assert self.alias.historically_trained == self.original_source
                assert self.copied_membership_ref() is self.alias.historically_trained
            else:
                assert self.copied_membership_ref() is None
                assert self.original_context_ref() is None
            if context is not None:
                assert context.historically_trained == set()
        else:
            assert self.copied_membership_ref() == self.original_source
            if context is not None:
                assert id(context) == self.original_context_id
                self.assert_full_ledger(context)
        if context is not None:
            assert context.manifest is self.manifest
            assert context.validation_keys is self.validation_keys
            assert context.stored_validation_keys is self.stored_keys
            assert context.max_train_entries == 100

    def run(self):
        train, validation = self.holder._prepare_training_split(use_train_cache=self.staged)
        assert self.activated is self.manifest
        assert self.manifest["validation_leakage"] == self.leakage
        if self.train_cache == "hit" and self.staged:
            assert train == []
            assert "train_parse" not in self.calls
            assert self.holder._preloaded_snapshot_dataset is self.cached_train
        else:
            assert [corpus.canonical_state_key(row.state) for row in train] == (
                self.expected_train_keys)
            assert self.holder._preloaded_snapshot_dataset is None
        if self.validation_cache == "hit" and self.staged:
            assert validation == []
            assert "validation_parse" not in self.calls
            assert self.holder._preloaded_validation_dataset is self.cached_validation
        else:
            assert [corpus.canonical_state_key(row.state) for row in validation] == [
                self.expected_validation_key]
            assert self.holder._preloaded_validation_dataset is None


@pytest.mark.parametrize("train_cache", ["hit", "miss", "disabled"])
@pytest.mark.parametrize("validation_cache", ["hit", "miss"])
def test_startup_retires_copied_ledger_before_train_cache_or_parse(
    tmp_path, monkeypatch, train_cache, validation_cache,
):
    harness = _StartupHarness(
        tmp_path, monkeypatch, train_cache=train_cache, validation_cache=validation_cache)
    harness.run()
    assert harness.calls[:4] == [
        "prepare", "reuse_identity", "validation_metadata", "validation_cache"]
    assert harness.holder._pending_validation_reuse_identity["leak_fingerprints"]
    assert ("train_cache" in harness.calls) == (train_cache != "disabled")


@pytest.mark.parametrize("validation_cache", ["hit", "miss"])
def test_startup_preserves_external_context_alias(tmp_path, monkeypatch, validation_cache):
    harness = _StartupHarness(
        tmp_path, monkeypatch, keep_alias=True, validation_cache=validation_cache)
    harness.run()
    # A caller may still use the original general-purpose context for validation.
    assert harness.manager.load_validation_entries(harness.alias)
    assert harness.alias.historically_trained == harness.original_source


@pytest.mark.parametrize("manager_kind,context_kind", [
    ("subclass", "exact"), ("custom", "exact"),
    ("exact", "subclass"), ("exact", "custom"),
])
def test_startup_preserves_custom_split_contracts(
    tmp_path, monkeypatch, manager_kind, context_kind,
):
    _StartupHarness(
        tmp_path, monkeypatch, manager_kind=manager_kind, context_kind=context_kind).run()


@pytest.mark.parametrize("failure", ["metadata", "cache", "parse"])
def test_validation_failure_prevents_train_work_and_activation(tmp_path, monkeypatch, failure):
    harness = _StartupHarness(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match="controlled validation failure") as error:
        harness.run()
    assert error.value is harness.failure
    assert not {"train_metadata", "train_cache", "train_parse", "activate"}.intersection(
        harness.calls)
    assert harness.source_membership == harness.original_source
    assert harness.activated is None


def test_startup_validation_parser_checks_ledger_beyond_stored_keys(tmp_path, monkeypatch):
    _StartupHarness(tmp_path, monkeypatch, hidden_validation=True).run()


def test_invalid_validation_cache_accounting_uses_full_ledger_fallback(tmp_path, monkeypatch):
    harness = _StartupHarness(tmp_path, monkeypatch, validation_cache="malformed")
    harness.run()
    assert harness.calls.index("validation_parse") < harness.calls.index("train_metadata")


def test_unstaged_general_split_keeps_full_context_semantics(tmp_path, monkeypatch):
    harness = _StartupHarness(tmp_path, monkeypatch, staged=False)
    harness.run()
    assert harness.calls == ["prepare", "validation_parse", "train_parse", "activate"]
