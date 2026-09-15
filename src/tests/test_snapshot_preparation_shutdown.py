"""Shutdown at staged snapshot boundaries must not begin unused work."""

import threading
from types import SimpleNamespace

import pytest

import dama.ai.ml.trainer as trainer_module
from dama.ai.ml.trainer import Trainer


_HANDOFF_FIELDS = (
    "_bg_selfplay_entries",
    "_bg_selfplay_dataset",
    "_bg_selfplay_incremental",
    "_bg_snapshot_manifest",
    "_bg_validation_entries",
    "_bg_validation_identity",
)


class _PreparationHarness:
    def __init__(self, monkeypatch, route, stop_stage=None):
        self.calls = []
        self.stage_entered = threading.Event()
        self.release_stage = threading.Event()
        self.release_next_cycle = threading.Event()
        self.stop_stage = stop_stage
        self.context = SimpleNamespace(manifest={
            "version": 2,
            "metrics": {"fresh_unique_state_rate": 0.6},
        })
        self.validation_entries = [object()]
        self.validation_identity = object()
        self.train_entries = [object()]
        self.dataset = [object(), object()]
        self.source_path = "snapshot/manifest.json"
        self.old_handoff = {field: object() for field in _HANDOFF_FIELDS}

        holder = object.__new__(Trainer)
        self.holder = holder
        holder.config = SimpleNamespace(
            replay_max_files=60,
            replay_max_entries=1_000_000,
            max_moves_per_sample=32,
        )
        holder.replay_buffer = SimpleNamespace(cleanup_old_files=lambda: 0)
        holder._bg_selfplay_thread = None
        holder._bg_selfplay_stop_event = threading.Event()
        holder._data_ready_event = threading.Event()
        holder._bg_selfplay_lock = threading.Lock()
        holder._stopped = False
        holder._paused = False
        holder._corpus_settings = lambda **_kwargs: ({}, {}, {})
        holder._train_tensor_window = object()
        holder._validation_tensor_identity = object()
        self.old_train_window = holder._train_tensor_window
        self.old_validation_identity = holder._validation_tensor_identity
        for field, value in self.old_handoff.items():
            setattr(holder, field, value)

        generated = False

        def generate(*_args, **_kwargs):
            nonlocal generated
            if generated:
                # Keep a healthy publication available for inspection until
                # the parent requests shutdown. No second admission can race it.
                self.release_next_cycle.wait(timeout=10)
                holder._bg_selfplay_stop_event.set()
                return 1, 23
            generated = True
            self.calls.append("generate")
            return 1, 23

        def consider(**_kwargs):
            self.calls.append("consider")
            return SimpleNamespace(
                admitted=True, manifest_path=self.source_path)

        def prepare(path, max_train_entries=0):
            self.calls.append(("prepare", path, max_train_entries))
            self._hold_stage("prepare")
            return self.context

        def validation(context):
            self.calls.append(("validation", context))
            self._hold_stage("validation")
            return self.validation_entries, self.validation_identity

        def train(context):
            self.calls.append(("train", context))
            return self.train_entries

        def per_file(context, should_abort=None):
            self.calls.append(("per_file", context))
            # Record dispatch even when its normal first check would abort;
            # a boundary stop should not call the next phase at all.
            if should_abort is not None and should_abort():
                return None
            return self.dataset

        def tensorize(entries, **kwargs):
            self.calls.append(("tensorize", entries, kwargs))
            return self.dataset

        manager = SimpleNamespace(
            consider_snapshot=consider,
            prepare_split=prepare,
            load_validation_entries=lambda _context: self.validation_entries,
            load_train_entries=train,
        )
        if route == "per_file":
            manager.load_train_file_entries = lambda *_args: []
            manager.train_cap_sample_indices = lambda *_args: None
        holder._snapshot_manager = manager
        holder.run_selfplay = generate
        holder._load_or_reuse_validation_entries = validation
        holder._load_or_reuse_train_dataset = per_file
        monkeypatch.setattr(
            trainer_module.CachedTensorDataset,
            "from_entries",
            staticmethod(tensorize),
        )

    def _hold_stage(self, stage):
        if stage == self.stop_stage:
            self.stage_entered.set()
            if not self.release_stage.wait(timeout=10):
                self.calls.append("stage_handshake_timed_out")
                self.holder._bg_selfplay_stop_event.set()

    def start(self):
        self.holder._start_background_selfplay(72)

    def close(self):
        self.holder._bg_selfplay_stop_event.set()
        self.release_stage.set()
        self.release_next_cycle.set()
        thread = self.holder._bg_selfplay_thread
        if thread is not None:
            thread.join(timeout=10)
            assert not thread.is_alive()

    def expected_prefix(self):
        return [
            "generate",
            "consider",
            ("prepare", self.source_path, 1_000_000),
        ]


@pytest.mark.parametrize("route", ["per_file", "fallback"])
@pytest.mark.parametrize("stop_stage", ["prepare", "validation"])
@pytest.mark.parametrize("stop_signal", ["flag", "event"])
def test_staged_snapshot_stop_preserves_previous_handoff(
    monkeypatch, capsys, route, stop_stage, stop_signal,
):
    harness = _PreparationHarness(monkeypatch, route, stop_stage)
    holder = harness.holder
    try:
        harness.start()
        assert harness.stage_entered.wait(timeout=10)
        if stop_signal == "flag":
            holder._stopped = True
        else:
            holder._bg_selfplay_stop_event.set()
        harness.release_stage.set()
        holder._bg_selfplay_thread.join(timeout=10)
        assert not holder._bg_selfplay_thread.is_alive()
        # In flag cases the independent producer event stays clear until
        # cleanup, proving the flag alone terminated the actual worker.
        assert holder._bg_selfplay_stop_event.is_set() == (stop_signal == "event")
    finally:
        harness.close()

    expected = harness.expected_prefix()
    if stop_stage == "validation":
        expected.append(("validation", harness.context))
    assert harness.calls == expected
    for field, value in harness.old_handoff.items():
        assert getattr(holder, field) is value
    assert holder._train_tensor_window is harness.old_train_window
    assert holder._validation_tensor_identity is harness.old_validation_identity
    assert not holder._data_ready_event.is_set()
    captured = capsys.readouterr()
    assert "Background snapshot self-play error" not in captured.out
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("route", ["per_file", "fallback"])
def test_staged_snapshot_without_stop_delivers_exact_objects(
    monkeypatch, capsys, route,
):
    harness = _PreparationHarness(monkeypatch, route)
    holder = harness.holder
    try:
        harness.start()
        assert holder._data_ready_event.wait(timeout=10)
        with holder._bg_selfplay_lock:
            assert holder._bg_selfplay_entries is None
            assert holder._bg_selfplay_dataset is harness.dataset
            assert holder._bg_selfplay_incremental is None
            assert holder._bg_snapshot_manifest is harness.context.manifest
            assert holder._bg_validation_entries is harness.validation_entries
            assert holder._bg_validation_identity is harness.validation_identity
    finally:
        harness.close()

    expected = harness.expected_prefix() + [("validation", harness.context)]
    if route == "per_file":
        expected.append(("per_file", harness.context))
    else:
        expected.extend([
            ("train", harness.context),
            ("tensorize", harness.train_entries, {
                "max_moves_per_sample": 32, "show_progress": True,
            }),
        ])
    assert harness.calls == expected
    captured = capsys.readouterr()
    assert "Background snapshot self-play error" not in captured.out
    assert "Traceback" not in captured.err
