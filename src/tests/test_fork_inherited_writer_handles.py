"""Fork children must not inherit a durable writer's open temporary.

On the WSL DrvFS project volume, a forked child that still holds a writer's
temporary keeps the replaced public name unopenable (ENOENT while it remains
listed) until that child exits. Self-play pools fork from the producer thread
while the training and checkpoint threads publish statistics and checkpoints.
The ENOENT itself only reproduces on DrvFS, so these tests pin the
platform-independent contract: a child forked while a registered writer is
open holds no descriptor on that writer's file.
"""

import json
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from dama.ai.ml import trainer as trainer_module
from dama.ai.ml.trainer import Trainer


pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork") or not Path("/proc/self/fd").is_dir(),
    reason="descriptor inheritance is observed through fork and /proc/self/fd",
)


def _identity(descriptor: int) -> tuple[int, int]:
    info = os.fstat(descriptor)
    return info.st_dev, info.st_ino


def _handles_on(identity: tuple[int, int]) -> int:
    count = 0
    for name in os.listdir("/proc/self/fd"):
        try:
            info = os.stat(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if (info.st_dev, info.st_ino) == tuple(identity):
            count += 1
    return count


def _in_fork_child(probe):
    """Fork now, run ``probe`` in the child and return its JSON result."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            os.write(write_fd, json.dumps(probe()).encode("utf-8"))
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        with os.fdopen(read_fd, "rb") as reader:
            payload = reader.read()
    finally:
        os.waitpid(pid, 0)
    return json.loads(payload)


def test_fork_child_releases_only_registered_writer_descriptors(
    tmp_path: Path,
) -> None:
    registered_path = tmp_path / "registered.tmp"
    unregistered_path = tmp_path / "unregistered.tmp"
    null_rdev = os.stat(os.devnull).st_rdev

    with registered_path.open("w") as registered, \
            unregistered_path.open("w") as unregistered:
        registered_fd = registered.fileno()
        registered_identity = _identity(registered_fd)
        unregistered_identity = _identity(unregistered.fileno())

        def probe():
            info = os.fstat(registered_fd)
            return {
                "registered": _handles_on(registered_identity),
                "unregistered": _handles_on(unregistered_identity),
                "number_holds_null_device": (
                    stat.S_ISCHR(info.st_mode) and info.st_rdev == null_rdev),
            }

        # Control: an unregistered writer is inherited on this platform, so
        # the registered case below can fail.
        assert _in_fork_child(probe)["registered"] == 1

        with trainer_module._fork_children_drop_fd(registered_fd):
            report = _in_fork_child(probe)
            registered.write("{}")
        assert trainer_module._FORK_CHILD_DROPPED_FDS == set()

    # The child keeps the number occupied, so a stray flush from its copied
    # file object cannot reach a newly opened file. The parent's writer is
    # unaffected.
    assert report == {
        "registered": 0,
        "unregistered": 1,
        "number_holds_null_device": True,
    }
    assert registered_path.read_text() == "{}"


def test_stats_writer_temporary_is_not_inherited_by_a_concurrent_fork(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "training_stats.json"
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        stats_file=str(stats_path),
        log_dir=str(tmp_path),
        recovery_enforced=False,
    )
    holder.stats = trainer_module.TrainingStats()
    holder.step = 300
    holder._stats_write_lock = threading.RLock()
    holder._stats_snapshot_generation = 0
    holder._stats_persisted_generation = -1
    holder._update_training_progress_report = lambda _path: None

    real_dump = json.dump
    child_handles = []

    def dump_while_a_pool_forks(obj, handle, *args, **kwargs):
        identity = _identity(handle.fileno())
        child_handles.append(_in_fork_child(lambda: _handles_on(identity)))
        return real_dump(obj, handle, *args, **kwargs)

    monkeypatch.setattr(trainer_module.json, "dump", dump_while_a_pool_forks)

    assert Trainer._save_stats(holder) is True

    assert child_handles == [0]
    assert trainer_module._FORK_CHILD_DROPPED_FDS == set()
    assert json.loads(stats_path.read_text())["total_steps"] == 300


@pytest.mark.parametrize("boundary", ["open", "close"])
def test_stats_writer_protects_temporary_lifecycle_from_forks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, boundary: str,
) -> None:
    """A pool can fork before registration or after its body unregisters."""
    stats_path = tmp_path / "training_stats.json"
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        stats_file=str(stats_path), log_dir=str(tmp_path),
        recovery_enforced=False,
    )
    holder.stats = trainer_module.TrainingStats()
    holder.step = 300
    holder._stats_write_lock = threading.RLock()
    holder._stats_snapshot_generation = 0
    holder._stats_persisted_generation = -1
    holder._update_training_progress_report = lambda _path: None

    boundary_reached = threading.Event()
    release_boundary = threading.Event()
    fork_started = threading.Event()
    fork_finished = threading.Event()
    identity = []
    child_handles = []
    failures = []
    real_temporary = trainer_module.tempfile.NamedTemporaryFile

    def pause_at_boundary(handle):
        identity[:] = [_identity(handle.fileno())]
        boundary_reached.set()
        assert release_boundary.wait(10), "temporary lifecycle was never released"

    class PausedTemporary:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            if boundary == "close":
                pause_at_boundary(self.handle)
            self.handle.close()

    def temporary(**kwargs):
        handle = real_temporary(**kwargs)
        if boundary == "open":
            pause_at_boundary(handle)
        return PausedTemporary(handle)

    def write_stats():
        try:
            assert Trainer._save_stats(holder) is True
        except BaseException as exc:
            failures.append(exc)

    def fork_pool():
        try:
            fork_started.set()
            child_handles.append(_in_fork_child(lambda: _handles_on(identity[0])))
        except BaseException as exc:
            failures.append(exc)
        finally:
            fork_finished.set()

    monkeypatch.setattr(trainer_module.tempfile, "NamedTemporaryFile", temporary)
    writer = threading.Thread(target=write_stats)
    forker = threading.Thread(target=fork_pool)
    writer.start()
    try:
        assert boundary_reached.wait(10)
        forker.start()
        assert fork_started.wait(10)
        # Old code allows this fork to finish with the writer still open.
        # Protected lifecycle changes hold it until registration/close finishes.
        fork_finished.wait(0.25)
    finally:
        release_boundary.set()
        writer.join(10)
        if forker.ident is not None:
            forker.join(10)
    assert not writer.is_alive() and not forker.is_alive()
    assert failures == []
    assert child_handles == [0]
    assert trainer_module._FORK_CHILD_DROPPED_FDS == set()
    assert json.loads(stats_path.read_text())["total_steps"] == 300


def test_numbered_checkpoint_temporary_is_not_inherited_by_a_concurrent_fork(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model_step_002000.pt"
    holder = object.__new__(Trainer)
    holder.config = trainer_module.TrainingConfig(
        checkpoint_dir=str(checkpoint_dir),
        latest_path=str(tmp_path / "latest.pt"),
    )
    holder.model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.tensor([1.0])},
        arch_params={},
    )
    holder.optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {}, "param_groups": []})
    holder.stats = trainer_module.TrainingStats()
    holder.step = 2000
    holder.epoch = 1
    holder.scheduler = None
    holder.scaler = None
    holder.stats_collector = None
    holder.log_file = str(tmp_path / "train.jsonl")
    holder.device = torch.device("cpu")
    holder._checkpoint_thread = None
    holder._active_snapshot_manifest = {}
    holder._evaluate_validation_loss = lambda: None
    holder._evaluate_teacher_promotion = lambda _path: None
    holder._live_optimizer_context = lambda: {}
    holder._snapshot_stats = lambda: {}
    holder._save_stats = lambda **_kwargs: None
    holder._put_status = lambda _message: None
    holder._prune_old_checkpoints = lambda _path: []

    real_fsync = os.fsync
    child_handles = []

    def fsync_while_a_pool_forks(descriptor):
        # The first regular-file sync happens while the numbered temporary is
        # open, the window in which a self-play pool fork used to inherit it.
        if not child_handles and stat.S_ISREG(os.fstat(descriptor).st_mode):
            identity = _identity(descriptor)
            child_handles.append(
                _in_fork_child(lambda: _handles_on(identity)))
        return real_fsync(descriptor)

    monkeypatch.setattr(trainer_module.os, "fsync", fsync_while_a_pool_forks)

    Trainer._save_checkpoint(holder, loss=0.5)
    Trainer._wait_for_checkpoint_writer(holder, timeout=30)

    assert child_handles == [0]
    assert trainer_module._FORK_CHILD_DROPPED_FDS == set()
    assert torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )["step"] == 2000


def test_copied_checkpoint_alias_writer_is_not_inherited_by_a_fork(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "numbered.pt"
    destination = tmp_path / "latest.pt"
    source.write_bytes(b"complete checkpoint")
    destination.write_bytes(b"previous checkpoint")
    child_handles = []
    real_fsync = os.fsync

    def fail_link(*_args):
        raise OSError("force alias copy fallback")

    def fsync_while_a_pool_forks(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            identity = _identity(descriptor)
            child_handles.append(_in_fork_child(lambda: _handles_on(identity)))
            assert destination.read_bytes() == b"previous checkpoint"
        return real_fsync(descriptor)

    monkeypatch.setattr(trainer_module.os, "link", fail_link)
    monkeypatch.setattr(trainer_module.os, "fsync", fsync_while_a_pool_forks)
    Trainer._publish_checkpoint_alias(source, destination)

    assert child_handles == [0]
    assert trainer_module._FORK_CHILD_DROPPED_FDS == set()
    assert destination.read_bytes() == source.read_bytes()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["latest.pt", "numbered.pt"]
