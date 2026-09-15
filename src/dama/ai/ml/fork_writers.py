"""Keep temporary writers out of concurrently forked worker processes.

DrvFS can hide an atomically replaced public name while a fork child retains
its writer. Serialize open/register and close/unregister against fork, then
redirect registered descriptors in each child without freeing their numbers.
"""

from contextlib import contextmanager
import errno
import os
import tempfile
import threading
from typing import Any, Iterator, NoReturn


_FORK_CHILD_DROPPED_FDS: set[int] = set()
_fork_writer_lifecycle_lock = threading.Lock()


def _abort_fork_child() -> NoReturn:
    """Release inherited writers and let the pool report the failed worker."""
    # At-fork exceptions are ignored, and reporting through a full stderr
    # pipe can block with writers still open. Exit directly to release them.
    os._exit(1)


def _drop_registered_fds_in_fork_child() -> None:
    """Point registered writer descriptors at the null device in a fork child."""
    descriptors = tuple(_FORK_CHILD_DROPPED_FDS)
    _FORK_CHILD_DROPPED_FDS.clear()
    if not descriptors:
        return
    try:
        try:
            null_fd = os.open(os.devnull, os.O_RDWR)
        except OSError as error:
            if error.errno != errno.EMFILE:
                raise
            # Only this thread survives fork. Free one writer slot so the
            # null device can reuse it even with a full descriptor table.
            os.close(descriptors[0])
            null_fd = os.open(os.devnull, os.O_RDWR)
    except OSError:
        _abort_fork_child()
    try:
        for descriptor in descriptors:
            # Keep the number occupied: a copied stream's stray flush must
            # never reach an unrelated file opened later by this worker.
            try:
                os.dup2(null_fd, descriptor)
            except OSError:
                _abort_fork_child()
    finally:
        # EMFILE recovery may put the null device in a registered writer's
        # slot. Leave that number occupied for its copied stream's lifetime.
        if null_fd not in descriptors:
            os.close(null_fd)


def _finish_fork_in_child() -> None:
    try:
        _drop_registered_fds_in_fork_child()
    finally:
        _fork_writer_lifecycle_lock.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_fork_writer_lifecycle_lock.acquire,
        after_in_parent=_fork_writer_lifecycle_lock.release,
        after_in_child=_finish_fork_in_child,
    )


@contextmanager
def _fork_children_drop_fd(descriptor: int) -> Iterator[None]:
    """Protect an already-open descriptor only for the duration of this block.

    The caller owns opening/closing gaps. Temporary writers should use
    ``fork_safe_temporary_file`` to protect their complete descriptor lifetime.
    """
    with _fork_writer_lifecycle_lock:
        _FORK_CHILD_DROPPED_FDS.add(descriptor)
    try:
        yield
    finally:
        with _fork_writer_lifecycle_lock:
            _FORK_CHILD_DROPPED_FDS.discard(descriptor)


@contextmanager
def fork_safe_mkstemp(**kwargs: Any) -> Iterator[tuple[int, str]]:
    """Own a raw temporary descriptor through close, excluding concurrent forks.

    Wrappers must use ``closefd=False``. The caller publishes and removes the
    temporary name after this block; a failed raw close must not be retried.
    """
    with _fork_writer_lifecycle_lock:
        descriptor, name = tempfile.mkstemp(**kwargs)
        try:
            _FORK_CHILD_DROPPED_FDS.add(descriptor)
        except BaseException:
            # No filename has reached the caller yet, so clean it here too.
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.unlink(name)
            except OSError:
                pass
            raise
    body_failed = True
    try:
        yield descriptor, name
        body_failed = False
    finally:
        with _fork_writer_lifecycle_lock:
            try:
                os.close(descriptor)
            except OSError:
                # Preserve construction/write errors over secondary cleanup.
                if not body_failed:
                    raise
            finally:
                _FORK_CHILD_DROPPED_FDS.discard(descriptor)


@contextmanager
def fork_safe_temporary_file(**kwargs: Any) -> Iterator[Any]:
    """Create a NamedTemporaryFile whose entire open lifetime is fork-safe."""
    with _fork_writer_lifecycle_lock:
        temporary = tempfile.NamedTemporaryFile(**kwargs)
        try:
            descriptor = temporary.fileno()
            _FORK_CHILD_DROPPED_FDS.add(descriptor)
        except BaseException:
            # The caller has not received the temporary name yet. Keep the
            # registration error and remove the name even with delete=False.
            try:
                temporary.close()
            except OSError:
                pass
            try:
                os.unlink(temporary.name)
            except OSError:
                pass
            raise
    body_failed = True
    try:
        yield temporary
        body_failed = False
    finally:
        # close() can release the GIL, including during buffered flushes. A
        # registration alone cannot protect this transition against os.fork().
        with _fork_writer_lifecycle_lock:
            try:
                temporary.close()
            except OSError:
                # A buffered close can fail after a write/interruption did.
                # Preserve that first failure, but refuse a failed healthy close.
                if not body_failed:
                    raise
            finally:
                _FORK_CHILD_DROPPED_FDS.discard(descriptor)
