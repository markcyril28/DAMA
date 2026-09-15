"""Fork cleanup must release writers even when the child has no spare fd."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from dama.ai.ml import fork_writers


pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="requires POSIX fork and descriptor limits",
)


_PROBE = r'''
import contextlib
import errno
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys

sys.path.insert(0, sys.argv[2])
from dama.ai.ml import fork_writers

directory = Path(sys.argv[1])
failure = sys.argv[3]
parent_pid = os.getpid()
null_rdev = os.stat(os.devnull).st_rdev
read_fd, write_fd = os.pipe()
with contextlib.ExitStack() as stack:
    raw_fd, raw_name = stack.enter_context(
        fork_writers.fork_safe_mkstemp(dir=directory))
    named = stack.enter_context(
        fork_writers.fork_safe_temporary_file(dir=directory, delete=False))
    descriptors = [raw_fd, named.fileno()]
    identities = [(os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in descriptors]
    occupied = []
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    if failure == "exhaustion":
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, limits[1]))
        while True:
            try:
                occupied.append(os.open(os.devnull, os.O_RDWR))
            except OSError as error:
                assert error.errno == errno.EMFILE
                break
    else:
        operation_name = "open" if failure == "full_stderr" else failure
        operation = getattr(os, operation_name)
        def fail_in_child(*args, **kwargs):
            if os.getpid() != parent_pid:
                raise OSError(errno.EIO, "injected child redirection failure")
            return operation(*args, **kwargs)
        setattr(fork_writers.os, operation_name, fail_in_child)

    saved_stderr = None
    if failure == "full_stderr":
        blocked_read_fd, blocked_write_fd = os.pipe()
        os.set_blocking(blocked_write_fd, False)
        while True:
            try:
                os.write(blocked_write_fd, b"x" * 4096)
            except BlockingIOError:
                break
        os.set_blocking(blocked_write_fd, True)
        saved_stderr = os.dup(2)
        os.dup2(blocked_write_fd, 2)

    child_pid = os.fork()
    if child_pid == 0:
        report = []
        for fd, identity in zip(descriptors, identities):
            info = os.fstat(fd)
            report.append({
                "retained_writer": (info.st_dev, info.st_ino) == identity,
                "holds_null_device": stat.S_ISCHR(info.st_mode) and info.st_rdev == null_rdev,
            })
            # A stray flush from an inherited stream must remain harmless.
            os.write(fd, b"child write must be discarded")
        os.write(write_fd, json.dumps(report).encode("ascii"))
        os._exit(0)

    if saved_stderr is not None:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(blocked_write_fd)
        # Keep the read end open without draining it. A diagnostic emitted
        # before exit would block; kill that child so this control stays bounded.
        signal.signal(signal.SIGALRM, lambda *_args: os.kill(child_pid, signal.SIGKILL))
        signal.alarm(2)
    for fd in occupied:
        os.close(fd)
    resource.setrlimit(resource.RLIMIT_NOFILE, limits)
    os.close(write_fd)
    child_report = os.read(read_fd, 4096)
    os.close(read_fd)
    _, status = os.waitpid(child_pid, 0)
    if saved_stderr is not None:
        signal.alarm(0)
        os.close(blocked_read_fd)
    os.write(raw_fd, b"parent raw")
    named.write(b"parent named")

assert Path(raw_name).read_bytes() == b"parent raw"
assert Path(named.name).read_bytes() == b"parent named"
assert not fork_writers._FORK_CHILD_DROPPED_FDS
Path(raw_name).unlink()
Path(named.name).unlink()
print(json.dumps({
    "exit_code": os.waitstatus_to_exitcode(status),
    "child_report": json.loads(child_report) if child_report else None,
}))
'''


@pytest.mark.parametrize("failure", ["exhaustion", "open", "dup2", "full_stderr"])
def test_fork_child_does_not_continue_with_inherited_writers(tmp_path, failure):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROBE), str(tmp_path),
         str(Path(fork_writers.__file__).resolve().parents[3]), failure],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    if failure == "exhaustion":
        assert report == {
            "exit_code": 0,
            "child_report": [
                {"retained_writer": False, "holds_null_device": True},
                {"retained_writer": False, "holds_null_device": True},
            ],
        }
        assert result.stderr == ""
    else:
        assert report == {"exit_code": 1, "child_report": None}
        assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
