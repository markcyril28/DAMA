"""Exercise evaluator startup in a fresh interpreter and forkserver."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or "forkserver" not in mp.get_all_start_methods(),
    reason="The preloaded evaluation forkserver is specific to Linux",
)
def test_evaluation_forkserver_runs_cpu_math_without_initializing_cuda(tmp_path):
    # A real main module and a fresh interpreter prevent an earlier pool or
    # pytest's imported libraries from hiding a broken forkserver preload.
    probe = tmp_path / "evaluation_startup_probe.py"
    probe.write_text(textwrap.dedent("""\
        import json
        import os
        from concurrent.futures import ProcessPoolExecutor


        def inspect_worker():
            import numpy as np
            import torch

            before_cuda = torch.cuda.is_initialized()
            left = [[1.0, 2.0], [3.0, 4.0]]
            right = [[5.0, 6.0], [7.0, 8.0]]
            numpy_result = (np.array(left) @ np.array(right)).tolist()
            torch_result = (torch.tensor(left) @ torch.tensor(right)).tolist()
            return {
                "pid": os.getpid(),
                "intra_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "before_cuda": before_cuda,
                "after_cuda": torch.cuda.is_initialized(),
                "numpy_result": numpy_result,
                "torch_result": torch_result,
            }


        if __name__ == "__main__":
            # Match the trainer's parent import order. The isolated server
            # still has to initialize its own numerical libraries correctly.
            import numpy
            import torch
            from dama.ai.ml.model_vs_algo import (
                _evaluation_worker_context,
                _evaluation_worker_init,
            )

            context = _evaluation_worker_context()
            assert context.get_start_method() == "forkserver"
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=_evaluation_worker_init,
            ) as pool:
                worker = pool.submit(inspect_worker).result(timeout=20)
            print(json.dumps({"parent_pid": os.getpid(), "worker": worker}))
        """), encoding="utf-8")

    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(project_root / "src"), str(project_root), env.get("PYTHONPATH"),
    )))
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_MAX_THREADS="1",
        # This is the local Conda runtime's observed threading layer. Pin it
        # only in the probe so ambient GNU/force overrides cannot mask failure.
        MKL_THREADING_LAYER="INTEL",
    )
    env.pop("MKL_SERVICE_FORCE_INTEL", None)
    with subprocess.Popen(
        [sys.executable, str(probe)],
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            # Pool shutdown may itself wait on a stuck worker. Stop this
            # probe's isolated group, including its forkserver and workers.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            pytest.fail("Evaluator startup timed out:\n" + stdout + stderr)

    assert process.returncode == 0, stdout + stderr
    payload = json.loads(stdout.strip().splitlines()[-1])
    worker = payload["worker"]
    assert worker["pid"] != payload["parent_pid"]
    assert worker["intra_threads"] == worker["interop_threads"] == 1
    assert worker["before_cuda"] is False
    assert worker["after_cuda"] is False
    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert worker["numpy_result"] == expected
    assert worker["torch_result"] == expected
