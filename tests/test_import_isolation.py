from __future__ import annotations

import subprocess
import sys


def test_mamba2_triton_import_does_not_initialize_mamba3_backends() -> None:
    script = r'''
import sys

from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state_varlen
from mamba_ssm import Mamba2

assert callable(chunk_state_varlen)
assert Mamba2.__name__ == "Mamba2"
assert "mamba_ssm.modules.mamba3" not in sys.modules
assert not any(name == "tilelang" or name.startswith("tilelang.") for name in sys.modules)
assert not any(name == "tvm" or name.startswith("tvm.") for name in sys.modules)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
