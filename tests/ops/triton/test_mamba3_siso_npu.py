"""Ascend NPU smoke test for the migrated Mamba-3 SISO operators.

This exercises the two public entry points used on Ascend --

``mamba3_siso_step`` (recurrent step) and ``mamba3_siso_combined``
(chunked forward + backward) -- on an Ascend NPU and compares them against the
pure-PyTorch reference implementations that ship with the upstream test module
``tests/ops/triton/test_mamba3_siso.py``.

**Which step implementation runs.** ``mamba3_siso_step`` now defaults to the
Ascend C operator ``mamba_ssm.ops.ascendc.mamba3_siso_step`` (the Triton
``mamba3_siso_step`` kernel replaced by ``mamba3_step_fn`` on the NPU).  Set
``MAMBA3_SISO_STEP_IMPL=triton`` to run the same cases against the previous
Triton implementation, which is kept for the CUDA path and for A/B runs.

The reference helpers (``create_mamba3_siso_inputs``, ``mamba3_siso_step_ref``,
``mamba3_siso_fwd_ref``, ``relative_error``) are device-agnostic and are reused
verbatim; only the device string is switched to ``"npu"``.

Run inside the NPU container (from the repo root):

    python tests/ops/triton/test_mamba3_siso_npu.py

Requires ``torch_npu`` and ``triton-ascend`` (plus the built
``libmamba3_decode_step_ops.so`` for the Ascend C step); exits non-zero on
failure.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import torch

try:  # register the "npu" device
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - CUDA-only environment
    torch_npu = None

import triton

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reference implementations and helpers from the upstream (device-agnostic) test.
from test_mamba3_siso import (  # noqa: E402
    create_mamba3_siso_inputs,
    mamba3_siso_fwd_ref,
    mamba3_siso_step_ref,
    relative_error,
)

# Step implementation: Ascend C operator by default, Triton on request.
SISO_STEP_IMPL = os.environ.get("MAMBA3_SISO_STEP_IMPL", "ascendc").strip().lower()
if SISO_STEP_IMPL == "triton":
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step  # noqa: E402
elif SISO_STEP_IMPL == "ascendc":
    from mamba_ssm.ops.ascendc.mamba3_siso_step import mamba3_siso_step  # noqa: E402
else:
    raise SystemExit(
        f"unknown MAMBA3_SISO_STEP_IMPL={SISO_STEP_IMPL!r} (expected 'ascendc' or 'triton')"
    )
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined  # noqa: E402

DEVICE = "npu"
DTYPE = torch.bfloat16


def _require_npu() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("Ascend NPU not available.")
    torch.npu.set_device(0)
    print(
        f"[env] torch={torch.__version__} triton={triton.__version__} "
        f"npu_devices={torch.npu.device_count()} siso_step_impl={SISO_STEP_IMPL}"
    )


def test_angle_dt(seqlen: int = 200, chunk_size: int = 64, rtol: float = 1e-2) -> None:
    """Single-operator check for ``angle_dt_fwd`` vs a pure-PyTorch cumsum.

    The output is ``mod 2*pi``, so equality is checked through ``sin``/``cos``
    (periodic equivalence) rather than directly.
    """
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd

    torch.random.manual_seed(0)
    batch, nheads, dim = 2, 4, 8
    angle = torch.randn(batch, seqlen, nheads, dim, device=DEVICE, dtype=torch.float32)
    dt = torch.rand(batch, nheads, seqlen, device=DEVICE, dtype=torch.float32) * 0.1

    out, output_state = angle_dt_fwd(
        angle, dt, chunk_size=chunk_size, return_output_state=True
    )

    two_pi = 2.0 * math.pi
    vals = torch.tanh(angle) * math.pi * dt.permute(0, 2, 1).unsqueeze(-1)
    ref = torch.cumsum(vals, dim=1)
    ref = ref - two_pi * torch.floor(ref / two_pi)

    sin_err = (torch.sin(out) - torch.sin(ref)).abs().max().item()
    cos_err = (torch.cos(out) - torch.cos(ref)).abs().max().item()
    print(f"[angle_dt] max |sin diff| = {sin_err:.3e}, max |cos diff| = {cos_err:.3e}")
    assert sin_err < rtol and cos_err < rtol, "angle_dt_fwd does not match reference"
    assert output_state.shape == (batch, nheads, dim)
    print("[ok] angle_dt_fwd\n")


def _small_inputs(seqlen, requires_grad=False):
    return create_mamba3_siso_inputs(
        batch=2, seqlen=seqlen, nheads=8, nheads_qk=2,
        headdim_qk=64, headdim_v=32,
        dtype=DTYPE, device=DEVICE,
        has_D=True, has_Z=True, has_input_states=True,
        requires_grad=requires_grad,
    )


def test_step(seqlen: int = 64, rtol: float = 5e-2) -> None:
    """Recurrent ``mamba3_siso_step`` vs reference recurrence."""
    torch.random.manual_seed(42)
    inp = _small_inputs(seqlen)

    angle_state, ssm_state, k_state, v_state = inp["Input_States"]
    outs = []
    for step in range(seqlen):
        o, states = mamba3_siso_step(
            inp["Q"][:, step].contiguous(), inp["K"][:, step].contiguous(),
            inp["V"][:, step].contiguous(),
            inp["ADT"][:, :, step].contiguous(), inp["DT"][:, :, step].contiguous(),
            inp["Trap"][:, :, step].contiguous(),
            inp["Q_bias"], inp["K_bias"], inp["Angles"][:, step].contiguous(),
            inp["D"], inp["Z"][:, step].contiguous(),
            (angle_state, ssm_state, k_state, v_state),
        )
        angle_state, ssm_state, k_state, v_state = states
        outs.append(o)
    outs = torch.stack(outs, dim=1)

    ref_out, ref_states = mamba3_siso_step_ref(
        inp["Q"], inp["K"], inp["V"], inp["ADT"], inp["DT"], inp["Trap"],
        inp["Q_bias"], inp["K_bias"], inp["Angles"], inp["D"], inp["Z"],
        Input_States=inp["Input_States"],
    )
    err = relative_error(outs, ref_out, name="step_out")
    print(f"[step] output relative error = {err:.2e}")
    assert err < rtol, f"step output error {err:.2e} >= {rtol}"

    names = ["angle", "ssm", "k", "v"]
    for name, ker, ref in zip(names, (angle_state, ssm_state, k_state, v_state), ref_states):
        e = relative_error(ker, ref, name=f"step_{name}", angle=(name == "angle"))
        print(f"[step] final {name} state error = {e:.2e}")
        assert e < rtol, f"step {name} state error {e:.2e} >= {rtol}"
    print("[ok] mamba3_siso_step\n")


def test_combined_forward_backward(seqlen: int = 128, chunk_size: int = 64,
                                   rtol: float = 5e-2) -> None:
    """Chunked ``mamba3_siso_combined`` forward + backward vs fwd reference."""
    torch.random.manual_seed(42)
    inp = _small_inputs(seqlen, requires_grad=True)
    inp_ref = copy.deepcopy(inp)

    ref_out, ref_final = mamba3_siso_fwd_ref(
        inp_ref["Q"], inp_ref["K"], inp_ref["V"], inp_ref["ADT"], inp_ref["DT"],
        inp_ref["Trap"], inp_ref["Q_bias"], inp_ref["K_bias"], inp_ref["Angles"],
        inp_ref["D"], inp_ref["Z"], inp_ref["Input_States"],
    )

    out_kernel, f_angle, f_ssm, f_k, f_v = mamba3_siso_combined(
        inp["Q"], inp["K"], inp["V"], inp["ADT"], inp["DT"], inp["Trap"],
        inp["Q_bias"], inp["K_bias"], inp["Angles"], inp["D"], inp["Z"],
        inp["Input_States"], chunk_size=chunk_size, return_final_states=True,
    )

    err = relative_error(out_kernel, ref_out, name="combined_out")
    print(f"[combined] forward output error = {err:.2e}")
    assert err < rtol, f"combined forward error {err:.2e} >= {rtol}"

    dO = torch.randn_like(ref_out)
    grads = [torch.randn_like(t) for t in (ref_final[0], ref_final[1], ref_final[2], ref_final[3])]
    torch.autograd.backward([ref_out, *ref_final], [dO, *grads])
    torch.autograd.backward([out_kernel, f_angle, f_ssm, f_k, f_v], [dO, *grads])

    for name in ["Q", "K", "V", "ADT", "DT", "Trap", "Q_bias", "K_bias", "Angles", "D", "Z"]:
        if inp[name] is None:
            continue
        e = relative_error(inp[name].grad, inp_ref[name].grad, name=f"d{name}")
        print(f"[combined] d{name} error = {e:.2e}")
        assert e < rtol, f"d{name} error {e:.2e} >= {rtol}"
    print("[ok] mamba3_siso_combined forward+backward\n")


def main() -> None:
    _require_npu()
    test_angle_dt()
    test_step()
    test_combined_forward_backward()
    print("[ALL PASSED]")


if __name__ == "__main__":
    main()
