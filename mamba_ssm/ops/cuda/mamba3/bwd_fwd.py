"""Hand-written CUDA replacement for the TileLang ``mamba_mimo_bwd_fwd`` kernel (see bwd_fwd.cu).

Opt-in with ``M3_CUDA_BWD=1`` (together with the CUDA bwd_bwd), same configuration rules: returns None outside
the Olala configuration, without the pre-gate RMSNorm fusion, or if the JIT build fails, so the caller keeps the
TileLang kernel. GH200, H=12 per TP rank, S=65536, two-level: Pass A 0.50 ms vs 0.55, Pass C 2.52 ms vs 2.93.
"""

import torch

from mamba_ssm.ops.cuda.mamba3.bwd_bwd import applies, build

# TileLang argument positions the CUDA kernel does not take: OUT_NORM_WEIGHT, DOUT_NORM_WEIGHT, DOUT_PRE_RMS, NS_ANCHOR
_UNUSED = (8, 10, 11, 25)


def cuda_bwd_fwd(B, H, G, N, P, R, hasZ, hasD, reduceO, fuse_pregate_headwise_rms_norm=False, isVarlen=True,
                 chunk_size=16, rotary_dim_divisor=4, dtype="bfloat16", states_dtype=torch.bfloat16,
                 has_init_state=False, state_only=False, blocked=False, **_):
    """A callable with the TileLang kernel's argument list, or None when this kernel does not apply."""
    if fuse_pregate_headwise_rms_norm or not applies(B, G, N, P, R, hasZ, hasD, reduceO, False, isVarlen, chunk_size,
                                                     rotary_dim_divisor, dtype, states_dtype):
        return None
    ext = build("bwd_fwd")
    if ext is None:
        return None
    flags = (bool(blocked), bool(has_init_state), bool(state_only))

    def run(*args):
        ext.bwd_fwd(*[a for i, a in enumerate(args) if i not in _UNUSED], *flags)

    return run
