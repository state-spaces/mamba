"""Hand-written CUDA (sm_90, wgmma) replacement for the TileLang ``mamba_mimo_fwd`` varlen kernel (see fwd.cu).

Opt-in with ``M3_CUDA_FWD=1``. Covers the Olala configuration (the one of the CUDA backward kernels, see bwd_bwd.py)
without per-sequence initial states, in its two forms: the full pass with ``return_state=False`` (stock call or
two-level Pass C, ``init_state`` = the state entering each block) and the state-only pass (two-level Pass A).
Anything else -- or a failed JIT build -- returns None and the caller keeps the TileLang kernel. The backward is
unaffected: the autograd function saves only inputs.
"""

import os

import torch
import triton
import triton.language as tl

from mamba_ssm.ops.cuda.mamba3.bwd_bwd import build


def applies(B, G, N, P, R, hasZ, hasD, reduceO, fuse_pregate_headwise_rms_norm, isVarlen, chunk_size,
            rotary_dim_divisor, dtype, has_initial_state, return_state, state_only):
    """M3_CUDA_FWD=1, the Olala configuration on an sm_90 GPU, and a mode the CUDA kernels implement."""
    return (
        os.environ.get("M3_CUDA_FWD", "0") == "1"
        and B == 1 and G == 1 and N == 128 and P == 64 and R == 4 and chunk_size == 16 and rotary_dim_divisor == 4
        and hasZ and hasD and reduceO and not fuse_pregate_headwise_rms_norm and isVarlen
        and str(dtype).replace("torch.", "") == "bfloat16"
        and not has_initial_state
        and (return_state if state_only else not return_state)
        and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
    )


@triton.jit
def _fwd_prep(
    DT, TRAP, DA_CS, CU, BLK_SEG, BLK_C0, BLK_NCH,
    TS, GAM, DACS, DTOT, CH_START, CH_LEN,
    S, H, BLOCKED: tl.constexpr, C: tl.constexpr,
):
    """Per (block, head): the trapezoid scales of the TileLang kernel (same bf16 roundings: DT and the shifted
    gamma held in bf16, the trap scale rounded by the CUDA kernel), the block-level log-decay cumsum rebuilt from
    the 16-token-group cumsums DA_CS (dacs, for the state-only pass) and its total, and the block's token range."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    seg = tl.load(BLK_SEG + b) if BLOCKED else b
    s0 = tl.load(CU + seg)
    s1 = tl.load(CU + seg + 1)
    c0 = tl.load(BLK_C0 + b) if BLOCKED else 0
    nch = tl.load(BLK_NCH + b) if BLOCKED else (s1 - s0 + C - 1) // C
    t0 = s0 + c0 * C
    ln = tl.minimum(nch * C, s1 - t0)
    if h == 0:
        tl.store(CH_START + b, t0)
        tl.store(CH_LEN + b, ln)
    i = tl.arange(0, C)
    run = 0.0
    for g in range(0, nch):
        t = t0 + g * C + i
        m = t < s1
        off = h * S + t
        dtb = tl.load(DT + off, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32)
        tr = tl.load(TRAP + off, mask=m, other=0.0).to(tl.float32)
        gam = dtb * tl.sigmoid(tr)
        nx = t + 1 < s1
        dtn = tl.load(DT + off + 1, mask=nx, other=0.0).to(tl.bfloat16).to(tl.float32)
        trn = tl.load(TRAP + off + 1, mask=nx, other=0.0).to(tl.float32)
        sh = tl.where(nx, (dtn * tl.sigmoid(-trn)).to(tl.bfloat16).to(tl.float32), 0.0)
        tl.store(TS + off, gam + sh, mask=m)
        tl.store(GAM + off, gam, mask=m)
        dcs = tl.load(DA_CS + off, mask=m, other=0.0)
        tl.store(DACS + off, run + dcs, mask=m)
        run += tl.sum(tl.where(t == tl.minimum(t0 + g * C + C, s1) - 1, dcs, 0.0), axis=0)
    tl.store(DTOT + b * H + h, run)


def cuda_fwd(B, H, G, N, P, R, hasZ, hasD, reduceO, fuse_pregate_headwise_rms_norm=False, isVarlen=True,
             chunk_size=16, rotary_dim_divisor=4, dtype="bfloat16", has_initial_state=False, return_state=False,
             state_only=False, **_):
    """A callable ``run(q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, D, mimo_z, angles, dA_cs, dA_cs_rev, dt, trap,
    cu_seqlens, blk_seg, blk_c0, blk_nch, init_state) -> (o, h, k_final)`` with the TileLang wrapper's outputs, or
    None when this kernel does not apply."""
    if not applies(B, G, N, P, R, hasZ, hasD, reduceO, fuse_pregate_headwise_rms_norm, isVarlen, chunk_size,
                   rotary_dim_divisor, dtype, has_initial_state, return_state, state_only):
        return None
    ext = build("fwd")
    if ext is None:
        return None

    def run(q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, D, mimo_z, angles, dA_cs, dA_cs_rev, dt, trap, cu_seqlens,
            blk_seg, blk_c0, blk_nch, init_state):
        S = q.shape[1]
        dev = q.device
        blocked = blk_seg is not None
        nblk = int(blk_seg.numel()) if blocked else cu_seqlens.numel() - 1
        f32 = dict(device=dev, dtype=torch.float32)
        ts, gam, dacs = (torch.empty(H, S, **f32) for _ in range(3))
        dtot = torch.empty(nblk, H, **f32)
        ch_start = torch.empty(nblk, device=dev, dtype=torch.int32)
        ch_len = torch.empty(nblk, device=dev, dtype=torch.int32)
        if nblk > 0:
            dummy = cu_seqlens
            _fwd_prep[(nblk, H)](
                dt, trap, dA_cs, cu_seqlens, blk_seg if blocked else dummy, blk_c0 if blocked else dummy,
                blk_nch if blocked else dummy, ts, gam, dacs, dtot, ch_start, ch_len, S, H,
                BLOCKED=blocked, C=chunk_size, num_warps=1,
            )
        Q = q.view(S * R, N)
        K = k.view(S * R, N)
        V = v.view(S, H, P)
        if state_only:
            h_out = torch.empty(nblk, H, N, P, **f32)
            ext.chunk_state(K, V, k_bias, mimo_v, angles.view(S, H, -1), dacs, ts, dtot, ch_start, ch_len, h_out, nblk)
            return None, h_out, None
        o = torch.empty(B, S, H, P, device=dev, dtype=v.dtype)
        ext.scan(Q, K, V, z.view(S, H, P), q_bias, k_bias, mimo_v, mimo_o, mimo_z, D, angles.view(S, H, -1), None,
                 dA_cs.view(H, S), dA_cs_rev.view(H, S), ts, gam, ch_start, ch_len, None,
                 init_state.contiguous() if init_state is not None else None, o.view(S, H, P), None, nblk)
        return o, None, None

    return run
