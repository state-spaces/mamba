"""Mamba-3 MIMO varlen forward on Blackwell: the tcgen05 prefill kernels of the vLLM Olala engine (scan_tc.cu).

Pipeline over 256-token chunks (a chunk never straddles a sequence):
  prep        (Triton) per (chunk, head): dacs = chunk-local cumsum of ADT and its total, the trapezoid scales
              gam = DT sigmoid(trap), ts = gam + DT_next sigmoid(-trap_next), the 32-token group cumsums of ADT
              (da_cs / da_cs_rev, clipped <= 0) used for the token masks, and the rotary phases relative to the
              chunk start (thl, from the raw angles: cumsum of tanh(angle) pi DT, 32-token groups wrapped mod 2 pi)
              with the chunk's total (asum): fp32 phases stay small, no sequence-long angle cumsum.
  chunk_state (tcgen05) the state every chunk produces from zero, in its local rotary frame.
  state_pass  (Triton) the state entering every chunk and the chunk's entering phase ain (mod 2 pi), one
              sequential walk per sequence and head.
  scan        (tcgen05) the outputs, 32-token sub-blocks carrying the state in tensor memory.
Same math as the TileLang kernel; the differences are fp32 roundings (states in fp32, chunk-local phases).
"""

import math

import torch
import triton
import triton.language as tl

from mamba_ssm.ops.cuda.mamba3.sm100.common import N, NQ, P, R, build, chunk_layout, config_ok

LS = 256  # tokens per chunk of the tcgen05 kernels
G32 = 32  # tokens per sub-block (the token-mask cumsums are 32-token group sums)
TWO_PI = 2.0 * math.pi


@triton.jit
def _wrap(x):
    return x - 6.283185307179586 * tl.floor(x * 0.15915494309189535)


@triton.jit
def _prep_kernel(ADT, DT, TRAP, ANG, CH_START, CH_LEN, CH_SEND,
                 DACS, TS, GAM, DTOT, DCS32, DCSR32, THL, ASUM, S, H,
                 LS: tl.constexpr, G: tl.constexpr, NQ: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    cs = tl.load(CH_START + c)
    cl = tl.load(CH_LEN + c)
    send = tl.load(CH_SEND + c)
    i = tl.arange(0, LS)
    tm = i < cl
    t = cs + i
    off = h * S + t
    adt = tl.load(ADT + off, mask=tm, other=0.0)
    dtv = tl.load(DT + off, mask=tm, other=0.0)
    trv = tl.load(TRAP + off, mask=tm, other=0.0).to(tl.float32)
    nm = tm & (t + 1 < send)
    dtn = tl.load(DT + off + 1, mask=nm, other=0.0)
    trn = tl.load(TRAP + off + 1, mask=nm, other=0.0).to(tl.float32)
    gam = dtv * tl.sigmoid(trv)
    tl.store(DACS + off, tl.cumsum(adt, axis=0), mask=tm)
    a2 = tl.reshape(adt, (LS // G, G))
    cs2 = tl.cumsum(a2, axis=1)
    rev2 = tl.cumsum(a2, axis=1, reverse=True) - a2
    tl.store(DCS32 + off, tl.reshape(tl.minimum(cs2, 0.0), (LS,)), mask=tm)
    tl.store(DCSR32 + off, tl.reshape(tl.minimum(rev2, 0.0), (LS,)), mask=tm)
    tl.store(TS + off, gam + dtn * tl.sigmoid(-trn), mask=tm)
    tl.store(GAM + off, gam, mask=tm)
    tl.store(DTOT + c * H + h, tl.sum(adt, axis=0))
    # phases: cumsum of tanh(angle) pi DT (as angle_dt) over 32-token groups, the carry wrapped mod 2 pi
    nq = tl.arange(0, NQ)
    ig = tl.arange(0, G)
    carry = tl.zeros([NQ], dtype=tl.float32)
    for g in tl.static_range(LS // G):
        tg = cs + g * G + ig
        mg = (g * G + ig) < cl
        a = tl.load(ANG + (tg[:, None] * H + h) * NQ + nq[None, :], mask=mg[:, None], other=0.0)
        dg = tl.load(DT + h * S + tg, mask=mg, other=0.0)
        inc = (tl.sigmoid(2.0 * a) * 2.0 - 1.0) * 3.141592653589793 * dg[:, None]
        tl.store(THL + (tg[:, None] * H + h) * NQ + nq[None, :], _wrap(tl.cumsum(inc, axis=0) + carry[None, :]),
                 mask=mg[:, None])
        carry = _wrap(carry + tl.sum(inc, axis=0))
    tl.store(ASUM + (c * H + h) * NQ + nq, carry)


@triton.jit
def _state_pass_kernel(SLOC, DTOT, ASUM, AIN, SEQ_C0, SEQ_NC, SINT, H,
                       N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr, PB: tl.constexpr):
    """SinT[c] (P x N, fp32) = state entering chunk c; Sloc[c] (N x P) in the chunk's local rotary frame;
    AIN[c] = phase entering chunk c (mod 2 pi, carried from the chunk totals ASUM)."""
    h = tl.program_id(0)
    s = tl.program_id(1)
    p0 = tl.program_id(2) * PB
    c0 = tl.load(SEQ_C0 + s)
    nc = tl.load(SEQ_NC + s)
    nq = tl.arange(0, NQ)
    p = p0 + tl.arange(0, PB)
    st0 = tl.zeros([NQ, PB], dtype=tl.float32)
    st1 = tl.zeros([NQ, PB], dtype=tl.float32)
    st2 = tl.zeros([NQ, PB], dtype=tl.float32)
    st3 = tl.zeros([NQ, PB], dtype=tl.float32)
    th = tl.zeros([NQ], dtype=tl.float32)
    for j in tl.range(0, nc, num_stages=4):
        c = c0 + j
        ob = SINT + (c * H + h).to(tl.int64) * N * P
        tl.store(ob + p[None, :] * N + nq[:, None], st0)
        tl.store(ob + p[None, :] * N + NQ + nq[:, None], st1)
        tl.store(ob + p[None, :] * N + 2 * NQ + nq[:, None], st2)
        tl.store(ob + p[None, :] * N + 3 * NQ + nq[:, None], st3)
        lb = SLOC + (c * H + h).to(tl.int64) * N * P
        l0 = tl.load(lb + nq[:, None] * P + p[None, :])
        l1 = tl.load(lb + (NQ + nq[:, None]) * P + p[None, :])
        l2 = tl.load(lb + (2 * NQ + nq[:, None]) * P + p[None, :])
        l3 = tl.load(lb + (3 * NQ + nq[:, None]) * P + p[None, :])
        if p0 == 0:
            tl.store(AIN + (c * H + h) * NQ + nq, th)
        cr = tl.cos(th)[:, None]
        sr = tl.sin(th)[:, None]
        dec = tl.exp(tl.load(DTOT + c * H + h))
        st0 = dec * st0 + (cr * l0 - sr * l2)
        st2 = dec * st2 + (sr * l0 + cr * l2)
        st1 = dec * st1 + l1
        st3 = dec * st3 + l3
        th = _wrap(th + tl.load(ASUM + (c * H + h) * NQ + nq))


def prep(adt, dt, trap, angles, NC, meta, S, H):
    """The per-chunk prep tensors (see the module docstring); ``angles`` are the raw angles [S, H, NQ]."""
    dev = adt.device
    f32 = dict(device=dev, dtype=torch.float32)
    x = dict(dacs=torch.empty(H, S, **f32), ts=torch.empty(H, S, **f32), gam=torch.empty(H, S, **f32),
             dtot=torch.empty(NC, H, **f32), dcs32=torch.empty(H, S, **f32), dcsr32=torch.empty(H, S, **f32),
             thl=torch.empty(S, H, NQ, **f32), asum=torch.empty(NC, H, NQ, **f32), ain=torch.empty(NC, H, NQ, **f32))
    _prep_kernel[(NC, H)](adt, dt, trap, angles, meta["start"], meta["len"], meta["send"],
                          x["dacs"], x["ts"], x["gam"], x["dtot"], x["dcs32"], x["dcsr32"], x["thl"], x["asum"], S, H,
                          LS=LS, G=G32, NQ=NQ, num_warps=2)
    return x


def entering_states(ext, k, v, k_bias, mimo_v, x, NC, meta, S, H):
    """SinT [NC, H, P, N] fp32: the state entering every chunk."""
    dev = v.device
    sloc = torch.empty(NC, H, N, P, device=dev, dtype=torch.float32)
    ext.chunk_state(k.view(S, R, N), v.view(S, H, P), k_bias, mimo_v, x["thl"], x["dacs"], x["ts"], x["dtot"],
                    meta["start"], meta["len"], sloc, NC)
    sint = torch.empty(NC, H, P, N, device=dev, dtype=torch.float32)
    # serial over a sequence's chunks: narrow P slices for long sequences (more programs, shorter steps)
    long_seq = meta["max_nc"] >= 32
    PB = 4 if long_seq else 16
    _state_pass_kernel[(H, meta["NS"], P // PB)](sloc, x["dtot"], x["asum"], x["ain"], meta["seq_c0"], meta["seq_nc"],
                                                sint, H,
                                                N=N, P=P, NQ=NQ, PB=PB, num_warps=1 if long_seq else 2)
    return sint


def fwd(q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, D, mimo_z, angles, adt, dt, trap, cu_list):
    """(out [1, S, H, P], thl [S, H, NQ], ain [NC, H, NQ]) of the reduceO + Z-gated forward (no states returned),
    from the raw angles [1, S, H, NQ]. thl / ain (the phases relative to each 256-token chunk and the chunk's
    entering phase) replace the sequence-long angle cumsum in the backward."""
    ext = build("mamba3_scan_tc", "scan_tc.cu")
    S, H = v.shape[1], v.shape[2]
    NC, meta = chunk_layout(cu_list, LS, v.device)
    out = torch.empty_like(v)
    if NC == 0:
        return out, torch.zeros(S, H, NQ, device=v.device), torch.zeros(1, H, NQ, device=v.device)
    adt, dt = adt.reshape(H, S).float().contiguous(), dt.reshape(H, S).float().contiguous()
    trap = trap.reshape(H, S).contiguous()
    f32 = lambda t: t.float().contiguous()  # noqa: E731
    q_bias, k_bias, mimo_v, mimo_o, mimo_z, D = map(f32, (q_bias, k_bias, mimo_v, mimo_o, mimo_z, D))
    x = prep(adt, dt, trap, angles.float().contiguous(), NC, meta, S, H)
    sint = entering_states(ext, k, v, k_bias, mimo_v, x, NC, meta, S, H)
    k_fin = torch.empty(NC, R, H, N, device=v.device, dtype=q.dtype)
    ext.scan(q.view(S, R, N), k.view(S, R, N), v.view(S, H, P), z.view(S, H, P), q_bias, k_bias, mimo_v, mimo_o, mimo_z,
             D, x["thl"], x["ain"], x["dcs32"], x["dcsr32"], x["ts"], x["gam"], meta["start"], meta["len"],
             meta["last"], sint, out.view(S, H, P), k_fin, NC)
    return out, x["thl"], x["ain"]


def available() -> bool:
    return build("mamba3_scan_tc", "scan_tc.cu") is not None


__all__ = ["fwd", "prep", "entering_states", "config_ok", "available", "LS"]
