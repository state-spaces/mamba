"""Mamba-3 MIMO varlen backward on Blackwell: chunk-parallel over 32-token sub-blocks, tcgen05 kernels (bwd_tc.cu).

  prep    (Triton) per (sub-block, head): l (inclusive local cumsum of ADT), f, gamma; the sub-block total
  local   (CUDA)   Sloc = khat^T (PsiV exp(l_end - l) f), dloc = qhat^T (dO exp(l))                 (bf16 N x P)
  scan    (Triton) h_b (state entering b) and D_b (gradient of the state leaving b), bf16: a two-level scan
                   (groups of 32 sub-blocks), segmented at the sequence boundaries, in both directions
  main    (CUDA)   per sub-block, the heads in turn, both sides on the same operands: dq / dk (summed over the
                   heads), dv, dz, the phase gradient, decay row sums, dgamma, dfactor, and the
                   per-head bias / mimo / D gradients
  finish  (Triton) dADT = local reverse cumsum + <h_{b+1}, D_b> (from the main kernel: no sequence-long cumsum);
                   and the dt / trap gradients (the math of bwd_dtrap_ddt_kernel_varlen, fused).
See bwd_tc.cu for the math. States are stored per 32-token sub-block in bf16 (TileLang: per 16-token chunk).
"""

import torch
import triton
import triton.language as tl

from mamba_ssm.ops.cuda.mamba3.sm100.common import N, NQ, P, R, build, chunk_layout

C = 32  # tokens per sub-block


@triton.jit
def _prep_kernel(ADT, DT, TRAP, CH_START, CH_LEN, CH_SEND, ELL, FF, GAM, TOT, S, H, C: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    cs = tl.load(CH_START + c)
    cl = tl.load(CH_LEN + c)
    send = tl.load(CH_SEND + c)
    i = tl.arange(0, C)
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
    tl.store(ELL + off, tl.cumsum(adt, axis=0), mask=tm)
    tl.store(FF + off, gam + dtn * tl.sigmoid(-trn), mask=tm)
    tl.store(GAM + off, gam, mask=tm)
    tl.store(TOT + c * H + h, tl.sum(adt, axis=0))


GS = 32  # sub-blocks per group of the two-level state scan (16: the sequential group pass is twice longer)


@triton.jit
def _scan_coef(TOT, FIRST, LAST, H, NB, h, j, REV: tl.constexpr):
    """Recurrence x_j = m_j x_{j-1} + u_j along the scan order (forward: j = sub-block, x = state entering it;
    reverse: j = NB - 1 - sub-block, x = gradient of the state leaving it). Returns (m_j, source sub-block of u_j or
    -1): a sequence boundary resets the state (m = 0, no source)."""
    if REV:
        b = NB - 1 - j
        cut = tl.load(LAST + b) != 0
        m = tl.where(cut, 0.0, tl.exp(tl.load(TOT + (b + 1) * H + h, mask=~cut, other=0.0)))
        return m, tl.where(cut, -1, b + 1)
    else:
        b = j
        cut = tl.load(FIRST + b) != 0
        m = tl.where(cut, 0.0, tl.exp(tl.load(TOT + (b - 1) * H + h, mask=~cut, other=0.0)))
        return m, tl.where(cut, -1, b - 1)


@triton.jit
def _scan_a(LOC, TOT, FIRST, LAST, GT, GA, H, NB, NG, N: tl.constexpr, P: tl.constexpr, NBR: tl.constexpr,
            GS: tl.constexpr, REV: tl.constexpr):
    """Per group: the affine map x_out = A x_in + T over its GS steps (T from a local scan starting at zero)."""
    h = tl.program_id(0)
    g = tl.program_id(1)
    nb = tl.program_id(2)
    o = (nb * NBR + tl.arange(0, NBR))[:, None] * P + tl.arange(0, P)[None, :]
    x = tl.zeros([NBR, P], dtype=tl.float32)
    A = 1.0
    for i in tl.range(0, GS, num_stages=3):
        j = g * GS + i
        if j < NB:
            m, src = _scan_coef(TOT, FIRST, LAST, H, NB, h, j, REV)
            u = tl.load(LOC + (tl.maximum(src, 0) * H + h).to(tl.int64) * N * P + o, mask=(src >= 0) & (o >= 0), other=0.0).to(tl.float32)
            x = m * x + u
            A = A * m
    tl.store(GT + ((g * H + h) * N * P + o).to(tl.int64), x)
    if nb == 0:
        tl.store(GA + g * H + h, A)


@triton.jit
def _scan_b(GT, GA, CIN, H, NG, N: tl.constexpr, P: tl.constexpr, NBR: tl.constexpr):
    """Sequential over the groups: CIN[g] = x entering group g."""
    h = tl.program_id(0)
    nb = tl.program_id(1)
    o = (nb * NBR + tl.arange(0, NBR))[:, None] * P + tl.arange(0, P)[None, :]
    c = tl.zeros([NBR, P], dtype=tl.float32)
    for g in tl.range(0, NG, num_stages=4):
        off = ((g * H + h) * N * P + o).to(tl.int64)
        tl.store(CIN + off, c)
        c = tl.load(GA + g * H + h) * c + tl.load(GT + off)


@triton.jit
def _scan_c(LOC, TOT, FIRST, LAST, CIN, OUT, H, NB, N: tl.constexpr, P: tl.constexpr, NBR: tl.constexpr,
            GS: tl.constexpr, REV: tl.constexpr):
    """Within each group from its entering value: OUT[sub-block] (bf16) = x."""
    h = tl.program_id(0)
    g = tl.program_id(1)
    nb = tl.program_id(2)
    o = (nb * NBR + tl.arange(0, NBR))[:, None] * P + tl.arange(0, P)[None, :]
    x = tl.load(CIN + ((g * H + h) * N * P + o).to(tl.int64))
    for i in tl.range(0, GS, num_stages=3):
        j = g * GS + i
        if j < NB:
            m, src = _scan_coef(TOT, FIRST, LAST, H, NB, h, j, REV)
            u = tl.load(LOC + (tl.maximum(src, 0) * H + h).to(tl.int64) * N * P + o, mask=(src >= 0) & (o >= 0), other=0.0).to(tl.float32)
            x = m * x + u
            b = (NB - 1 - j) if REV else j
            tl.store(OUT + (b * H + h).to(tl.int64) * N * P + o, x.to(tl.bfloat16))


def _state_scan(loc, tot, first, last, out, NB, H, rev):
    NG = triton.cdiv(NB, GS)
    NBR = 16
    nblk = N // NBR
    dev = loc.device
    gt = torch.empty(NG, H, N, P, device=dev, dtype=torch.float32)
    ga = torch.empty(NG, H, device=dev, dtype=torch.float32)
    cin = torch.empty_like(gt)
    _scan_a[(H, NG, nblk)](loc, tot, first, last, gt, ga, H, NB, NG, N=N, P=P, NBR=NBR, GS=GS, REV=rev, num_warps=4)
    _scan_b[(H, nblk)](gt, ga, cin, H, NG, N=N, P=P, NBR=NBR, num_warps=4)
    _scan_c[(H, NG, nblk)](loc, tot, first, last, cin, out, H, NB, N=N, P=P, NBR=NBR, GS=GS, REV=rev, num_warps=4)


@triton.jit
def _finish_kernel(DLROW, DFAC, DGAM, FF, CROSS, NXT, FIRST, DT, TRAP, CH_START, CH_LEN, DADT, DDT, DTRAP, S, H,
                   C: tl.constexpr):
    """dADT = local reverse cumsum of dl + <h_{b+1}, D_b> (every pair of tokens straddling the sub-block end);
    ddt / dtrap from the trapezoid gradients (f_t = gamma_t + DT_{t+1} sigmoid(-trap_{t+1}), gamma = DT sigmoid(trap)):
    the math of bwd_dtrap_ddt_kernel_varlen."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    t0 = tl.load(CH_START + b)
    nv = tl.load(CH_LEN + b)
    nx = tl.load(NXT + b)
    i = tl.arange(0, C)
    tm = i < nv
    so = h * S + t0 + i
    dl = tl.load(DLROW + so, mask=tm, other=0.0) - tl.load(FF + so, mask=tm, other=0.0) * tl.load(DFAC + so, mask=tm, other=0.0)
    cross = tl.where(nx >= 0, tl.load(CROSS + b * H + h), 0.0)  # <h_{b+1}, D_b> from the main kernel
    tl.store(DADT + so, tl.cumsum(dl, axis=0, reverse=True) + cross, mask=tm)
    pm = tm & ((i > 0) | (tl.load(FIRST + b) == 0))  # token t - 1 in the same sequence
    dfp = tl.load(DFAC + so - 1, mask=pm, other=0.0)
    dg = tl.load(DFAC + so, mask=tm, other=0.0) + tl.load(DGAM + so, mask=tm, other=0.0)
    dtv = tl.load(DT + so, mask=tm, other=0.0)
    trv = tl.load(TRAP + so, mask=tm, other=0.0).to(tl.float32)
    sp, sm = tl.sigmoid(trv), tl.sigmoid(-trv)
    tl.store(DDT + so, sp * dg + sm * dfp, mask=tm)
    tl.store(DTRAP + so, ((dg - dfp) * dtv * sp * sm).to(DTRAP.dtype.element_ty), mask=tm)


_NEXT: dict = {}
_FLAGS: dict = {}


def _seq_flags(meta, NB, cu_list, dev):
    """int32 [NB] flags: first / last sub-block of its sequence."""
    key = (tuple(cu_list), str(dev))
    fl = _FLAGS.get(key)
    if fl is None:
        if len(_FLAGS) > 64:
            _FLAGS.clear()
        first = torch.zeros(NB, device=dev, dtype=torch.int32)
        last = torch.zeros(NB, device=dev, dtype=torch.int32)
        first[meta["seq_c0"].long()] = 1
        last[(meta["seq_c0"] + meta["seq_nc"] - 1).long()] = 1
        fl = _FLAGS[key] = (first, last)
    return fl


def _next_sb(meta, NB, cu_list, dev):
    key = (tuple(cu_list), str(dev))
    nxt = _NEXT.get(key)
    if nxt is None:
        if len(_NEXT) > 64:
            _NEXT.clear()
        nxt = torch.arange(1, NB + 1, device=dev, dtype=torch.int32)
        nxt[(meta["seq_c0"] + meta["seq_nc"] - 1).long()] = -1
        _NEXT[key] = nxt
    return nxt


def _sb_chunk(cu_list, ls, dev):
    """int32 [NB]: the ``ls``-token forward chunk of every sub-block (both layouts restart at the sequences)."""
    key = (tuple(cu_list), ls, str(dev))
    hit = _SBC.get(key)
    if hit is None:
        out, c0 = [], 0
        for a, b in zip(cu_list[:-1], cu_list[1:]):
            out += [c0 + (x - a) // ls for x in range(a, b, C)]
            c0 += -(-(b - a) // ls)
        if len(_SBC) > 64:
            _SBC.clear()
        hit = _SBC[key] = torch.tensor(out or [0], dtype=torch.int32).to(dev, non_blocking=True)
    return hit


_SBC: dict = {}


def bwd(dout, q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, mimo_z, angles, adt, dt, trap, D, cu_list, cu_seqlens,
        ain=None, ain_chunk=256):
    """The TileLang backward's 15 outputs (dq, dk, dv, dADT, ddt, dtrap, dq_bias, dk_bias, dmimo_v, dmimo_z,
    dmimo_o, dangles_cumsum, dD, dz, None) for the reduceO + Z + D configuration. ``angles``: the angle cumsum
    [.., S, H, NQ], or with ``ain`` the phases relative to the ``ain_chunk``-token forward chunks (fwd_tc.fwd's thl,
    ain)."""
    ext = build("mamba3_bwd_tc", "bwd_tc.cu")
    S, H = v.shape[1], v.shape[2]
    dev = v.device
    NB, meta = chunk_layout(cu_list, C, dev)
    f32 = dict(device=dev, dtype=torch.float32)
    fl = lambda t: t.float().contiguous()  # noqa: E731
    mv, mo, mz, Dp = map(fl, (mimo_v, mimo_o, mimo_z, D))
    qb, kb = q_bias.to(torch.bfloat16).contiguous(), k_bias.to(torch.bfloat16).contiguous()
    adt_, dt_ = adt.reshape(H, S).float().contiguous(), dt.reshape(H, S).float().contiguous()
    trap_ = trap.reshape(H, S).contiguous()
    ang = angles.contiguous().view(S, H, NQ)
    Q, K, V = q.contiguous().view(S, R, N), k.contiguous().view(S, R, N), v.view(S, H, P)
    Z, DO = z.contiguous().view(S, H, P), dout.contiguous().view(S, H, P)
    ell, ff, gam = (torch.empty(H, S, **f32) for _ in range(3))
    tot = torch.empty(NB, H, **f32)
    _prep_kernel[(NB, H)](adt_, dt_, trap_, meta["start"], meta["len"], meta["send"], ell, ff, gam, tot, S, H,
                          C=C, num_warps=1)
    if ain is None:
        ain_t, sbc = torch.zeros(1, H, NQ, **f32), torch.zeros(max(NB, 1), device=dev, dtype=torch.int32)
    else:
        ain_t, sbc = ain.contiguous(), _sb_chunk(cu_list, ain_chunk, dev)
    common = (Q, K, V, DO, Z, ang, qb, kb, mv, mo, mz, Dp, ell, ff, gam, tot, meta["start"], meta["len"], ain_t, sbc)
    sloc = torch.empty(NB, H, N, P, device=dev, dtype=torch.bfloat16)
    dloc = torch.empty_like(sloc)
    ext.local(*common, sloc, dloc, NB)
    hs = torch.empty(NB, H, N, P, device=dev, dtype=torch.bfloat16)
    ds = torch.empty_like(hs)
    first, last = _seq_flags(meta, NB, cu_list, dev)
    _state_scan(sloc, tot, first, last, hs, NB, H, rev=False)
    _state_scan(dloc, tot, first, last, ds, NB, H, rev=True)
    del sloc, dloc
    grid = ext.grid(NB)
    nacc = ext.nacc()
    acc_q = torch.zeros(grid, H, nacc, **f32)
    acc_kv = torch.zeros(grid, H, nacc, **f32)
    dq = torch.empty(1, S, R, 1, N, device=dev, dtype=q.dtype)
    dk = torch.empty(1, S, R, 1, N, device=dev, dtype=k.dtype)
    dz, dv = torch.empty_like(v), torch.empty_like(v)
    dang = torch.empty(1, S, H, NQ, **f32)  # phase gradient, written by the main kernel
    dlrow, dgam, dfac = (torch.empty(1, H, S, **f32) for _ in range(3))
    cross = torch.empty(NB, H, **f32)
    ext.main(*common, hs, ds, dq, dk, dz.view(S, H, P), dv.view(S, H, P), dang, dlrow, dgam, dfac, acc_q, acc_kv,
             cross, NB)
    del hs, ds
    dadt = torch.empty(1, H, S, **f32)
    ddt = torch.empty(1, H, S, device=dev, dtype=dt.dtype)
    dtrap = torch.empty(1, H, S, device=dev, dtype=trap.dtype)
    _finish_kernel[(NB, H)](dlrow, dfac, dgam, ff, cross, _next_sb(meta, NB, cu_list, dev), first, dt_, trap_,
                            meta["start"], meta["len"], dadt, ddt, dtrap, S, H, C=C, num_warps=1)
    aq, akv = acc_q.sum(0), acc_kv.sum(0)
    RN, RP = R * N, R * P
    dqb, dkb = aq[:, :RN].view(H, R, N), akv[:, :RN].view(H, R, N)
    dmo, dmz = aq[:, RN:RN + RP].view(H, R, P), aq[:, RN + RP:RN + 2 * RP].view(H, R, P)
    dmv, dD = akv[:, RN:RN + RP].view(H, R, P), akv[:, RN + 2 * RP]
    return (dq, dk, dv, dadt, ddt, dtrap, dqb, dkb, dmv, dmz, dmo, dang, dD, dz, None)
