"""Two-level (block-decomposed) scan for mamba3_mimo — production module.

Replaces the single forward launch with three:

    Pass A   blocked grid, state_only=True,  INIT_STATE=0      -> C_local
    Pass B   one Triton kernel, scans the block transfers      -> S_start
    Pass C   blocked grid, state_only=False, INIT_STATE=S_start -> Out

WHY. The stock grid is one thread-block per (head, SEGMENT), so duration is
set by the LONGEST segment: a measured 2.45-6.07x forward penalty for skew at
fixed total work. The blocked grid is work-shaped, so sequential depth per
block is capped at `scan_block` regardless of segment length. Measured: the
imbalance goes to 1.00x of the uniform case, i.e. it is removed, not reduced.
See RESULTS-twolevel-2026-08-27.md.

EXACTNESS. The per-chunk decay is a SCALAR, a_i = exp(da_cs[chunk_end]), so
    S_end = A_block * S_start + C_local,   A_block = prod a_i
and — because it is scalar — Pass C needs no correction pass: seeding the
state and running the ordinary loop is sufficient. Validated bitwise against
the single-launch kernel for scan_block >= 8 and to bf16 noise below that.

SAFE BY CONSTRUCTION FOR THE BACKWARD. mamba3_mimo's autograd Function saves
only INPUTS (Q, K, V, ADT, DT, Trap, biases, Angles, D, Z, MIMO_*, cu_seqlens)
— nothing the forward produced. The backward recomputes DA_CS/Segsum from
those, so it cannot observe how Out was computed. Gradients are therefore
bit-identical to stock for identical inputs and identical dout; tests assert
exactly that.

ON BY DEFAULT (scan_block 32) where it pays (see two_level_pays); M3_SCAN_BLOCK=0 takes the stock
path untouched, so this is reversible without redeploying code.
"""

import os
import weakref

import torch
import triton
import triton.language as tl

__all__ = ["two_level_forward", "two_level_bwd_fwd", "get_plan",
           "scan_block_from_env", "clear_plan_cache"]

# ---------------------------------------------------------------- Pass B ----


@triton.jit
def _pass_b(A, C, S, SEG_START, SEG_NB,
            H: tl.constexpr, NP: tl.constexpr, BLOCK: tl.constexpr,
            REV: tl.constexpr):
    """Exclusive scan of S(j) = A(j-1)*S(j-1) + C(j-1) along the block axis.

    Sequential only along that axis: every (h, n, p) carries its own
    independent scalar scan — H*N*P = 98304 of them at production shape — so
    one thread walks the block axis while adjacent threads walk adjacent `p`,
    keeping every access coalesced.

    NOT the cumprod closed form S_j = P_j*cumsum(C_k/P_{k+1}): A_block is
    exp(sum of da_cs over hundreds of tokens), so P_j underflows fp32 to zero
    within a few blocks and C_k/P_{k+1} overflows. The recurrence is fine when
    A underflows; only the closed form dies.
    """
    si = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)

    base = tl.load(SEG_START + si)
    n = tl.load(SEG_NB + si)

    off = t * BLOCK + tl.arange(0, BLOCK)
    m = off < NP

    # REV walks the block axis backwards, for bwd_bwd's reverse recurrence
    # D_i = a_i * D_{i+1} + g_i. Same write-then-accumulate exclusive scan --
    # only the visiting order changes, because a_i is the same scalar decay.
    s = tl.zeros([BLOCK], dtype=tl.float32)
    for j in range(0, n):
        ib = base + (n - 1 - j) if REV else base + j
        p = (ib * H + h) * NP + off
        tl.store(S + p, s, mask=m)                   # exclusive: write first
        a = tl.load(A + ib * H + h).to(tl.float32)
        c = tl.load(C + p, mask=m, other=0.0)
        s = a * s + c


def pass_b_triton(A, C_local, seg_start, seg_nb, BLOCK=256, rev=False):
    """A: [NBLK, H]   C_local: [NBLK, H, N, P] fp32   ->   S_start, same shape."""
    NBLK, H, N, P = C_local.shape
    NP = N * P
    assert C_local.is_contiguous()
    S = torch.empty_like(C_local)
    grid = (seg_start.numel(), H, triton.cdiv(NP, BLOCK))
    _pass_b[grid](A.to(torch.float32).contiguous(), C_local, S,
                  seg_start, seg_nb, H=H, NP=NP, BLOCK=BLOCK, REV=rev)
    return S


# ------------------------------------------------------------ the tables ----


def build_plan(cu_list, chunk, scan_block, device):
    """Block mapping tables for one packing. Fully vectorised except a loop
    over SEGMENTS (13 at production shape, not blocks).

    The Python version of the chunk-end gather cost 10.4 ms/call — 302 ms per
    step across 29 Mamba layers, which would have eaten a fifth of the gain.
    Hence the tensor form below, plus the per-step cache in `get_plan`.
    """
    NS = len(cu_list) - 1
    seg, c0s, nchs, seg_start, seg_nb = [], [], [], [], []
    for si in range(NS):
        L = cu_list[si + 1] - cu_list[si]
        nch = (L + chunk - 1) // chunk
        seg_start.append(len(seg))
        nb = 0
        for c0 in range(0, nch, scan_block):
            seg.append(si)
            c0s.append(c0)
            nchs.append(min(scan_block, nch - c0))
            nb += 1
        seg_nb.append(nb)

    # Built on the CPU and transferred once. Doing this with CUDA tensor ops
    # is SLOWER (measured 16.9 ms vs 10.4 ms for the naive Python loop): the
    # problem is ~128 blocks, far too small to amortise launch overhead, so it
    # is pure launch cost. CPU + one transfer is ~0.1 ms.
    seg_c = torch.tensor(seg, dtype=torch.int32)
    c0_c = torch.tensor(c0s, dtype=torch.int32)
    nch_c = torch.tensor(nchs, dtype=torch.int32)

    # chunk-end absolute positions:
    #   idx[ib, j] = cu[seg[ib]] + min((c0[ib]+j+1)*chunk, L[seg[ib]]) - 1
    # the index the kernel reads in BOTH of its branches, so the segment-tail
    # case falls out for free.
    cu_c = torch.tensor(cu_list, dtype=torch.int64)
    s_of = cu_c[seg_c.long()]
    L_of = cu_c[seg_c.long() + 1] - s_of
    K = int(nch_c.max())
    j = torch.arange(K, dtype=torch.int64)
    ends = torch.minimum((c0_c.long()[:, None] + j[None, :] + 1) * chunk,
                         L_of[:, None]) - 1
    idx_c = s_of[:, None] + ends
    msk_c = (j[None, :] < nch_c.long()[:, None]).to(torch.float64)

    # Per-BLOCK segment bounds. The kernel could derive these from
    # CU_SEQLENS[BLK_SEG[i_blk]], but that nested indirection gives WRONG
    # results (measured: 11 of 14 gradients materially wrong in bwd_fwd) --
    # the per-sequence bounds sit inside `if NS > 1`, a RUNTIME traced branch
    # because NS is T.dynamic, and an unmaterialised load as the index breaks
    # there. Materialising the index into a scalar did not help either.
    # Precomputing the bounds host-side removes i_ns from the blocked path
    # altogether, so no indirection remains.
    s0_c = cu_c[seg_c.long()].to(torch.int32)                  # start_seq_ind
    len_c = (cu_c[seg_c.long() + 1] - cu_c[seg_c.long()]).to(torch.int32)
    ch0_c = (s0_c.long() // chunk + seg_c.long()).to(torch.int32)  # start_chunk_ind

    to = lambda t: t.to(device, non_blocking=True)
    seg_t, c0_t, nch_t = to(seg_c), to(c0_c), to(nch_c)
    idx, msk = to(idx_c), to(msk_c)

    return dict(seg=seg_t, c0=c0_t, nch=nch_t, idx=idx, msk=msk,
                s0=to(s0_c), len=to(len_c), ch0=to(ch0_c),
                seg_start=to(torch.tensor(seg_start, dtype=torch.int32)),
                seg_nb=to(torch.tensor(seg_nb, dtype=torch.int32)),
                last_blk=to(torch.tensor(
                    [st + nb - 1 for st, nb in zip(seg_start, seg_nb)],
                    dtype=torch.int64)),
                nblk=len(seg))


_PLAN_CACHE = {}
_ZERO_CACHE = {}
_CACHE_MAX = 8


def clear_plan_cache():
    _PLAN_CACHE.clear()
    _ZERO_CACHE.clear()


_CU_HOST: dict = {}


def cu_host_list(cu_seqlens) -> list:
    """Host copy of cu_seqlens, read ONCE per packing.

    Every Mamba layer of a step (fwd and bwd) receives the same cu_seqlens
    tensor, so the copy is cached on the tensor's identity: a weakref (the
    entry dies with the tensor, so a recycled id() never aliases) plus its
    version counter (an in-place write invalidates it). One device sync per
    step instead of two per layer per direction.
    """
    key = id(cu_seqlens)
    hit = _CU_HOST.get(key)
    if hit is not None:
        ref, version, values = hit
        if ref() is cu_seqlens and version == cu_seqlens._version:
            return values
    values = cu_seqlens.tolist()
    if len(_CU_HOST) >= _CACHE_MAX:
        _CU_HOST.pop(next(iter(_CU_HOST)))
    _CU_HOST[key] = (weakref.ref(cu_seqlens), cu_seqlens._version, values)
    return values


def get_plan(cu_seqlens, chunk, scan_block):
    """Cached `build_plan`. All 29 Mamba layers of a step share one packing,
    so the build happens once per step rather than 29 times."""
    cu_list = cu_host_list(cu_seqlens)
    key = (tuple(cu_list), chunk, scan_block, str(cu_seqlens.device))
    pl = _PLAN_CACHE.get(key)
    if pl is None:
        if len(_PLAN_CACHE) >= _CACHE_MAX:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
        pl = build_plan(cu_list, chunk, scan_block, cu_seqlens.device)
        _PLAN_CACHE[key] = pl
    return pl


def _zeros(shape, device):
    """Shared read-only zero INIT_STATE for Pass A (never written by it)."""
    key = (shape, str(device))
    z = _ZERO_CACHE.get(key)
    if z is None:
        if len(_ZERO_CACHE) >= _CACHE_MAX:
            _ZERO_CACHE.pop(next(iter(_ZERO_CACHE)))
        z = torch.zeros(shape, device=device, dtype=torch.float32)
        _ZERO_CACHE[key] = z
    return z


def a_block(dA_cs, idx, msk):
    """A_block[ib, h] = exp(sum of dA_cs at the block's chunk ends).

    Not a kernel output: a pure function of dA_cs and the chunk grid, so the
    host gets it exactly — and Pass B needs it host-side anyway. fp64 because
    it is a sum of log-decays over hundreds of tokens.
    """
    d = dA_cs[0].double()                              # [H, S]; varlen: B = 1
    g = d[:, idx.reshape(-1)].reshape(d.shape[0], *idx.shape)
    return (g * msk).sum(-1).exp().transpose(0, 1).contiguous()


# ---------------------------------------------------------- orchestration ----


def scan_block_from_env(default=0):
    try:
        return int(os.environ.get("M3_SCAN_BLOCK", default))
    except ValueError:
        return default


_SPLIT_CACHE: dict = {}


def two_level_pays(cu_seqlens, nheads, device) -> bool:
    """Whether the block decomposition beats the stock scan for this packing.

    The stock scan's time is set by its longest serial chain (one block per
    (head, segment) walking L_max tokens); its throughput floor is the total
    work spread over the SMs (S_total * H / #SM). Splitting only pays when the
    chain is the bottleneck, L_max * #SM > S_total * H. GH200, H=12,
    S=65536: 2x32768 (ratio 5.5) 85.7 -> 19.2 ms fwd+bwd, one 20k doc + 45
    short (3.4) 59.5 -> 24.0, 64x1024 (0.17) 23.2 -> 25.8 if split anyway.
    Cached per packing (all layers of a step share it); the cu_seqlens read
    is the same host sync get_plan() already does.
    """
    cu = cu_host_list(cu_seqlens)
    key = (tuple(cu), nheads, str(device))
    hit = _SPLIT_CACHE.get(key)
    if hit is None:
        if len(_SPLIT_CACHE) >= _CACHE_MAX:
            _SPLIT_CACHE.pop(next(iter(_SPLIT_CACHE)))
        l_max = max((b - a for a, b in zip(cu[:-1], cu[1:])), default=0)
        n_sm = torch.cuda.get_device_properties(device).multi_processor_count
        hit = l_max * n_sm > cu[-1] * nheads
        _SPLIT_CACHE[key] = hit
    return hit


def two_level_forward(fwd_fn, kwargs, scan_block):
    """Run Pass A -> B -> C in place of one `mamba_mimo_forward_varlen` call.

    `fwd_fn(**kwargs)` must be the blocked kernel host entry, i.e. it accepts
    blk_seg / blk_c0 / blk_nch / init_state / state_only.
    Returns the same (Out, Final_SSM_State, Final_K) triple as the stock call.
    """
    cu = kwargs["cu_seqlens"]
    chunk = kwargs["chunk_size"]
    dA_cs = kwargs["dA_cs"]
    pl = get_plan(cu, chunk, scan_block)

    H = dA_cs.shape[1]
    N = kwargs["q"].shape[-1]
    P = kwargs["v"].shape[-1]
    z0 = _zeros((pl["nblk"], H, N, P), dA_cs.device)

    blocked = dict(kwargs)
    blocked.update(blk_seg=pl["seg"], blk_c0=pl["c0"], blk_nch=pl["nch"])

    # Pass A. return_state=True is required: it emits C_local AND activates the
    # segment-tail masking, which is what makes the last chunk of a segment
    # right. Never reads z0, so the shared zero buffer is safe.
    _, C_local, _ = fwd_fn(**dict(blocked, state_only=True,
                                  return_state=True, init_state=z0))

    S_start = pass_b_triton(a_block(dA_cs, pl["idx"], pl["msk"]),
                            C_local, pl["seg_start"], pl["seg_nb"])

    # Pass C. The tail branch that return_state gates affects only the carried
    # state, never Out, so return_state can follow the caller.
    out, h, k_final = fwd_fn(**dict(blocked, state_only=False,
                                    init_state=S_start))

    # FINAL_STATE is per-BLOCK under the blocked grid, so the caller's
    # per-SEGMENT contract is the LAST block of each segment. FINAL_K is
    # already segment-indexed (the kernel writes it under i_ns), so it passes
    # through. Production runs return_state=False and never enters this.
    if h is not None:
        h = h[pl["last_blk"]].contiguous()
    return out, h, k_final


def two_level_bwd_fwd(make_kernel, base_args, pl, dA_cs, H, N, P):
    """Pass A -> B -> C for `bwd_fwd`, in place of its single launch.

    `make_kernel(**flags)` builds the bwd_fwd kernel; `base_args` is everything
    up to and including cu_seqlens. The three block tables, INIT_STATE and
    FINAL_STATE are appended here.

    Two things differ from the forward:

    * Pass A must NOT write STATES. That buffer is [B, H, max_nchunks, N, P] --
      ~808 MB per layer per micro-batch at chunk 16 -- and Pass C rewrites every
      element of it, so writing it twice would cost more bandwidth than the
      whole decomposition saves. The `state_only` guard covers it.
    * Pass C's FINAL_STATE is never read, so it aliases `c_local`, whose
      contents Pass B has already consumed by then. Saves one NBLK x H x N x P
      fp32 buffer (~55 MB at production shape).

    Returns the plan, whose `nblk` the caller needs to size the per-block
    DMIMO accumulators.
    """
    dev = dA_cs.device
    tabs = (pl["seg"], pl["c0"], pl["nch"], pl["s0"], pl["len"], pl["ch0"])
    z0 = _zeros((pl["nblk"], H, N, P), dev)          # shared, read-only
    c_local = torch.empty((pl["nblk"], H, N, P), device=dev, dtype=torch.float32)

    make_kernel(has_init_state=True, state_only=True, blocked=True)(
        *base_args, *tabs, z0, c_local)

    s_start = pass_b_triton(a_block(dA_cs, pl["idx"], pl["msk"]),
                            c_local, pl["seg_start"], pl["seg_nb"])

    make_kernel(has_init_state=True, state_only=False, blocked=True)(
        *base_args, *tabs, s_start, c_local)
    return pl


def two_level_bwd_bwd(make_kernel, base_args, pl, dA_cs, H, N, P):
    """Pass A -> B -> C for `bwd_bwd`, whose recurrence runs BACKWARDS.

        D_i = a_i * D_{i+1} + g_i,        a_i = exp(da_cs[chunk_end])

    Same scalar coefficient as the forward, so `A_block` is the *same* table
    and the decomposition is identical up to direction: each block is traversed
    right-to-left, and Pass B scans the block transfers right-to-left too
    (`rev=True`).

    Pass A is more expensive here than in bwd_fwd: the reverse state update
    needs `q_shared` and `dPhiO_shared`, so it must keep the rotary-Q and the
    DOUT projection, not just K/V. Expect a worse overhead ratio.
    """
    dev = dA_cs.device
    tabs = (pl["seg"], pl["c0"], pl["nch"], pl["s0"], pl["len"], pl["ch0"])
    z0 = _zeros((pl["nblk"], H, N, P), dev)
    d_local = torch.empty((pl["nblk"], H, N, P), device=dev, dtype=torch.float32)

    make_kernel(has_init_state=True, state_only=True, blocked=True)(
        *base_args, *tabs, z0, d_local)

    d_in = pass_b_triton(a_block(dA_cs, pl["idx"], pl["msk"]),
                         d_local, pl["seg_start"], pl["seg_nb"], rev=True)

    make_kernel(has_init_state=True, state_only=False, blocked=True)(
        *base_args, *tabs, d_in, d_local)
    return pl
