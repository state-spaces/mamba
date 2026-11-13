"""Mamba-3 MIMO chunked backward for Ascend 910B (A2).

Reverse-mode port of ``mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py``, sharing
the host-side plan of the forward port.  Two phases, mirroring the CUDA
operator's recompute kernel + gradient kernel split:

* phase 1 re-runs the forward chunk loop and records what the reverse pass needs
  (``states_in``, ``q_rot``, ``k_rot``, ``k_scaled``, ``PsiV``, the two masks,
  ``exp(dA_cs)``/``exp(dA_cs_rev)``, the gating coefficients and ``o4``);
* phase 2 walks the chunks in reverse.  Every Jacobian is either ``x @ y^T``
  (:func:`mm_nt`) or an elementwise kernel from :mod:`a2_vec`; no NN GEMM is used
  because that form miscomputes on this backend -- ``nn_128x128x64`` measured
  rel 1.2e+00 where the same shapes as NT are exact -- so the right operand is
  transposed on the host instead (``MIGRATION.md`` §4.4).

Reverse order per chunk is ``dv -> dq -> dk``, with ``G == 1``.  Five
accumulators (``DFACTOR`` / ``DGAMMA_DIAG`` / ``DDA_CS`` / ``DDA_CS_REV`` /
``DSSDA``) collect per-tensor partials; the host then composes
``ddt`` / ``dtrap`` / ``ddA`` from them::

    dA_cs      : o4 = o3*exp(dA_cs)             -> sum_{r,p} do4*o4, + cross-chunk decay
    dA_cs_rev  : ks_e = k_scaled*exp(dA_cs_rev) -> sum_{r,n} dks_e*ks_e
    segsum     : dsegsum[i,j] = rank-block sums of (dprod * masked)
    gamma_diag : diagonal term -> row/col sums of the same-step R x R block
    gamma      : k_scaled = k_rot*trap_scale    -> sum_{r,n} dks*k_rot

``dks`` accumulates from *both* consumers of ``k_scaled`` (the intra-chunk
``qk_intra`` and the recurrent state) -- the one place the reverse pass is easy
to get wrong.  The state path yields ``d/d ks_e``, and
``ks_e = k_scaled*exp(dA_cs_rev)``, so it must be scaled by ``exp(dA_cs_rev)``
before it can be added to the intra path's ``d/d k_scaled``.  Dropping that
factor breaks ``dk``/``dk_bias``/``dangles``/``dtrap`` together while
``dA_cs_rev`` stays correct; that asymmetry is the fingerprint of this bug.
The same chain is written up in ``MIGRATION.md`` §5.2.

Discretisation chains ``ddt``/``dtrap``/``ddA_cs``/``ddA_cs_rev``/``dsegsum``:
the CUDA side computes those in separate triton helpers, so here they are
host-side torch maths (see ``_segsum_ddt``, ``_ddA_cs_ddt``, ``_ddA_cs_rev_ddt``,
``_dtrap_ddt``), deliberately mirroring ``mamba_ssm.ops.triton.mamba3.
mamba3_mimo_utils.*_ref``.  Two details were only settled by diffing against an
A100 golden bundle:

* none of the three chain helpers may apply the ``-log2(e)`` factor that the
  ``*_ref`` functions carry -- the CUDA kernel's own output is unscaled (the
  host adds it straight into ``ddA``), confirmed by ``ddA_final - DDA_pass2``
  reproducing the chain over ``-log2e`` to rel_l2 1.1e-7;
* ``ddt`` is the trapezoid path alone; the chain belongs to ``ddA`` only.
"""

import math
import os

import torch

from mamba_ssm.ops.tilelang_ascend.mamba3 import a2_prims as P
from mamba_ssm.ops.tilelang_ascend.mamba3 import a2_vec as V
from mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo_fwd import (
    _chunk_scalar, _dt_shifted, _outproj_weight, _pad_rows, _rotary_apply,
    _rotary_layout, _silu_mimo)

_DO_SYNC = bool(int(os.environ.get("A2_SYNC", "1")))


def _sync():
    if _DO_SYNC:
        torch.npu.synchronize()


#: The GEMM kernels are compiled for one element type, so every operand must be
#: cast to it.  Feeding fp32 buffers to a kernel built for fp16 reads the memory
#: as garbage (NaN), which is why the cast is centralised here.
_DT = [torch.float16]


def _c16(x):
    return x.to(_DT[0]).contiguous()


def mm_nt(x, y):
    """``x @ y^T`` for ``x:[M,K]``, ``y:[N,K]`` (returns fp32)."""

    out = P.run_gemm(_c16(x), _c16(y), x.shape[0], y.shape[0], x.shape[1],
                     transpose_b=True,
                     dtype=str(_DT[0]).replace("torch.", ""))
    _sync()
    return out


def mm_tn(x, y):
    """``x^T @ y`` for ``x:[K,M]``, ``y:[K,N]`` (returns fp32)."""

    out = P.run_gemm(_c16(x.T), _c16(y.T), x.shape[1], y.shape[1], x.shape[0],
                     transpose_b=True,
                     dtype=str(_DT[0]).replace("torch.", ""))
    _sync()
    return out


def mm_nn(x, y):
    """``x @ y`` expressed as ``mm_tn(x^T, y)`` to stay on the validated NT path."""

    return mm_tn(x.T, y)


def rot_adjoint(dy1, dy2, cos, sin):
    """Transpose of ``y1 = a1*c - a2*s, y2 = a1*s + a2*c``."""

    return dy1 * cos + dy2 * sin, -dy1 * sin + dy2 * cos


def rot_angle_grads(a1, a2, dy1, dy2, cos, sin):
    """``d/dtheta`` of one rotary pair, kept elementwise over the rotary axis.

    ``cos``/``sin`` are per rotary position, so the result is per (row, n2); only
    the rank axis is reduced later.  Reducing here as well silently turns the
    gradient into a scalar per row.
    """

    dcos = dy1 * a1 + dy2 * a2
    dsin = -dy1 * a2 + dy2 * a1
    return -sin * dcos + cos * dsin


def _rot_blocks(x, d, ha):
    """The ``d`` blocks of the ``(d, ha)`` head view of ``x``."""

    return x.reshape(-1, d, ha).unbind(1)


def _rot_pair(x, d, ha):
    """The two blocks the rotary actually rotates: ``0`` and ``d // 2``."""

    blocks = _rot_blocks(x, d, ha)
    return blocks[0], blocks[d // 2]


def _rot_pair_scatter(x, y0, yh, d, ha):
    """``x`` with blocks ``0`` / ``d // 2`` replaced by ``y0`` / ``yh``."""

    out = list(_rot_blocks(x, d, ha))
    out[0], out[d // 2] = y0, yh
    return torch.cat(out, dim=1)


def _silu_grad(u):
    """``d/du [g*tanh(g) + g]`` with ``g = u/2``."""

    g = u * 0.5
    th = torch.tanh(g)
    return 0.5 * (th + g * (1.0 - th * th) + 1.0)


class _Grads:
    """fp32 accumulators, indexed on the flattened token axis."""

    def __init__(self, B, S, R, G, N, H, Pd, dev, n_ang=None):
        z = torch.zeros
        self.dq = z((B * S, R, G, N), device=dev, dtype=torch.float32)
        self.dk = z((B * S, R, G, N), device=dev, dtype=torch.float32)
        self.dv = z((B * S, H, Pd), device=dev, dtype=torch.float32)
        self.dz = z((B * S, H, Pd), device=dev, dtype=torch.float32)
        self.dangles = z((B * S, H, (N // 2) if n_ang is None else n_ang),
                         device=dev, dtype=torch.float32)
        self.dqb = z((H, R, N), device=dev, dtype=torch.float32)
        self.dkb = z((H, R, N), device=dev, dtype=torch.float32)
        self.dpsi = z((H, R, Pd), device=dev, dtype=torch.float32)
        self.dphi = z((H, R, Pd), device=dev, dtype=torch.float32)
        self.dzeta = z((H, R, Pd), device=dev, dtype=torch.float32)
        #: fused pre-gate RMSNorm: DOUT_NORM_WEIGHT is per (batch, head, rank, p)
        #: in the CUDA kernel; the batch axis is summed by the host caller.
        self.doutw = z((B, H, R, Pd), device=dev, dtype=torch.float32)
        self.dD = z((H,), device=dev, dtype=torch.float32)
        # discretisation intermediates (filled by _bwd_sequence)
        self.dacs = z((B * S, H), device=dev, dtype=torch.float32)
        self.dacsrev = z((B * S, H), device=dev, dtype=torch.float32)
        # the chunk-to-chunk state-decay term belongs to the dA chain only:
        # the CUDA golden ddA_cs buffer excludes it.
        self.dacs_chunk = z((B * S, H), device=dev, dtype=torch.float32)
        self.dfactor = z((B * S, H), device=dev, dtype=torch.float32)
        self.dgamma_diag = z((B * S, H), device=dev, dtype=torch.float32)
        self.dsegsum = None


def _recompute(s, i_b, i_h, start, seq_len, seg0):
    """Forward chunk loop that also returns the per-chunk intermediates."""

    C, R, F, N, Pd = s["C"], s["R"], s["F"], s["N"], s["P"]
    dt_, dev = s["dt_"], s["dev"]
    base = i_b * s["S"] if s["dense"] else 0
    seq_end = start + seq_len
    nchunks = math.ceil(seq_len / C)
    d_rot, N2 = _rotary_layout(N, s["rot"])
    i_h_qk = i_h // (s["H"] // s["G"])
    qF, kF, vF, zF, angF = s["qF"], s["kF"], s["vF"], s["zF"], s["angF"]

    step_i = torch.arange(F, device=dev)[:, None] // R
    step_j = torch.arange(F, device=dev)[None, :] // R
    causal = step_i > step_j
    same_step = step_i == step_j
    zero = torch.zeros((), device=dev, dtype=torch.float32)
    f32 = torch.float32

    states = torch.zeros((N, Pd), device=dev, dtype=f32)
    rec = []
    for i in range(nchunks):
        c0 = start + i * C
        n_valid = min(C, seq_end - c0)
        dt_c = _chunk_scalar(s["dt"][i_b, i_h], c0 - base, n_valid, C)
        trap_c = _chunk_scalar(s["trap"][i_b, i_h], c0 - base, n_valid, C)
        gamma = dt_c * torch.sigmoid(trap_c)
        gamma_s = _dt_shifted(s["dt"][i_b, i_h], s["trap"][i_b, i_h],
                             seq_end - base, c0 - base, C)
        trap_scale = gamma + gamma_s
        exp_da = torch.exp(_chunk_scalar(s["dacs"][i_b, i_h], c0 - base,
                                        n_valid, C))
        exp_rev = torch.exp(_chunk_scalar(s["dacsrev"][i_b, i_h], c0 - base,
                                         n_valid, C))
        last_tok = (seq_end - 1 if n_valid < C else c0 + C - 1) - base
        da_last = float(s["dacs"][i_b, i_h, last_tok])

        ang = _pad_rows(angF[c0:c0 + n_valid, i_h, :], n_valid, C)
        cos_c = ang.cos().repeat_interleave(R, 0).to(dt_)
        sin_c = ang.sin().repeat_interleave(R, 0).to(dt_)

        # PsiV kept in fp32 as well, so dv/dPsi do not inherit the GEMM cast
        v_chunk = _pad_rows(vF[c0:c0 + n_valid, i_h, :], n_valid, C).to(f32)
        psiv32 = (v_chunk[:, None, :]
                  * s["psi"][i_h].to(f32)[None, :, :]).reshape(F, Pd)
        psiv = psiv32.to(dt_)

        sc = s["segsum"][i_b, i_h, seg0 + i].to(f32)
        seg_coef = torch.where(
            causal,
            torch.exp(sc[step_i.clamp(max=C - 1), step_j.clamp(max=C - 1)]),
            zero).to(dt_)
        diag_coef = torch.where(
            same_step, gamma.repeat_interleave(R)[:, None].expand(F, F),
            zero).to(dt_)

        q_p = _pad_rows(qF[c0:c0 + n_valid, :, i_h_qk, :], n_valid,
                        C).reshape(F, N).to(dt_)
        k_p = _pad_rows(kF[c0:c0 + n_valid, :, i_h_qk, :], n_valid,
                        C).reshape(F, N).to(dt_)
        bias_q = s["qbias"][i_h].to(f32).repeat(C, 1)
        bias_k = s["kbias"][i_h].to(f32).repeat(C, 1)
        q_p = V.run_ewise("add", q_p.contiguous(), bias_q.to(dt_)); _sync()
        k_p = V.run_ewise("add", k_p.contiguous(), bias_k.to(dt_)); _sync()

        qk_dot = P.run_gemm(q_p, k_p, F, F, N, transpose_b=True).to(dt_)
        _sync()

        q_rot = _rotary_apply(q_p, cos_c, sin_c, d_rot, N2, V.run_rotary)
        _sync()
        k_rot = _rotary_apply(k_p, cos_c, sin_c, d_rot, N2, V.run_rotary)
        _sync()

        # q_rot @ states (no transpose on the state), so use the NN helper
        o_inter = mm_nn(q_rot, states.to(dt_))
        ts_f = trap_scale.repeat_interleave(R, 0)[:, None].expand(F, N).to(dt_)
        k_scaled = V.run_ewise("mul", k_rot, ts_f.contiguous()); _sync()
        qk_intra = P.run_gemm(q_rot, k_scaled, F, F, N, transpose_b=True).to(dt_)
        _sync()
        masked = V.run_ewise("mul", qk_intra, seg_coef); _sync()
        o_intra = mm_nt(masked, psiv.T)
        diagonal = V.run_ewise("mul", qk_dot, diag_coef); _sync()
        o_diag = mm_nt(diagonal, psiv.T)

        # exp(dA_cs) belongs to the inter term only (reference L1003-1004):
        # intra / diagonal / v*D carry their own decay and must not be rescaled.
        e_f = exp_da.repeat_interleave(R, 0)[:, None].expand(F, Pd)
        o4 = o_inter * e_f + o_intra + o_diag
        if s["D"] is not None:
            o4 = o4 + psiv32 * s["D"][i_h].to(f32)

        gate32 = None
        z_chunk = None
        zexp32 = None
        rstd = None
        if s["reduceO"]:
            if zF is not None:
                z_chunk = _pad_rows(zF[c0:c0 + n_valid, i_h, :], n_valid,
                                    C).to(f32)
                # keep the [C, R, P] shape until after the per-rank multiply
                zexp32 = _silu_mimo(z_chunk[:, None, :],
                                    s["zeta"][i_h].to(f32)[None, :, :])
                gate32 = (zexp32 * s["phi"][i_h].to(f32)[None, :, :]
                          ).reshape(F, Pd)
            else:
                gate32 = s["phi"][i_h].to(f32)[None, :, :].expand(
                    C, R, Pd).reshape(F, Pd)
            if s["fuse_rms"]:
                # rstd is recorded once per chunk: the fused epilogue needs it
                # in both the reduced-output and the RMSNorm-backward terms.
                raw_c = o4.reshape(C, R, Pd)
                rstd = torch.rsqrt(raw_c.pow(2).mean(dim=-1, keepdim=True)
                                   + s["eps"])

        rec.append(dict(c0=c0, n_valid=n_valid, states_in=states.clone(),
                        q_p=q_p, k_p=k_p, q_rot=q_rot, k_rot=k_rot,
                        k_scaled=k_scaled, psiv=psiv, psiv32=psiv32,
                        masked=masked, qk_dot=qk_dot, seg_coef=seg_coef,
                        qk_intra=qk_intra,
                        diag_coef=diag_coef, trap_scale=trap_scale,
                        exp_da=exp_da, exp_rev=exp_rev, da_last=da_last,
                        o4=o4, o_inter=o_inter, gate=gate32, z_chunk=z_chunk, cos=cos_c,
                        sin=sin_c, zexp=zexp32, rstd=rstd))
        ks_e = (k_scaled.to(f32)
                * exp_rev.repeat_interleave(R, 0)[:, None].expand(F, N))
        states = states * math.exp(da_last) + mm_tn(ks_e, psiv32)
    return rec


def _bwd_sequence(g, s, i_b, i_h, start, seq_len, seg0, dstate):
    """One (batch entry, head, sequence): recompute, then reverse."""

    C, R, F, N, Pd = s["C"], s["R"], s["F"], s["N"], s["P"]
    dt_, dev, f32 = s["dt_"], s["dev"], torch.float32
    d_rot, N2 = _rotary_layout(N, s["rot"])
    i_h_qk = i_h // (s["H"] // s["G"])
    doF = s["doF"]
    rec = _recompute(s, i_b, i_h, start, seq_len, seg0)

    dstate_cur = (torch.zeros((N, Pd), device=dev, dtype=f32) if dstate is None
                  else dstate.to(f32))
    for it in range(len(rec) - 1, -1, -1):
        r = rec[it]
        c0, n_valid = r["c0"], r["n_valid"]

        # ---- dO -> do4 (and dgate) ---------------------------------------
        if s["reduceO"]:
            dO_c = _pad_rows(doF[c0:c0 + n_valid, i_h, :], n_valid, C).to(f32)
            dO_r = dO_c[:, None, :]
            o4_c = r["o4"].reshape(C, R, Pd)
            if s["fuse_rms"]:
                # Reverse of the fused tail.  With
                #   xhat = raw * rstd,  dnorm = dO * MIMO_O * gate,
                #   y    = xhat * OUT_NORM_WEIGHT * gate,
                #   out  = sum_r y * MIMO_O,
                # the shared raw/rstd path makes this a plain RMSNorm backward:
                #   dW    = sum_{cs,r} dnorm * xhat        (DOUT_NORM_WEIGHT)
                #   dmean = mean_p(dnorm * W * xhat)
                #   d raw = rstd * (dnorm * W - xhat * dmean)
                gate_c = r["zexp"]
                rstd = r["rstd"]
                phi32 = s["phi"][i_h].to(f32)[None, :, :]
                w32 = s["outw"][i_h].to(f32)[None, None, :]
                xhat = o4_c * rstd
                dnorm = dO_r * phi32 * gate_c
                g.doutw[i_b, i_h] += (dnorm * xhat)[:n_valid].sum(0)
                dmean = (dnorm * xhat * w32).sum(dim=-1, keepdim=True) / Pd
                do4 = (rstd * (dnorm * w32 - xhat * dmean)).reshape(F, Pd)
                # dL/d(gate) before the SiLU; MIMO_O is already folded in
                dgate = (dO_r * phi32 * xhat * w32).reshape(F, Pd)
                # dMIMO_O = sum_cs dO * y with y = xhat * W * gate, i.e. *before*
                # the MIMO_O projection -- so no MIMO_O factor here (unlike dgate)
                g.dphi[i_h] += (dO_r * gate_c * xhat * w32)[:n_valid].sum(0)
            else:
                gate_c = r["gate"].reshape(C, R, Pd)
                do4 = (dO_r * gate_c).reshape(F, Pd)
                dgate = (dO_r * o4_c).reshape(F, Pd)
        else:
            do4 = _pad_rows(doF[c0:c0 + n_valid, :, i_h, :], n_valid,
                            C).reshape(F, Pd).to(f32)
            dgate = None

        e_f = r["exp_da"].repeat_interleave(R, 0)[:, None].expand(F, Pd)
        do3 = do4 * e_f          # inter branch only
        do3_16 = do3.to(dt_)
        do4_16 = do4.to(dt_)     # intra / diagonal / v*D branches

        # ---- recurrent state ---------------------------------------------
        dstates_in = dstate_cur * math.exp(r["da_last"])
        ks_e = (r["k_scaled"].to(f32)
                * r["exp_rev"].repeat_interleave(R, 0)[:, None].expand(F, N))
        # dks[f,n] = sum_p PsiV[f,p] * dstate[n,p]  ->  PsiV @ dstate^T
        dks = mm_nt(r["psiv32"], dstate_cur)
        dpsiv = mm_nn(ks_e, dstate_cur)
        # exp(dA_cs_rev) only scales ks_e, so this is the whole dA_cs_rev chain
        _sl = slice(c0, c0 + n_valid)
        g.dacsrev[_sl, i_h] = g.dacsrev[_sl, i_h] + (
            dks * r["k_scaled"].to(f32)).reshape(C, R, N).sum((1, 2))[:n_valid]
        # ... but dks above is d/d ks_e, while the intra-chunk path below
        # contributes d/d k_scaled.  Chain the state contribution through
        # exp(dA_cs_rev) before the two are summed.
        dks = dks * r["exp_rev"].repeat_interleave(R, 0)[:, None].expand(F, N)
        # the chunk-to-chunk decay exp(dA_cs[last token]) multiplies states_in
        last_tok = (start + seq_len - 1 if n_valid < C else c0 + C - 1)
        g.dacs_chunk[last_tok, i_h] += float(
            (dstate_cur * r["states_in"]).sum())

        # ---- o3 decomposition --------------------------------------------
        dqrot = mm_nt(do3_16, r["states_in"].to(dt_))
        dstates_in = dstates_in + mm_tn(r["q_rot"], do3_16)
        dpsiv = dpsiv + mm_tn(r["masked"], do4_16)
        dprod = mm_nt(do4_16, r["psiv"])
        if s["D"] is not None:
            dpsiv = dpsiv + do4 * s["D"][i_h].to(f32)
            g.dD[i_h] += float((do4 * r["psiv32"]).sum())

        dqk_intra = dprod * r["seg_coef"].to(f32)
        diag_c = r["diag_coef"].to(f32)

        # ---- discretisation intermediates ---------------------------------
        # seg_coef = exp(segsum) and masked = qk_intra * seg_coef, so the
        # gradient wrt segsum is dprod * masked directly.
        # segsum lives at token resolution, so the rank block of each (i, j)
        # pair is summed: d/d segsum[i,j] = sum_{r,r''} dprod * masked
        # The (i, j) pairs above are (query, key); the CUDA reference
        # (bwd_segsum_ddt_from_dSSdA_ref) consumes the transposed layout, so
        # store it transposed here.  Without this the reverse-cumsum + tril(-1)
        # mask in _segsum_ddt keeps the wrong triangle and the whole segsum
        # contribution silently evaluates to exactly zero (measured against the
        # A100 golden: the piece was bit-zero before this transpose).
        # The CUDA kernel stores the UNSCALED product here (``lkq_masked *
        # dk_intrachunk``, "the unmasked KQ product"); the consumer
        # (``bwd_segsum_dadt_kernel``) applies ``exp(Segsum)`` itself.  Storing
        # ``masked`` (which already contains exp(segsum)) double-counted the
        # factor and left an 8.5% error in this buffer.
        g.dsegsum[i_b, i_h, seg0 + it] = (
            dprod * r["qk_intra"].to(f32)).reshape(C, R, C, R).sum((1, 3)).transpose(0, 1)
        # o4 = o3 * exp(dA_cs) -> d/d dA_cs picks up one more factor of o4
        _sl = slice(c0, c0 + n_valid)
        g.dacs[_sl, i_h] = g.dacs[_sl, i_h] + (
            do4 * r["o_inter"]).reshape(C, R, Pd).sum((1, 2))[:n_valid]
        # the diagonal term adds gamma[step] to every rank pair of that step
        step_of = torch.arange(F, device=dev) // R
        same_step = (step_of[:, None] == step_of[None, :]).to(f32)
        g.dgamma_diag[_sl, i_h] = g.dgamma_diag[_sl, i_h] + (
            dprod * r["qk_dot"].to(f32) * same_step
        # [C, C] after the rank reduction; the mask leaves only the diagonal
        # block of each step, so the column sum is the per-step total
        ).reshape(C, R, C, R).sum((1, 3)).sum(1)[:n_valid]
        dqk_dot = dprod * diag_c
        dpsiv = dpsiv + mm_tn((r["qk_dot"].to(f32) * diag_c).to(dt_), do4_16)

        # ---- qk_intra / qk_dot -------------------------------------------
        dqrot = dqrot + mm_nn(dqk_intra.to(dt_), r["k_scaled"])
        dks = dks + mm_tn(dqk_intra.to(dt_), r["q_rot"])
        dqt = mm_nn(dqk_dot.to(dt_), r["k_p"])
        dkt = mm_tn(dqk_dot.to(dt_), r["q_p"])

        # ---- k_scaled = k_rot * trap_scale -------------------------------
        # every consumer of k_scaled has contributed to dks by now, so this is
        # the complete gradient wrt the trapezoid factor of each step
        g.dfactor[_sl, i_h] = g.dfactor[_sl, i_h] + (
            dks * r["k_rot"].to(f32)).reshape(C, R, N).sum((1, 2))[:n_valid]
        dkrot = dks * r["trap_scale"].repeat_interleave(R, 0)[:, None].expand(F, N)

        # ---- rotary adjoint ----------------------------------------------
        # Only blocks (0, d//2) of the head are rotated, so the remaining blocks
        # carry the gradient through unchanged and the angle gradient involves
        # those two blocks alone.
        cosf, sinf = r["cos"].to(f32), r["sin"].to(f32)
        qp0, qph = _rot_pair(r["q_p"].to(f32), d_rot, N2)
        kp0, kph = _rot_pair(r["k_p"].to(f32), d_rot, N2)
        dq0, dqh = _rot_pair(dqrot, d_rot, N2)
        dk0, dkh = _rot_pair(dkrot, d_rot, N2)
        dqt = dqt + _rot_pair_scatter(
            dqrot, *rot_adjoint(dq0, dqh, cosf, sinf), d_rot, N2)
        dkt = dkt + _rot_pair_scatter(
            dkrot, *rot_adjoint(dk0, dkh, cosf, sinf), d_rot, N2)

        ang_q = rot_angle_grads(qp0, qph, dq0, dqh, cosf, sinf)
        ang_k = rot_angle_grads(kp0, kph, dk0, dkh, cosf, sinf)
        g.dangles[c0:c0 + n_valid, i_h, :] += (
            ang_q + ang_k).reshape(C, R, N2).sum(1)[:n_valid]

        # ---- activation gradients ----------------------------------------
        dqt_c = dqt.reshape(C, R, N)
        dkt_c = dkt.reshape(C, R, N)
        g.dq[c0:c0 + n_valid, :, i_h_qk, :] += dqt_c[:n_valid]
        g.dk[c0:c0 + n_valid, :, i_h_qk, :] += dkt_c[:n_valid]
        g.dqb[i_h] += dqt_c.sum(0)
        g.dkb[i_h] += dkt_c.sum(0)

        # ---- PsiV -> dv, dPsi --------------------------------------------
        dpsiv_c = dpsiv.reshape(C, R, Pd)
        v_chunk = _pad_rows(s["vF"][c0:c0 + n_valid, i_h, :], n_valid, C).to(f32)
        g.dv[c0:c0 + n_valid, i_h, :] += (
            dpsiv_c[:n_valid] * s["psi"][i_h].to(f32)[None, :, :]).sum(1)
        g.dpsi[i_h] += (dpsiv_c[:n_valid]
                        * v_chunk[:n_valid][:, None, :]).sum(0)

        # ---- gating ------------------------------------------------------
        if dgate is not None:
            dgate_c = dgate.reshape(C, R, Pd)
            phi32 = s["phi"][i_h].to(f32)[None, :, :]
            if r["z_chunk"] is not None:
                zc = r["z_chunk"].to(f32)
                zeta32 = s["zeta"][i_h].to(f32)[None, :, :]
                ds = _silu_grad(zc[:, None, :] * zeta32)
                if s["fuse_rms"]:
                    # MIMO_O is already inside dgate (its gradient was summed
                    # with the pre-projection y above)
                    dzexp = dgate_c
                else:
                    dzexp = dgate_c * phi32
                    g.dphi[i_h] += (dgate_c * _silu_mimo(zc[:, None, :], zeta32)
                                    )[:n_valid].sum(0)
                g.dz[c0:c0 + n_valid, i_h, :] += (
                    dzexp * ds * zeta32)[:n_valid].sum(1)
                g.dzeta[i_h] += (dzexp * ds * zc[:, None, :])[:n_valid].sum(0)
            else:
                g.dphi[i_h] += dgate_c[:n_valid].sum(0)

        dstate_cur = dstates_in


# ---------------------------------------------------------------------------
# Discretisation chain.  These mirror the ``*_ref`` host functions that the CUDA
# operator calls after its kernel (``mamba3_mimo_utils``), so they stay on the
# host and need no device kernel.
# ---------------------------------------------------------------------------
_LOG2E = math.log2(math.e)


def _segsum_ddt(d_ssda, dA_cs, C):
    """segsum -> dadt contribution.  ``d_ssda`` is ``[n,H,nchunks,C,C]``.

    Mirrors ``bwd_segsum_ddt_from_dSSdA_ref`` except for the final
    ``-log2(e)``: the CUDA host adds this straight into ``ddA`` without scaling,
    and the golden confirms it (``ddA_final - DDA_pass2`` reproduces the chain
    over ``-log2e`` with rel_l2 1.1e-7).
    """

    n, H, nch, _, _ = d_ssda.shape
    da = dA_cs.view(n, H, nch, C)
    # The CUDA kernel exponentiates the STORED Segsum, whose strictly-upper
    # triangle is zeroed (exp -> 1) rather than exp(da_i - da_j).  Masking the
    # exponent the same way makes this exactly bwd_segsum_dadt_kernel.
    e = torch.tril(da[..., :, None] - da[..., None, :], -1)
    x = d_ssda.permute(0, 1, 2, 4, 3) * torch.exp(e)
    # reverse cumsum along the token axis, then keep the strictly-lower triangle
    x = torch.flip(torch.cumsum(torch.flip(x, dims=[-2]), dim=-2), dims=[-2])
    tri = torch.tril(torch.ones(C, C, device=x.device, dtype=x.dtype), -1)
    return ((x * tri).sum(-1)).reshape(n, H, nch * C)


def _ddA_cs_ddt(ddA_cs, dA_cs, C):
    """Reverse-cumsum path of ``dA_cs`` -> dadt (no ``-log2e``, see above)."""

    n, H, S = ddA_cs.shape
    nch = S // C
    x = (torch.exp(dA_cs) * ddA_cs).view(n, H, nch, C)
    x = torch.flip(torch.cumsum(torch.flip(x, dims=[-1]), dim=-1), dims=[-1])
    return x.reshape(n, H, S)


def _ddA_cs_rev_ddt(ddA_cs_rev, dA_cs_rev, C):
    """Forward-exclusive-cumsum path of ``dA_cs_rev`` -> dadt (no ``-log2e``)."""

    n, H, S = ddA_cs_rev.shape
    nch = S // C
    x = (torch.exp(dA_cs_rev) * ddA_cs_rev).view(n, H, nch, C)
    x = torch.cumsum(x, dim=-1)
    x = torch.cat([torch.zeros_like(x[..., :1]), x[..., :-1]], dim=-1)
    return x.reshape(n, H, S)


def _dtrap_ddt(dfactor, dgamma_diag, trap, dt):
    """Trapezoid parameterisation: ``(ddt, dtrap)`` for the direct path.

    ``dfactor`` is the gradient of the *shifted* gamma, ``dgamma_diag`` the
    gradient of the unshifted one (the diagonal term).
    """

    trap_s = torch.sigmoid(trap)
    strap = torch.nn.functional.pad(trap_s[:, :, 1:], (0, 1), value=0.0)
    sdt = torch.nn.functional.pad(dt[:, :, 1:], (0, 1), value=0.0)
    dgamma = dfactor + dgamma_diag
    dsgamma = dfactor
    dsdt = (1.0 - strap) * dsgamma
    dstrap = -sdt * dsgamma
    ddt = torch.nn.functional.pad(dsdt[:, :, :-1], (1, 0), value=0.0)
    dtrap = torch.nn.functional.pad(dstrap[:, :, :-1], (1, 0), value=0.0)
    dtrap = dtrap + dgamma * dt
    dtrap = dtrap * trap_s * torch.sigmoid(-trap)
    ddt = ddt + dgamma * trap_s
    return ddt, dtrap


def mamba_mimo_backward(dout, q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, mimo_z,
                        angles, dA_cs, dA_cs_rev, dt, trap, D, segsum,
                        chunk_size, rotary_dim_divisor, dtype=torch.float16,
                        cu_seqlens=None, dstate=None,
                        fuse_pregate_headwise_rms_norm=False,
                        outproj_norm_weight=None, outproj_norm_eps=1e-5):
    """Backward for the dense and packed-varlen forward.

    ``dstate`` is the optional gradient of the final recurrent state, shaped
    ``[ns, H, N, P]`` (or ``[H, N, P]`` when there is one sequence).  The
    discretisation gradients -- ``ddt``, ``dtrap``, ``ddA``, ``ddA_cs``,
    ``ddA_cs_rev``, ``dsegsum``, ``dfactor``, ``dgamma_diag`` -- come from the
    host-side chain described in the module docstring, not from a device kernel.

    With ``fuse_pregate_headwise_rms_norm`` the fused epilogue is reversed here:
    the incoming ``dout`` is pushed through the headwise RMSNorm (including the
    ``OUT_NORM_WEIGHT`` gradient, returned as ``dout_norm_weight`` shaped
    ``[B, H, R, P]`` like the CUDA kernel's ``DOUT_NORM_WEIGHT``) and through
    the SiLU gate before the usual reverse chunk loop consumes the gradient of
    the raw MIMO output.
    """

    B, S, R, G, N = q.shape
    H, Pd = v.shape[-2], v.shape[-1]
    C = chunk_size
    dev = q.device
    dt_ = dtype if isinstance(dtype, torch.dtype) else torch.float16
    reduceO = mimo_o is not None
    assert G == 1, "A2 port assumes G == 1"
    if fuse_pregate_headwise_rms_norm:
        # Same preconditions as the CUDA wrapper.
        if not reduceO:
            raise ValueError(
                "fuse_pregate_headwise_rms_norm=True requires mimo_o.")
        if z is None or mimo_z is None:
            raise ValueError(
                "fuse_pregate_headwise_rms_norm=True requires z and mimo_z.")

    s = dict(C=C, R=R, F=C * R, N=N, P=Pd, H=H, G=G, S=S, B=B, dev=dev, dt_=dt_,
             rot=rotary_dim_divisor, reduceO=reduceO, D=D, psi=mimo_v,
             phi=mimo_o, zeta=mimo_z, qbias=q_bias, kbias=k_bias, dt=dt,
             trap=trap, dacs=dA_cs, dacsrev=dA_cs_rev, segsum=segsum,
             fuse_rms=bool(fuse_pregate_headwise_rms_norm),
             outw=(_outproj_weight(outproj_norm_weight, H, Pd, dev)
                   if fuse_pregate_headwise_rms_norm else None),
             eps=float(outproj_norm_eps),
             qF=q.reshape(B * S, R, G, N), kF=k.reshape(B * S, R, G, N),
             vF=v.reshape(B * S, H, Pd),
             zF=None if z is None else z.reshape(B * S, H, Pd),
             angF=angles.reshape(B * S, H, angles.shape[-1]),
             doF=(dout.reshape(B * S, H, Pd) if reduceO
                  else dout.reshape(B * S, R, H, Pd)))

    _DT[0] = dt_
    g = _Grads(B, S, R, G, N, H, Pd, dev, n_ang=angles.shape[-1])
    g.dsegsum = torch.zeros(segsum.shape, device=dev, dtype=torch.float32)
    if cu_seqlens is None:
        items = [(i_b, i_b * S, S, 0) for i_b in range(B)]
        s["dense"] = True
    else:
        assert B == 1, "varlen packing expects a single batch entry"
        cu = cu_seqlens.to(torch.int64).cpu()
        items = [(0, int(cu[i]), int(cu[i + 1]) - int(cu[i]), int(cu[i]) // C + i)
                 for i in range(len(cu) - 1)]
        s["dense"] = False

    for ns, (i_b, start, seq_len, seg0) in enumerate(items):
        for i_h in range(H):
            if dstate is None:
                ds = None
            elif dstate.dim() == 4:
                ds = dstate[ns, i_h]
            else:
                ds = dstate
            _bwd_sequence(g, s, i_b, i_h, start, seq_len, seg0, ds)

    # ---- discretisation chain (host side, per sequence) -------------------
    dacs = g.dacs.reshape(B, S, H).permute(0, 2, 1).contiguous()      # [B,H,S]
    dacsrev = g.dacsrev.reshape(B, S, H).permute(0, 2, 1).contiguous()
    dacsc = g.dacs_chunk.reshape(B, S, H).permute(0, 2, 1).contiguous()
    dfac = g.dfactor.reshape(B, S, H).permute(0, 2, 1).contiguous()
    dgd = g.dgamma_diag.reshape(B, S, H).permute(0, 2, 1).contiguous()
    trap_f, dt_f = trap.to(torch.float32), dt.to(torch.float32)

    ddt_direct = torch.zeros((B, H, S), device=dev, dtype=torch.float32)
    dtrap_out = torch.zeros((B, H, S), device=dev, dtype=torch.float32)
    ddA_out = torch.zeros((B, H, S), device=dev, dtype=torch.float32)
    for i_b, start, seq_len, seg0 in items:
        loc0 = 0 if s["dense"] else start
        sl = slice(loc0, loc0 + seq_len)
        nch = math.ceil(seq_len / C)
        P = nch * C

        def _pad(x):
            out = torch.zeros((1, H, P), device=dev, dtype=torch.float32)
            out[:, :, :seq_len] = x[i_b:i_b + 1, :, sl]
            return out

        # NOTE: the second argument is the FORWARD cumsum `dA_cs`, not the
        # gradient buffer `dacs`.  Passing the latter makes the exponent
        # exp(ddA_cs_i - ddA_cs_j) instead of exp(dA_cs_i - dA_cs_j).
        d1 = _segsum_ddt(g.dsegsum[i_b:i_b + 1, :, seg0:seg0 + nch],
                         _pad(dA_cs.to(torch.float32)), C)
        d2 = _ddA_cs_rev_ddt(_pad(dacsrev), _pad(dA_cs_rev.to(torch.float32)), C)
        d3 = _ddA_cs_ddt(_pad(dacs) + _pad(dacsc),
                         _pad(dA_cs.to(torch.float32)), C)
        dt_direct, dtrap_seq = _dtrap_ddt(_pad(dfac), _pad(dgd),
                                          _pad(trap_f), _pad(dt_f))
        ddA_out[i_b:i_b + 1, :, sl] = (d1 + d2 + d3)[:, :, :seq_len]
        ddt_direct[i_b:i_b + 1, :, sl] = dt_direct[:, :, :seq_len]
        dtrap_out[i_b:i_b + 1, :, sl] = dtrap_seq[:, :, :seq_len]
    # CUDA semantics: ddt is the trapezoid/dt path only.  The decay chain above
    # lands in ddA (the CUDA host does ``ddA += dadt_fused(...)``), so adding it
    # to ddt as well double-counts it.  Measured on the A100 golden:
    # ``golden_ddt == compute_dtrap_ddt_ref(...)`` to rel_l2 7e-8, and
    # ``A2 ddt - A2 ddA`` matched golden ddt to 4.9% (the fp16 noise floor)
    # where ``A2 ddt`` alone was 35% off.
    ddt_out = ddt_direct

    return dict(
        dq=g.dq.reshape(q.shape).to(dt_),
        dk=g.dk.reshape(k.shape).to(dt_),
        dv=g.dv.reshape(v.shape).to(dt_),
        dq_bias=g.dqb, dk_bias=g.dkb,
        dmimo_v=g.dpsi, dmimo_o=g.dphi, dmimo_z=g.dzeta,
        dz=g.dz.reshape(v.shape).to(dt_), dD=g.dD,
        dangles=g.dangles.reshape(angles.shape),
        # chain intermediates (host-side maths, see the module docstring).  ddA
        # here is the decay-chain part only: this entry point has no ``A``/``ADT``
        # input, so the CUDA kernel's direct dA term has no counterpart.
        ddA_cs=g.dacs.reshape(B, S, H).permute(0, 2, 1).contiguous(),
        ddA_cs_rev=g.dacsrev.reshape(B, S, H).permute(0, 2, 1).contiguous(),
        dsegsum=g.dsegsum,
        dfactor=g.dfactor.reshape(B, S, H).permute(0, 2, 1).contiguous(),
        dgamma_diag=g.dgamma_diag.reshape(B, S, H).permute(0, 2, 1).contiguous(),
        ddt=ddt_out, dtrap=dtrap_out, ddA=ddA_out,
        # separate parts, so the autograd comparison can isolate each path
        ddt_direct=ddt_direct,
        # fused pre-gate RMSNorm: DOUT_NORM_WEIGHT at the kernel's granularity
        # ([B, H, R, P]); the host caller sums the batch axis like the CUDA
        # wrapper does.  Absent (``None``) when the fused epilogue did not run --
        # that is the CUDA entry point's contract too, and the upstream test
        # asserts it (``assert dout_norm_weight is None``).
        dout_norm_weight=(g.doutw if fuse_pregate_headwise_rms_norm else None))
