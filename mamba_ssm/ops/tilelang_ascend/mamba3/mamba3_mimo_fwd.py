"""Mamba-3 MIMO chunked (dense and variable-length) forward for Ascend 910B (A2).

Ported from ``mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`` and
``..._fwd_varlen.py`` (CUDA TileLang).  The chunk algebra, the workspace layout
and the discretisation maths are unchanged; the execution model is not:

* the CUDA kernels are one fused per-``(head, batch[, sequence])`` kernel with
  register fragments, ``T.Pipelined`` and a 2-D/3-D grid.  The A2 backend has no
  fragment level, ``npuir_dot`` is one-shot (a kernel may contain exactly one
  dot), and vector and cube work run on different cores;
* so this module keeps the host-side plan -- packing, the per-token scalars, the
  segsum mask, the rotary angles, ``PsiV`` and the gating product, all of which
  are pure functions of the operator inputs -- and expresses each chunk as a
  short sequence of A2 kernels from :mod:`a2_prims` and :mod:`a2_vec`;
* every broadcast is pre-expanded and every GEMM uses the NT form: the NN form
  miscomputes (``nn_128x128x64`` measured rel 1.2e+00) and unsupported
  broadcasts silently return wrong values (``MIGRATION.md`` §4.4-4.5).

Math per chunk ``i`` of one ``(batch b, head h)``, with ``C`` chunk steps,
``R`` MIMO ranks, ``F = C*R``, state dim ``N``, head dim ``P``::

    q  = Q[b, c0:c0+C, :, hqk, :] + q_bias[h]           # [F, N]
    k  = K[b, c0:c0+C, :, hqk, :] + k_bias[h]           # [F, N]
    qk_dot = q @ k^T                                     # pre-rotary, [F, F]
    q, k  <- rotary(q, k; cos, sin)                      # per-step angles
    o     = q @ states                                   # inter-chunk
    k    *= trap_scale[step]
    qk_intra = q @ k^T
    masked   = qk_intra * seg_mask                       # strictly causal * exp(segsum)
    o       += masked @ PsiV
    diag     = (qk_dot * block_diag_gamma) @ PsiV        # same-step R x R block
    o       += diag + D * PsiV
    o       *= exp(dA_cs[step])
    o       *= phi * silu(z * zeta)                      # when reduceO / gating on
    O        = sum_r o                                   # rank reduction
    states   = states * exp(dA_cs[last]) + (k * exp(dA_cs_rev))^T @ PsiV

Variable length packs several sequences into the token axis; each sequence runs
the identical chunk loop over ``[cu_seqlens[ns], cu_seqlens[ns+1])`` with its
tail chunk zero padded, and reads ``SEGSUM`` at the global chunk index
``cu_seqlens[ns] // C + ns + i``.  Zero padding makes every padded row contribute
nothing, so no extra masking is needed.

Known gaps (``MIGRATION.md`` §7): the epilogue rank reduction -- and the
per-rank headwise RMS statistics of the ``fuse_pregate_headwise_rms_norm``
epilogue -- run on the host.
"""

import math
import os

import torch

from mamba_ssm.ops.tilelang_ascend.mamba3 import a2_prims as P
from mamba_ssm.ops.tilelang_ascend.mamba3 import a2_vec as V

#: Consecutive A2 kernel launches are not ordered tightly enough for a dependent
#: chain: without a device sync between them the consumer sometimes reads a
#: half-written buffer (the failure moves between stages run to run and shows up
#: as NaN).  ``A2_SYNC=0`` disables the sync once launch ordering is understood.
_DO_SYNC = bool(int(os.environ.get("A2_SYNC", "1")))
_DEBUG = bool(int(os.environ.get("A2_DEBUG", "0")))


def _sync():
    if _DO_SYNC:
        torch.npu.synchronize()


def _dbg(tag, t):
    if _DEBUG:
        f = t.float()
        print(f"  [a2] {tag}: max={float(f.abs().max()):.4g} "
              f"nan={int(torch.isnan(f).sum())}")


def _gemm(*a, **k):
    r = P.run_gemm(*a, **k)
    _sync()
    return r


def _ew(op, *a, **k):
    r = V.run_ewise(op, *a, **k)
    _sync()
    return r


def _rot(*a, **k):
    r = V.run_rotary(*a, **k)
    _sync()
    return r


def _rotary_layout(N, rot):
    """Block layout of the reference rotary: ``(d, N/d)`` with ``d = rot``.

    The reference implements exactly two conventions.  ``d=2`` pairs block 0
    with block 1 of a ``(2, N/2)`` head view (LLaMA halves); ``d=4`` pairs block
    0 with block 2 of a ``(4, N/4)`` view and leaves blocks 1 and 3 untouched
    (``apply_rotary_emb_rotate_half``).  Both apply the same angles, so the
    common rule is: view the head as ``(d, N/d)`` and rotate block 0 against
    block ``d // 2``.  Any other divisor the reference rejects outright.
    """

    d = int(rot)
    if d not in (2, 4) or N % d:
        raise ValueError(
            f"unsupported rotary_dim_divisor={rot} for headdim_qk={N}: the "
            "reference implements 2 (halves) and 4 (rotate_half) only")
    return d, N // d


def _rotary_apply(x, cos, sin, d, ha, rot_fn):
    """Rotate block 0 against block ``d // 2`` of the ``(d, ha)`` head view.

    ``rot_fn`` is :func:`a2_vec.run_rotary` on the device path.
    """

    blocks = list(x.reshape(-1, d, ha).unbind(1))
    y0, yh = rot_fn(blocks[0].contiguous(), blocks[d // 2].contiguous(), cos, sin)
    out = list(blocks)
    out[0], out[d // 2] = y0, yh
    return torch.cat(out, dim=1).contiguous()


def _dt_shifted(dt_row, trap_row, seq_end, c0, chunk_size):
    """``dt[cs+1] * sigmoid(-trap[cs+1])``, zero at the last step of a sequence.

    ``dt_row``/``trap_row`` are 1-D views of the whole token axis and ``seq_end``
    is the exclusive end of the current sequence, so the shift never crosses into
    the next packed sequence.
    """

    idx = torch.arange(c0 + 1, c0 + chunk_size + 1, device=dt_row.device)
    valid = idx < seq_end
    nxt = idx.clamp(max=seq_end - 1)
    out = dt_row[nxt] * torch.sigmoid(-trap_row[nxt])
    return torch.where(valid, out, torch.zeros_like(out))


def _silu_mimo(z, zeta):
    """``silu(z * zeta)`` written the way the CUDA kernel does it."""

    g = z * zeta * 0.5
    return g * torch.tanh(g) + g


def _outproj_weight(outproj_norm_weight, H, Pd, device, dtype=torch.float32):
    """``OUT_NORM_WEIGHT`` as ``[H, P]`` fp32 (or ``dtype``), defaulting to all ones.

    Mirrors the CUDA host wrapper: ``None`` means an all-ones weight, and the
    caller may hand over either ``[H, P]`` or the flattened ``[H * P]`` buffer
    the model keeps for its output ``RMSNorm``.  ``dtype`` exists only so the
    fp64 reference can keep its whole graph in fp64; the default is unchanged.
    """

    if outproj_norm_weight is None:
        return torch.ones((H, Pd), dtype=dtype, device=device)
    w = outproj_norm_weight
    if w.ndim == 1:
        if w.numel() != H * Pd:
            raise ValueError(
                f"Expected flattened outproj_norm_weight to have {H * Pd} "
                f"elements, got {w.numel()}.")
        w = w.reshape(H, Pd)
    elif tuple(w.shape) != (H, Pd):
        raise ValueError(
            f"Expected outproj_norm_weight to have shape ({H}, {Pd}) or "
            f"({H * Pd},), got {tuple(w.shape)}.")
    return w.to(dtype)


def _pad_rows(x, n_valid, rows):
    """Zero pad ``x[:n_valid]`` to ``rows`` along the leading axis."""

    if n_valid == rows:
        return x
    out = torch.zeros((rows,) + tuple(x.shape[1:]), device=x.device,
                      dtype=x.dtype)
    out[:n_valid] = x[:n_valid]
    return out


def _chunk_scalar(row, c0, n_valid, C, dtype=torch.float32):
    """``[C]`` view of a per-token scalar row, zero padded past the sequence end.

    A tail chunk only has ``n_valid < C`` real tokens left in the token axis, so
    slicing ``C`` entries there would run past the tensor.
    """

    return _pad_rows(row[c0:c0 + n_valid].to(dtype), n_valid, C)


def _psiv(s, c0, n_valid, C, dtype=None, acc_dtype=torch.float32):
    """``PsiV[cs*R + r, :] = V[token, h, :] * Psi[h, r, :]`` for one chunk.

    ``acc_dtype`` is the dtype the product is formed in; it defaults to fp32 (the
    device path) and the fp64 reference passes ``torch.float64`` instead.
    """

    Pd, R = s["P"], s["R"]
    v_chunk = _pad_rows(s["v_flat"][c0:c0 + n_valid, s["_h"], :], n_valid,
                        C).to(acc_dtype)
    psiv = v_chunk[:, None, :] * s["psi"][s["_h"]].to(acc_dtype)[None, :, :]
    psiv = psiv.reshape(C * R, Pd).contiguous()
    return psiv if dtype is None else psiv.to(dtype)


def _forward_sequence(s, i_b, i_h, start, seq_len, seg0, states, o_flat):
    """Chunk loop for one packed sequence; returns the final recurrent state."""

    C, R, F, N, Pd = s["C"], s["R"], s["F"], s["N"], s["P"]
    dt_, dev = s["dt_"], s["q"].device
    seq_end = start + seq_len
    nchunks = math.ceil(seq_len / C)
    d_rot, N2 = _rotary_layout(N, s["rot"])
    s["_h"] = i_h
    # ``dt``/``trap``/``dA_cs*`` keep a batch axis; the chunk offset is global for
    # the flattened token views, so the scalars need a batch-local index.
    base = i_b * s["S"] if s["dense"] else 0

    qF, kF, vF = s["q_flat"], s["k_flat"], s["v_flat"]
    zF, angF = s["z_flat"], s["ang_flat"]
    i_h_qk = i_h // (s["H"] // s["G"])
    reduceO = s["reduceO"]

    step_i = torch.arange(F, device=dev)[:, None] // R
    step_j = torch.arange(F, device=dev)[None, :] // R
    causal = step_i > step_j
    same_step = step_i == step_j
    zero = torch.zeros((), device=dev, dtype=torch.float32)

    for i in range(nchunks):
        c0 = start + i * C
        n_valid = min(C, seq_end - c0)

        # ---------------- host-side plan (input-only maths) ----------------
        dt_c = _chunk_scalar(s["dt"][i_b, i_h], c0 - base, n_valid, C)
        trap_c = _chunk_scalar(s["trap"][i_b, i_h], c0 - base, n_valid, C)
        gamma = dt_c * torch.sigmoid(trap_c)
        gamma_s = _dt_shifted(s["dt"][i_b, i_h], s["trap"][i_b, i_h],
                              seq_end - base, c0 - base, C)
        trap_scale = gamma + gamma_s                              # [C]
        exp_da = torch.exp(_chunk_scalar(s["dacs"][i_b, i_h], c0 - base, n_valid, C))
        exp_da_rev = torch.exp(_chunk_scalar(s["dacsrev"][i_b, i_h], c0 - base, n_valid, C))
        last_tok = (seq_end - 1 if n_valid < C else c0 + C - 1) - base
        da_last = float(s["dacs"][i_b, i_h, last_tok])

        ang = _pad_rows(angF[c0:c0 + n_valid, i_h, :], n_valid, C
                        ).to(torch.float32)                       # [C, N2]
        cos_c = ang.cos().repeat_interleave(R, 0).to(dt_)         # [F, N2]
        sin_c = ang.sin().repeat_interleave(R, 0).to(dt_)

        sc = s["segsum"][i_b, i_h, seg0 + i].to(torch.float32)    # [C, C]
        seg_coef = torch.where(
            causal,
            torch.exp(sc[step_i.clamp(max=C - 1), step_j.clamp(max=C - 1)]),
            zero,
        ).to(dt_)                                                 # [F, F]
        diag_coef = torch.where(
            same_step,
            gamma.repeat_interleave(R)[:, None].expand(F, F),
            zero,
        ).to(dt_)                                                 # [F, F]

        psiv = _psiv(s, c0, n_valid, C, dt_)                      # [F, P]

        if reduceO:
            zexp = None
            if zF is not None:
                z_chunk = _pad_rows(zF[c0:c0 + n_valid, i_h, :], n_valid,
                                    C).to(torch.float32)
                zexp = _silu_mimo(z_chunk[:, None, :],
                                  s["zeta"][i_h].to(torch.float32)[None, :, :])
            if s["fuse_rms"]:
                # The fused epilogue applies MIMO_O *after* the headwise RMS
                # scaling, so the gate is kept unprojected and combined with
                # ``rstd`` in the epilogue below.
                gate_coef = None
            elif zexp is not None:
                gate_coef = (zexp * s["phi"][i_h].to(torch.float32)[None, :, :]
                             ).to(dt_)
            else:
                gate_coef = s["phi"][i_h].to(torch.float32)[None, :, :].expand(
                    C, R, Pd).to(dt_)
            if gate_coef is not None:
                gate_coef = gate_coef.reshape(F, Pd).contiguous()  # [F, P]

        # ---------------- device stages ------------------------------------
        q_p = _pad_rows(qF[c0:c0 + n_valid, :, i_h_qk, :], n_valid,
                        C).reshape(F, N).to(dt_)
        k_p = _pad_rows(kF[c0:c0 + n_valid, :, i_h_qk, :], n_valid,
                        C).reshape(F, N).to(dt_)
        q_p = _ew("add", q_p.contiguous(),
                  s["qbias"][i_h].to(torch.float32).repeat(C, 1).to(dt_))
        k_p = _ew("add", k_p.contiguous(),
                  s["kbias"][i_h].to(torch.float32).repeat(C, 1).to(dt_))
        _dbg("q_p", q_p)

        qk_dot = _gemm(q_p, k_p, F, F, N, transpose_b=True).to(dt_)
        _dbg("qk_dot", qk_dot)

        # ``rot=2`` pairs block 0 with block 1 of the (2, N/2) head view;
        # ``rot=4`` pairs block 0 with block 2 of the (4, N/4) view, leaving
        # blocks 1 and 3 unrotated (``apply_rotary_emb_rotate_half``).
        q_rot = _rotary_apply(q_p, cos_c, sin_c, d_rot, N2, _rot)
        _dbg("q_rot", q_rot)
        k_rot = _rotary_apply(k_p, cos_c, sin_c, d_rot, N2, _rot)

        # NT only: the NN path miscomputes (``nn_128x128x64`` -> rel 1.2e+00), so
        # the right operand is transposed on the host instead.
        o_acc = _gemm(q_rot, states.to(dt_).T.contiguous(), F, Pd, N,
                      transpose_b=True)
        # exp(dA_cs) belongs to the inter term ONLY (reference L1003-1004):
        # intra / diagonal / v*D already carry their own decay.
        o_acc = _ew("mul", o_acc.to(dt_),
                    exp_da.repeat_interleave(R)[:, None].expand(
                        F, Pd).to(dt_).contiguous())

        k_scaled = _ew("mul", k_rot,
                       trap_scale.repeat_interleave(R)[:, None].expand(
                           F, N).to(dt_).contiguous())
        qk_intra = _gemm(q_rot, k_scaled, F, F, N, transpose_b=True).to(dt_)
        masked = _ew("mul", qk_intra, seg_coef)
        o_acc = _ew("add", o_acc.to(dt_),
                    _gemm(masked, psiv.T.contiguous(), F, Pd, F,
                          transpose_b=True).to(dt_))

        diagonal = _ew("mul", qk_dot, diag_coef)
        o_acc = _ew("add", o_acc,
                    _gemm(diagonal, psiv.T.contiguous(), F, Pd, F,
                          transpose_b=True).to(dt_))

        if s["D"] is not None:
            d_val = s["D"][i_h].to(torch.float32)
            o_acc = _ew("add", o_acc, (psiv.to(torch.float32) * d_val).to(dt_))

        _dbg("o_scaled", o_acc)

        # ---------------- epilogue -----------------------------------------
        if reduceO:
            if s["fuse_rms"]:
                # CUDA source L361-413:  rstd = 1/sqrt(mean_p(raw^2) + eps) is
                # per (chunk step, rank) -- the RMS runs over the head dim P and
                # never over the rank axis -- and
                #   y[cs, r, p] = raw * rstd * OUT_NORM_WEIGHT[p] * gate[cs,r,p]
                #   out[cs, p]  = sum_r y[cs, r, p] * MIMO_O[r, p]
                raw32 = o_acc.to(torch.float32).reshape(C, R, Pd)
                rstd = torch.rsqrt(raw32.pow(2).mean(dim=-1, keepdim=True)
                                   + s["eps"])
                coef = (zexp
                        * rstd
                        * s["outw"][i_h].to(torch.float32)[None, None, :]
                        ).reshape(F, Pd).to(dt_).contiguous()
                o_acc = _ew("mul", o_acc, coef)
                o_rank = (o_acc.to(torch.float32).reshape(C, R, Pd)
                          * s["phi"][i_h].to(torch.float32)[None, :, :]
                          ).sum(dim=1)
            else:
                o_acc = _ew("mul", o_acc, gate_coef)
                o_rank = o_acc.to(torch.float32).reshape(C, R, Pd).sum(dim=1)
            o_flat[c0:c0 + n_valid, i_h, :] = o_rank[:n_valid].to(dt_)
        else:
            o_flat[c0:c0 + n_valid, :, i_h, :] = o_acc.reshape(C, R, Pd)[:n_valid]

        # ---------------- recurrent state ----------------------------------
        # The CUDA kernel builds this from k_shared *after* the trap scaling
        # (source line 307 then 441), not from the rotary k.
        k_state = _ew(
            "mul", k_scaled,
            exp_da_rev.repeat_interleave(R)[:, None].expand(
                F, N).to(dt_).contiguous())
        st_new = _gemm(k_state.T.contiguous(), psiv.T.contiguous(), N, Pd, F,
                       transpose_b=True)
        states = states * math.exp(da_last) + st_new

        if i == nchunks - 1 and s["want_k"]:
            # ``k_rot`` is laid out as token*R + r, so the row slice must use the
            # R-major count: ``k_rot[:n_valid][-R:]`` would take token R-1 of the
            # chunk instead of the last token (verified against the A100 golden:
            # the old code matched rope(K[token 3]) with cos 0.9956, the fix
            # matches rope(K[token C-1])).
            s["final_k"][s["ns_index"], :, i_h, :] = \
                k_rot[:n_valid * R][-R:, :].reshape(R, N)

    return states


def _prepare(q, k, v, z, angles, N):
    B, S, R, G, _ = q.shape
    H, Pd = v.shape[-2], v.shape[-1]
    return dict(q=q, k=k, v=v, z=z, B=B, S=S, R=R, G=G, N=N, H=H, P=Pd,
                q_flat=q.reshape(B * S, R, G, N),
                k_flat=k.reshape(B * S, R, G, N),
                v_flat=v.reshape(B * S, H, Pd),
                z_flat=None if z is None else z.reshape(B * S, H, Pd),
                ang_flat=angles.reshape(B * S, H, angles.shape[-1]))


def mamba_mimo_forward(q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, D, mimo_z,
                       angles, dA_cs, dA_cs_rev, dt, trap, segsum, chunk_size,
                       rotary_dim_divisor, dtype=torch.float16,
                       return_state=False,
                       fuse_pregate_headwise_rms_norm=False,
                       outproj_norm_weight=None, outproj_norm_eps=1e-5,
                       **kwargs):
    """Dense forward; signature mirrors the CUDA operator's host entry point."""

    return _forward_common(
        q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, D, mimo_z, angles, dA_cs,
        dA_cs_rev, dt, trap, segsum, chunk_size, rotary_dim_divisor, dtype,
        return_state, fuse_pregate_headwise_rms_norm, outproj_norm_weight,
        outproj_norm_eps, cu_seqlens=None)


def mamba_mimo_forward_varlen(q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, D,
                              mimo_z, angles, dA_cs, dA_cs_rev, dt, trap, segsum,
                              chunk_size, rotary_dim_divisor,
                              dtype=torch.float16, cu_seqlens=None,
                              return_state=False,
                              fuse_pregate_headwise_rms_norm=False,
                              outproj_norm_weight=None, outproj_norm_eps=1e-5,
                              **kwargs):
    """Packed-sequence forward; ``cu_seqlens=None`` degrades to the dense path."""

    return _forward_common(
        q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, D, mimo_z, angles, dA_cs,
        dA_cs_rev, dt, trap, segsum, chunk_size, rotary_dim_divisor, dtype,
        return_state, fuse_pregate_headwise_rms_norm, outproj_norm_weight,
        outproj_norm_eps, cu_seqlens=cu_seqlens)


def _forward_common(q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, D, mimo_z,
                    angles, dA_cs, dA_cs_rev, dt, trap, segsum, chunk_size,
                    rotary_dim_divisor, dtype, return_state, fuse_rmsnorm,
                    outproj_norm_weight, outproj_norm_eps, cu_seqlens):
    N = q.shape[-1]
    s = _prepare(q, k, v, z, angles, N)
    B, S, R, H, Pd = s["B"], s["S"], s["R"], s["H"], s["P"]
    assert s["G"] == 1, "A2 port currently assumes G == 1 (no GQA grouping)"
    reduceO = mimo_o is not None
    if fuse_rmsnorm:
        # Same preconditions as the CUDA host wrapper.
        if not reduceO:
            raise ValueError(
                "fuse_pregate_headwise_rms_norm=True requires mimo_o.")
        if z is None or mimo_z is None:
            raise ValueError(
                "fuse_pregate_headwise_rms_norm=True requires z and mimo_z.")
    dev = q.device
    dt_ = dtype if isinstance(dtype, torch.dtype) else torch.float16
    C = chunk_size
    F = C * R
    s.update(C=C, F=F, dt_=dt_, reduceO=reduceO, D=D, rot=rotary_dim_divisor,
             dt=dt, trap=trap, dacs=dA_cs, dacsrev=dA_cs_rev, segsum=segsum,
             qbias=q_bias, kbias=k_bias, psi=mimo_v,
             phi=mimo_o if reduceO else None,
             zeta=mimo_z if z is not None else None, want_k=return_state,
             fuse_rms=bool(fuse_rmsnorm),
             outw=(_outproj_weight(outproj_norm_weight, H, Pd, dev)
                   if fuse_rmsnorm else None),
             eps=float(outproj_norm_eps))

    # One work item per (batch entry, sequence): dense runs one sequence per batch
    # entry, varlen packs them into the token axis of a single batch entry.
    s["dense"] = cu_seqlens is None
    if cu_seqlens is None:
        items = [(i_b, i_b * S, S, 0) for i_b in range(B)]
    else:
        assert B == 1, "varlen packing expects a single batch entry"
        cu = cu_seqlens.to(torch.int64).cpu()
        items = [(0, int(cu[i]), int(cu[i + 1]) - int(cu[i]),
                  int(cu[i]) // C + i) for i in range(len(cu) - 1)]
    ns_count = len(items)

    shape_o = (B, S, H, Pd) if reduceO else (B, S, R, H, Pd)
    o = torch.zeros(shape_o, device=dev, dtype=dt_)
    o_flat = o.reshape(B * S, H, Pd) if reduceO else o.reshape(B * S, R, H, Pd)
    final_state = torch.zeros((ns_count, H, N, Pd), device=dev,
                              dtype=torch.float32) if return_state else None
    final_k = torch.zeros((ns_count, R, H, N), device=dev,
                          dtype=dt_) if return_state else None
    s["final_k"] = final_k

    for ns, (i_b, start, seq_len, seg0) in enumerate(items):
        s["ns_index"] = ns
        for i_h in range(H):
            states = torch.zeros((N, Pd), device=dev, dtype=torch.float32)
            states = _forward_sequence(s, i_b, i_h, start, seq_len, seg0, states,
                                       o_flat)
            if return_state:
                final_state[ns, i_h] = states
    return o, final_state, final_k
