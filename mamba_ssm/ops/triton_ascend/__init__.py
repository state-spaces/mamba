"""Ascend NPU port of the Mamba-3 Triton operators (Triton-Ascend / CANN).

This package is the *destination* of the NPU migration that used to be applied
in place to ``mamba_ssm.ops.triton``.  Keeping it separate means:

* ``mamba_ssm.ops.triton`` stays byte-identical to upstream, so the CUDA path is
  always the reference implementation and never grows backend conditionals;
* everything Ascend-specific (CANN libdevice, pointer-based loads, UB tiling,
  ``coreDim`` grid-stride launches) lives here and can be reviewed as one unit.

Layout mirrors the upstream package::

    triton_ascend/
        layernorm_gated.py   # gated RMSNorm/LayerNorm, coreDim grid-stride
        layer_norm.py        # non-gated LayerNorm/RMSNorm (kept for parity)
        mamba3/
            npu_compat.py                  # backend probe + UB/config policy
            utils.py                       # tanh/sin/cos/sech2 on CANN
            angle_dt.py                    # angle cumsum forward/backward
            grouped_head_reduction.py      # per-head bias reduction
            mamba3_siso_fwd.py             # chunked forward (+ PHASE split)
            mamba3_siso_bwd.py             # dzdo / dqkv / dqktheta / ddt
            mamba3_siso_combined.py        # autograd wrapper
            mamba3_siso_step.py            # single-token decode
            mamba3_mimo_rotary_step.py     # rotary applied in the decode step

Nothing here is imported by ``mamba_ssm.ops.triton``.  The only place that
chooses between the two packages is ``mamba_ssm.ops.backend``; model code asks
that module, while Ascend-only code (``ops/ascendc``, ``ops/tilelang_ascend``,
NPU tests) imports this package directly.  The MIMO forward is the ``tilelang``
family (``ops.tilelang`` / ``ops.tilelang_ascend``), so it is not mirrored here;
the same dispatcher serves all three families.

``mamba3/mamba3_mimo_utils.py`` is deliberately absent: the MIMO segsum helpers
are shared verbatim with the CUDA side (``ops/triton/mamba3/mamba3_mimo_utils.py``)
so both backends run identical discretisation maths instead of two copies that
can drift.

The kernels keep their ``npu_compat.IS_CANN`` branches even though this package
is the Ascend side: on a build without CANN libdevice they fall back to the CUDA
formulation instead of raising, which keeps CPU-side static checks and forced
A/B runs (``MAMBA3_TRITON_BACKEND=ascend`` on a GPU box) working.
"""

__all__: list[str] = []
