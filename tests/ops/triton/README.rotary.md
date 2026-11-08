# SISO rotary stability regression

This backport starts at the clean training dependency
`49bba41eba081cf4a682972e737547a0ddc702b4`. It incorporates the kernel changes
from state-spaces/mamba PRs [1059](https://github.com/state-spaces/mamba/pull/1059),
[1070](https://github.com/state-spaces/mamba/pull/1070), and
[1071](https://github.com/state-spaces/mamba/pull/1071). Upstream authors retain
credit. Patch context was adapted to the older pin; no unrelated optimization
or model default is changed.

Phases, final phase state and raw rates remain FP32. Fixed-length backward
uses the state-derived chunk carry plus the final-chunk anchor instead of
accumulating noisy phase gradients over all later chunks. Variable-length
execution retains the ordinary reverse sum; the phase precision fixes still
apply. MIMO is outside this change.

Run in the approved CUDA container:

```bash
python -m pytest -q -s tests/ops/triton/test_mamba3_rotary_stability.py
```

The FP64 chunk reference is checked against a separate token recurrence on
CPU, including all nine input gradients. GPU cases check length 73, 4096 and
8192, true FP32 rates, partial rotary, output/all gradients, expanded strides,
and FP32 phase/final-state storage with both BF16 and FP32 rate inputs.
Five tests passed on H100 with Torch 2.8.0+cu128 / Triton 3.7.1 on November 7,
2026. Angle-gradient relative L2 errors were 0.472%, 0.737%, and 0.659% for
those three lengths. The longest test fails the old reverse-sum formulation.

The exact same three-file backport was previously profiled on the CB4
softplus-dt MVA geometry (batch 1, length 4096, model width 2048, 64 heads,
state/value widths 64). Full mixer forward/backward median increased from
6.960 to 7.067 ms; angle-gradient error on fresh block inputs dropped from
3.727% to 0.953% versus FP64. This is a numerical diagnostic, not proof of
improved optimizer trajectories. The source and logs are recorded by Zoo's
`20261107-mamba3-rotary-numerics` experiment. Existing frozen training pins
are not silently replaced by this branch.
