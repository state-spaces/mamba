# SSD combined-operator acceptance (forward), full-batch comparison.
#
# NOTE (fixed vs the original journey script): mamba_chunk_scan_combined with
# the default return_final_states=False returns a SINGLE tensor; unpacking it
# as `out, final = ...` silently splits the batch dim. The recorded 910B
# numbers (out ~1.37e-06, final ~1.01e-06) came from that half-batch variant
# (both sides sliced identically, so verdicts were still valid). This version
# passes return_final_states=True explicitly, asserts shapes, and compares the
# full batch against the pure-torch reference.
import torch
import torch_npu
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, ssd_chunk_scan_combined_ref
from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state_ref
from mamba_ssm.ops.triton.ssd_state_passing import state_passing_ref

torch.manual_seed(0)
device = "npu"
dtype = torch.float32
b, s, h, p, n = 2, 64, 2, 32, 16

x = torch.randn(b, s, h, p, device=device, dtype=dtype) * 0.5
dt = torch.rand(b, s, h, device=device, dtype=dtype) * 0.1 + 0.01
A = -torch.rand(h, device=device, dtype=dtype) - 0.5
B = torch.randn(b, s, 1, n, device=device, dtype=dtype) * 0.5
C = torch.randn(b, s, 1, n, device=device, dtype=dtype) * 0.5
D = torch.randn(h, device=device, dtype=dtype)
dt_bias = torch.rand(h, device=device, dtype=dtype) * 0.1 - 0.05

out, final = mamba_chunk_scan_combined(x, dt, A, B, C, 16, D=D, dt_bias=dt_bias, dt_softplus=True,
                                       return_final_states=True)
assert tuple(out.shape) == (b, s, h, p), f"out shape {tuple(out.shape)}"
assert tuple(final.shape) == (b, h, p, n), f"final shape {tuple(final.shape)}"

ref = ssd_chunk_scan_combined_ref(x, dt, A, B, C, 16, D=D, dt_bias=dt_bias, dt_softplus=True)
assert tuple(ref.shape) == (b, s, h, p), f"ref shape {tuple(ref.shape)}"

# Reference final states, mirroring ssd_chunk_scan_combined_ref internals:
# dt -> (bias, softplus) -> dA cumsum per chunk; chunk_state -> state passing,
# take final_states from the last chunk.
dt_t = rearrange(dt, "b (c z) h -> b h c z", c=(s + 15) // 16).float()
dt_t = torch.nn.functional.softplus(dt_t + rearrange(dt_bias, "h -> h 1 1"))
dA_cumsum = torch.cumsum(dt_t * rearrange(A, "h -> h 1 1"), dim=-1)
chunk_states = chunk_state_ref(B, x, dt_t, dA_cumsum)  # (b, c, h, p, n)
_, ref_final = state_passing_ref(rearrange(chunk_states, "b c h p n -> b c h (p n)"),
                                 dA_cumsum[:, :, :, -1])
ref_final = rearrange(ref_final, "b h (p n) -> b h p n", n=n)

torch.npu.synchronize()
diff = (out - ref).abs().max().item()
fdiff = (final - ref_final).abs().max().item()
print("out shape:", tuple(out.shape), "ref shape:", tuple(ref.shape))
print("max abs diff out:", diff)
print("max abs diff final states:", fdiff)
rel = diff / ref.abs().max().item()
rel_f = fdiff / ref_final.abs().max().item()
print("rel diff:", rel, "/", rel_f)
print("PASS" if (rel < 1e-3 and rel_f < 1e-3) else "MISMATCH")
