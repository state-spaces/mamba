import sys
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def zeros_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)

mode = sys.argv[1] if len(sys.argv) > 1 else "ub"
if mode == "ub":
    N, BLOCK_SIZE = 65536, 65536      # 单核 2097152 bits 溢出（910B UB=1572864 bits 时应报错）
else:
    N, BLOCK_SIZE = 1073741824, 1024  # 文档 coreDim 案例
out = torch.empty(N, device="npu", dtype=torch.float32)
grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
zeros_kernel[grid](out, N, BLOCK_SIZE=BLOCK_SIZE)
torch.npu.synchronize()
print("kernel done, sum =", out.sum().item())
