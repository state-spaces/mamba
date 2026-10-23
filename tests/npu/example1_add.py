import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
x = torch.rand(98432, device="npu")
y = torch.rand(98432, device="npu")
print("versions:", torch.__version__, torch_npu.__version__, triton.__version__)
d = (add(x, y) - (x + y)).abs().max().item()
print("max diff:", d)
