import itertools
from math import sqrt

import pandas
import torch
from tqdm import tqdm
import triton

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
except:
    pass




def benchmark_mamba(batch, head, length, dim_head, d_state, selective_scan_cuda, *args):
   from einops import rearrange, repeat

   d_model = dim_head * head
   expand = 2
   d_inner = d_model * expand
   device = "cuda"

   # S4D real initialization
   A = repeat(
       torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
       "n -> d n",
       d=d_inner,
   ).contiguous()
   A_log = torch.log(A)  # Keep A_log in fp32

   x = torch.rand(
       (batch, d_inner, length), device=device, dtype=torch.bfloat16
   ).requires_grad_(True)
   z = torch.rand(
       (batch, d_inner, length), device=device, dtype=torch.bfloat16
   ).requires_grad_(True)
   delta = torch.rand(
       (batch, d_inner, length), device=device, dtype=torch.bfloat16
   ).requires_grad_(True)
   delta_bias = torch.randn(d_inner).to("cuda").requires_grad_(True)
   A = -torch.exp(A_log.float())  # (d_inner, d_state)
   B = (
       torch.randn(batch, 1, d_state, length)
       .to("cuda")
       .to(torch.bfloat16)
       .requires_grad_(True)
   )
   C = (
       torch.randn(batch, 1, d_state, length)
       .to("cuda")
       .to(torch.bfloat16)
       .requires_grad_(True)
   )
   D = torch.ones(d_inner, device=device)  # Keep in fp32
   delta_softplus = True

   ms = triton.testing.do_bench(
       lambda: selective_scan_cuda.fwd(
           x, delta, A, B, C, D, z, delta_bias, delta_softplus, *args
       ),
       warmup=100,
   )
   return ms


def get_inputs(B, H, L, E=64, ret_padding_mask=False, dtype=torch.float32):
    q = torch.rand((B, H, L, E), device="cuda", dtype=dtype)
    k = torch.rand((B, H, L, E), device="cuda", dtype=dtype)
    v = torch.rand((B, H, L, E), device="cuda", dtype=dtype)

    input_lengths = torch.randint(1, L, (B,), device=q.device).long()
    input_lengths[-1] = L
    padding_mask = torch.zeros((B, L), dtype=q.dtype, device=q.device)
    padding_mask[
        (
            torch.arange(padding_mask.shape[0], device=padding_mask.device),
            input_lengths - 1,
        )
    ] = 1
    padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
    if not ret_padding_mask:
        padding_mask = None
    return (q, k, v), padding_mask


def flash_attn_forward(queries, keys, values, padding_mask=None):
    qkv = torch.stack([queries, keys, values], dim=2)
    qkv = qkv.permute(0, 3, 2, 1, 4)
    B, T, _, H, D = qkv.shape
    scale = 1.0 / sqrt(D)

    if padding_mask is not None:
        # unpad_input expectes True to correspond to valid indices and False to invalid
        qkv, indices, cu_q_lens, max_s = unpad_input(qkv, ~padding_mask)
        packed_res = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_q_lens,
            max_s,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            alibi_slopes=None,
            deterministic=False,
        )
        res = pad_input(packed_res, indices, B, T)
        res = res.transpose(1, 2)
    else:
        res = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            alibi_slopes=None,
            deterministic=False,
        )
        res = res.transpose(1, 2)  # B x T x H x D -> B x H x T x D
    return res

    
def benchmark_flash(q, k, v, padding_mask):
    dim_E = q.shape[-1]
    H = q.shape[1]
    E = dim_E * H
    ms = triton.testing.do_bench(
        lambda: flash_attn_forward(q, k, v, padding_mask=padding_mask), warmup=100
    )
    return ms


def build_selective_scan_fn(selective_scan_cuda: object = None, mode="mamba_ssm", tag=None):
    MODE = mode

    class SelectiveScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, nrows=1, backnrows=-1):
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if z is not None and z.stride(-1) != 1:
                z = z.contiguous()
            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True
            if D is not None and (D.dtype != torch.float):
                ctx._d_dtype = D.dtype
                D = D.float()
            if delta_bias is not None and (delta_bias.dtype != torch.float):
                ctx._delta_bias_dtype = delta_bias.dtype
                delta_bias = delta_bias.float()

            assert u.shape[1] % (B.shape[1] * nrows) == 0 
            assert nrows in [1, 2, 3, 4] # 8+ is too slow to compile

            if backnrows > 0:
                assert u.shape[1] % (B.shape[1] * backnrows) == 0 
                assert backnrows in [1, 2, 3, 4] # 8+ is too slow to compile
            else:
                backnrows = nrows
            ctx.backnrows = backnrows
            
            if MODE in ["mamba_ssm"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

            elif MODE in ["sscore"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            elif MODE in ["sstest"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, nrows)
            else:
                raise NotImplementedError

            ctx.delta_softplus = delta_softplus
            ctx.has_z = z is not None

            last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
            if not ctx.has_z:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
                return out if not return_last_state else (out, last_state)
            else:
                ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
                if MODE in ["mamba_ssm", "sstest"]:
                    out_z = rest[0]
                    return out_z if not return_last_state else (out_z, last_state)
                elif MODE in ["sscore"]:
                    return out if not return_last_state else (out, last_state)

        @staticmethod
        def backward(ctx, dout, *args):
            if not ctx.has_z:
                u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
                z = None
                out = None
            else:
                u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
            # backward of selective_scan_cuda with the backward of chunk).
            # Here we just pass in None and dz will be allocated in the C++ code.
            if MODE in ["mamba_ssm"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
                    False # option to recompute out_z, not used here
                )
            elif MODE in ["sstest"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
                    False, ctx.backnrows  # option to recompute out_z, not used here
                )
            elif MODE in ["sscore"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.backnrows
                )
            else:
                raise NotImplementedError
            
            dz = rest[0] if ctx.has_z else None
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            
            _dD = None
            if D is not None:
                if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                    _dD = dD.to(ctx._d_dtype)
                else:
                    _dD = dD

            _ddelta_bias = None
            if delta_bias is not None:
                if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                    _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
                else:
                    _ddelta_bias = ddelta_bias

            return (du, ddelta, dA, dB, dC,
                        dD if D is not None else None,
                        dz,
                        ddelta_bias if delta_bias is not None else None,
                        None, None, None, None)

    def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, nrows=1, backnrows=-1):
        """if return_last_state is True, returns (out, last_state)
        last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
        not considered in the backward pass.
        """
        return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, nrows, backnrows)

    selective_scan_fn.__repr__ = lambda *_ :f"selective_scan_fn | {mode} | {tag}"
    # print(repr(selective_scan_fn), "==", selective_scan_fn.__repr__())

    return selective_scan_fn


def benchmark_mamba_fwdbwd(batch, head, length, dim_head, d_state, selective_scan_fn, *args):
   from einops import rearrange, repeat

   d_model = dim_head * head
   expand = 2
   d_inner = d_model * expand
   device = "cuda"

   # S4D real initialization
   A = repeat(
       torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
       "n -> d n",
       d=d_inner,
   ).contiguous()
   A_log = torch.log(A)  # Keep A_log in fp32

   x = torch.rand(
       (batch, d_inner, length), device=device, dtype=torch.bfloat16
   ).requires_grad_(True)
   z = torch.rand(
       (batch, d_inner, length), device=device, dtype=torch.bfloat16
   ).requires_grad_(True)
   delta = torch.rand(
       (batch, d_inner, length), device=device, dtype=torch.bfloat16
   ).requires_grad_(True)
   delta_bias = torch.randn(d_inner).to("cuda").requires_grad_(True)
   A = -torch.exp(A_log.float())  # (d_inner, d_state)
   B = (
       torch.randn(batch, 1, d_state, length)
       .to("cuda")
       .to(torch.bfloat16)
       .requires_grad_(True)
   )
   C = (
       torch.randn(batch, 1, d_state, length)
       .to("cuda")
       .to(torch.bfloat16)
       .requires_grad_(True)
   )
   D = torch.ones(d_inner, device=device)  # Keep in fp32
   delta_softplus = True

   ms = triton.testing.do_bench(
       lambda: selective_scan_fn(
           x, delta, A, B, C, D, z, delta_bias, delta_softplus, False, *args
       )[0].sum().backward(),
       warmup=100,
   )
   return ms


def test_bench(with_flash=False, with_mamba_fwd=False, with_mamba_fwdbwd=False):
    batch_sizes = [16]
    heads = [12, 16, 32]
    time_steps = [50, 100, 1000, 1600, 3200, 6400]
    get_padding_masks = [True, False]
    # d_states = [2, 4, 8, 16]
    d_states = [2, 16] # to save space, otherwise, too many columns would display
    dtypes = [torch.bfloat16]
    E = 64
    fwdnrows = [0, 1, 2, 4] # 64 // 3 != 0
    bwdnrows = [0, 1, 2, 4] # 64 // 3 != 0

    results = []

    if with_flash:
        for B, H, L, pm, dtype in tqdm(
            itertools.product(batch_sizes, heads, time_steps, get_padding_masks, dtypes)
        ):
            (q, k, v), padding_mask = get_inputs(
                B, H, L, E=64, ret_padding_mask=pm, dtype=dtype
            )
            ms = benchmark_flash(q, k, v, padding_mask)
            results.append(
                {
                    "name": "flash",
                    "batch_size": B,
                    "nheads": H,
                    "seq_len": L,
                    "dim": H * E,
                    "padding": pm,
                    "dtype": dtype,
                    "ms": ms,
                }
            )

    if with_mamba_fwd:
        for B, H, L, pm, d_state, dtype, fwdnrow in tqdm(
            itertools.product(
                batch_sizes, heads, time_steps, get_padding_masks, d_states, dtypes, fwdnrows
            )
        ):
            (q, k, v), padding_mask = get_inputs(
                B, H, L, E=64, ret_padding_mask=pm, dtype=dtype
            )

            if fwdnrow == 0:
                import selective_scan_cuda
                ms = benchmark_mamba(B, H, L, E, d_state, selective_scan_cuda)
            else:
                import selective_scan_cuda_test
                ms = benchmark_mamba(B, H, L, E, d_state, selective_scan_cuda_test, fwdnrow)
            results.append(
                {
                    "name": f"mamba-{d_state}-{fwdnrow}",
                    "batch_size": B,
                    "nheads": H,
                    "seq_len": L,
                    "dim": H * E,
                    "padding": pm,
                    "dtype": dtype,
                    "ms": ms,
                }
            )

    if with_mamba_fwdbwd:
        for B, H, L, pm, d_state, dtype, fwdnrow, bwdnrow in tqdm(
            itertools.product(
                batch_sizes, heads, time_steps, get_padding_masks, d_states, dtypes, fwdnrows, bwdnrows
            )
        ):
            (q, k, v), padding_mask = get_inputs(
                B, H, L, E=64, ret_padding_mask=pm, dtype=dtype
            )

            if fwdnrow == 0:
                if bwdnrow == 0:
                    import selective_scan_cuda
                    ms = benchmark_mamba_fwdbwd(B, H, L, E, d_state, build_selective_scan_fn(selective_scan_cuda, mode="mamba_ssm"))
                else:
                    continue
            else:
                import selective_scan_cuda_test
                ms = benchmark_mamba_fwdbwd(B, H, L, E, d_state, build_selective_scan_fn(selective_scan_cuda_test, mode="sstest"), fwdnrow)
            results.append(
                {
                    "name": f"mamba-{d_state}-{fwdnrow}-{bwdnrow}",
                    "batch_size": B,
                    "nheads": H,
                    "seq_len": L,
                    "dim": H * E,
                    "padding": pm,
                    "dtype": dtype,
                    "ms": ms,
                }
            )

    df = pandas.DataFrame(results)
    piv = df.pivot(
        columns="name",
        values="ms",
        index=["dtype", "padding", "batch_size", "nheads", "seq_len"],
    )
    pandas.set_option('display.width', 1000)
    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None)
    print(piv.sort_index().round(3))


if __name__ == "__main__":
    test_bench(with_mamba_fwd=True)
    test_bench(with_mamba_fwdbwd=True)


"""
Thank you very much, @apoorv2904. 
You are right, and I nearly failed to reproduce the results I have observed before.
These days, I kept working on it. (the environment I use is 4090 24G, with py310+cu121+torch2.2)
1. I added `nrow` feature in backward to better compare with different `nrow` settings.
2. I compared my code (`selective_scan_test` here, or `selective_scan_core` in VMamba) with `mamba_ssm` rather than `selective_scan_ref`, and keeps no difference (tested all pass with [test file](https://github.com/MzeroMiko/mamba/blob/main/kernel/test_selective_scan_new2old.py)).
3. I realised that the [issue]`https://github.com/alxndrTL/mamba.py/issues/8` proves nothing here, since raising `d_state` only inference the flops in SSM (nearly equals selective scan) while raising `d_model` or `seqlen` inferences the whole mamba model. As SSM is fast compared to `the whole model + data loading`, the speed difference is small and hard to observe (which is one possibility to that issue).
4. I used my newly written [`simple benchmark`](https://github.com/MzeroMiko/mamba/blob/main/kernel/test_selective_scan_speed.py), and found the results are consistent with yours. It seems that raissing nrows would only make the code slower, until I finally realised that ***the test which shows raise the nrow will raise the speed, was done in 7x7 images, which means seqlen is 49! extremely small!***. Then I add `seqlen=64` in testing, and found in some `fwdnrow+bwdnrow` patterns, the speed is fast, see [log](https://github.com/MzeroMiko/mamba/blob/main/kernel/test_selective_scan_speed.log) for details. Though I still do not know how bwd codes inferences the fwd procedure.
5. I modified your [`benchmark`](https://github.com/MzeroMiko/mamba/blob/main/kernel/test_selective_scan_benchmark.py), and the results are consistent with `test_selective_scan_speed`, see [log](https://github.com/MzeroMiko/mamba/blob/main/kernel/test_selective_scan_benchmark.log) for details.
To conclude, with short `seqlen`, bigger `nrow` may leads to faster speed, but the reason remains unknown.
"""