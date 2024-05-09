import torch
import triton
import triton.language as tl

assert triton.__version__ != '2.1.0', 'Triton 2.1.0 is missing enable_fp_fusion. Triton 2.2.0 is required for numerical stability of this implementation.'


# manual tuple packing by @jackd from https://github.com/openai/triton/issues/2359
@triton.jit
def unpack64(merged):
    tl.static_assert(merged.dtype == tl.uint64)
    b = (merged & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    a = (merged >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    return a, b


@triton.jit
def pack64(a, b):
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a = a.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    a = a << 32
    b = b.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    return a | b


@triton.jit()
def first_order_op(l, r):
    """
    See https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf Section 1.4.1
    """
    xl, fl = unpack64(l)
    xr, fr = unpack64(r)
    x = xl * fr + xr
    f = fl * fr
    return pack64(x, f)


@triton.jit
def forward_scan(
    gates,
    tokens,
    outputs,
    SEQUENCE_LENGTH: tl.constexpr,
):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    strides = tl.arange(0, SEQUENCE_LENGTH) + sequence_id * SEQUENCE_LENGTH

    tokens_ = tl.load(tokens + strides)
    gates_ = tl.load(gates + strides)

    tuples = pack64(tokens_, gates_)
    output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=first_order_op)
    output_tokens_, output_gates_ = unpack64(output_tuples_)
    tl.store(outputs + strides, output_tokens_)


@triton.jit
def backward_scan(
    gates,
    tokens,
    outputs,
    SEQUENCE_LENGTH: tl.constexpr,
):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    forward_strides = tl.arange(0, SEQUENCE_LENGTH) + sequence_id * SEQUENCE_LENGTH
    reverse_strides = (tl.num_programs(axis=0) * tl.num_programs(axis=1) * SEQUENCE_LENGTH - 1) - forward_strides

    tokens_ = tl.load(tokens + reverse_strides)
    gates_ = tl.load(gates + reverse_strides)

    tuples = pack64(tokens_, gates_)
    output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=first_order_op)
    output_tokens_, output_gates_ = unpack64(output_tuples_)
    tl.store(outputs + reverse_strides, output_tokens_)


class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, tokens):
        B, C, T = gates.shape
        assert tokens.shape == (B, C, T)
        assert gates.is_contiguous()
        assert tokens.is_contiguous()

        states = torch.zeros_like(tokens)
        forward_scan[(B,C)](gates, tokens, states, SEQUENCE_LENGTH=T, enable_fp_fusion=False)

        ctx.save_for_backward(states, gates)
        return states

    # backward scan is a padded reverse scan
    # See https://arxiv.org/abs/1709.04057 Section 2.2
    @staticmethod
    def backward(ctx, grad_output):
        states, gates = ctx.saved_tensors
        B, C, T = gates.shape

        grad_output = grad_output.contiguous()
        assert states.is_contiguous()
        assert gates.is_contiguous()

        d_states = torch.empty_like(states)
        padded_shifted_gates = torch.cat([gates, torch.ones_like(gates[:, :, :1])], dim=-1)[:, :, 1:].contiguous()
        backward_scan[(B,C)](padded_shifted_gates, grad_output, d_states, SEQUENCE_LENGTH=T, enable_fp_fusion=False)

        padded_outputs = torch.cat([torch.zeros_like(states[:, :, :1]), states], dim=-1)[:, :, :-1]
        d_gates = padded_outputs * d_states

        d_tokens = d_states
        return d_gates, d_tokens


def scan(gates, tokens):
    """Solve a first-order recurrence relation:

    .. math::
        x_t = a_t x_{t-1} + b_t

    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    Arguments:
        gates (torch.Tensor): shape (B, C, T), must be contiguous. T must be a power of 2.
        tokens (torch.Tensor): shape (B, C, T), must be contiguous. T must be a power of 2.

    Returns:
        (torch.Tensor): shape (B, C, T)
    """
    return Scan.apply(gates, tokens)
