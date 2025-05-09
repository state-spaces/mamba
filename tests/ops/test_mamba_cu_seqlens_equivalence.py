import copy
import random
import torch

from mamba_ssm.modules.mamba_simple import Mamba


'''
unpack function: convert packed_hidden_states (batch_size=1) to hidden_states
'''
def unpack(packed_hidden_states, cu_seqlens):
    batch_size = cu_seqlens.shape[0] - 1
    seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    hidden_dim = packed_hidden_states.shape[2]
    hidden_states = torch.zeros(batch_size, seq_len, hidden_dim, dtype=packed_hidden_states.dtype, device=packed_hidden_states.device)
    for i in range(batch_size):
        hidden_states[i, : cu_seqlens[i + 1] - cu_seqlens[i], :] = packed_hidden_states[:, cu_seqlens[i] : cu_seqlens[i + 1], :]
    return hidden_states


'''
pack function: convert hidden_states to packed_hidden_states (batch_size=1)
'''
def pack(hidden_states, cu_seqlens):
    batch_size, seq_len, hidden_dim = hidden_states.shape
    seq_len_list = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_len_list_3d = seq_len_list.unsqueeze(1).unsqueeze(2)
    indices_3d = (
        torch.arange(seq_len, device=hidden_states.device)
        .unsqueeze(0)
        .unsqueeze(2)
        .repeat(batch_size, 1, hidden_dim)
    )
    mask_3d = indices_3d < seq_len_list_3d
    packed_hidden_states = hidden_states[mask_3d].view(-1, hidden_dim)
    return packed_hidden_states


'''
Generate random cu_seqlens for testing
'''
def generate_random_cu_seqlens(seq_len, batch_size):
    if batch_size > 1:
        ret = sorted(random.sample(range(1, seq_len), batch_size - 1))
    else:
        ret = []
    cu_seqlens = [0] + ret + [seq_len]
    assert batch_size == len(cu_seqlens) - 1
    return cu_seqlens


def main():
    # config tested with A100
    hidden_dim = 2048
    seq_len = 1024
    batch_size = 8
    device='cuda'
    
    itype = torch.float32
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # If we have z, the errors on the weights seem higher
    rtolw = max(rtolw, rtol)
    atolw = max(atolw, atol)

    # Generate random cu_seqlens for testing
    cu_seqlens = generate_random_cu_seqlens(seq_len, batch_size)
    cu_seqlens = torch.tensor(cu_seqlens, device=device)
    print(f'Generate random cu_seqlens = {cu_seqlens.tolist()}')
    
    # Generate packed_hidden_states with random values for testing
    # packed_hidden_states (batch_size=1) should be forwarded with cu_seqlens
    hidden_states_list = [torch.randn(l, hidden_dim, device=device) for l in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()]
    packed_hidden_states = torch.cat(hidden_states_list, dim=0).unsqueeze(0)
    # hidden_states should be forwarded without cu_seqlens
    hidden_states = unpack(packed_hidden_states, cu_seqlens)

    # Check: sum of seq_len of item in hidden_states_list should be equal to seq_len of packed_hidden_states
    assert sum([hs.shape[0] for hs in hidden_states_list]) == packed_hidden_states.shape[1]
    # Check: max of seq_len of item in hidden_states_list should be equal to seq_len of hidden_states
    assert max([hs.shape[0] for hs in hidden_states_list]) == hidden_states.shape[1]

    # creat one simple mamba block
    mamba_ref = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=hidden_dim, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to(device)

    # reference output for forwardding hidden_states
    out_ref_original = mamba_ref(hidden_states)
    out_ref = pack(out_ref_original, cu_seqlens).unsqueeze(0)

    # output for forwardding packed_hidden_states with cu_seqlens
    mamba = copy.deepcopy(mamba_ref)
    out = mamba(packed_hidden_states, cu_seqlens)

    # Testing the max/mean diff
    print(f'max diff for output in varlen_mamba fwd pass: {(out - out_ref).abs().max().item()}')
    print(f'mean diff for output in varlen_mamba fwd pass: {(out - out_ref).abs().mean().item()}')
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    # bwd for mamba w/ cu_seqlens
    g = torch.randn_like(out)
    out.backward(g)
    mamba_grad = {name: param.grad.clone() for name, param in mamba.named_parameters()}

    # bwd for mamba wo/ cu_seqlens
    g_ref = unpack(g, cu_seqlens)
    out_ref_original.backward(g_ref)
    mamba_ref_grad = {name: param.grad.clone() for name, param in mamba_ref.named_parameters()}

    # check bwd pass
    assert set(mamba_grad.keys()) == set(mamba_ref_grad.keys())
    for name in mamba_ref_grad:
        print(f'max diff for {name} in varlen_mamba bwd pass: {(mamba_grad[name] - mamba_ref_grad[name]).abs().max().item()}')
        print(f'mean diff for {name} in varlen_mamba bwd pass: {(mamba_grad[name] - mamba_ref_grad[name]).abs().mean().item()}')
        assert torch.allclose(mamba_grad[name], mamba_ref_grad[name], rtol=rtol, atol=atol)

if __name__ == "__main__":
    main()
