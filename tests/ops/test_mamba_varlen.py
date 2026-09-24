import random
import pytest
import torch

from torch import nn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


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


class NLayerMambaModel(nn.Module):
    def __init__(self, layer_num, hidden_dim, device):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=hidden_dim, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=2,    # Block expansion factor
                    layer_idx=layer_idx,
                ).to(device) for layer_idx in range(layer_num)
            ]
        )
        
    def forward(self, x, cu_seqlens=None, seq_idx=None, position_ids=None):
        residual = x
        for layer in self.layers:
            x = layer(x, cu_seqlens, seq_idx=seq_idx, position_ids=position_ids)
        return x + residual


'''
Generate random cu_seqlens for testing
'''
def generate_random_cu_seqlens(seq_len, batch_size=None):
    if batch_size is None:
        batch_size = random.randint(1, seq_len)
    if batch_size > 1:
        ret = sorted(random.sample(range(1, seq_len), batch_size - 1))
    else:
        ret = []
    cu_seqlens = [0] + ret + [seq_len]
    assert batch_size == len(cu_seqlens) - 1
    return cu_seqlens


@pytest.mark.parametrize('itype', [torch.float32])
@pytest.mark.parametrize('layer_num', [1, 2, 4, 8])
@pytest.mark.parametrize("hidden_dim", [2048])
@pytest.mark.parametrize('seq_len', [1024, 2048, 4096, 8192])
def test_mamba_varlen(itype, layer_num, hidden_dim, seq_len):
    device='cuda'
    if itype == torch.float32:
        rtol, atol = (6e-4, 2e-3)
    elif itype == torch.bfloat16:
        rtol, atol = (3e-2, 5e-2)
    else:
        rtol, atol = (3e-3, 5e-3)

    # Generate random cu_seqlens for testing
    cu_seqlens = generate_random_cu_seqlens(seq_len)
    cu_seqlens = torch.tensor(cu_seqlens, device=device)
    print(f'Generate random cu_seqlens = {cu_seqlens.tolist()}')
    
    # Generate packed_hidden_states with random values for testing
    # packed_hidden_states (packed_batch_size=1) should be forwarded with cu_seqlens
    hidden_states_list = [torch.randn(l, hidden_dim, device=device) for l in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()]
    packed_hidden_states = torch.cat(hidden_states_list, dim=0).unsqueeze(0)
    # hidden_states should be forwarded without cu_seqlens
    hidden_states = unpack(packed_hidden_states, cu_seqlens)

    # Check: sum of seq_len of item in hidden_states_list should be equal to seq_len of packed_hidden_states
    assert sum([hs.shape[0] for hs in hidden_states_list]) == packed_hidden_states.shape[1]
    # Check: max of seq_len of item in hidden_states_list should be equal to seq_len of hidden_states
    assert max([hs.shape[0] for hs in hidden_states_list]) == hidden_states.shape[1]

    # creat one simple mamba block
    mamba_ref = NLayerMambaModel(layer_num, hidden_dim, device)
    mamba = NLayerMambaModel(layer_num, hidden_dim, device)
    mamba.load_state_dict(mamba_ref.state_dict())
    print(f"show reference model for testing: {mamba_ref}", flush=True)

    # reference output for forwardding hidden_states
    out_ref_original = mamba_ref(hidden_states)
    out_ref = pack(out_ref_original, cu_seqlens).unsqueeze(0)
    
    # In production, cu_seqlens/seq_idx/position_ids should be prepared in the dataloader
    seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32, device=cu_seqlens.device) 
            for i, s in enumerate(cu_seqlens[1:]-cu_seqlens[:-1])], dim=0).unsqueeze(0)
    position_ids = (torch.arange((cu_seqlens[1:] - cu_seqlens[:-1]).sum(), device=cu_seqlens.device) 
                    - torch.repeat_interleave(cu_seqlens[:-1], (cu_seqlens[1:] - cu_seqlens[:-1]))).to(torch.int32).unsqueeze(0)
    # output for forwardding packed_hidden_states
    out = mamba(packed_hidden_states, cu_seqlens=cu_seqlens, seq_idx=seq_idx, position_ids=position_ids)

    # Testing the max/mean diff
    print(f"max diff for output in varlen_mamba fwd pass: {(out - out_ref).abs().max().item()}", flush=True)
    print(f"mean diff for output in varlen_mamba fwd pass: {(out - out_ref).abs().mean().item()}", flush=True)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    # Generate random loss for backward testing
    loss_fn = nn.CrossEntropyLoss()
    g = torch.randn_like(out)
    g_ref = unpack(g, cu_seqlens)
    loss = loss_fn(out, g)
    loss_ref = loss_fn(out_ref_original, g_ref)
    loss.backward()
    loss_ref.backward()

    # Check weight grad
    all_grads_match = True
    for (name_ref, param_ref), (name_packed, param_packed) in zip(
        mamba_ref.named_parameters(), mamba.named_parameters()
    ):
        grad_match = torch.allclose(param_ref.grad, param_packed.grad, rtol=rtol, atol=atol)
        if not grad_match:
            print(f"Gradient mismatch in {name_ref} and {name_packed}! Max diff: {(param_ref.grad - param_packed.grad).abs().max().item()}", flush=True)
            all_grads_match = False
    print(f"All gradients match: {all_grads_match}", flush=True)


@pytest.mark.parametrize('wtype', [torch.float32])
@pytest.mark.parametrize('itype', [torch.float32])
@pytest.mark.parametrize('seq_len', [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("return_last_state", [True])
@pytest.mark.parametrize('has_delta_bias', [True])
@pytest.mark.parametrize('delta_softplus', [True])
@pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize('has_D', [True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("is_variable_C", [True])
@pytest.mark.parametrize("is_variable_B", [True])
def test_selective_scan_varlen(is_variable_B, is_variable_C, varBC_groups, has_D, has_z, has_delta_bias,
                        delta_softplus, return_last_state, seq_len, itype, wtype):
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = 'cuda'
    if itype == torch.float32:
        rtol, atol = (6e-4, 2e-3)
    elif itype == torch.bfloat16:
        rtol, atol = (3e-2, 5e-2)
    else:
        rtol, atol = (3e-3, 5e-3)
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    torch.random.manual_seed(0)
    packed_batch_size = 1
    dim = 768
    dstate = 8
    is_complex = wtype == torch.complex64
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
    if not is_variable_B:
        B_shape = (dim, dstate)
    elif varBC_groups == 1:
        B_shape = (packed_batch_size, dstate, seq_len if not is_complex else seq_len * 2)
    else:
        B_shape = (packed_batch_size, varBC_groups, dstate, seq_len if not is_complex else seq_len * 2)
    B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype,
                    requires_grad=True)
    if not is_variable_C:
        C_shape = (dim, dstate)
    elif varBC_groups == 1:
        C_shape = (packed_batch_size, dstate, seq_len if not is_complex else seq_len * 2)
    else:
        C_shape = (packed_batch_size, varBC_groups, dstate, seq_len if not is_complex else seq_len * 2)
    C = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype,
                    requires_grad=True)
    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None
    if has_z:
        z = torch.randn(packed_batch_size, dim, seq_len, device=device, dtype=itype, requires_grad=True)
    else:
        z = None
    if has_delta_bias:
        delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_()
    else:
        delta_bias = None
    u = torch.randn(packed_batch_size, dim, seq_len, device=device, dtype=itype, requires_grad=True)
    delta = (0.5 * torch.rand(packed_batch_size, dim, seq_len, device=device, dtype=itype)).requires_grad_()
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    u_ref = u.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    
    # In production, cu_seqlens/seq_idx/position_ids should be prepared in the dataloader
    cu_seqlens = torch.tensor(generate_random_cu_seqlens(seq_len), device=device)
    # seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32, device=cu_seqlens.device) 
    #         for i, s in enumerate(cu_seqlens[1:]-cu_seqlens[:-1])], dim=0).unsqueeze(0)
    position_ids = (torch.arange((cu_seqlens[1:] - cu_seqlens[:-1]).sum(), device=cu_seqlens.device) 
                    - torch.repeat_interleave(cu_seqlens[:-1], (cu_seqlens[1:] - cu_seqlens[:-1]))).to(torch.int32).unsqueeze(0)
    
    out, *rest = selective_scan_fn(
        u, delta, A, B, C, D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
        return_last_state=return_last_state, cu_seqlens=cu_seqlens, position_ids=position_ids
    )
    if return_last_state:
        state = rest[0]
    out_ref, *rest = selective_scan_ref(
        u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z=z_ref,
        delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
        return_last_state=return_last_state, cu_seqlens=cu_seqlens, position_ids=position_ids
    )
    if return_last_state:
        state_ref = rest[0]
    # dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    if return_last_state:
        print(f'State max diff: {(state - state_ref).abs().max().item()}')
        assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f'du max diff: {(u.grad - u_ref.grad).abs().max().item()}')
    print(f'ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}')
    print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
    print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
    print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
    if has_D:
        print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
    if has_z:
        print(f'dz max diff: {(z.grad - z_ref.grad).abs().max().item()}')
    if has_delta_bias:
        print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')

    assert torch.allclose(u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    assert torch.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
                          atol=atolw if not is_variable_B else atol)
    assert torch.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
                          atol=atolw if not is_variable_C else atol)
    if has_D:
        assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    if has_z:
        assert torch.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
    if has_delta_bias:
        assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)
    