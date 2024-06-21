import random
import torch

from mamba_ssm.modules.mamba_simple import Mamba

class AlignTimer:
    def __init__(self, message='kernel_no_name'):
        self.message = message

    def __enter__(self):
        torch.cuda.synchronize()  
        self.starter = torch.cuda.Event(enable_timing=True)
        self.starter.record()
        return self

    def __exit__(self, type, value, traceback):
        self.ender = torch.cuda.Event(enable_timing=True)
        self.ender.record()
        torch.cuda.synchronize()  
        self.time = self.starter.elapsed_time(self.ender)
        print('{} uses time {:.4f} ms'.format(self.message, self.time))
'''
unpack function: convert packed_hidden_states (batch_size=1) to hidden_states
'''
def unpack(packed_hidden_states, cu_seqlens):
    batch_size = packed_hidden_states.shape[0]
    package_num = cu_seqlens.shape[0] - 1
    seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    hidden_dim = packed_hidden_states.shape[2]
    hidden_states = torch.zeros(package_num * batch_size, seq_len, hidden_dim, dtype=packed_hidden_states.dtype, device=packed_hidden_states.device)
    for j in range(batch_size):
        for i in range(package_num):
            line = j * package_num + i
            hidden_states[line, : cu_seqlens[i + 1] - cu_seqlens[i], :] = packed_hidden_states[j, cu_seqlens[i] : cu_seqlens[i + 1], :]
    return hidden_states


'''
pack function: convert hidden_states to packed_hidden_states (batch_size=1)
'''
def pack(hidden_states, cu_seqlens, batch_size):
    package_num, seq_len, hidden_dim = hidden_states.shape
    seq_len_list = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_len_list_3d = seq_len_list.unsqueeze(1).unsqueeze(2)
    indices_3d = (
        torch.arange(seq_len, device=hidden_states.device)
        .unsqueeze(0)
        .unsqueeze(2)
        .repeat(package_num, 1, hidden_dim)
    )
    mask_3d = indices_3d < seq_len_list_3d.repeat(batch_size, 1, 1)
    packed_hidden_states = hidden_states[mask_3d].view(batch_size,-1, hidden_dim)
    return packed_hidden_states

    
'''
Generate random cu_seqlens for testing
'''
def generate_random_cu_seqlens(seq_len, packages_num = 2):
    
    if packages_num > 1:
        ret = sorted(random.sample(range(1, seq_len), packages_num - 1))
    else:
        ret = []
    cu_seqlens = [0] + ret + [seq_len]
    assert packages_num == len(cu_seqlens) - 1
    index = []
    for i in range(1, len(cu_seqlens)):
        token_len = cu_seqlens[i] - cu_seqlens[i-1]
        index.extend(list(range(token_len)))
    return cu_seqlens, index


def main():
    # config tested with A100
    hidden_dim = 4
    seq_len = 1024
    batch_size = 2
    device='cuda'
    
    itype = torch.half
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # If we have z, the errors on the weights seem higher
    rtolw = max(rtolw, rtol)
    atolw = max(atolw, atol)
    packages_num = 8
    # Generate random cu_seqlens for testing
    cu_seqlens, index = generate_random_cu_seqlens(seq_len, packages_num = packages_num)
    cu_seqlens = torch.tensor(cu_seqlens).cuda()
    index = torch.tensor(index, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1).contiguous().cuda()
    print("cu_seqlens:", cu_seqlens, "index:",index)
    # Generate packed_hidden_states with random values for testing
    # packed_hidden_states (batch_size=1) should be forwarded with cu_seqlens
    hidden_states_list = [torch.randn(l, hidden_dim, device=device) for l in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()]
    packed_hidden_states = torch.cat(hidden_states_list, dim=0).unsqueeze(0)
    packed_hidden_states = packed_hidden_states.expand(batch_size, -1,-1).contiguous()
    # hidden_states should be forwarded without cu_seqlens
    hidden_states = unpack(packed_hidden_states, cu_seqlens)


    # Check: sum of seq_len of item in hidden_states_list should be equal to seq_len of packed_hidden_states
    assert sum([hs.shape[0] for hs in hidden_states_list]) == packed_hidden_states.shape[1]
    # Check: max of seq_len of item in hidden_states_list should be equal to seq_len of hidden_states
    assert max([hs.shape[0] for hs in hidden_states_list]) == hidden_states.shape[1]


    grads = {}
    

    # creat one simple mamba block
    mamba = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=hidden_dim, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to(device)
    
    mamba_ref = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=hidden_dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    ).to(device)
    mamba_ref.load_state_dict(mamba.state_dict())
    
    # reference output for forwardding hidden_states
    with AlignTimer("pack_fwd"):
        out = mamba(packed_hidden_states, index)
    
    with AlignTimer("unpack_fwd"):
        out_ref = mamba_ref(hidden_states)
    out_ref_pack = pack(out_ref, cu_seqlens, batch_size)
    
    # with AlignTimer("unpack"):
    #     out_ref = mamba_ref(hidden_states)
    # out_ref_pack = pack(out_ref, cu_seqlens, batch_size)
    # output for forwardding packed_hidden_states with cu_seqlens


    # Testing the max/mean diff
    import numpy as np
    np.testing.assert_allclose(out.detach().cpu().numpy(), out_ref_pack.detach().cpu().numpy(), rtol = rtol, atol=atol)
    print(f'Output max diff: {(out - out_ref_pack).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref_pack).abs().mean().item()}')
    assert torch.allclose(out, out_ref_pack, rtol=rtol, atol=atol)
    
    g = torch.randn(out.shape).to(device)  
    with AlignTimer("pack_bwd"):
        out.backward(g)
    gradients = {name: param.grad.clone() for name, param in mamba.named_parameters()}

    g_ref = unpack(g, cu_seqlens)
    with AlignTimer("unpack_bwd"):
        out_ref.backward(g_ref)
    gradients_ref = {name: param.grad.clone() for name, param in mamba_ref.named_parameters()}
    
        
    # 比较两组梯度
    for name in gradients_ref:
        if name in gradients:
            is_equal = torch.allclose(gradients_ref[name], gradients[name], rtol=rtol, atol=atol)
            print(f"Gradients for {name} are {'equal' if is_equal else 'not equal'}")
            if not is_equal:
                print(f"Gradient difference for {name}: {torch.abs(gradients_ref[name] - gradients[name]).max()}")
        else:
            print(f"Parameter {name} not found in the second set of gradients")
    
    # grad_results = torch.load('use_position_grad_results.pt')
    # grad_results_ref = torch.load('no_position_grad_results.pt')
    # print(grad_results)
    # for name in grad_results_ref:
    #     if name in grad_results and grad_results[name] is not None:
    #         is_equal = torch.allclose(grad_results_ref[name], grad_results[name], rtol=rtol, atol=atol)
    #         print(f"Gradients for {name} are {'equal' if is_equal else 'not equal'}")
    #         if not is_equal:
    #             print(f"Gradient difference for {name}: {torch.abs(grad_results_ref[name] - grad_results[name]).max()}")
    #     else:
    #         print(f"Parameter {name} not found in the second set of gradients")


if __name__ == "__main__":
    main()