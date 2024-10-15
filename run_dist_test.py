from collections import defaultdict
import pandas as pd
from mamba_ssm import Mamba2
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
#import torch.distributed.autograd as dist_autograd
#from einops import rearrange
if not dist.is_available():
    raise Exception("Distributed note abval")
import argparse

def send_and_receive_(x, receive_buffer, send_to_rank, receive_from_rank):
    assert send_to_rank or receive_from_rank
    ops = []
    if send_to_rank is not None:
        ops.append(dist.P2POp(dist.isend, x, send_to_rank))
    if receive_from_rank is not None:
        ops.append(dist.P2POp(dist.irecv, receive_buffer, receive_from_rank))

    reqs = dist.batch_isend_irecv(ops)

    for req in reqs:
        req.wait()
    dist.barrier()

class SequenceParallelMixerFn(Function):
    @staticmethod
    def forward(ctx, x, padding):
        #Prepends the last n_padding tokens from layer_n to layer_{n+1}
        #These are mixed into subsequent tokens of layer n+1 by convolution, but their index is then discarded
        # the convolution is causal, so the mixing only goes in one direction
        rank, world_size = dist.get_rank(), dist.get_world_size()
        ctx.padding = padding
        if world_size == 1:
            return x

        send_to_rank = rank + 1 if rank < world_size - 1 else None
        receive_from_rank = rank - 1 if rank > 0 else None
        #print('dist', rank, send_to_rank, receive_from_rank)
        #_, pre_tokens = x.split(x.shape[1]-self.padding, dim=1)
        pre_tokens = x[:,-ctx.padding:].contiguous()
        print('dist',rank,pre_tokens.requires_grad)
        assert pre_tokens.shape[1] == ctx.padding
        receive_buffer = torch.zeros_like(pre_tokens, requires_grad=True).contiguous() #TODO this isn't used by rank=0
        send_and_receive_(pre_tokens, receive_buffer, send_to_rank, receive_from_rank)
        if rank > 0:
            x = F.pad(x, (0, 0, ctx.padding, 0), 'constant', 0)
            x[:,:ctx.padding] = receive_buffer
            print('dist',rank,'receive_buffer grad',receive_buffer.requires_grad)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        print('grad_x', rank, grad_x.shape)
        if world_size == 1:
            return grad_x
        send_to_rank = rank -1 if rank > 0 else None
        receive_from_rank = rank + 1 if rank < world_size - 1 else None
        pre_tokens_grad = x[:,:ctx.padding].contiguous()
        assert pre_tokens_grad.shape[1] == ctx.padding
        receive_buffer = torch.zeros_like(pre_tokens, requires_grad=True).contiguous() #TODO this isn't used by rank=0
        send_and_receive_(pre_tokens, receive_buffer, send_to_rank, receive_from_rank)
        if rank < world_size -1:
            grad_x_out = grad_x.clone()
            grad_x_out[:,-ctx.padding:] += receive_buffer
        return grad_x_out

class SequenceParallelMixerLayer(nn.Module):
    def __init__(self, padding = 0):
        super(SequenceParallelMixerLayer, self).__init__()
        self.padding = padding
    def forward(self,x):
        return SequenceParallelMixerFn.apply(x, self.padding)


parser = argparse.ArgumentParser()
# This is always passed in by default
#parser.add_argument("--local_rank", type=int)
# These are your own arguments
#parser.add_argument("--master_addr", type=str)
parser.add_argument("--nproc_per_node", type=int)
parser.add_argument("--random_seed", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--iterations", type=int)
parser.add_argument("--num_layers", type=int)
args = parser.parse_args()
print(args)
torch.manual_seed(args.random_seed)
num_gpus = args.nproc_per_node
num_layers = args.num_layers
batch = args.batch_size
iterations = args.iterations
mesh_1d = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(num_gpus,))
#print(mesh_1d.get_group().bound_device_id)
#if dist.get_rank() == 0:
    #N.B. must use contiguous when splitting tensors for distributed ops!
    #sequence = rearrange(seq, 'i (n j) k -> i n j k', n = dist.get_world_size())
    #sequence = [sequence[:,i,:,:].contiguous() for i in range(dist.get_world_size())]
    #sequence = [seq[:,seq_per_gpu*x:seq_per_gpu*(x+1),:] for x in range(dist.get_world_size())]
    #sequence = [(torch.ones([batch,seq_per_gpu,256],dtype=torch.float32)*x).cuda() for x in range(dist.get_world_size())]
    #torch.save(sequence, f'sequence_{dist.get_rank()}.pt')
    #print('0',sequence[0].shape)
#else:
    #sequence = None
    
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved

#print(dist.get_rank(), input_tensor.shape)
#dist.scatter(input_tensor, sequence, src=0)
#print(input_tensor[0,0,0], dist.get_rank())
world_size, rank = dist.get_world_size(), dist.get_rank()


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

res_forward = list()
res_backward = list()
layers = []
for _ in range(num_layers):
    mamba_layer = Mamba2(256)
    padding = mamba_layer.d_conv - 1
    layers.append(SequenceParallelMixerLayer(padding))
    layers.append(mamba_layer)
model = nn.Sequential(*layers).cuda()
if dist.get_rank() == 0:
    print(model)

for s in range(10,11):
    length = 2**s
    seq = torch.randn([iterations,batch,length*8,256],device='cpu')
    torch.save(seq,'seq.pt')
    #seq = torch.cat([(torch.ones([batch,length,256],dtype = torch.float32)*x).cuda() for x in range(num_gpus)], dim=1)
    assert seq.shape[1]%num_gpus == 0
    seq_per_gpu = seq.shape[2]//num_gpus
    #print('running on ',dist.get_rank(), ' with ', seq_per_gpu)
    #Equal split sequences - easy test
    #sequence = rearrange(seq, 'i b (n j) k -> i n b j k', n = world_size)
    #sequence = [sequence[:,i,:,:].contiguous() for i in range(world_size)]
    #Split with padded repeats for 1d conv overlap
    #sequence = [seq[:, :, seq_per_gpu*r:seq_per_gpu*(r+1)+padding] for r in range(world_size)]
    sequence = [seq[:, :, seq_per_gpu * r:seq_per_gpu * (r + 1)] for r in range(world_size)] #Don't need padding with Mixer layer
    #with dist_autograd.context() as context_id:
    for i in range(iterations):
        #input_tensor = sequence[i,rank].cuda()
        #print(f"{sequence[rank].shape = }")
        input_tensor = sequence[rank][i].cuda().contiguous()
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start.record()
        output = model(input_tensor)
        end.record()
        torch.cuda.synchronize()
        r = torch.cuda.memory_reserved(rank)
        a = torch.cuda.memory_allocated(rank)
        t = start.elapsed_time(end)
        res_forward.append({'exp':s,'it':i,'res':r,'all':a,'time':t})
        if rank == 0:
            print("forward",rank,i, a/10**9, r/10**9, 'GB')
            #print(rank,prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=3))
            print("forward",rank,i,t, 'ms')
        model.zero_grad()
        start.record()
        #dist_autograd.backward(context_id, [output[:,-1,:].sum()]) #For RPC only
        output.sum().backward()
        end.record()
        torch.cuda.synchronize()
        r = torch.cuda.memory_reserved(rank)
        a = torch.cuda.memory_allocated(rank)
        t = start.elapsed_time(end)
        res_backward.append({'exp':s,'it':i,'res':r,'all':a,'time':t})
        if rank == 0:
            print("backward",rank,i, a/10**9, r/10**9, 'GB')
            #print(rank,prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=3))
            print("backward",rank,i,t, 'ms')
        dist.barrier()
    torch.save(input_tensor,f'input_{rank}.pt')
    torch.save(output, f"output_{rank}.pt")
    torch.save({x[0]:x[1].grad for x in model.named_parameters()}, f"grad_dict_{rank}.pt")
pd.DataFrame(res_forward).to_csv(f'res_fw_{rank}.csv')
pd.DataFrame(res_backward).to_csv(f'res_bw_{rank}.csv')
dist.destroy_process_group()

exit()

if dist.get_world_size() > 1:
    tensor_list = gather(dist.get_rank(),output)
    #print(dist.get_rank(), [x[0,0,0] for x in tensor_list])

    if dist.get_rank() == 0:
        torch.save(tensor_list, f'output.pt')
else:
    torch.save(output, f'output.pt')


def gather(rank, tensor):
        #group = dist.new_group(list(range(rank + 1)))
        #shape = tensor.shape
        tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, tensor) #  group=group)
        return tensor_list

#input_tensor = torch.zeros([batch,seq_per_gpu,256], device='cuda')
