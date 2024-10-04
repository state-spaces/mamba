from collections import defaultdict
import pandas as pd
from mamba_ssm import Mamba2
import torch
import torch.distributed as dist
from einops import rearrange
if not dist.is_available():
    raise Exception("Distributed note abval")
import argparse
parser = argparse.ArgumentParser()
# This is always passed in by default
#parser.add_argument("--local_rank", type=int)
# These are your own arguments
#parser.add_argument("--master_addr", type=str)
parser.add_argument("--nproc_per_node", type=int)
parser.add_argument("--random_seed", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--iterations", type=int)
args = parser.parse_args()
print(args)
torch.manual_seed(args.random_seed)
num_gpus =  args.nproc_per_node
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

layer = Mamba2(256).cuda()
for s in range(10,11):
    length = 2**s
    seq = torch.randn([iterations,batch,length*8,256],device='cpu')
    #seq = torch.cat([(torch.ones([batch,length,256],dtype = torch.float32)*x).cuda() for x in range(num_gpus)], dim=1)
    assert seq.shape[1]%num_gpus == 0
    seq_per_gpu = seq.shape[1]//num_gpus
    print('running on ',dist.get_rank(), ' with ', seq_per_gpu)
    sequence = rearrange(seq, 'i b (n j) k -> i n b j k', n = world_size)
    #sequence = [sequence[:,i,:,:].contiguous() for i in range(world_size)]
    for i in range(iterations):
        input_tensor = sequence[i,rank].cuda()
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start.record()
        output = layer(input_tensor)
        end.record()
        torch.cuda.synchronize()
        r = torch.cuda.memory_reserved(rank)
        a = torch.cuda.memory_allocated(rank)
        t = start.elapsed_time(end)
        res_forward.append({'exp':s,'it':i,'res':r,'all':a,'time':t})
        print("forward",rank,i, a/10**9, r/10**9, 'GB')
        #print(rank,prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=3))
        print("forward",rank,i,t, 'ms')
        if True: #world_size == 1:
            layer.zero_grad()
            start.record()
            output[:,0,:].sum().backward()
            end.record()
            torch.cuda.synchronize()
            r = torch.cuda.memory_reserved(rank)
            a = torch.cuda.memory_allocated(rank)
            t = start.elapsed_time(end)
            res_backward.append({'exp':s,'it':i,'res':r,'all':a,'time':t})
            print("backward",rank,i, a/10**9, r/10**9, 'GB')
            #print(rank,prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=3))
            print("backward",rank,i,t, 'ms')
    torch.save(input_tensor,f'input_{rank}.pt')
    torch.save(output, f"output_{rank}.pt")
    torch.save({x[0]:x[1].grad for x in layer.named_parameters()}, f"grad_dict_{rank}.pt")
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
