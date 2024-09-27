from mamba_ssm import Mamba2
import torch
import torch.distributed as dist

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
args = parser.parse_args()
print(args)
torch.manual_seed(args.random_seed)
num_gpus =  args.nproc_per_node

print('running on ',num_gpus)
seq = torch.randn([2,1024*8,256],device='cuda')
assert seq.shape[1]%num_gpus == 0
seq_per_gpu = seq.shape[1]//num_gpus
mesh_1d = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(num_gpus,))

print(dist.get_rank())

if dist.get_rank() == 0:
    sequence = [seq[:,seq_per_gpu*x:seq_per_gpu*(x+1)] for x in range(dist.get_world_size())]
else:
    sequence = None

def gather(rank, tensor):
        #group = dist.new_group(list(range(rank + 1)))
        #shape = tensor.shape
        tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, tensor) #  group=group)
        return tensor_list

input_tensor = torch.zeros([2,seq_per_gpu,256], device='cuda')
dist.scatter(input_tensor, sequence, src=0)
print(input_tensor[0,0,0], dist.get_rank())

layer = Mamba2(256).cuda()
output = layer(input_tensor)

tensor_list = gather(dist.get_rank(),output)
print(dist.get_rank(), [x[0,0,0] for x in tensor_list])

if dist.get_rank() == 0:
    torch.save(tensor_list, f'output_{num_gpus}.pt')


dist.destroy_process_group()

