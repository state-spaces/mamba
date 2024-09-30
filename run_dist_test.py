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
args = parser.parse_args()
print(args)
torch.manual_seed(args.random_seed)
num_gpus =  args.nproc_per_node
batch = args.batch_size
length = 1024
seq = torch.randn([batch,length*8,256],device='cuda')
#seq = torch.cat([(torch.ones([batch,length,256],dtype = torch.float32)*x).cuda() for x in range(num_gpus)], dim=1)
assert seq.shape[1]%num_gpus == 0
seq_per_gpu = seq.shape[1]//num_gpus
mesh_1d = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(num_gpus,))
print('running on ',dist.get_rank(), ' with ', seq_per_gpu)

if dist.get_rank() == 0:
    #N.B. must use contiguous when splitting tensors for distributed ops!
    sequence = rearrange(seq, 'i (n j) k -> i n j k', n = dist.get_world_size())
    sequence = [sequence[:,i,:,:].contiguous() for i in range(dist.get_world_size())]
    #sequence = [seq[:,seq_per_gpu*x:seq_per_gpu*(x+1),:] for x in range(dist.get_world_size())]
    #sequence = [(torch.ones([batch,seq_per_gpu,256],dtype=torch.float32)*x).cuda() for x in range(dist.get_world_size())]
    torch.save(sequence, f'sequence_{dist.get_rank()}.pt')
    print('0',sequence[0].shape)
else:
    sequence = None

def gather(rank, tensor):
        #group = dist.new_group(list(range(rank + 1)))
        #shape = tensor.shape
        tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, tensor) #  group=group)
        return tensor_list

input_tensor = torch.zeros([batch,seq_per_gpu,256], device='cuda')
print(dist.get_rank(), input_tensor.shape)
dist.scatter(input_tensor, sequence, src=0)
#print(input_tensor[0,0,0], dist.get_rank())

layer = Mamba2(256).cuda()
torch.save(input_tensor,f'input_{dist.get_rank()}.pt')
output = layer(input_tensor)

tensor_list = gather(dist.get_rank(),output)
#print(dist.get_rank(), [x[0,0,0] for x in tensor_list])

if dist.get_rank() == 0:
    torch.save(tensor_list, f'output.pt')


dist.destroy_process_group()

