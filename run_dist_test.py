from mamba_ssm import Mamba2
import torch
import torch.distributed as dist
torch.manual_seed(0)

if not dist.is_available():
    raise Exception("Distributed note abval")

mesh_1d = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(8,))

print(dist.get_rank())

if dist.get_rank() == 0:
    sequence = [torch.randn([2,1024,256],device='cuda') for x in range(dist.get_world_size())]
else:
    sequence = None

def gather(rank, tensor):
        #group = dist.new_group(list(range(rank + 1)))
        #shape = tensor.shape
        tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, tensor) #  group=group)
        return tensor_list

input_tensor = torch.zeros([2,1024,256], device='cuda')
dist.scatter(input_tensor, sequence, src=0)
print(input_tensor[0,0,0], dist.get_rank())

layer = Mamba2(256).cuda()
output = layer(input_tensor)

tensor_list = gather(dist.get_rank(),output)
print(dist.get_rank(), [x[0,0,0] for x in tensor_list])



dist.destroy_process_group()

