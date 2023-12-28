import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import Pennies
from torchtext.data import BucketIterator
from mamba_ssm.models.mamba_lm_head_model import MambaLMHeadModel
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from torch.nn.parallel import DistributedDataParallel, FullyShardedDataParallel

class TinyShakespeareDataset(Dataset):
    def __init__(self, file_path, ctx_len):
        self.file_path = file_path
        self.lines = self._preprocess_dataset(file_path)
        self.ctx_len = ctx_len

    def _preprocess_dataset(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.read().split('\n')
        lines = [line.lower() for line in lines if line.strip()]
        return lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        return torch.tensor(line)

def load_dataset(batch_size, num_workers, ctx_len):
    train_data, val_data = Pennies.splits(root='./data')
    train_data, val_data = train_data.remove_empty(), val_data.remove_empty()
    train_iter = BucketIterator(TinyShakespeareDataset(train_data, ctx_len=ctx_len), batch_size=batch_size, num_workers=num_workers)
    val_iter = BucketIterator(TinyShakespeareDataset(val_data, ctx_len=ctx_len), batch_size=batch_size, num_workers=num_workers)
    return train_iter, val_iter

def create_model(config_name, device, dtype):
    config_data = load_config_hf(config_name)
    config = MambaConfig(**config_data)
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    model.load_state_dict(load_state_dict_hf(config_name, device=device, dtype=dtype))
    return model

def train(model, train_iter, val_iter, device, optimizer, scheduler, num_epochs):
    model.train()
    total_loss = 0
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            loss = model(batch.text, inference_params=None).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch + 1}, Loss: {total_loss / (i + 1):.4f}")
        val_loss = evaluate(model, val_iter, device)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
    return model

def evaluate(model, val_iter, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            loss = model(batch.text, inference_params=None).loss
            total_loss += loss.item()
    return total_loss / (i + 1)

def fsdp_wrapper(model, device, dtype):
    model = model.to(device).type(dtype)
    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    model = FullyShardedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    return model

def main(config_name, device, dtype, num_epochs, batch_size, num_workers, ctx_len):
    train_iter, val_iter = load_dataset(batch_size, num_workers, ctx_len)
    model = create_model(config_name, device, dtype)
    model = fsdp_wrapper(model, device, dtype)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    model = train(model, train_iter, val_iter, device, optimizer, scheduler, num_epochs)

if __name__ == "__main__":
    config_name = "mamba_mixer_config"
    device = "cuda:0"
    dtype = torch.bfloat16
    num_epochs = 10
    batch_size = 32
    num_workers = 4
    ctx_len = 1024  # Specify the desired context length
    file_path = "./input.txt"
    main(config_name, device, dtype, num_epochs, batch_size, num_workers, ctx_len)
