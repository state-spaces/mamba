import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from mamba_ssm.models.mamba_lm_head_model import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
import logging
import random
import numpy as np
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
import torch.distributed as dist
import os
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self._load_data(file_path)

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return [tokens[i:i + self.block_size] for i in range(0, len(tokens), self.block_size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

def collate_batch(batch):
    return pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)

def load_dataset(file_path, tokenizer, block_size, batch_size, num_workers, split_ratio=0.8):
    dataset = TextDataset(file_path, tokenizer, block_size)
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    return train_loader, val_loader

def create_model(device, dtype):
    mamba_config = MambaConfig(
        d_model=2560,
        n_layer=64,
        vocab_size=50277,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8
    )
    model = MambaLMHeadModel(mamba_config, device=device, dtype=dtype)
    return model

def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch.to(device)
        outputs = model(input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch.to(device)
            outputs = model(input_ids)
            total_loss += outputs.loss.item()
    return total_loss / len(data_loader)

def setup_distributed(gpu, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=gpu, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def main_worker(gpu, args):
    setup_distributed(gpu, args.world_size)
    torch.cuda.set_device(gpu)
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    train_loader, val_loader = load_dataset(args.file_path, tokenizer, args.block_size, args.batch_size, args.num_workers)

    model = create_model(gpu, torch.float16)
    model = FSDP(model)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, gpu)
        val_loss = evaluate(model, val_loader, gpu)
        logging.info(f"GPU {gpu}, Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--file_path", type=str, default="./input.txt")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    args = parser.parse_args()
    args.world_size = args.num_gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.multiprocessing.spawn(main_worker, nprocs=args.num_gpus, args=(args,))
