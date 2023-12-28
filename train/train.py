import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import WikiText2
from torchtext.data import BucketIterator, Field
from mamba_ssm.models.mamba_lm_head_model import MambaLMHeadModel
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from torch.nn.parallel import DistributedDataParallel

def tokenize(text):
    return text.split()

# Define how to process the text data
TEXT = Field(tokenize=tokenize, lower=True, batch_first=True)

def load_dataset(batch_size, num_workers):
    train_data, val_data, test_data = WikiText2.splits(TEXT, root='./data')
    TEXT.build_vocab(train_data)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_data, val_data, test_data), 
        batch_size=batch_size,
        device=device,
        sort_within_batch=True
    )
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
    return model

def main(config_name, device, dtype, num_epochs, batch_size, num_workers):
    train_iter, val_iter = load_dataset(batch_size, num_workers)
    model = create_model(config_name, device, dtype)
    model = fsdp_wrapper(model, device, dtype)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    model = train(model, train_iter, val_iter, device, optimizer, scheduler, num_epochs)

if __name__ == "__main__":
    config_name = "mamba_mixer_config"
    device = "cuda:0"
    dtype = torch.bfloat16
    num_epochs = 10
    batch_size = 32
    num_workers = 4
    main(config_name, device, dtype, num_epochs, batch_size, num_workers)
