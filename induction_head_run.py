import torch.nn.functional as F
from tqdm import tqdm
import wandb
import torch
from torch import optim
import itertools
import numpy as np
from dataclasses import dataclass

from ds.datasets import InductionHead
from simple_mamba.mamba_lm import MambaLM, MambaLMConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

# Assumptions: 'model', 'dataloader', 'device', 'optim' (optimizer) are already defined
def train(config, model, data_loader, optimizer):

    # Setup tqdm for the outer loop
    pbar = tqdm(total=config.epochs, desc="Epoch Progress", position=0)

    # Training Loop
    for epoch in range(config.epochs):
        avg_loss = 0
        for data, labels in data_loader:
            data = data.to(device).long()  # Ensure data is on the correct device and dtype
            labels = labels.to(device).long()  # Ensure labels are on the correct device and converted to long        
            
            # Forward pass
            logits = model(data)  # [batch_size, seq_len, cat_num]
            
            # Compute loss
            loss = F.cross_entropy(logits[:, -config.induction_len:, :].reshape(-1, config.n_categories), 
                                   labels[:, -config.induction_len:].reshape(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
        
        avg_loss /= len(data_loader)
        wandb.log({"epoch": epoch,
                    "loss": avg_loss})
        
        if config.stop_on_loss and avg_loss < config.stop_on_loss:
            break
    pbar.close()


@dataclass
class Config:
    ssm_type: str
    d_model: int
    n_layers: int
    n_categories: int
    seq_len: int
    induction_len: int
    num_triggers: int
    batch_size: int
    epoch_size: int
    epochs: int
    lr: float
    stop_on_loss: float
    seed: int
    comment: str

def experiments(kwargs):
    # Extract argument names and their corresponding value lists
    arg_names = kwargs.keys()
    value_lists = kwargs.values()

    # Iterate over the Cartesian product of value lists
    for values in itertools.product(*value_lists):
        # Yield a dictionary mapping argument names to values
        yield dict(zip(arg_names, values))

def run_experiment(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    mamba_config = MambaLMConfig(
                           ssm_type=config.ssm_type,
                           d_model=config.d_model, 
                           n_layers=config.n_layers, 
                           vocab_size=config.n_categories, 
                           pad_vocab_size_multiple=config.n_categories)
    
    dataset = InductionHead(config.epoch_size, 
                            config.seq_len, 
                            config.n_categories,
                            config.num_triggers,
                            config.induction_len)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=config.batch_size, 
                                              shuffle=True)
    model = MambaLM(mamba_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)    
    train(config, model, data_loader, optimizer)

def name(config):
    # short name for display on wandb
    return f"Induction-{config.ssm_type}-seq_len{config.seq_len}-ind_len{config.induction_len}"

def main():
    for i, config in enumerate(experiments({
            "ssm_type":              ["S4D-Complex", "conv", "S4D-Real", "S6-Real"],
            "d_model":               [64],
            "n_layers":              [2],
            "n_categories":          [16],
            "seq_len":               [256],
            "induction_len":         [2, 4, 8],
            "num_triggers":          [1],
            "batch_size":            [8],
            "epochs":                [1600],
            "epoch_size":            [256],
            "lr":                    [1e-3],
            "stop_on_loss":          [0.01],
            "seed":                  [1],   
            })):
        config.update({"comment": ""})
        exp_name = name(Config(**config))
        wandb.init(
            project="mamba",
            entity="complex-team",
            name=exp_name,
            config=config
        )
        run_experiment(Config(**config))
        wandb.finish()
    print('Finished Training')

if __name__=='__main__':
    main()