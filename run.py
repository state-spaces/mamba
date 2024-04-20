import torch.nn.functional as F
from tqdm import tqdm
import wandb
import torch
from torch import optim
from ds.datasets import DynamicCategoricalDataset
from simple_mamba.mamba_lm import MambaLM, MambaLMConfig
import itertools
from dataclasses import dataclass


device = "cuda" if torch.cuda.is_available() else "cpu"

# Assumptions: 'model', 'dataloader', 'device', 'optim' (optimizer) are already defined
def train(config, model, data_loader, optimizer):

    # Setup tqdm for the outer loop
    pbar = tqdm(total=config.epochs, desc="Epoch Progress", position=0)

    # Training Loop
    for epoch in range(config.epochs):
        for data, labels in data_loader:
            data = data.to(device).long()  # Ensure data is on the correct device and dtype
            labels = labels.to(device).long()  # Ensure labels are on the correct device and converted to long        
            
            # Forward pass
            logits = model(data)  # [batch_size, seq_len, cat_num]
            
            # Compute loss
            loss = F.cross_entropy(logits[:, config.lag:, :].reshape(-1, config.n_categories), labels[:, config.lag:].reshape(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store loss for this batch to average later
        wandb.log({"epoch": epoch,
                    "loss": loss.item()})
    pbar.close()


@dataclass
class Config:
    ssm_type: str
    d_model: int
    n_layers: int
    n_categories: int
    lag: int
    extra: int
    batch_size: int
    epoch_size: int
    epochs: int
    lr: float
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
    mamba_config = MambaLMConfig(
                           ssm_type=config.ssm_type,
                           d_model=config.d_model, 
                           n_layers=config.n_layers, 
                           vocab_size=config.n_categories, 
                           pad_vocab_size_multiple=config.n_categories)
    
    dataset = DynamicCategoricalDataset(config.epoch_size, 
                                        config.extra + config.lag, 
                                        config.n_categories, 
                                        config.lag)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=config.batch_size, 
                                              shuffle=True)
    model = MambaLM(mamba_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)    
    train(config, model, data_loader, optimizer)

def name(config):
    # short name for display on wandb
    return f"{config.ssm_type}-lag{config.lag}-extra{config.extra}"

def main():
    for i, config in enumerate(experiments({
            "ssm_type":              ["S4D-Complex", "S4D-Real", "conv", "S6-Real"],
            "d_model":               [16, 64],
            "n_layers":              [2],
            "n_categories":          [16],
            "lag":                   [32, 64, 96, 128, 160,192],
            "extra":                 [1],
            "batch_size":            [8],
            "epochs":                [int(1600*2)],
            "epoch_size":            [128],
            "lr":                    [1e-3],
            })):
        config.update({"comment": ""})
        exp_name = name(Config(**config))
        wandb.init(
            project="RealVSComplex",
            name=exp_name,
            config=config
        )
        run_experiment(Config(**config))
        wandb.finish()
    print('Finished Training')

if __name__=='__main__':
    main()