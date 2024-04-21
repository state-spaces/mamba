import torch.nn.functional as F
from tqdm import tqdm
import wandb
import torch
from torch import optim
from ds.datasets import DynamicCategoricalDataset
from simple_mamba.mamba_lm import MambaLM, MambaLMConfig
import itertools
import numpy as np
from dataclasses import dataclass


device = "cuda" if torch.cuda.is_available() else "cpu"

# Assumptions: 'model', 'dataloader', 'device', 'optim' (optimizer) are already defined
def train(config, model, data_loader, optimizer):

    # Setup tqdm for the outer loop
    pbar = tqdm(total=config.epochs, desc="Epoch Progress", position=0)

    # Training Loop
    for epoch in range(config.epochs):
        avg_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        total_correct_sequences = 0
        for data, labels in data_loader:
            data = data.to(device).long()  # Ensure data is on the correct device and dtype
            labels = labels.to(device).long()  # Ensure labels are on the correct device and converted to long

            # Forward pass
            logits = model(data)  # [batch_size, seq_len, cat_num]

            # Compute loss
            loss = F.cross_entropy(logits[:, config.lag:, :].reshape(-1, config.n_categories),
                                   labels[:, config.lag:].reshape(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            # Calculate predictions
            _, predicted = torch.max(logits, dim=2)  # [batch_size, seq_len]

            # Mask to focus only on relevant positions
            relevant_labels = labels[:, config.lag:]
            relevant_predicted = predicted[:, config.lag:]

            # Calculate correct predictions per token
            correct_tokens = (relevant_predicted == relevant_labels).sum()
            total_correct_tokens += correct_tokens.item()
            total_tokens += relevant_labels.numel()  # Total number of evaluated tokens

            # Calculate correct predictions per sequence
            correct_sequences = (relevant_predicted == relevant_labels).all(dim=1).sum()
            total_correct_sequences += correct_sequences.item()

        total_sequences = sum(len(labels) for _, labels in data_loader)
        avg_loss /= len(data_loader)
        avg_accuracy_per_token = total_correct_tokens / total_tokens
        avg_accuracy_per_sequence = total_correct_sequences / total_sequences

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "avg_accuracy_per_token": avg_accuracy_per_token,
            "avg_accuracy_per_sequence": avg_accuracy_per_sequence
        })

        if config.stop_on_loss and avg_loss < config.stop_on_loss:
            break
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
    stop_on_loss: float
    seed: int
    comment: str

def fexperiments(kwargs):
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
    return f"{config.ssm_type}-lag{config.lag}-extra{config.extra}-dim{config.d_model}"

def main():
    for i, config in enumerate(experiments({
            "ssm_type":              ["S4D-Complex", "S4D-Real", "conv", "S6-Real"],
            "d_model":               [16, 64],
            "n_layers":              [2],
            "n_categories":          [16],
            "lag":                   [64, 128],
            "extra":                 [32, 64, 128],
            "batch_size":            [8],
            "epochs":                [int(1600*2)],
            "epoch_size":            [128],
            "lr":                    [1e-3],
            "stop_on_loss":          [0.01],
            "seed":                  [42],   
            })):
        config.update({"comment": ""})
        exp_name = name(Config(**config))
        wandb.init(
            project="RealVSComplexDelay",
            entity="yuv-milo",
            name=exp_name,
            config=config
        )
        run_experiment(Config(**config))
        wandb.finish()
    print('Finished Training')

if __name__=='__main__':
    main()