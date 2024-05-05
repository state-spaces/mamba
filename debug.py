import torch.nn.functional as F
# import wandb
import torch
from torch import optim
from ds.datasets import DynamicCategoricalDataset
from simple_mamba.mamba_lm import MambaLM, MambaLMConfig
import itertools
import numpy as np
from dataclasses import dataclass
import os
# os.environ["WANDB_SILENT"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Assumptions: 'model', 'dataloader', 'device', 'optim' (optimizer) are already defined
def train(config, model, data_loader, optimizer):

    # Training Loop
    if config.deterministic:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

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

            # N = 0
            # all_parameters = list(model.parameters())
            # for param in all_parameters[:-N]:
            #     param.requires_grad = False
            print("\n".join(f"{name}: {param.size()}, requires_grad={param.requires_grad}" for name, param in model.named_parameters()))
            # Set requires_grad=False for all except the last N parameters
            loss2 = loss

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
        # wandb.log({
        #     "epoch": epoch,
        #     "loss": avg_loss,
        #     "avg_accuracy_per_token": avg_accuracy_per_token,
        #     "avg_accuracy_per_sequence": avg_accuracy_per_sequence
        # })

        if config.stop_on_loss and avg_loss < config.stop_on_loss:
            break
    # pbar.close()


@dataclass
class Config:
    ssm_type: str
    initA_real: str
    initA_imag: str
    discretizationA: str
    discretizationB: str
    param_A_imag: str
    A_imag_using_weight_decay: str
    dt_is_selective: str
    channel_sharing: str
    deterministic: bool
    pscan: bool
    bias: bool
    d_model: int
    d_state: int
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
    bias:bool


def experiments(kwargs):
    # Extract argument names and their corresponding value lists
    arg_names = [k[0] for k in kwargs]
    value_lists = [k[1] for k in kwargs]

    # Iterate over the Cartesian product of value lists
    for values in itertools.product(*value_lists):
        # Yield a dictionary mapping argument names to values
        yield dict(zip(arg_names, values))

def run_experiment(config):
    exp_name = name(config)

    # wandb.init(
    #     project="complex-mamba-copy-s4",
    #     entity="complex-team",
    #     name=exp_name,
    #     config=config
    # )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    mamba_config = MambaLMConfig(
        ssm_type=config.ssm_type,
        discretizationA=config.discretizationA,
        discretizationB=config.discretizationB,
        initA_imag=config.initA_imag,
        initA_real=config.initA_real,
        param_A_imag=config.param_A_imag,
        A_imag_using_weight_decay=config.A_imag_using_weight_decay,
        dt_is_selective=config.dt_is_selective,
        channel_sharing=config.channel_sharing,
        d_model=config.d_model,
        d_state=config.d_state,
        n_layers=config.n_layers,
        vocab_size=config.n_categories,
        pad_vocab_size_multiple=config.n_categories,
        bias=config.bias,
        pscan=config.pscan,
        deterministic = config.deterministic)

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
    # wandb.finish()

def name(config):
    # short name for display on wandb
    return f"{config.ssm_type}-lag{config.lag}-extra{config.extra}-dim{config.d_model}"

def main():
    S4 = False
    batch_size = 32
    n_categories = 16
    lag = 128
    extra = 32
    lr = 1
    epochs = 100
    if S4:
        settings_options = [
            ["seed", [2]],
            ["ssm_type", ["S4D-Complex"]],
            ["discretizationA", ["normal"]],
            ["discretizationB", ["s6"]],
            ["d_model", [64]],
            ["d_state", [16]],
            ["lag", [lag,]],
            ["extra", [extra,]],
            ["n_layers", [2]],
            ["n_categories", [n_categories]],
            ["batch_size", [batch_size]],
            ["epochs", [epochs]],  # [int(1600 * 6]],
            ["epoch_size", [128 * 4]],
            ["lr", [lr,]],
            ["stop_on_loss", [0.01]],
            ["initA_imag", ["S4", ]],
            ["initA_real", ["S4", ]],
            ["param_A_imag", ["normal", ]],
            ["A_imag_using_weight_decay", ["False", ]],
            ["dt_is_selective", ["False", ]],
            ["channel_sharing", [False]],
            ["bias", [False]],
            ["deterministic", [True]],
            ["pscan", [False]],
        ]

    else:
        settings_options = [
            ["seed", [2]],
            ["ssm_type", ["S6-Real-complex-bias"]],
            ["discretizationA", ["normal"]],
            ["discretizationB", ["s6"]],
            ["d_model", [64]],
            ["d_state", [8]],
            ["lag", [lag, ]],
            ["extra", [extra, ]],
            ["n_layers", [2]],
            ["n_categories", [n_categories]],
            ["batch_size", [batch_size]],
            ["epochs", [epochs]],  # [int(1600 * 6]],
            ["epoch_size", [128 * 4]],
            ["lr", [lr,]],
            ["stop_on_loss", [0.01]],
            ["initA_imag", ["S4", ]],
            ["initA_real", ["S4", ]],
            ["param_A_imag", ["normal", ]],
            ["A_imag_using_weight_decay", ["False", ]],
            ["dt_is_selective", ["False", ]],
            ["channel_sharing", [False]],
            ["bias", [False]],
            ["deterministic", [True]],
            ["pscan", [True]],
        ]

    tasks = []
    for i, config in enumerate(experiments(settings_options)):
        print(i)
        config.update({"comment": "Delete this"})
        tasks.append(run_experiment(Config(**config)))
    print("finished running all")

if __name__ == '__main__':
    main()