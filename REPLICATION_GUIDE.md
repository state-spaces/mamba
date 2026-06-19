# Replication Guide: Mamba DNA Project

This document outlines the steps taken to set up the environment, prepare the data, train, and evaluate the Mamba model for DNA sequence modeling. Following this guide will replicate the baseline experiment and provide the foundation for further development.

## 1. Project Setup & Environment Configuration

These steps cover cloning the repository and configuring the necessary environment and dependencies.

### 1.1. Clone the Repository

First, clone the project from the GitHub repository.

```bash
git clone https://github.com/Kaushikj-7/mamba.git
cd mamba
```

### 1.2. Environment Setup (WSL)

All commands should be executed within a Windows Subsystem for Linux (WSL) terminal.

### 1.3. Python Virtual Environment

Create and activate a Python 3.12 virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.4. Install Dependencies

Install the required Python packages. The `mamba-ssm` package requires a special environment variable to compile correctly on systems with limited memory.

```bash
export MAX_JOBS=1
pip install -r requirements.txt
```

**Note:** `MAX_JOBS=1` is critical to prevent out-of-memory errors during the compilation of the `mamba-ssm` CUDA kernels.

## 2. Data Preparation

This section describes how to download and process the HG38 human genome dataset.

### 2.1. Download HG38 Dataset

Run the provided script to download the raw `hg38.fa` file.

```bash
bash scripts/download_hg38.sh
```

### 2.2. Process and Tokenize Data

Run the `prepare_hg38.py` script to tokenize the genome and create data shards. The `PYTHONPATH` must be set to the project root for the script to correctly locate its module dependencies.

```bash
export PYTHONPATH=.
python src/data/prepare_hg38.py
```

## 3. Model Training

The following steps detail the training process and the necessary code modifications.

### 3.1. Code Modifications

The training script `src/training/train_mamba.py` was modified to improve usability and fix a bug.

#### 3.1.1. Added `tqdm` for Progress Tracking

A `tqdm` progress bar was added to the training loop to visualize epoch progress.

```python
// In src/training/train_mamba.py

// ... existing code ...
from tqdm import tqdm
// ... existing code ...
        # train for one epoch
        model.train()
        
        pbar = tqdm(range(len(train_loader)))
        for i, (x, y) in enumerate(iter(train_loader)):
// ... existing code ...
```

#### 3.1.2. Fixed Hyperparameter Type Error

Hyperparameters loaded from the YAML config were incorrectly interpreted as strings. They were cast to `float` to resolve a `TypeError`.

```python
// In src/training/train_mamba.py

// ... existing code ...
    # Get learning rate from config
    lr = float(config['train_params']['lr'])
    min_lr = float(config['train_params']['min_lr'])
    weight_decay = float(config['train_params']['weight_decay'])
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
// ... existing code ...
```

### 3.2. Run Training

Execute the training script using the `mamba_1024.yaml` configuration.

```bash
python src/training/train_mamba.py --config configs/mamba_1024.yaml
```

## 4. Model Evaluation

After training, the model was evaluated for performance, and results were benchmarked.

### 4.1. Code Modifications

The evaluation script `src/eval/benchmark_comparison.py` was modified to fix a runtime error.

#### 4.1.1. Fixed Tensor Reshaping Error

A `RuntimeError` related to tensor views was fixed by replacing `.view(-1)` with `.reshape(-1)`, which handles both contiguous and non-contiguous tensors.

```python
// In src/eval/benchmark_comparison.py

# ... existing code ...
        # loss calculation
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, 5) # 5 is vocab_size
        shift_labels = shift_labels.reshape(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
# ... existing code ...
```

### 4.2. Run Evaluation

An evaluation script `run_evaluations.sh` was created to automate the process.

**`run_evaluations.sh`:**
```bash
#!/bin/bash

# Ensure python can find the modules
export PYTHONPATH=.

# Run generation
echo "Running generation..."
python src/eval/generate.py

# Run benchmarking
echo "Running benchmark comparison..."
python src/eval/benchmark_comparison.py
```

Make it executable and run it:
```bash
chmod +x run_evaluations.sh
./run_evaluations.sh
```

## 5. Results and Documentation

### 5.1. Baseline Performance

The 2.63M parameter model, trained for 4 epochs, achieved the following result:
- **Bits Per Base (BPB):** 1.6198

### 5.2. Parameter Count

A script `count_params.py` was created to calculate the model's parameter count.

**`count_params.py`:**
```python
import yaml
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Load the configuration
with open('configs/mamba_1024.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract model parameters from the config
d_model = config['model_params']['d_model']
n_layer = config['model_params']['n_layer']
vocab_size = config['model_params']['vocab_size']

# Create the Mamba configuration object
mamba_config = MambaConfig(d_model=d_model, n_layer=n_layer, vocab_size=vocab_size)

# Create the model
model = MambaLMHeadModel(config=mamba_config)

# Calculate and print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.2f}M")
```

### 5.3. Experiment Log

An `experiment_log.md` file was created to document results.

**`experiment_log.md`:**
```markdown
# Experiment Log

## Baseline: 2.63M Parameter Model

- **Config:** `configs/mamba_1024.yaml`
- **Epochs:** 4
- **Bits Per Base (BPB):** 1.6198
```
