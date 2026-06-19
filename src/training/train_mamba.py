import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from mamba_ssm.models.config_mamba import MambaConfig
from src.model.mamba_wrapper import MambaDNA
from src.data.dataset import DNADataset
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.eval.eval_metrics import MetricsTracker

def train(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Hyperparameters
    model_params = config['model_params']
    train_params = config['train_params']
    data_params = config.get('data_params', {})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    mamba_config = MambaConfig(
        d_model=model_params['d_model'],
        n_layer=model_params['n_layer'],
        vocab_size=model_params['vocab_size']
    )
    model = MambaDNA(mamba_config).to(device)
    
    # Data
    train_dataset = DNADataset(data_params.get('train_path', 'data/processed/train.npy'), model_params.get('seq_len', 1024))
    val_dataset = DNADataset(data_params.get('val_path', 'data/processed/val.npy'), model_params.get('seq_len', 1024))
    
    train_loader = DataLoader(train_dataset, batch_size=train_params.get('batch_size', 8), shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_params.get('batch_size', 8), shuffle=False, num_workers=4, pin_memory=True)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])
    
    num_epochs = train_params.get('epochs', 10)
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    # Metrics
    metrics = MetricsTracker()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Checkpointing
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Resume from checkpoint if exists
    start_epoch = 0
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_checkpoint):
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        metrics.reset()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            logits = output.logits
            
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, model_params['vocab_size']), target.view(-1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            metrics.update(loss, logits, target)
            
            pbar.set_postfix(metrics.get_metrics())
            
        # Validation
        model.eval()
        val_metrics = MetricsTracker()
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                logits = output.logits
                loss = criterion(logits.view(-1, model_params['vocab_size']), target.view(-1))
                val_metrics.update(loss, logits, target)
        
        print(f"Epoch {epoch+1} Validation Metrics: {val_metrics.get_metrics()}")
        
        # Save Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item(),
        }
        torch.save(checkpoint, latest_checkpoint)
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)
