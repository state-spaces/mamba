import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import FSDPStrategy
import time
from collections import OrderedDict
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

class TextDataset(Dataset):
    def __init__(self, file_path, context_size, eos_token=0):
        self.context_size = context_size
        self.eos_token = eos_token
        self.data_file = file_path
        self.mmap_array = np.memmap(self.data_file, dtype='uint8', mode='r')
        self.seed = 42

    def __len__(self):
        return len(self.mmap_array) - self.context_size + 1

    def __getitem__(self, idx):
        generator = random.Random(self.seed + idx)
        start = generator.randint(0, len(self.mmap_array) - 1)
        end = start + self.context_size
        if end > len(self.mmap_array):
            padding_size = end - len(self.mmap_array)
            data_slice = np.concatenate(
                (self.mmap_array[start:], np.full(padding_size, self.eos_token, dtype='uint8'))
            )
        else:
            data_slice = self.mmap_array[start:end]
        return torch.tensor(data_slice, dtype=torch.long)

def collate_batch(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

class MambaDataModule(pl.LightningDataModule):
    def __init__(self, file_path, context_size, batch_size, num_workers, split_ratio=0.8):
        super().__init__()
        self.file_path = file_path
        self.context_size = context_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        dataset = TextDataset(self.file_path, self.context_size)
        train_size = int(len(dataset) * self.split_ratio)
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_batch, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_batch, pin_memory=True)

class MambaModel(pl.LightningModule):
    def __init__(self, mamba_config):
        super().__init__()
        self.model = MambaLMHeadModel(mamba_config)
        self.last_step_end_time = time.time()

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        start_time = self.last_step_end_time
        input_ids = batch
        with torch.cuda.amp.autocast(): # mixed precision training
            outputs = self(input_ids)
            labels = input_ids[:, 1:].contiguous()
            logits = outputs.logits[:, :-1, :].contiguous()
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            perplexity = torch.exp(loss)
            tokens_in_batch = input_ids.numel()

        self.last_step_end_time = time.time()
        elapsed_time = self.last_step_end_time - start_time
        if elapsed_time > 0:
            tokens_per_second = tokens_in_batch / elapsed_time
            self.log('tokens_per_second', tokens_per_second, on_step=True, on_epoch=False, sync_dist=True)

        self.log('train_loss', loss, sync_dist=True)
        self.log('train_perplexity', perplexity, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch
        with torch.cuda.amp.autocast(): # mixed precision training
            outputs = self(input_ids)
            labels = input_ids[:, 1:].contiguous()
            logits = outputs.logits[:, :-1, :].contiguous()
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log('val_loss', loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-5)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}

    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs)

def main(args):
    pl.seed_everything(42)

    os.makedirs(args.output_dir, exist_ok=True)
    
    mamba_config = MambaConfig(
        d_model=2560,
        n_layer=64,
        vocab_size=256, # byte level
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8
    )

    model = MambaModel(mamba_config)
    data_module = MambaDataModule(args.file_path, args.context_size, args.batch_size, args.num_workers)

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, monitor='val_loss', save_top_k=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger = TensorBoardLogger("tb_logs", name="mamba_model")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        log_every_n_steps=1,
        accelerator='gpu',
        strategy=FSDPStrategy(state_dict_type="full"),
        use_distributed_sampler=False,
        devices=args.num_gpus,
        callbacks=[checkpoint_callback, lr_monitor],
        precision='16-mixed'
    )

    trainer.fit(model, datamodule=data_module)

    if trainer.is_global_zero:
        print(f"Saving model to {os.path.join(args.output_dir, args.model_name)}")
        checkpoint = torch.load(checkpoint_callback.best_model_path)
        model = MambaLMHeadModel(mamba_config).to('cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('model.'):
                k = k[6:]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.save_pretrained(os.path.join(args.output_dir, args.model_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--context_size", type=int, default=1024)
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--model_name", type=str, default="mamba_model")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()

    main(args)
