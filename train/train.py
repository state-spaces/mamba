import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Set the environment variable for tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set float32 matrix multiplication precision for CUDA device with Tensor Cores
torch.set_float32_matmul_precision('medium')  # or 'high'

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size, stride):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride
        self.data = self._load_data(file_path)

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return [tokens[i:i + self.block_size] for i in range(0, len(tokens), self.stride)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

def collate_batch(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

class MambaDataModule(pl.LightningDataModule):
    def __init__(self, file_path, tokenizer_name, block_size, stride, batch_size, num_workers, split_ratio=0.8):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.block_size = block_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        dataset = TextDataset(self.file_path, self.tokenizer, self.block_size, self.stride)
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

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            outputs = self(input_ids)
            labels = input_ids[:, 1:].contiguous()
            logits = outputs.logits[:, :-1, :].contiguous()
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            perplexity = torch.exp(loss)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_perplexity', perplexity, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            outputs = self(input_ids)
            labels = input_ids[:, 1:].contiguous()
            logits = outputs.logits[:, :-1, :].contiguous()
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log('val_loss', loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-5)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}

def main(args):
    pl.seed_everything(42)

    mamba_config = MambaConfig(
        d_model=1280,
        n_layer=32,
        vocab_size=50277,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8
    )

    model = MambaModel(mamba_config)
    data_module = MambaDataModule(args.file_path, args.tokenizer_name, args.block_size, args.stride, args.batch_size, args.num_workers)

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, monitor='val_loss', save_top_k=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger = TensorBoardLogger("tb_logs", name="mamba_model")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        log_every_n_steps=1,
        accelerator='gpu',
        devices=args.num_gpus,
        callbacks=[checkpoint_callback, lr_monitor],
        precision='16-mixed'  # Using 16 for mixed precision training while keeping model parameters in float32
    )
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--file_path", type=str, default="./input.txt")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    args = parser.parse_args()

    main(args)
