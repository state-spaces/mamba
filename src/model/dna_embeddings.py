import torch.nn as nn

class DNAEmbedding(nn.Module):
    """
    Embedding layer for DNA tokens.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x)
