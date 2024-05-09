from torch import nn

# reference implementation of the matching layer used in the LRA retrieval task
# https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/models/layers/common_layers.py#L197

class MATCH(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # dimensions are hardcoded to the values used in the original LRA code
        self.encoder = nn.Linear(input_dim, 512)
        self.middle = nn.Linear(512, 256)
        self.decoder = nn.Linear(256, output_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.middle(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x