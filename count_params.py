import yaml
import argparse
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Set up argument parser
parser = argparse.ArgumentParser(description='Calculate model parameters.')
parser.add_argument('--config', type=str, default='configs/mamba_1024.yaml',
                    help='Path to the configuration file.')
args = parser.parse_args()

# Load the configuration
with open(args.config, 'r') as f:
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
