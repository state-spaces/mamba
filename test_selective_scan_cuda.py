import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from parameterized import parameterized
from utils import experiments, Config
import simple_mamba.mamba as mamba

batch_size = 32
n_categories = 16

# Check where S4-Complex Fails
lags = [20, ]
extras = [10, ]
epochs = 100

settings_options_s6complex = [
    ["seed", [2]],
    ["ssm_type", ["S6-Real", "S6-Complex"]],
    ["d_model", [64]],
    ["d_state", [8]],
    ["lag", lags],
    ["extra", extras],
    ["n_layers", [2]],
    ["n_categories", [n_categories]],
    ["batch_size", [batch_size]],
    ["epochs", [epochs]],  # [int(1600 * 6]],
    ["epoch_size", [128 * 4]],
    ["stop_on_loss", [0.01]],
    ["lr", [1e-3]],
    ["A_imag_using_weight_decay", ["True"]],
    ["initA_imag", ["S4"]],
    ["param_A_imag", ["normal",]],
    ["discretizationB", ["zoh"]],
    ["discretizationA", ["normal"]],
    ["initA_real", ["S6"]],
    ["dt_is_selective", [True, False]],
    ["channel_sharing", [True]],
    ["bias", [False]],
    ["deterministic", [True]],
    ["pscan", [False, True]],
    ["comment", [""]]
]

configs = list(experiments(settings_options_s6complex))
configs = [Config(**config) for config in configs]


class TestMambaModels(unittest.TestCase):
    @parameterized.expand(configs)
    def test_models(self, config):
        self.mamba_config = mamba.MambaConfig(
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
            deterministic=config.deterministic,
            bias=config.bias,
            pscan=config.pscan)
        self.mamba_config_cuda = mamba.MambaConfig(
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
            deterministic=config.deterministic,
            bias=config.bias,
            pscan=config.pscan,
            use_cuda=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models from both files
        model1 = mamba.Mamba(self.mamba_config).to(self.device)
        model2 = mamba.Mamba(self.mamba_config_cuda).to(self.device)

        # Define a dummy input tensor
        input_tensor = torch.randn(10, 10, config.d_model).to(self.device)

        # Define a dummy target tensor for the loss calculation
        target = torch.randn(10, 10, config.d_model).to(self.device)

        # Define the optimizer for both models
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

        # Loss function
        criterion = nn.MSELoss()

        # Forward pass and loss calculation
        output1 = model1(input_tensor)
        loss1 = criterion(output1, target)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        output2 = model2(input_tensor)
        loss2 = criterion(output2, target)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        # Check if the outputs are close enough
        torch.testing.assert_allclose(output1, output2, rtol=1e-4, atol=1e-5)

        # Check if model parameters are close after one update step
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_allclose(param1, param2, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()