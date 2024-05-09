import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from parameterized import parameterized
from utils import experiments, Config
import simple_mamba.mamba as mamba
from copy import deepcopy


settings_options_s6complex = [
    ["ssm_type", ["S6-Real", "S6-Complex"]],
    ["d_model", [2, 4, 5]],
    ["d_state", [2, 4, 8]],
    ["n_layers", [2]],
    ["deterministic", [True]],
    ["pscan", [True, False]]
]

configs = list(experiments(settings_options_s6complex))
configs = [mamba.MambaConfig(**config) for config in configs]


class TestSelectiveScan(unittest.TestCase):
    @parameterized.expand(configs)
    def test_models(self, config):
        self.mamba_config = deepcopy(config)
        self.mamba_config.use_cuda = False
        self.mamba_config_cuda = deepcopy(config)
        self.mamba_config_cuda.use_cuda = True
        print("config: ", config)
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