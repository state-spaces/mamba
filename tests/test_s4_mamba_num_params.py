import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from parameterized import parameterized
from utils import experiments, Config
import simple_mamba.mamba as mamba
from copy import deepcopy


settings_options_s6complex = [
    ["ssm_type", ["S6-Real"]],
    ["d_model", [2]],
    ["d_state", [8]],
    ["n_layers", [1]],
    ["deterministic", [True]],
    ["pscan", [True]]
]

configs = list(experiments(settings_options_s6complex))
configs = [mamba.MambaConfig(**config) for config in configs]

class TestS4MambaNumParams(unittest.TestCase):
    @parameterized.expand(configs)
    def test_models(self, config):
        print(config)
        self.mamba_config_real = deepcopy(config)
        self.mamba_config_complex = deepcopy(config)
        self.s4_config_real = deepcopy(config)
        self.s4_config_complex = deepcopy(config)
        self.mamba_config_real.ssm_type = "S6-Real"
        self.mamba_config_complex.ssm_type = "S6-Complex"
        self.s4_config_real.ssm_type = "S4D-Real"
        self.s4_config_complex.ssm_type = "S4D-Complex"

        self.s4_config_real.__post_init__()
        self.s4_config_complex.__post_init__()
        self.mamba_config_complex.__post_init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models from both files
        model1 = mamba.Mamba(self.mamba_config_real).to(self.device)
        model2 = mamba.Mamba(self.s4_config_real).to(self.device)

        # check if the number of parameters are the same.
        for n, p in model1.named_parameters():
            print(n, p.numel(), p.shape)
        for n, p in model2.named_parameters():
            print(n, p.numel(), p.shape)
        self.assertEqual(sum(p.numel() for p in model1.parameters()), sum(p.numel() for p in model2.parameters()))


if __name__ == '__main__':
    unittest.main()