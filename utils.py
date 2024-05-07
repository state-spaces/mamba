from dataclasses import dataclass
import itertools


@dataclass
class Config:
    ssm_type: str
    initA_real: str
    initA_imag: str
    discretizationA: str
    discretizationB: str
    param_A_imag: str
    A_imag_using_weight_decay: str
    dt_is_selective: str
    channel_sharing: str
    deterministic:bool
    pscan: bool
    d_model: int
    d_state: int
    n_layers: int
    n_categories: int
    lag: int
    extra: int
    batch_size: int
    epoch_size: int
    epochs: int
    lr: float
    stop_on_loss: float
    seed: int
    comment: str
    bias:bool

def experiments(kwargs):
    # Extract argument names and their corresponding value lists
    arg_names = [k[0] for k in kwargs]
    value_lists = [k[1] for k in kwargs]

    # Iterate over the Cartesian product of value lists
    for values in itertools.product(*value_lists):
        # Yield a dictionary mapping argument names to values
        yield dict(zip(arg_names, values))