import torch


def delay_l2(lag):
    def delay_loss_func(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean((output[:, lag:, :] - target[:, lag:, :]) ** 2)
        return loss

    return delay_loss_func


class NormalNoiseSignalGenerator():
    def __init__(self, std=1, mean=0):
        self.std = std
        self.mean = mean

    def generate(self, num_signals, signal_length, signal_dim=1):
        if signal_dim != 1:
            raise NotImplementedError("ConstSignalGenerator only supports signal dim 1")

        signals = torch.randn([num_signals, signal_length])
        return signals


class DelayedSignalDatasetRegenerated(torch.utils.data.TensorDataset):

    def __init__(self, samples_num=1, seq_length=10000, lag_length=1000,
                 signal_generator=None, lag_type="zero"):
        assert lag_length < seq_length
        assert lag_type in ["zero"]

        if signal_generator is None:
            signal_generator = NormalNoiseSignalGenerator()
            # raise ValueError("signal_generator must be provided")

        self.signal_generator = signal_generator
        self.samples_num = samples_num
        self.seq_length = seq_length
        self.lag_length = lag_length
        self.lag_type = lag_type
        super().__init__()

    def __getitem__(self, index):
        X = self.signal_generator.generate(num_signals=1,
                                           signal_length=self.seq_length)
        X = X.unsqueeze(-1)

        if self.lag_type == "zero":
            Y = torch.zeros(X.shape)
            Y[:, self.lag_length:, :] = X[:, :-self.lag_length, :]

        return X[0, :, :], Y[0, :, :]

    def __len__(self):
        return self.samples_num