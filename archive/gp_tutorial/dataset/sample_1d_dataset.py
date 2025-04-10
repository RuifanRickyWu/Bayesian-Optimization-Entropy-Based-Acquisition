import math
import torch


class Dataset1D:

    def __init__(self):
        pass

    def get_initial_training_set(self, n_samples: int, noise_variance: float):
        # True function is sin(2*pi*x) with Gaussian noise
        train_x = torch.linspace(0, 1, n_samples)
        noise = torch.randn(n_samples) * torch.sqrt(torch.tensor(noise_variance))
        train_y = torch.sin(train_x * (2 * math.pi)) + noise
        return train_x, train_y

    def get_observation(self, x: torch.Tensor, noise_variance):
        x1 = x
        noise = torch.randn(1) * torch.sqrt(torch.tensor(noise_variance))
        y = torch.sin(x1 * (2 * math.pi)) + noise
        return y