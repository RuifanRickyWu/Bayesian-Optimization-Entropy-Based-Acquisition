import numpy
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split


class Dataset2D:
    _initial_training_x: torch.tensor
    _initial_training_y: torch.tensor
    _test_set_x: torch.tensor
    _test_set_y: torch.tensor
    _true_objective: numpy.ndarray
    _full_set_x1: torch.tensor
    _full_set_x2: torch.tensor
    _full_set_y: torch.tensor

    def __init__(self, sqrt_n_sample: int, noise_variance: float):
        # the true objective function is
        #   (x1−3)2+2x1x2+(2x2+3)2−3
        # Create a meshgrid for x1 and x2 similar to the numpy version
        np.random.seed(17)
        xv = torch.linspace(-5., 5., sqrt_n_sample)
        yv = torch.linspace(-5., 5., sqrt_n_sample)
        x1, x2 = torch.meshgrid(xv, yv, indexing="ij")

        self._full_set_x1 = x1
        self._full_set_x2 = x2

        # Noise and data generation
        noise = torch.randn(x1.shape) * torch.sqrt(torch.tensor(noise_variance))

        # Compute train_y based on the given formula
        train_y = (x1 - 3) ** 2 + 2 * x1 * x2 + (2 * x2 + 3) ** 2 - 3 + noise
        self._true_objective = ((x1 - 3) ** 2 + 2 * x1 * x2 + (2 * x2 + 3) ** 2 - 3).numpy()

        # Stack the inputs into a single tensor
        train_x = torch.stack((x1, x2), dim=-1)
        train_x = train_x.reshape(-1, 2)
        train_y = train_y.reshape(-1)

        # Combine train_x and train_y into a TensorDataset
        dataset = TensorDataset(train_x, train_y)

        # Define split sizes
        test_size = int(0.8 * len(dataset))  # 95% for testing
        train_size = len(dataset) - test_size
        self._test_set_x = train_x

        # Perform random split
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Extract tensors from the split datasets
        self._initial_training_x, self._initial_training_y = train_dataset[:]

    def get_initial_training_set(self):
        return self._initial_training_x, self._initial_training_y

    def get_full_test_set(self):
        return self._test_set_x

    def get_observation(self, x: torch.Tensor, noise_variance):
        x1 = x[:, 0]
        x2 = x[:, 1]
        noise = torch.randn(1) * torch.sqrt(torch.tensor(noise_variance))
        y = (x1 - 3) ** 2 + 2 * x1 * x2 + (2 * x2 + 3) ** 2 - 3 + noise
        return y

    def get_objective(self):
        return self._true_objective

    def get_axis(self):
        return self._full_set_x1, self._full_set_x2

