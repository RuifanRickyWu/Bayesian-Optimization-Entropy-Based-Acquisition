import logging
import math

import torch
import torch.nn.functional as f
from torch.utils.data import random_split, TensorDataset


class DatasetCreator:

    _noise_variance: float
    _dimension: int
    _objective_function: f
    _dataset_random_seed: int

    def __init__(self, training_noise_variance: float, dimension: int, objective_function_name = "schwefel", dataset_random_seed = 30):
        self._noise_variance = training_noise_variance
        self._dimension = dimension
        self._objective_function = self._create_objective_function(objective_function_name, dimension)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._dataset_random_seed = dataset_random_seed


    def get_objective_function(self) -> f:
        return self._objective_function

    def _create_objective_function(self, func_type: str, dimension: int):
        if dimension not in (1, 5, 10, 20, 30):
            raise ValueError("only support 1, 5, 10, 20, 30 dimension.")

        func_type = func_type.lower()

        if func_type == "rosenbrock":
            return lambda *x: sum(
                [100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(dimension - 1)],
                torch.zeros_like(x[0])
            )

        elif func_type == "ackley":
            return lambda *x: (
                    -20 * torch.exp(-0.2 * torch.sqrt(torch.stack([xi ** 2 for xi in x], dim=0).sum(dim=0) / dimension))
                    - torch.exp(torch.stack([torch.cos(2 * torch.pi * xi) for xi in x], dim=0).sum(dim=0) / dimension)
                    + 20 + torch.tensor(math.e, dtype=x[0].dtype, device=x[0].device)
            )

        elif func_type == "schwefel":
            return lambda *x: 418.9829 * dimension - torch.stack(
                [xi * torch.sin(torch.sqrt(torch.abs(xi))) for xi in x], dim=0
            ).sum(dim=0)

        else:
            raise ValueError("Undefined function type, should be in 'rosenbrock', 'ackley' or 'schwefel'.")


    def create_samples(self, dimension: int,  n_samples: int, sample_upper_bound: torch.Tensor, sample_lower_bound: torch.Tensor):
        generator = torch.Generator()
        generator.manual_seed(self._dataset_random_seed)
        sample_x = torch.rand(n_samples, dimension, generator=generator) * (sample_upper_bound - sample_lower_bound) + sample_lower_bound
        sample_x, _ = torch.sort(sample_x, dim=0)
        sample_y = self._objective_function(*[sample_x[:, i] for i in range(dimension)])
        noise = torch.randn_like(sample_y) * self._noise_variance
        sample_y += noise

        return sample_x, sample_y

    def make_observation(self, x):
        raw_observation = self._objective_function(*x)
        noise = torch.randn_like(raw_observation) * self._noise_variance
        new_observation = raw_observation+ noise
        self.logger.info(f"New Observation is ->: {new_observation}")
        return new_observation


