import torch
import torch.nn.functional as f
from torch.utils.data import random_split, TensorDataset


class DatasetCreator:
    def __init__(self):
        pass

    def _get_objective_function(self, dimension: int) -> f:
        if dimension == 1:
            return lambda x: torch.sin(x * torch.pi * 2)
        elif dimension == 2:
            return lambda x, y: (x - 3) ** 2 + 2 * x * y + (2 * y + 3) ** 2 - 3
        else:
            raise ValueError("Dimension not defined")

    def create_training_set(self, dimension: int, n_samples: int, training_noise_variance: float, scope_lb: torch.Tensor, scope_up: torch.Tensor):
        objective_function = self._get_objective_function(dimension)
        x_values = [torch.linspace(scope_lb[i].item(), scope_up[i].item(), n_samples) for i in range(dimension)]
        meshgrid = torch.meshgrid(*x_values, indexing="ij")
        train_x = torch.stack([g.flatten() for g in meshgrid], dim=1)
        train_y = objective_function(*meshgrid).flatten()
        noise = torch.randn_like(train_y) * training_noise_variance
        train_y += noise

        test_x = train_x.clone()
        test_y = train_y.clone()


        dataset = TensorDataset(train_x, train_y)

        test_size = int(0.2 * len(dataset))
        train_size = len(dataset) - test_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_x, train_y = map(torch.stack, zip(*train_dataset))

        return train_x, train_y, test_x, test_y, objective_function

    def create_new_observation(self, objective_function, x, observation_noise):
        observed_y = objective_function(x)
        noise = torch.randn_like(observed_y) * observation_noise
        return observed_y + noise


