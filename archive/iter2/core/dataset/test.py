import torch
import torch.nn.functional as f
from torch.utils.data import random_split, TensorDataset


class DatasetCreator:
    def __init__(self):
        pass

    def get_objective_function(self, dimension: int) -> f:
        if dimension == 1:
            return lambda x: torch.sin(x * torch.pi * 2)
        elif dimension == 2:
            return lambda x, y: (x - 3) ** 2 + 2 * x * y + (2 * y + 3) ** 2 - 3
        else:
            raise ValueError("Dimension not defined")

    def create_training_set(self, dimension: int, n_samples: int, training_noise_variance: float):
        objective_function = self.get_objective_function(dimension)
        #Hardcoding starting ending edge for x_values?
        x_values = [torch.linspace(-5, 5, n_samples) for _ in range(dimension)]
        meshgrid = torch.meshgrid(*x_values, indexing="ij")
        train_x = torch.stack([g.flatten() for g in meshgrid], dim=1)
        train_y = objective_function(*meshgrid).flatten()
        noise = torch.randn_like(train_y) * training_noise_variance
        train_y += noise

        test_x = train_x.clone()
        test_y = train_y.clone()


        dataset = TensorDataset(train_x, train_y)

        test_size = int(0.2 * len(dataset))  # 20% test size
        train_size = len(dataset) - test_size  # 80% train size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_x, train_y = map(torch.stack, zip(*train_dataset))

        return train_x, train_y, test_x, test_y, objective_function
