from archive.iter2.core.gp_model.base_gp_model import ExactGPModel
import torch


class LCBAcquisition:
    _kappa: float
    def __init__(self, kappa: float):
        self._kappa = kappa

    def _create_random_search_space(self, dimension: int, search_lower_bound: torch.Tensor, search_upper_bound: torch.Tensor,
                                   num_samples: int) -> torch.Tensor:

        return search_lower_bound + (search_upper_bound - search_lower_bound) * torch.rand(num_samples, dimension)


    def next_point(self, model: ExactGPModel, dimension, search_lower_bound, search_upper_bound):
        search_space = self._create_random_search_space(dimension, search_lower_bound, search_upper_bound, 100)
        lcb_values = self.evaluate(model, search_space)
        next_idx = torch.argmin(lcb_values)
        return search_space[next_idx].unsqueeze(0)

    def evaluate(self,model: ExactGPModel, search_space: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            posterior = model(search_space)
            mean = posterior.mean
            std_dev = posterior.stddev

        # LCB formulation: mean - kappa * std_dev
        return mean - self._kappa * std_dev
