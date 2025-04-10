import logging

import numpy as np
import torch


class LCBAcquisition:
    _kappa: float
    _lr: float
    _gradient_steps: int

    def __init__(self, kappa: float):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._kappa = kappa
        self._lr = 0.001
        self._gradient_steps = 10

    def acq(self, x, model, likelihood):
        model.eval()
        likelihood.eval()
        pred = likelihood(model(x))
        mean = pred.mean
        std = pred.variance.sqrt()
        return mean - self._kappa * std

    def find_next_point(self, num_trial: int, dimension: int, scope_lower_bound: torch.tensor,
                        scope_upper_bound: torch.tensor, model, likelihood):
        best_x = None
        best_acq = float("inf")
        starting_candidates = torch.rand(num_trial, dimension) * (
                    scope_upper_bound - scope_lower_bound) + scope_lower_bound
        ending_candidates = []

        for x_init in starting_candidates:
            x_init = x_init.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([x_init], lr=self._lr)
            for _ in range(self._gradient_steps):
                optimizer.zero_grad()
                loss = self.acq(x_init.view(1, -1), model, likelihood)
                loss.backward()
                optimizer.step()
                x_init.data = torch.clamp(x_init.data, scope_lower_bound, scope_upper_bound)

            final_acq_value = self.acq(x_init.view((1, -1)), model, likelihood).item()
            ending_candidates.append((x_init.detach(), final_acq_value))
            if final_acq_value < best_acq:
                best_x = x_init.detach()
                best_acq = final_acq_value
        self.logger.info(f"All Ending Candidates ->: {ending_candidates}")
        self.logger.info(f"Selected Point ->: {best_x} and the Corresponding Acq Value: {best_acq}")
        return best_x, starting_candidates, ending_candidates
