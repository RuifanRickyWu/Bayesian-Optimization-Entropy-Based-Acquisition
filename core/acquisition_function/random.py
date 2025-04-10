import logging

import numpy as np
import torch


class RandomAcquisition:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def find_next_point(self, num_trial: int, dimension: int, scope_lower_bound: torch.tensor,
                        scope_upper_bound: torch.tensor, model, likelihood):
        next_random_point = (scope_upper_bound - scope_lower_bound) * torch.rand(dimension) + scope_lower_bound

        return next_random_point, None, None
