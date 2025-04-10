import logging
import torch
import gpytorch
from torch.distributions import Normal
from torch.special import ndtr
from botorch.acquisition import ProbabilityOfImprovement
from core.acquisition_function.wrapper.bo_torch_wrapper import BOTorchWrapper


class MESAcquisition:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._xi = 0.5
        self._pi_gradient_steps = 10
        self._pi_learning_rate = 0.01
        self._monte_carlo_sample_amount = 500
        self._max_value_sample_amount = 10
        self._initial_points_from_pi = 500

    def _sample_y_star_with_gumbel(self, mu: torch.Tensor, sigma: torch.Tensor, num_samples: int):
        N = mu.shape[0]
        device = mu.device
        normal = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

        gumbel_noise = -torch.log(-torch.log(torch.rand(num_samples, N, device=device)))
        z = mu.unsqueeze(0) + sigma.unsqueeze(0) * normal.sample((num_samples, N))
        gumbelized = -z + gumbel_noise
        min_indices = gumbelized.argmin(dim=1)
        y_star_samples = z[torch.arange(num_samples), min_indices]
        return y_star_samples

    def acq(self, candidate_points, model, likelihood):
        device = candidate_points.device
        N = candidate_points.shape[0]

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = likelihood(model(candidate_points))
            mu = posterior.mean
            sigma = posterior.stddev.clamp(min=1e-6)

        y_star_samples = self._sample_y_star_with_gumbel(mu, sigma, self._max_value_sample_amount)

        z = (y_star_samples.view(-1, 1) - mu.view(1, -1)) / sigma.view(1, -1)
        log_cdf = torch.log(torch.clamp(ndtr(z), min=1e-10))
        info_gain = -log_cdf.mean(dim=0)

        return info_gain

    def find_next_point(self, num_points: int, dimension: int,
                        scope_lower_bound: torch.Tensor, scope_upper_bound: torch.Tensor,
                        model, likelihood):
        device = scope_lower_bound.device
        starting_candidates = torch.rand(self._initial_points_from_pi, dimension, device=device) * \
                              (scope_upper_bound - scope_lower_bound) + scope_lower_bound
        ending_candidate = []

        model.eval()
        likelihood.eval()

        wrapped_model = BOTorchWrapper(model, likelihood)
        train_y = model.train_targets
        best_f = train_y.min()

        pi_acq = ProbabilityOfImprovement(wrapped_model, best_f=best_f, maximize=False)

        with torch.no_grad():
            pi_vals = pi_acq(starting_candidates.unsqueeze(1))

        top_K_indices = torch.topk(pi_vals.squeeze(-1), k=num_points).indices
        pi_candidate = starting_candidates[top_K_indices]

        info_gain = self.acq(pi_candidate, model, likelihood)
        best_idx = torch.argmax(info_gain)
        best_x = pi_candidate[best_idx].detach()

        for candidate, info in zip(pi_candidate, info_gain):
            ending_candidate.append((candidate.detach(), info))

        return best_x, pi_candidate, ending_candidate
