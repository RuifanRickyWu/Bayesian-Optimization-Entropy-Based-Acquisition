import logging
import copy

import gpytorch
import torch
from botorch.acquisition import ProbabilityOfImprovement
from core.acquisition_function.wrapper.bo_torch_wrapper import BOTorchWrapper


class PESAcquisition:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._xi = 0.5
        self._pi_gradient_steps = 10
        self._pi_learning_rate = 0.01
        self._monte_carlo_sample_amount = 500
        self._observation_for_each_monte_carlo_process = 5
        self._initial_points_from_pi = 500

    def _sample_fmin_values(self, model, likelihood, candidate_points, S):
        try:
            model.eval()
            pred = likelihood(model(candidate_points))
            samples = pred.rsample(sample_shape=torch.Size([S]))  # [S, N]
            if torch.isnan(samples).any():
                self.logger.warning("NaN detected in GP samples")
                return torch.full((S,), float("inf"), device=candidate_points.device)
            min_values, _ = torch.min(samples, dim=1)  # [S]
            return min_values
        except Exception as e:
            self.logger.error(f"Sampling fmin values failed: {e}")
            return torch.full((S,), float("inf"), device=candidate_points.device)

    def _compute_entropy_of_min_values(self, min_values: torch.Tensor):
        min_val = min_values.min().item()
        max_val = min_values.max().item()

        if torch.isclose(torch.tensor(min_val), torch.tensor(max_val), atol=1e-6):
            return torch.tensor(0.0, device=min_values.device)

        hist = torch.histc(min_values, bins=30, min=min_val, max=max_val)
        p = hist / hist.sum()
        p = p.clamp(min=1e-12)
        entropy = -torch.sum(p * torch.log(p))

        if not torch.isfinite(entropy):
            self.logger.warning(f"Entropy is not finite: {entropy.item()}")
            entropy = torch.tensor(0.0, device=min_values.device)

        return entropy

    def _simulate_evaluation_with_gp_update(self, model, likelihood, train_X, train_Y, candidate_x, y_sim):
        new_train_X = torch.cat([train_X, candidate_x.unsqueeze(0)], dim=0)
        new_train_Y = torch.cat([train_Y, y_sim.unsqueeze(0)], dim=0)

        model_copy = copy.deepcopy(model)
        likelihood_copy = copy.deepcopy(likelihood)

        try:
            model_copy.set_train_data(new_train_X, new_train_Y, strict=False)

            model_copy.train()
            likelihood_copy.train()

            optimizer = torch.optim.Adam([
                {'params': model_copy.parameters()},
                {'params': likelihood_copy.parameters()},
            ], lr=0.01)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_copy, model_copy)

            for _ in range(10):
                optimizer.zero_grad()
                output = model_copy(new_train_X)
                if torch.isnan(output.mean).any():
                    self.logger.warning("NaN in GP output during retraining.")
                    return model, likelihood
                with gpytorch.settings.cholesky_jitter(1e-3):
                    loss = -mll(output, new_train_Y)
                if not torch.isfinite(loss):
                    self.logger.warning("NaN/Inf loss during GP update.")
                    return model, likelihood
                loss.backward()
                optimizer.step()

            model_copy.eval()
            likelihood_copy.eval()
            return model_copy, likelihood_copy

        except Exception as e:
            self.logger.error(f"GP retrain failed with exception: {e}")
            return model, likelihood

    def acq(self, candidate_points, model, likelihood):
        device = candidate_points.device
        N = candidate_points.shape[0]

        model.eval()
        likelihood.eval()

        train_X = model.train_inputs[0]
        train_Y = model.train_targets

        min_values = self._sample_fmin_values(model, likelihood, candidate_points, self._monte_carlo_sample_amount)
        H_fmin = self._compute_entropy_of_min_values(min_values)

        info_gain = torch.zeros(N, device=device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = likelihood(model(candidate_points))
            mu = posterior.mean
            sigma = posterior.stddev

        for i in range(N):
            candidate_x = candidate_points[i]
            mu_i = mu[i]
            sigma_i = sigma[i]

            if not torch.isfinite(mu_i) or not torch.isfinite(sigma_i) or sigma_i <= 0:
                self.logger.warning(f"Skipping candidate {i} due to invalid mean or std.")
                continue

            y_sim_values = torch.normal(mu_i.expand(self._observation_for_each_monte_carlo_process),
                                        sigma_i.expand(self._observation_for_each_monte_carlo_process))

            H_cond_sum = 0.0
            for y_sim in y_sim_values:
                new_model, new_likelihood = self._simulate_evaluation_with_gp_update(
                    model, likelihood, train_X, train_Y, candidate_x, y_sim
                )
                min_values_cond = self._sample_fmin_values(
                    new_model, new_likelihood, candidate_points, self._monte_carlo_sample_amount
                )
                H_cond = self._compute_entropy_of_min_values(min_values_cond)
                H_cond_sum += H_cond

            H_cond_avg = H_cond_sum / self._observation_for_each_monte_carlo_process
            info_gain[i] = H_fmin - H_cond_avg

        return info_gain

    def find_next_point(self, num_points: int, dimension: int, scope_lower_bound: torch.Tensor,
                        scope_upper_bound: torch.Tensor, model, likelihood):
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
            pi_vals = pi_acq(starting_candidates.unsqueeze(1))  # shape: [M, 1]

        top_K_indices = torch.topk(pi_vals.squeeze(-1), k=num_points).indices
        pi_candidate = starting_candidates[top_K_indices]

        info_gain = self.acq(pi_candidate, model, likelihood)

        if torch.isnan(info_gain).any() or not torch.isfinite(info_gain).all():
            self.logger.warning("Invalid info gain detected. Falling back to first candidate.")
            return pi_candidate[0].detach(), pi_candidate, [(x.detach(), 0.0) for x in pi_candidate]

        best_idx = torch.argmax(info_gain)
        best_x = pi_candidate[best_idx].detach()

        for candidate, info in zip(pi_candidate, info_gain):
            ending_candidate.append((candidate.detach(), info))

        return best_x, pi_candidate, ending_candidate
