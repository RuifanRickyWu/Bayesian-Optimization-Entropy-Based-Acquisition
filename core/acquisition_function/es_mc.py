import logging
import copy

import gpytorch
import torch
import torch.distributions
from botorch.acquisition import ProbabilityOfImprovement
from core.acquisition_function.wrapper.bo_torch_wrapper import BOTorchWrapper


class ESAcquisition:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._xi = 0.5
        self._pi_gradient_steps = 10
        self._pi_learning_rate = 0.01
        self._monte_carlo_sample_amount = 500
        self._observation_for_each_monte_carlo_process = 5
        self._initial_points_from_pi = 500

    def _sample_GP_functions(self, model, likelihood, candidates, S):
        model.eval()
        pred = likelihood(model(candidates))
        samples = pred.rsample(sample_shape=torch.Size([S]))
        return samples

    def _compute_p_min_from_samples(self, samples):
        S, N = samples.shape
        min_indices = torch.argmin(samples, dim=1)
        counts = torch.zeros(N, device=samples.device)
        for idx in min_indices:
            counts[idx] += 1
        p_min = counts / S
        return p_min

    def _compute_entropy(self, p):
        p_clamped = p.clamp(min=1e-12)
        return -torch.sum(p_clamped * torch.log(p_clamped))

    def _simulate_evaluation_with_gp_update(self, model, likelihood, train_X, train_Y, candidate_x, y_sim):
        new_train_X = torch.cat([train_X, candidate_x.unsqueeze(0)], dim=0)
        new_train_Y = torch.cat([train_Y, y_sim.unsqueeze(0)], dim=0)

        model_copy = copy.deepcopy(model)
        likelihood_copy = copy.deepcopy(likelihood)

        model_copy.set_train_data(new_train_X, new_train_Y, strict=False)

        model_copy.train()
        likelihood_copy.train()

        optimizer = torch.optim.Adam([
            {'params': model_copy.parameters()},
            {'params': likelihood_copy.parameters()},
        ], lr=0.01)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_copy, model_copy)

        training_iter = 10
        for _ in range(training_iter):
            optimizer.zero_grad()
            output = model_copy(new_train_X)
            loss = -mll(output, new_train_Y)
            loss.backward()
            optimizer.step()

        model_copy.eval()
        likelihood_copy.eval()

        return model_copy, likelihood_copy

    def acq(self, candidate_points, model, likelihood):
        device = candidate_points.device
        N = candidate_points.shape[0]

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = likelihood(model(candidate_points))
            mu = posterior.mean
            sigma = posterior.stddev

        samples = self._sample_GP_functions(model, likelihood, candidate_points, self._monte_carlo_sample_amount)
        current_p_min = self._compute_p_min_from_samples(samples)
        H0 = self._compute_entropy(current_p_min)

        info_gain = torch.zeros(N, device=device)

        train_X = model.train_inputs[0]
        train_Y = model.train_targets

        for i in range(N):
            mu_i = mu[i]
            sigma_i = sigma[i]
            candidate_x = candidate_points[i]
            y_sim_values = torch.normal(mu_i.expand(self._observation_for_each_monte_carlo_process), sigma_i.expand(self._observation_for_each_monte_carlo_process))

            H_avg = 0.0
            for y_sim in y_sim_values:
                new_model, new_likelihood = self._simulate_evaluation_with_gp_update(
                    model, likelihood, train_X, train_Y, candidate_x, y_sim
                )

                new_samples = self._sample_GP_functions(new_model, new_likelihood, candidate_points, self._monte_carlo_sample_amount)

                new_p_min = self._compute_p_min_from_samples(new_samples)
                H_y = self._compute_entropy(new_p_min)
                H_avg += H_y

            H_avg /= self._observation_for_each_monte_carlo_process
            info_gain[i] = H0 - H_avg

        return info_gain

    def find_next_point(self, num_points: int, dimension: int, scope_lower_bound: torch.Tensor, scope_upper_bound: torch.Tensor, model, likelihood):
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
        best_idx = torch.argmax(info_gain)
        best_x = pi_candidate[best_idx].detach()

        for candidate, info in zip(pi_candidate, info_gain):
            ending_candidate.append((candidate.detach(), info))

        return best_x, pi_candidate, ending_candidate