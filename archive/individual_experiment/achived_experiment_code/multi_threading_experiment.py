import concurrent.futures
import math
import traceback

import gpytorch
import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
import time
import csv
import os

from core.gp_model.base_gp_model import ExactGPModel
from core.settings.setup import Setup
from core.dataset.dataset_creator import DatasetCreator
from core.acquisition_function.mes import MESAcquisition
from core.acquisition_function.lcb import LCBAcquisition
from core.acquisition_function.es import ESAcquisition

#Parameters
training_noise = 0.04
optimizer_iter = 3
gp_iter = 10
bo_iter = 1

training_sample_size = 10
testing_sample_size = 300
scope_lb = torch.tensor([-1, -1, -1, -1, -1])
scope_ub = torch.tensor([1, 1, 1, 1, 1])
#scope_lb = torch.tensor([-1])
#scope_ub = torch.tensor([1])
dimension = len(scope_lb)


setup = Setup(__name__)
setup.setup_logger()
logger = setup.get_logger()


dataset = DatasetCreator(training_noise, dimension)
mes = MESAcquisition()
es = ESAcquisition()
lcb = LCBAcquisition(2)

train_x, train_y = dataset.create_samples(dimension, training_sample_size, scope_ub, scope_lb)
test_x, test_y = dataset.create_samples(dimension, testing_sample_size, scope_ub, scope_lb)
objective_function = dataset.get_objective_function()

def run_experiment_with_acquisition(acquisition_function, name):
    local_train_x = train_x.clone()
    local_train_y = train_y.clone()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    results = []

    min_value, min_index = torch.min(local_train_y, dim=0)
    f_best = min_value
    f_best_x = local_train_x[min_index]
    number_of_improvement = 0
    for i in range(bo_iter):
        logger.info(f"[{name}] Starting BO Iteration {i}")

        gp = ExactGPModel(local_train_x, local_train_y, likelihood)
        optimizer = torch.optim.LBFGS(
            gp.parameters(),
            lr=0.1,
            max_iter=optimizer_iter
        )
        mll = ExactMarginalLogLikelihood(likelihood, gp)

        def closure():
            optimizer.zero_grad()
            output = gp(local_train_x)
            loss = -mll(output, local_train_y)
            loss.backward(retain_graph=True)
            return loss

        final_loss = None
        for j in range(gp_iter):
            gp.train()
            likelihood.train()
            loss = optimizer.step(closure)
            final_loss = loss.item()
            lengthscale = gp.covar_module.base_kernel.lengthscale.detach().numpy()
            noise = gp.likelihood.noise.item()
            logger.info(f"[{name}] Iter {j + 1}/{gp_iter} - Loss: {loss.item():.3f}  Lengthscale: {lengthscale}  Noise: {noise}")

        gp.eval()
        likelihood.eval()
        next_x, _, _ = acquisition_function.find_next_point(
            3, dimension, scope_lb, scope_ub, gp, likelihood
        )
        next_y = dataset.make_observation(next_x)
        local_train_x = torch.cat([local_train_x, next_x.unsqueeze(0)], dim=0)
        local_train_y = torch.cat([local_train_y, next_y.unsqueeze(0)], dim=0)

        current_observation = next_y.unsqueeze(0)
        if current_observation < f_best:
            f_best = current_observation
            f_best_x = next_x.unsqueeze(0)
            number_of_improvement +=1



        results.append({
            "iteration": i,
            "number_of_points": i + training_sample_size + 1,
            "new_sampled_point_observation" : current_observation,
            "new_sampled_point": next_x.unsqueeze(0),
            "best_observation" : f_best,
            "best_point": f_best_x,
            "number_of_improvement": number_of_improvement
        })
        logger.info(f"[{name}] Iteration {i} with final loss {final_loss:.4f}")

    # Save results
    iterations = np.array([r["iteration"] for r in results])
    number_of_points = np.array([r["number_of_points"] for r in results])
    new_sampled_observations = torch.stack([r["new_sampled_point_observation"] for r in results])
    new_sampled_points = torch.cat([r["new_sampled_point"] for r in results])
    best_observations = torch.stack([r["best_observation"] for r in results])
    best_points = torch.stack([r["best_point"] for r in results])
    num_improvements = np.array([r["number_of_improvement"] for r in results])

    npz_path = f"bo_experiment_results_{name}.npz"
    np.savez(
        npz_path,
        iterations=iterations,
        number_of_points=number_of_points,
        new_sampled_observations=new_sampled_observations.numpy(),
        new_sampled_points=new_sampled_points.numpy(),
        best_observations=best_observations.numpy(),
        best_points=best_points.numpy(),
        number_of_improvement=num_improvements
    )
    logger.info(f"[{name}] Results saved to {os.path.abspath(npz_path)}")


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        futures.append(executor.submit(run_experiment_with_acquisition, lcb, "LCB"))
        #futures.append(executor.submit(run_experiment_with_acquisition, mes, "MES"))
        #futures.append(executor.submit(run_experiment_with_acquisition, es, "ES"))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                trace = traceback.format_exc()
                logger.error(trace)
                logger.error(f"Experiment failed: {e}")