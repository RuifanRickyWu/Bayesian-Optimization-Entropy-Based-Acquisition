import concurrent.futures
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
from core.acquisition_function.es_mc import ESAcquisition
from core.acquisition_function.pi import PIAcquisition
from core.acquisition_function.pes import PESAcquisition
from core.acquisition_function.ei import EIAcquisition
from core.acquisition_function.random import RandomAcquisition

# Parameters
training_noise = 0.04
optimizer_iter = 5
gp_iter = 10
bo_iter = 200

training_sample_size = 10
testing_sample_size = 300
scope_lb = torch.full((5,), -1.0)
scope_ub = torch.full((5,), 1.0)
dimension = len(scope_lb)

setup = Setup(__name__)
setup.setup_logger()
logger = setup.get_logger()

dataset = DatasetCreator(training_noise, dimension, dataset_random_seed= 10)
mes = MESAcquisition()
es = ESAcquisition()
lcb = LCBAcquisition(0.01)
pi = PIAcquisition(0.01)
pes = PESAcquisition()
ei = EIAcquisition()
random = RandomAcquisition()

train_x, train_y = dataset.create_samples(dimension, training_sample_size, scope_ub, scope_lb)
test_x, test_y = dataset.create_samples(dimension, testing_sample_size, scope_ub, scope_lb)
objective_function = dataset.get_objective_function()


def run_experiment_with_acquisition(acquisition_function, name, turn):
    local_train_x = train_x.clone()
    local_train_y = train_y.clone()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    results = []

    f_best = torch.tensor(float("inf"))
    f_best_x = None
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
            logger.info(
                f"[{name}] Iter {j + 1}/{gp_iter} - Loss: {loss.item():.3f}  Lengthscale: {lengthscale}  Noise: {noise}")

        gp.eval()
        likelihood.eval()
        next_x, _, _ = acquisition_function.find_next_point(
            100, dimension, scope_lb, scope_ub, gp, likelihood
        )
        next_y = dataset.make_observation(next_x)
        local_train_x = torch.cat([local_train_x, next_x.unsqueeze(0)], dim=0)
        local_train_y = torch.cat([local_train_y, next_y.unsqueeze(0)], dim=0)

        current_observation = next_y.unsqueeze(0)
        if current_observation.item() < f_best.item():
            f_best = current_observation.clone()
            f_best_x = next_x.unsqueeze(0)
            number_of_improvement += 1

        results.append({
            "iteration": i,
            "number_of_points": i + training_sample_size + 1,
            "new_sampled_point_observation": current_observation,
            "new_sampled_point": next_x.unsqueeze(0),
            "best_observation": f_best,
            "best_point": f_best_x,
            "number_of_improvement": number_of_improvement
        })
        logger.info(f"[{name}] Iteration {i} with final loss {final_loss:.4f}")

    # Create a folder for the acquisition function if it doesn't exist.
    folder_path = os.path.join("/content/results", name)
    os.makedirs(folder_path, exist_ok=True)
    csv_path = os.path.join(folder_path, f"bo_experiment_results_{name}_{turn}.csv")

    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = [
            "iteration",
            "number_of_points",
            "new_sampled_point_observation",
            "new_sampled_point",
            "best_observation",
            "best_point",
            "number_of_improvement"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({
                "iteration": row["iteration"],
                "number_of_points": row["number_of_points"],
                "new_sampled_point_observation": row["new_sampled_point_observation"].item(),
                "new_sampled_point": row["new_sampled_point"].squeeze(0).tolist(),
                "best_observation": row["best_observation"].item(),
                "best_point": row["best_point"].tolist(),
                "number_of_improvement": row["number_of_improvement"]
            })

    logger.info(f"[{name}] Results saved to {os.path.abspath(csv_path)}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    acquisition_functions = [
        (lcb, "LCB"),
        (ei, "EI"),
        (pi, "PI"),
        (es, "ES"),
        (pes, "PES"),
        (mes, "MES"),
        (random, "Random")
    ]

    for i in range(10):
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(acquisition_functions)) as executor:
            futures = []
            for acq_func, name in acquisition_functions:
                futures.append(executor.submit(run_experiment_with_acquisition, acq_func, name, str(i)))

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    trace = traceback.format_exc()
                    logger.error(trace)
                    logger.error(f"Experiment failed: {e}")