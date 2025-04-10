import gpytorch
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
from core.acquisition_function.pi import PIAcquisition

#Parameters
training_noise = 0.04
optimizer_iter = 3
gp_iter = 10
bo_iter = 10

training_sample_size = 5
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
lcb = LCBAcquisition(2)
pi = PIAcquisition(2)

train_x, train_y = dataset.create_samples(dimension, training_sample_size, scope_ub, scope_lb)
test_x, test_y = dataset.create_samples(dimension, testing_sample_size, scope_ub, scope_lb)
objective_function = dataset.get_objective_function()

likelihood = gpytorch.likelihoods.GaussianLikelihood()

# For storing results
results = []


def single_experiment(acquisition_function, train_x, train_y):
    results = []
    for i in range(bo_iter):
        start_time = time.time()
        logger.info(f"Starting BO Iteration {i}")

        # GP setup
        gp = ExactGPModel(train_x, train_y, likelihood)
        optimizer = torch.optim.LBFGS(
            gp.parameters(),
            lr=0.1,
            max_iter=optimizer_iter)
        mll = ExactMarginalLogLikelihood(likelihood, gp)

        def closure():
            optimizer.zero_grad()
            output = gp(train_x)
            loss = -mll(output, train_y)
            loss.backward(retain_graph=True)
            return loss

        # GP training loop
        final_loss = None
        for j in range(gp_iter):
            gp.train()
            likelihood.train()
            loss = optimizer.step(closure)
            final_loss = loss.item()
            # lengthscale logging
            lengthscale = gp.covar_module.base_kernel.lengthscale.detach().numpy()
            noise = gp.likelihood.noise.item()
            logger.info(f"Iter {j + 1}/{gp_iter} - Loss: {loss.item():.3f}  Lengthscale: {lengthscale}  Noise: {noise}")

        # BO acquisition
        gp.eval()
        likelihood.eval()
        next_x, starting_candidate, ending_candidate = acquisition_function.find_next_point(
            3, dimension, scope_lb, scope_ub, gp, likelihood)
        next_y = dataset.make_observation(next_x)
        train_x = torch.cat([train_x, next_x.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, next_y.unsqueeze(0)], dim=0)

        # Record time and loss
        elapsed_time = time.time() - start_time
        results.append({
            "iteration": i,
            "final_loss": final_loss,
            "time_sec": elapsed_time
        })
        logger.info(f"Iteration {i} completed in {elapsed_time:.2f}s with final loss {final_loss:.4f}")

    # Save results to CSV
    results_file = "bo_experiment_results.csv"
    with open(results_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "final_loss", "time_sec"])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to {os.path.abspath(results_file)}")


single_experiment(pi, train_x, train_y)
