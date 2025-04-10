import multiprocessing
import concurrent.futures
import traceback
import os
import csv
import torch
import gpytorch
from gpytorch import ExactMarginalLogLikelihood

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

def create_acquisition_function(name):
    return {
        "LCB": LCBAcquisition(0.01),
        "EI": EIAcquisition(),
        "PI": PIAcquisition(0.01),
        "ES": ESAcquisition(),
        "PES": PESAcquisition(),
        "MES": MESAcquisition(),
        "Random": RandomAcquisition()
    }[name]

def run_experiment_with_acquisition_by_name(acq_name: str, turn: int):
    try:
        setup = Setup(__name__)
        setup.setup_logger()
        logger = setup.get_logger()

        training_noise = 0.01
        optimizer_iter = 5
        gp_iter = 10
        bo_iter = 200
        training_sample_size = 20
        scope_lb = torch.full((20,), -1.0)
        scope_ub = torch.full((20,), 1.0)
        dimension = len(scope_lb)

        dataset = DatasetCreator(training_noise, dimension, dataset_random_seed=15)
        train_x, train_y = dataset.create_samples(dimension, training_sample_size, scope_ub, scope_lb)
        acquisition_function = create_acquisition_function(acq_name)

        local_train_x = train_x.clone()
        local_train_y = train_y.clone()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        results = []

        f_best = torch.tensor(float("inf"))
        f_best_x = None
        number_of_improvement = 0

        for i in range(bo_iter):
            logger.info(f"[{acq_name}] Starting BO Iteration {i}")

            gp = ExactGPModel(local_train_x, local_train_y, likelihood)
            optimizer = torch.optim.LBFGS(gp.parameters(), lr=0.01, max_iter=optimizer_iter)
            mll = ExactMarginalLogLikelihood(likelihood, gp)

            def closure():
                optimizer.zero_grad()
                output = gp(local_train_x)
                with gpytorch.settings.cholesky_jitter(1e-3):
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
                    f"[{acq_name}] Iter {j + 1}/{gp_iter} - Loss: {loss.item():.3f}  Lengthscale: {lengthscale}  Noise: {noise}")

            gp.eval()
            likelihood.eval()
            next_x, _, _ = acquisition_function.find_next_point(100, dimension, scope_lb, scope_ub, gp, likelihood)
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
            logger.info(f"[{acq_name}] Iteration {i} with final loss {final_loss:.4f}")

        folder_path = os.path.join("../../results", acq_name)
        os.makedirs(folder_path, exist_ok=True)
        csv_path = os.path.join(folder_path, f"bo_experiment_results_{acq_name}_{turn}.csv")

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

        logger.info(f"[{acq_name}] Results saved to {os.path.abspath(csv_path)}")
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error in acquisition {acq_name} turn {turn}:\n{error_msg}")
        raise e

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    acquisition_names = ["LCB", "EI", "PI", "ES", "PES", "MES", "Random"]

    for i in range(10):
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(acquisition_names)) as executor:
            futures = []
            for name in acquisition_names:
                futures.append(executor.submit(run_experiment_with_acquisition_by_name, name, i))

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Experiment failed: {e}")
