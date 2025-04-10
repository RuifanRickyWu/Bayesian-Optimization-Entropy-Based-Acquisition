import logging

import gpytorch
import torch
from gpytorch import ExactMarginalLogLikelihood
from torch.optim import Adam

from archive.iter2.core.gp_model.base_gp_model import ExactGPModel
from archive.iter2.core.dataset.dataset_creator import DatasetCreator
from archive.iter2.core.visual_creator.visualization_creator import VisualCreator
from archive.gp_tutorial.dataset.sample_1d_dataset import Dataset1D


class BayesianOptimization:

    _dataset_creator: DatasetCreator
    _visual_creator: VisualCreator

    def __init__(self):
        self.setup_logger()
        self.logger = logging.getLogger(self.__class__.__name__)


        self._dataset_creator = DatasetCreator()
        self._visual_creator = VisualCreator()
        self.d1d = Dataset1D()

    def GP_nD_adam(self, gp_iter: int, optimizer_iter: int, training_sample_size: int, training_noise_variance: float, dimension: int):
        train_x, train_y, test_x, test_y = self._dataset_creator.create_training_set(dimension, training_sample_size, training_noise_variance)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp = ExactGPModel(train_x, train_y, likelihood)

        optimizer = Adam([{'params': gp.parameters()}], lr=0.1)
        mll = ExactMarginalLogLikelihood(likelihood, gp)

        for i in range(gp_iter):
            gp.train()
            likelihood.train()
            optimizer.zero_grad()
            pred = gp(train_x)
            loss = -mll(pred, train_y)
            loss.backward()

            # Logging
            lengthscale = gp.covar_module.base_kernel.lengthscale.detach().numpy()
            noise = gp.likelihood.noise.item()

            self.logger.info(f"Iter {i + 1}/{gp_iter} - Loss: {loss.item():.3f}  Lengthscale: {lengthscale}  Noise: {noise}")
            optimizer.step()


    def GP_nD(self, gp_iter: int, optimizer_iter: int, training_sample_size: int, training_noise_variance: float, dimension: int):
        train_x, train_y, test_x, test_y, objective_function = self._dataset_creator.create_training_set(dimension, training_sample_size, training_noise_variance, -1, 1)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp = ExactGPModel(train_x, train_y, likelihood)
        optimizer = torch.optim.LBFGS(
            gp.parameters(),
            lr=0.1,
            max_iter=optimizer_iter
        )
        mll = ExactMarginalLogLikelihood(likelihood, gp)

        def closure():
            optimizer.zero_grad()
            output = gp(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            return loss

        for i in range(gp_iter):
            gp.train()
            likelihood.train()
            loss = optimizer.step(closure)

            # Logging
            lengthscale = gp.covar_module.base_kernel.lengthscale.detach().numpy()
            noise = gp.likelihood.noise.item()

            self.logger.info(f"Iter {i + 1}/{gp_iter} - Loss: {loss.item():.3f}  Lengthscale: {lengthscale}  Noise: {noise}")

        gp.eval()
        likelihood.eval()

        self._visual_creator.GP_plotting(dimension, gp, likelihood, train_x, train_y, test_x, test_y, objective_function)



    def setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("app.log", mode="w")
            ]
        )


bo = BayesianOptimization()
bo.GP_nD(10, 2, 30, 0.04, 1)
#bo.GP_nD(10, 2, 10, 0.04, 2)
#bo.GP_nD_adam(10, 2, 100, 0.04, 1)