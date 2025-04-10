import logging
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Dimension for Training Data -> X: {train_x.shape[0]}")
        self.logger.info(f"Size for Training Data -> X: {train_x.shape[1]}")
        self.logger.info(f"Dimension for Training Result -> y: {train_x.shape[0]}")

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

        self.logger.info(f"Number of ARD dimensions: {self.covar_module.base_kernel.ard_num_dims}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
