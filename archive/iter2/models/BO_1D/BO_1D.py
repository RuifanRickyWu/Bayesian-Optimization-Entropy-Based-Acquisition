import gpytorch
import torch
from gpytorch import ExactMarginalLogLikelihood

from archive.iter2.core.gp_model.base_gp_model import ExactGPModel
from archive.iter2.core.dataset.dataset_creator import DatasetCreator
from archive.iter2.core.settings.setup import Setup
from archive.iter2.core.visual_creator.visualization_creator import VisualCreator
from archive.iter2.core.acquisition_function.lcb import LCBAcquisition

#Initializing Components
setup = Setup(__name__)
setup.setup_logger()
logger = setup.get_logger()

dataset_creator = DatasetCreator()
visual_creator = VisualCreator()
lcb_acquisition = LCBAcquisition(0.5)

#Variables

dimension = 1
training_sample_size = 30
training_noise_variance = 0.04
optimizer_iter = 2
gp_iter = 10
scope_lb = torch.tensor([-1])
scope_ub = torch.tensor([1])
bo_iter = 3


train_x, train_y, test_x, test_y, objective_function = dataset_creator.create_training_set(dimension, training_sample_size, training_noise_variance, scope_lb, scope_ub)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

for i in range(bo_iter):
    #Setup and train the model
    logger.info(f"Starting BO Iteration {i}")
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
        loss.backward()
        return loss

    for j in range(gp_iter):
        gp.train()
        likelihood.train()
        loss = optimizer.step(closure)

        #lengthscale logging
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach().numpy()
        noise = gp.likelihood.noise.item()
        logger.info(f"Iter {j + 1}/{gp_iter} - Loss: {loss.item():.3f}  Lengthscale: {lengthscale}  Noise: {noise}")

    #Eval and find new point
    gp.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        next_x = lcb_acquisition.next_point(gp, dimension, scope_lb, scope_ub)
        next_y = dataset_creator.create_new_observation(objective_function, next_x, training_noise_variance).unsqueeze(-1)
        logger.info(f"With BO Iter {i + 1}/{bo_iter} - The next sampled X value is {next_x.detach().numpy()}, the corresponding observed y value is {next_y.detach().numpy()}")
        train_x = torch.cat([train_x, next_x], dim=0)
        print(train_y)
        print(next_y)
        train_y = torch.cat([train_y, next_y], dim=0)


    visual_creator.GP_plotting(dimension, gp, likelihood, train_x, train_y, test_x, test_y, objective_function,
                               scope_lb, scope_ub, 0)