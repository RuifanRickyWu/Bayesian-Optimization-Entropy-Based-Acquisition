import gpytorch
import torch
from gpytorch import ExactMarginalLogLikelihood

from archive.iter2.core.gp_model.base_gp_model import ExactGPModel
from archive.iter2.core.dataset.dataset_creator import DatasetCreator
from archive.iter2.core.settings.setup import Setup
from archive.iter2.core.visual_creator.visualization_creator import VisualCreator

#Initializing Components
setup = Setup(__name__)
setup.setup_logger()
logger = setup.get_logger()

dataset_creator = DatasetCreator()
visual_creator = VisualCreator()

#Variables

dimension = 2
training_sample_size = 20
training_noise_variance = 0.04
optimizer_iter = 2
gp_iter = 10
scope_lb = torch.tensor([-5, -5])
scope_ub = torch.tensor([5, 5])


train_x, train_y, test_x, test_y, objective_function = dataset_creator.create_training_set(dimension, training_sample_size, training_noise_variance, scope_lb, scope_ub)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
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

for i in range(gp_iter):
    gp.train()
    likelihood.train()
    loss = optimizer.step(closure)

    # Logging
    lengthscale = gp.covar_module.base_kernel.lengthscale.detach().numpy()
    noise = gp.likelihood.noise.item()

    logger.info(f"Iter {i + 1}/{gp_iter} - Loss: {loss.item():.3f}  Lengthscale: {lengthscale}  Noise: {noise}")

gp.eval()
likelihood.eval()

visual_creator.GP_plotting(dimension, gp, likelihood, train_x, train_y, test_x, test_y, objective_function, scope_lb, scope_ub, training_sample_size)