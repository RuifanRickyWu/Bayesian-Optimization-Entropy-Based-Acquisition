import gpytorch
import torch
from gpytorch import ExactMarginalLogLikelihood

from core.gp_model.base_gp_model import ExactGPModel
from core.settings.setup import Setup
from core.dataset.dataset_creator import DatasetCreator
from core.visualization_creator.visual_creator import VisualCreator
from core.acquisition_function.pes import PESAcquisition

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
visual = VisualCreator()
pes = PESAcquisition()

train_x, train_y = dataset.create_samples(dimension, training_sample_size, scope_ub, scope_lb)
test_x, test_y = dataset.create_samples(dimension, testing_sample_size, scope_ub, scope_lb)
objective_function = dataset.get_objective_function()

likelihood = gpytorch.likelihoods.GaussianLikelihood()

for i in range(bo_iter):
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
        loss.backward(retain_graph=True)
        return loss

    for j in range(gp_iter):
        gp.train()
        likelihood.train()
        loss = optimizer.step(closure)

        #lengthscale logging
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach().numpy()
        noise = gp.likelihood.noise.item()
        logger.info(f"Iter {j + 1}/{gp_iter} - Loss: {loss.item():.3f}  Lengthscale: {lengthscale}  Noise: {noise}")

    gp.eval()
    likelihood.eval()
    next_x, starting_candidate, ending_candidate = pes.find_next_point(3, dimension, scope_lb, scope_ub, gp, likelihood)
    next_y = dataset.make_observation(next_x)
    train_x = torch.cat([train_x, next_x.unsqueeze(0)], dim=0)
    train_y = torch.cat([train_y, next_y.unsqueeze(0)], dim=0)

    if dimension == 1:
        visual.GP_plotting_1D(gp, likelihood, train_x, train_y, test_x, next_x, next_y, objective_function, scope_lb, scope_ub)
        #visual.aquisition_plotting_1D(es, gp, likelihood, scope_lb, scope_ub, next_x, starting_candidate, ending_candidate)


