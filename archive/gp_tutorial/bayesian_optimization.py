import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from gpytorch.mlls import ExactMarginalLogLikelihood
from archive.gp_tutorial.dataset.sample_2d_dataset import Dataset2D
from archive.gp_tutorial.dataset.sample_1d_dataset import Dataset1D
from archive.gp_tutorial.gp_model.base_gp_model import ExactGPModel
from archive.iter2.core.acquisition_function.lcb import LCBAcquisition
import matplotlib.lines as mlines


class BayesianOptimization:
    _dataset_2d: Dataset2D
    _dataset_1d: Dataset1D
    _lcb_acquisition: LCBAcquisition

    def __init__(self):
        self._dataset_2d = Dataset2D(20, 0.04)
        self._dataset_1d = Dataset1D()
        self._lcb_acquisition = LCBAcquisition(0.5)

    def GP_1D(self, n_iter):
        train_x, train_y = self._dataset_1d.get_initial_training_set(100, 0.04)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_prior = ExactGPModel(train_x, train_y, likelihood)
        optimizer = Adam([{'params': gp_prior.parameters()}], lr=0.1)
        mll = ExactMarginalLogLikelihood(likelihood, gp_prior)

        for i in range(n_iter):
            gp_prior.train()
            likelihood.train()
            optimizer.zero_grad()
            pred = gp_prior(train_x)
            loss = -mll(pred, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, n_iter, loss.item(),
                gp_prior.covar_module.base_kernel.lengthscale.item(),
                gp_prior.likelihood.noise.item()
            ))
            # now the gp_prior becomes posterier
            optimizer.step()
        gp_prior.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 1, 51)
            observed_pred = likelihood(gp_prior(test_x))

        with torch.no_grad():
            f, ax = plt.subplots(1, 1, figsize=(4, 3))
            lower, upper = observed_pred.confidence_region()
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.show()

    def GP_2D(self, n_iter):
        train_x, train_y = self._dataset_2d.get_initial_training_set()
        test_x = self._dataset_2d.get_full_test_set()
        full_x1, full_x2 = self._dataset_2d.get_axis()
        objective_function = self._dataset_2d.get_objective()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_prior = ExactGPModel(train_x, train_y, likelihood)
        optimizer = Adam([{'params': gp_prior.parameters()}], lr=0.1)
        mll = ExactMarginalLogLikelihood(likelihood, gp_prior)

        for i in range(n_iter):
            gp_prior.train()
            likelihood.train()
            optimizer.zero_grad()
            pred = gp_prior(train_x)
            loss = -mll(pred, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, n_iter, loss.item(),
                gp_prior.covar_module.base_kernel.lengthscale.item(),
                gp_prior.likelihood.noise.item()
            ))
            # now the gp_prior becomes posterier
            optimizer.step()
        gp_prior.eval()
        likelihood.eval()

        with torch.no_grad():
            observed_pred = likelihood(gp_prior(test_x))
            test_mean = observed_pred.mean.numpy().reshape(full_x1.shape)
            test_variance = observed_pred.variance.numpy().reshape(full_x1.shape)

        fig, ax = plt.subplots(figsize=(6., 4.))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        ax.set_aspect('equal')
        ax.set_xticks((-5, 0, 5))
        ax.set_yticks((-5, 0, 5))
        ax.grid(False)

        lev = np.linspace(0., 250., 6)
        ax.contour(full_x1, full_x2, objective_function, lev, colors='k')  # Truth
        ax.plot(*train_x.T, 'o', color="C1")  # Training samples
        ax.contour(full_x1, full_x2, test_mean, lev, colors='C0', linestyles='dashed')  # Posterior mean
        truth_line = mlines.Line2D([], [], color='black', label='True $z(x,y)$')
        sample_line = mlines.Line2D([], [], color='C1', marker="o", linestyle="none", label='Train samples')
        mean_line = mlines.Line2D([], [], color='C0', linestyle="--", label='Posterior mean')
        ax.legend(handles=[truth_line, sample_line, mean_line], bbox_to_anchor=(1.05, 1), loc="upper left")

        # Write out
        plt.tight_layout()
        plt.savefig('gpr_posterior_mean.svg')
        plt.show()

        fig, ax = plt.subplots(figsize=(6., 4.))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        ax.set_aspect('equal')
        ax.set_xticks((-5, 0, 5))
        ax.set_yticks((-5, 0, 5))
        ax.grid(False)

        # Do the plotting
        ax.plot(*train_x.T, 'o', color="C1")  # Training samples
        lev = np.linspace(0, 5, 11)
        hc = ax.contourf(full_x1, full_x2, test_variance, lev)  # Posterior std
        for hci in hc.collections:
            hci.set_edgecolor("face")

        # Colorbar
        hcb = plt.colorbar(hc)
        hcb.ax.grid(False)
        hcb.set_label('Posterior standard deviation')

        # Write out
        plt.tight_layout()
        plt.savefig('gpr_posterior_std.svg')
        plt.show()

    def BO_1D(self, n_iter):
        train_x, train_y = self._dataset_1d.get_initial_training_set(3, 0.000000001)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        for i in range(n_iter):
            gp_prior = ExactGPModel(train_x, train_y, likelihood)
            optimizer = Adam([{'params': gp_prior.parameters()}], lr=0.1)
            mll = ExactMarginalLogLikelihood(likelihood, gp_prior)
            for j in range(10):
                gp_prior.train()
                likelihood.train()
                optimizer.zero_grad()
                pred = gp_prior(train_x)
                loss = -mll(pred, train_y)
                loss.backward()
                #now the gp_prior becomes posterier
                optimizer.step()
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    j + 1, 10, loss.item(),
                    gp_prior.covar_module.base_kernel.lengthscale.item(),
                    gp_prior.likelihood.noise.item()
                ))

            gp_prior.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                next_x = self._lcb_acquisition.next_point(gp_prior, bounds=(0.0, 0.1))
                next_y = self._dataset_1d.get_observation(next_x, 0.0000001)

                train_x = torch.cat([train_x, next_x], dim=0)
                train_y = torch.cat([train_y, next_y], dim=0)

                print(f"Iteration {i + 1}/{n_iter}:")
                print(f"  Next x = {next_x.item():.4f}, Next y = {next_y.item():.4f}")

            gp_prior.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.linspace(0, 1, 51)
                observed_pred = likelihood(gp_prior(test_x))

            with torch.no_grad():
                f, ax = plt.subplots(1, 1, figsize=(4, 3))
                lower, upper = observed_pred.confidence_region()
                ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
                # Plot predictive means as blue line
                ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
                # Shade between the lower and upper confidence bounds
                ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
                ax.set_ylim([-3, 3])
                ax.legend(['Observed Data', 'Mean', 'Confidence'])
                plt.show()


bo = BayesianOptimization()
#bo.GP_1D(10)
#bo.BO_1D(10)
bo.GP_2D(100)
