import gpytorch
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


class VisualCreator:

    def __init__(self):
        pass

    def GP_plotting(self, dimension: int, model, likelihood, train_x, train_y, test_x, test_y, objective_function, scope_lb, scope_ub, test_size):
        if dimension == 1:
            self._GP_plotting_1D(model, likelihood, train_x, train_y, test_x, test_y, scope_lb, scope_ub)
        if dimension == 2:
            self._GP_plotting_2D(model, likelihood, train_x, train_y, test_x, test_y, objective_function, scope_lb, scope_ub, test_size)

    def _GP_plotting_1D(self, model, likelihood, train_x, train_y, test_x, test_y, scope_lb, scope_ub):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))

        with torch.no_grad():
            f, ax = plt.subplots(1, 1, figsize=(4, 3))
            lower, upper = observed_pred.confidence_region()
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Training Data', 'Model Prediction', 'Confidence'])
            plt.savefig('gpr_posterior_mean.svg')
            plt.show()

    def _GP_plotting_2D(self, model, likelihood, train_x, train_y, test_x, test_y, objective_function, scope_lb, scope_ub, test_size):

        xv = torch.linspace(scope_lb[0].item(), scope_ub[0].item(), test_size)
        yv = torch.linspace(scope_lb[1].item(), scope_ub[1].item(), test_size)
        full_x1, full_x2 = torch.meshgrid(xv, yv, indexing="ij")
        z = objective_function(full_x1, full_x2)

        with torch.no_grad():
            observed_pred = likelihood(model(test_x))
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
        ax.contour(full_x1, full_x2, z, lev, colors='k')  # Truth
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
        lev = np.linspace(np.min(test_variance), np.max(test_variance), 5)
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

    def BO_plotting_1D(self):
        pass