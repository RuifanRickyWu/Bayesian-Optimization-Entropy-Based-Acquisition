import gpytorch.settings
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.lines as mlines
from core.acquisition_function.pi import PIAcquisition
from core.acquisition_function.lcb import LCBAcquisition
from core.acquisition_function.ei import EIAcquisition


class VisualCreator:

    def __init__(self):
        pass

    def GP_plotting_1D(self, model, likelihood, train_x, train_y, test_x, next_x, next_y, objective_function, scope_lb, scope_ub):

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            observed_mean = observed_pred.mean.numpy()

        objective_x = torch.linspace(scope_lb[0], scope_ub[0], 1000)
        objective_y = objective_function(objective_x)
        objective_x = objective_x.numpy()
        objective_y = objective_y.numpy()

        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        lower, upper = observed_pred.confidence_region()
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        ax.plot(objective_x, objective_y, color = 'orange')
        ax.plot(test_x.numpy(), observed_mean, 'b')
        ax.plot(next_x.numpy(), next_y.numpy(), "r*")
        ax.fill_between(test_x.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.legend(['Training Points', 'Objective Function', 'Model Prediction', "Next Point"])
        plt.savefig('gpr_posterior_mean.svg')
        plt.show()

    def GP_plotting_2D(self, model, likelihood, objective_function, train_x, train_y, test_x, scope_lb, scope_ub):
        xv = torch.linspace(scope_lb[0], scope_ub[0], 1000)
        yv = torch.linspace(scope_lb[1], scope_ub[1], 1000)
        objective_xv, objective_yv = torch.meshgrid(xv, yv)
        z = objective_function(objective_xv, objective_yv)

        with torch.no_grad():
            observed_pred = likelihood(model(test_x))
            test_mean = observed_pred.mean.numpy()
            test_variance = observed_pred.variance.numpy()

        f, ax = plt.subplots(1, 1, figsize=(4,3))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_xlim((scope_lb[0], scope_ub[0]))
        ax.set_ylim((scope_ub[1], scope_ub[1]))
        ax.set_aspect('equal')
        ax.grid(False)

        lev = np.linspace(0., 250., 6)
        ax.contour(objective_xv, objective_yv, z, lev, colors='k')
        ax.plot(*train_x.T, 'o', color="C1")
        truth_line = mlines.Line2D([], [], color='black', label='True $z(x,y)$')
        sample_line = mlines.Line2D([], [], color='C1', marker="o", linestyle="none", label='Train samples')
        mean_line = mlines.Line2D([], [], color='C0', linestyle="--", label='Posterior mean')
        ax.legend(handles=[truth_line, sample_line, mean_line], bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig('gpr_posterior_mean.svg')
        plt.show()


    def aquisition_plotting_1D(self, acq_function, model, likelihood, scope_lb, scope_ub, next_x, starting_candidate, ending_candidate):
        pi = PIAcquisition(0.5)
        lcb = LCBAcquisition(0.5)
        ei = EIAcquisition()
        es_mc = ESAcquisition()

        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            all_x = torch.linspace(scope_lb[0], scope_ub[0], 1000)
            all_y = []
            for x in all_x:
                if isinstance(pi, type(acq_function)):
                    all_y.append(acq_function.acq(x.unsqueeze(0), model, likelihood, model.train_targets.min().item()))
                if isinstance(lcb, type(acq_function)):
                    all_y.append(acq_function.acq(x.unsqueeze(0), model, likelihood))
                if isinstance(ei, type(acq_function)):
                    all_y.append(acq_function.acq(x.unsqueeze(0), model, likelihood))
                if isinstance(es_mc, type(acq_function)):
                    all_y.append(acq_function.acq(x.unsqueeze(0), model, likelihood))
            if isinstance(pi,type(acq_function)):
                next_x_acq = acq_function.acq(next_x, model, likelihood, model.train_targets.min().item())
            if isinstance(lcb, type(acq_function)):
                next_x_acq = acq_function.acq(next_x, model, likelihood)
            if isinstance(ei, type(acq_function)):
                next_x_acq = acq_function.acq(next_x, model, likelihood)
            if isinstance(es_mc, type(acq_function)):
                next_x_acq = acq_function.acq(next_x, model, likelihood)

            start_acq = []
            for start in starting_candidate:
                if isinstance(pi, type(acq_function)):
                    start_acq.append(acq_function.acq(start.unsqueeze(0), model, likelihood, model.train_targets.min().item()))
                if isinstance(lcb, type(acq_function)):
                    start_acq.append(acq_function.acq(start.unsqueeze(0), model, likelihood))
                if isinstance(ei, type(acq_function)):
                    start_acq.append(acq_function.acq(start.unsqueeze(0), model, likelihood))
                if isinstance(es_mc, type(acq_function)):
                    start_acq.append(acq_function.acq(start.unsqueeze(0), model, likelihood))

            end = []
            end_acq = []
            for tuple in ending_candidate:
                if tuple[0].numpy()[0] != next_x:
                    end.append(tuple[0].numpy())
                    end_acq.append(tuple[1])


        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(all_x.numpy(), all_y, 'orange')
        ax.plot(next_x, next_x_acq, "r*")
        ax.plot(starting_candidate.detach().numpy(), start_acq, "b*")
        ax.plot(end, end_acq, "g*")
        ax.legend(['Acquisition Function', 'Next Point(Best Candidate)', 'Starting Candidates', "Ending Choices"])
        plt.show()


