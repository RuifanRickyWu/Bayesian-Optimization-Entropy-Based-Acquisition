import gpytorch
import torch
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior


class BOTorchWrapper(Model):
    def __init__(self, model: gpytorch.models.ExactGP, likelihood):
        super().__init__()
        self._model = model
        self._likelihood = likelihood

        self._train_inputs = model.train_inputs
        self._train_targets = model.train_targets.unsqueeze(-1)  # shape: [N, 1]

        self._num_outputs = 1
        self.outcome_transform = None

    @property
    def train_inputs(self):
        return self._train_inputs

    @property
    def train_targets(self):
        return self._train_targets

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        self._model.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mvn = self._likelihood(self._model(X))
        return GPyTorchPosterior(mvn)

    def condition_on_observations(self, X, Y, **kwargs):
        raise NotImplementedError("Conditioning on observations is not implemented for BOTorchWrapper.")

    def subset_output(self, output_indices):
        return self

    def to(self, *args, **kwargs):
        self._model.to(*args, **kwargs)
        self._likelihood.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self._model.train(mode)
        self._likelihood.train(mode)
        return self

    def eval(self):
        self._model.eval()
        self._likelihood.eval()
        return self
