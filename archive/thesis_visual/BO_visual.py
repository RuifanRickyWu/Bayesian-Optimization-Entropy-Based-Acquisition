import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def f(x):
    return np.sin(2 * np.pi * x)

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

# Set random seed and generate input space
np.random.seed(42)
X = np.atleast_2d(np.linspace(0, 1, 1000)).T
X_sample = np.random.rand(2, 1)
Y_sample = f(X_sample)

# GP model with RBF kernel
kernel = RBF(length_scale=0.1) + WhiteKernel(noise_level=1e-5)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

steps = []

# Bayesian Optimization iterations
for t in range(3):
    gpr.fit(X_sample, Y_sample)
    mu, std = gpr.predict(X, return_std=True)
    ei = expected_improvement(X, X_sample, Y_sample, gpr)

    x_next = X[np.argmax(ei)]
    y_next = f(x_next)

    steps.append((X_sample.copy(), Y_sample.copy(), mu.copy(), std.copy(), ei.copy(), x_next.copy(), y_next.copy()))

    X_sample = np.vstack((X_sample, x_next))
    Y_sample = np.vstack((Y_sample, y_next))

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for i, ax in enumerate(axes):
    Xs, Ys, mu, std, ei, x_next, y_next = steps[i]

    ax.plot(X, f(X), 'r-', label="True function $f(x)$")
    ax.plot(X, mu, 'k--', label="GP posterior mean")
    ax.fill_between(X.ravel(), mu - 3*std, mu + 3*std, alpha=0.2, color='orange', label="GP posterior std. dev.")
    ax.plot(Xs, Ys, 'ko', label="Observations")
    ax.plot(x_next, y_next, 'ro', markerfacecolor='red', markeredgecolor='black', label="Next point")
    ax.set_title(f"Iteration $t={i+1}$")
    ax.set_xlim([0, 1])
    ax.set_ylim([-1.5, 1.5])
    ax.set_ylabel("Function value")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(X.ravel(), ei, color='green', alpha=0.8, label="Expected Improvement (EI)")
    ax2.fill_between(X.ravel(), ei, color='green', alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Expected Improvement", color='green')
    ax2.legend(loc="upper right")

fig.tight_layout()
plt.xlabel("x")
plt.show()
