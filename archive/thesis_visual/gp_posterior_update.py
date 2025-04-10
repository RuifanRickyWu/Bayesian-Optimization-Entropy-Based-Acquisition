
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor


def f_true(x):
    return np.sin(2 * np.pi * x)


x = np.linspace(0, 1, 500).reshape(-1, 1)

X_train_sets = [
    np.array([[0.2], [0.8]]),
    np.array([[0.2], [0.8], [0.5]]),
    np.array([[0.2], [0.8], [0.5], [0.35], [0.65]])
]
y_train_sets = [f_true(X).ravel() for X in X_train_sets]

kernel = RBF(length_scale=0.15)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
titles = ["(a) 2 Observations", "(b) 3 Observations", "(c) 5 Observations"]

for i in range(3):
    X_train, y_train = X_train_sets[i], y_train_sets[i]
    gp.fit(X_train, y_train)
    y_pred, sigma = gp.predict(x, return_std=True)

    axs[i].plot(x, f_true(x), 'r--', label="True function")
    axs[i].plot(x, y_pred, 'k', label="GP mean")
    axs[i].fill_between(x.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
                        alpha=0.2, color='gray', label="95% CI")
    axs[i].scatter(X_train, y_train, c='black', s=30, zorder=10, label="Observations")
    axs[i].set_ylim(-2, 2)
    axs[i].set_title(titles[i])
    axs[i].set_xlabel("$x$")
    if i == 0:
        axs[i].set_ylabel("$f(x)$")
    axs[i].legend(loc='upper right')

plt.tight_layout()
plt.show()
