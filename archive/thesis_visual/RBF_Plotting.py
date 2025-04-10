import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

x = np.linspace(-1, 1, 500).reshape(-1, 1)

length_scales = [0.1, 0.25, 0.5, 1.0]  # these are ℓ in the kernel
sigma_f = 1.0  # standard deviation of the GP prior

np.random.seed(42)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()

for i, ℓ in enumerate(length_scales):
    kernel = sigma_f**2 * RBF(length_scale=ℓ)
    K = kernel(x)  # covariance matrix from the kernel

    samples = np.random.multivariate_normal(mean=np.zeros(x.shape[0]), cov=K, size=20)

    for y in samples:
        axs[i].plot(x, y, color='gray', alpha=0.5)
    axs[i].axhline(0, color='black', linestyle='--', linewidth=1)
    axs[i].set_xlim(-1, 1)
    axs[i].set_ylim(-3, 3)
    axs[i].set_title(rf"$\ell = {ℓ}$")
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("f(x)")

plt.tight_layout()
plt.suptitle("Samples from a GP Prior with Squared Exponential Kernel", fontsize=14, y=1.02)
plt.show()
