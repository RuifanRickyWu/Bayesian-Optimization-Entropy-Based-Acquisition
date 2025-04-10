import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern

x = np.linspace(-1, 1, 500).reshape(-1, 1)

np.random.seed(42)

matern_params = [
    (0.2, 0.5, r"(a) Matérn-$\frac{1}{2}$"),
    (0.2, 1.5, r"(b) Matérn-$\frac{3}{2}$"),
    (0.2, 2.5, r"(c) Matérn-$\frac{5}{2}$")
]

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

for i, (length_scale, nu, title) in enumerate(matern_params):
    kernel = Matern(length_scale=length_scale, nu=nu)
    K = kernel(x)
    samples = np.random.multivariate_normal(mean=np.zeros(x.shape[0]), cov=K, size=20)

    for y in samples:
        axs[i].plot(x, y, color='black', alpha=0.3)
    axs[i].axhline(0, color='black', linestyle='--', linewidth=1)
    axs[i].set_xlim(-1, 1)
    axs[i].set_ylim(-3, 3)
    axs[i].set_xlabel("$x$", fontsize=12)
    axs[i].set_title(title, fontsize=13)
    axs[i].set_facecolor('#f0f0f0')
    axs[i].grid(False)

axs[0].set_ylabel("$f(x)$", fontsize=12)

plt.tight_layout()
plt.savefig("matern_gp_samples.png", dpi=300, bbox_inches='tight')
plt.show()
