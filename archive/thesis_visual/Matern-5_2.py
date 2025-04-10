import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern

x = np.linspace(-1, 1, 500).reshape(-1, 1)

kernel = Matern(length_scale=0.2, nu=1.5)

K = kernel(x)

np.random.seed(42)

samples = np.random.multivariate_normal(mean=np.zeros(x.shape[0]), cov=K, size=20)

plt.figure(figsize=(6, 5))
for y in samples:
    plt.plot(x, y, color='black', alpha=0.3)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlim(-1, 1)
plt.ylim(-3, 3)
plt.xlabel("$x$", fontsize=14)
plt.ylabel("$f$", fontsize=14)
plt.title(r"(a) Matérn-$\frac{1}{2}$", fontsize=14)
plt.grid(False)
plt.gca().set_facecolor('#f0f0f0')
plt.tight_layout()
plt.show()
