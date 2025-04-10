import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500).reshape(-1, 1)

def periodic_kernel(x1, x2, sigma_f=1.0, length_scale=0.5, period=1.0):
    dists = np.pi * np.abs(x1 - x2.T) / period
    return sigma_f**2 * np.exp(-2 * np.sin(dists)**2 / length_scale**2)

sigma_f = 1.0
length_scale = 0.3
period = 1.0

K = periodic_kernel(x, x, sigma_f=sigma_f, length_scale=length_scale, period=period)

np.random.seed(42)
samples = np.random.multivariate_normal(mean=np.zeros(x.shape[0]), cov=K, size=5)

plt.figure(figsize=(10, 5))
for y in samples:
    plt.plot(x, y, color='black', alpha=0.7)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("$x$", fontsize=12)
plt.ylabel("$f(x)$", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
