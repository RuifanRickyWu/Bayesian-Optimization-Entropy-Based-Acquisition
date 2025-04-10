import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
import matplotlib.cm as cm

x1 = np.linspace(-2, 2, 50)
x2 = np.linspace(-2, 2, 50)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

x0 = np.array([[0.0, 0.0]])

lengthscales = [
    (1.0, 1.0),
    (0.5, 2.0),
    (2.0, 0.5),
    (0.3, 3.0)
]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()
cmap = cm.get_cmap("Greys")

for i, (l1, l2) in enumerate(lengthscales):
    kernel = RBF(length_scale=[l1, l2])
    K = kernel(X_grid, x0).reshape(50, 50)

    im = axs[i].imshow(K, extent=(-2, 2, -2, 2), origin='lower', cmap=cmap, vmin=0, vmax=1)
    axs[i].set_title(rf"$\ell_1={l1}, \ell_2={l2}$", fontsize=12)
    axs[i].set_xlabel("$x_1$")
    axs[i].set_ylabel("$x_2$")
    axs[i].set_aspect('equal')

    fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04, ticks=[])

plt.tight_layout()
plt.suptitle("SE Kernel with ARD: Anisotropy via Length-Scales (Grayscale)", fontsize=14, y=1.05)

plt.savefig("se_kernel_ard_bw.png", dpi=300, bbox_inches='tight')
plt.show()
