import numpy as np
import gpflow
import tensorflow as tf
from conjugate_gradient import conjugate_gradient
import matplotlib.pyplot as plt


def conjugate_gradient_playground():
    seed = 1000
    tf.random.set_seed(seed)
    np.random.seed(seed)

    lengthscale = [0.2]
    variance = 0.555
    kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale, variance=variance)
    # kernel = gpflow.kernels.Matern32(lengthscales=lengthscale, variance=variance)
    n = 50
    m = 10
    x = np.linspace(1, m, n).reshape(-1, 1)
    y = np.cos(x) + 0.01 * np.random.randn(n, 1)
    y = y.T
    initial_solution = np.random.rand(1, n)
    error_threshold = 0.01
    max_iterations = 1000

    n_test = n * 10
    x_test = np.linspace(x.min(), x.max(), n_test).reshape(-1, 1)

    kxx = kernel(x)
    kxx_eig_vals, kxx_eig_vecs = tf.linalg.eigh(kxx)
    kxx_eig_vals = kxx_eig_vals.numpy().astype(np.float64)
    kxx_eig_vecs = kxx_eig_vecs.numpy().astype(np.float64)

    max_eig_val = kxx_eig_vals.max()
    min_eig_val = kxx_eig_vals.min()
    condition_number = max_eig_val / min_eig_val
    print(f"Eigenvalues: min={min_eig_val:0.4f}, max={max_eig_val:0.4f}")
    print(f"Condition number {condition_number:0.4f}")

    solution, stats = conjugate_gradient(kxx, y, initial_solution, error_threshold, max_iterations)

    solution_base = tf.linalg.solve(kxx, y.T)

    solution = solution.numpy().T
    solution_base = solution_base.numpy()

    cg_steps, cg_error = stats
    cg_steps = cg_steps.numpy()
    cg_error = cg_error.numpy()

    print(f"Conjugate gradient ran {cg_steps} iterations.")
    print(f"Conjugate gradient finished with {cg_error:0.4f} error.")

    error = np.mean((solution_base - solution) ** 2)
    print(f"Error comparing to the 'ground truth' solver: {error:0.4f}")

    k_test = kernel(x_test, x)
    f_test = k_test @ solution
    f_base_test = k_test @ solution_base

    figsize = (8, 12)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    ax1.scatter(x, y, alpha=0.25)
    ax1.plot(x_test, f_base_test, alpha=0.5)
    ax1.plot(x_test, f_test, color="tab:blue")

    for i in range(n):
        ax2.plot(x_test, solution[i,0] * k_test[:,i], alpha=0.5)
    ax2.stem(x, solution, markerfmt=" ")

    for i in range(n):
        ax3.plot(x_test, solution_base[i,0] * k_test[:,i], alpha=0.5)
    ax3.stem(x, solution_base, markerfmt=" ")

    ax1.set_title("Data and regression curve")
    ax2.set_title("CG canonical basis functions and weights")
    ax3.set_title("Cholesky canonical basis functions and weights")
    ax1.set_ylabel("$f$ and $y$")
    ax2.set_ylabel("$v$ and $k(x,\cdot)$")
    ax3.set_ylabel("$v$ and $k(x,\cdot)$")
    ax1.set_xlabel("$x$")
    ax2.set_xlabel("$x$")
    ax3.set_xlabel("$x$")
    plt.tight_layout()
    plt.savefig("conjugate_gradient.pdf")
    plt.show()

    print()


if __name__ == "__main__":
    conjugate_gradient_playground()
