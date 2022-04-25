import numpy as np
import gpflow
import tensorflow as tf
from conjugate_gradient import conjugate_gradient


def conjugate_gradient_playground():
    seed = 1000
    tf.random.set_seed(seed)
    np.random.seed(seed)

    lengthscale = [0.333]
    variance = 0.555
    kernel = gpflow.kernels.Matern32(lengthscales=lengthscale, variance=variance)
    n = 1000
    m = n // 10
    x = np.linspace(1, m, n).reshape(-1, 1)
    y = np.cos(x) + 0.01 * np.random.randn(n, 1)
    y = y.T
    initial_solution = np.random.rand(1, n)
    error_threshold = 0.01
    max_iterations = 1000

    kxx = kernel(x)
    kxx_eig_vals, kxx_eig_vecs = tf.linalg.eig(kxx)
    kxx_eig_vals = kxx_eig_vals.numpy().astype(np.float64)
    kxx_eig_vecs = kxx_eig_vecs.numpy().astype(np.float64)

    max_eig_val = kxx_eig_vals.max()
    min_eig_val = kxx_eig_vals.min()
    condition_number = max_eig_val / min_eig_val
    print(f"Eigenvalues: min={min_eig_val}, max={max_eig_val}")
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
    print()


if __name__ == "__main__":
    conjugate_gradient_playground()
