from pathlib import Path
from typing import Callable, Tuple, Optional
import numpy as np
import tensorflow as tf
import gpflow

from distance import DistanceType
from conjugate_gradient import ConjugateGradient, conjugate_gradient
from models import ClusterGP, CGGP
from cli_utils import create_model_and_covertree_update_fn, kernel_fn
from utils import add_diagonal, jit

import matplotlib.pyplot as plt


Tensor = tf.Tensor


def gen_data(seed: Optional[int] = None, noise: float = 0.001):
    rng = np.random.RandomState(seed)
    n = 1000
    l = 2 * np.pi
    x = 2 * l * rng.rand(n, 1) - l

    def fn(x):
        x2 = x**2
        return np.sin(x2) ** 2 + np.cos(x)

    std = np.sqrt(noise)
    y = fn(x) + rng.randn(n, 1) * std
    return x, y


def wasserstein2(moments1, moments2):
    mu1, cov1 = moments1
    mu2, cov2 = moments2

    # jitter = tf.convert_to_tensor(1e-10, dtype=cov2.dtype)
    jitter = tf.convert_to_tensor(0.0, dtype=cov2.dtype)

    def matrix_sqrt(m):
        eigvals, eigvecs = tf.linalg.eigh(m)
        eigvals = tf.where(eigvals < 0.0, jitter, eigvals)
        sqrt_eigvals = tf.sqrt(eigvals)
        sqrt_m = eigvecs @ tf.linalg.diag(sqrt_eigvals)
        return sqrt_m

    sqrt_cov1 = matrix_sqrt(cov1)
    cov1_cov2_term = sqrt_cov1 @ cov2 @ sqrt_cov1
    cov1_cov2_sqrt_term = matrix_sqrt(cov1_cov2_term)

    norm_term = tf.norm(mu1 - mu2) ** 2
    trace_term = tf.linalg.trace(cov1 + cov2 - 2.0 * cov1_cov2_sqrt_term)

    distance = norm_term + trace_term
    return distance


def gen_wasserstein_condition_numbers(
    gpr,
    data,
    resolutions,
    distance_type: DistanceType = "euclidean",
    use_jit: bool = True,
):
    noise = gpr.likelihood.variance.numpy()
    lengthscale = gpr.kernel.lengthscales.numpy()

    x, y = data

    def covertree_setup(spatial_resolution):
        return create_model_and_covertree_update_fn(
            model_cls, data, spatial_resolution, distance_type=distance_type
        )

    def model_cls(kernel, likelihood, iv, **kwargs):
        error_threshold = 1e-6
        conjugate_gradient = ConjugateGradient(error_threshold)
        return CGGP(kernel, likelihood, iv, conjugate_gradient, **kwargs)

    gpr_predict_fn = jit(use_jit)(gpr.predict_f)
    mu, cov = gpr_predict_fn(x, full_cov=True)
    gpr_moments = (mu, cov[0])

    condition_numbers = []
    num_inducing_points = []
    wasserstein_distances = []
    cg_iterations = []

    for resolution in resolutions:
        model_and_update_fn: Tuple[ClusterGP, Callable] = covertree_setup(resolution)
        model, update_fn = model_and_update_fn
        model.likelihood.variance.assign(noise)
        kernel = model.kernel
        kernel.lengthscales.assign(lengthscale)

        update_fn()
        iv = model.inducing_variable
        diag_lambda = model.diag_variance[..., 0]
        num_inducing = np.array(iv.num_inducing)
        num_inducing_points.append(num_inducing)
        u = model.pseudo_u

        kuu = gpflow.covariances.Kuu(iv, kernel, jitter=0.0)
        kuu_lambda = add_diagonal(kuu, diag_lambda)

        cg = model.conjugate_gradient
        _, cg_stats = stats_conjugate_gradient(cg, kuu_lambda, u)

        eigvals = tf.linalg.eigvalsh(kuu_lambda).numpy()
        eig_min = eigvals.min()
        eig_max = eigvals.max()
        condition_number = eig_max / eig_min
        condition_numbers.append(condition_number)

        predict_fn = jit(use_jit)(model.predict_f)
        mu_approx, cov_approx = predict_fn(x, full_cov=True)
        model_moments = (mu_approx, cov_approx[0])
        distance = wasserstein2(gpr_moments, model_moments)
        wasserstein_distances.append(distance)
        cg_iterations.append(np.array(cg_stats[0]))

    return condition_numbers, num_inducing_points, wasserstein_distances, cg_iterations


def sample_gpr_prior(kernel, inputs, num_samples: int = 1):
    mean = tf.zeros((1, tf.shape(inputs)[0]), dtype=inputs.dtype)
    cov = kernel(inputs, full_cov=True)
    # config = gpflow.config.Config(jitter=0.0)
    config = gpflow.config.Config()
    cov = cov[None, ...]
    with gpflow.config.as_context(config):
        output = gpflow.conditionals.util.sample_mvn(
            mean, cov, full_cov=True, num_samples=num_samples
        )
    output_samples = output[:, 0, ...]
    return output_samples[..., None]


def paper_visualization():
    seed = 333
    tf.random.set_seed(seed)
    np.random.seed(seed)

    noise = 0.02
    lengthscale = 0.5
    dtype = gpflow.config.default_float()
    distance_type = "euclidean"
    use_jit = True
    resolutions = np.linspace(0.1, 2.0, 10)

    n = 1000
    b = 5
    # dims = [1, 2, 4, 8]
    dims = [1, 2, 4, 8]
    num_samples = 5
    data_frame = {"resolutions": resolutions}

    for dim in dims:
        print(f">>> Start processing dim = {dim}")
        rng = np.random.RandomState(seed)
        lengthscales = [lengthscale * np.sqrt(dim)] * dim
        x = rng.rand(n, dim) * 2 * b - b
        xt = tf.convert_to_tensor(x, dtype=dtype)
        kernel = kernel_fn(dim)
        kernel.lengthscales.assign(lengthscales)

        yt = sample_gpr_prior(kernel, xt, num_samples=num_samples)
        for s in range(num_samples):
            print(f">>> Start processing sample = {s}")
            data = (xt, yt[s])
            gpr = gpflow.models.GPR(data, kernel, noise_variance=noise)

            metrics = gen_wasserstein_condition_numbers(
                gpr, data, resolutions, distance_type=distance_type, use_jit=use_jit
            )

            (
                condition_numbers,
                num_inducing_points,
                wasserstein_distances,
                cg_iterations,
            ) = metrics

            data_frame[f"condition_numbers_dim{dim}_s{s}"] = np.array(condition_numbers)
            data_frame[f"num_inducing_points_{dim}_{s}"] = np.array(num_inducing_points)
            data_frame[f"wasserstein_distances_{dim}_{s}"] = np.array(wasserstein_distances)
            data_frame[f"cg_iterations_{dim}_{s}"] = np.array(cg_iterations)

            plot(data, noise, resolutions, *metrics)

    dirpath = str(Path(Path(__file__).parent))
    store_metrics(dirpath, noise, lengthscale, data_frame)
    print()


def store_metrics(dirpath, noise, lengthscale, storage):
    import pandas as pd

    df = pd.DataFrame(data=storage)
    filename = f"metric_data_noise_{noise}_lengthscale_{lengthscale}.csv"
    filepath = str(Path(dirpath, filename))
    df.to_csv(filepath)


def plot(
    data,
    noise,
    resolutions,
    condition_numbers,
    num_inducing_points,
    wasserstein_distances,
):
    x, y = data
    x, y = np.array(x), np.array(y)
    dim = x.shape[-1]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(f"Data from GP prior samples, $d={dim}$, " + r"$\sigma^2=" + f"{noise}$")

    # ax3.scatter(x, y, s=5)
    # ax3.set_ylabel(r"$f(x) + \epsilon$, $\sigma^2 =" + f"{noise})$")
    # ax3.set_xlabel("$x$")

    ax1.plot(resolutions, condition_numbers)
    ax2.plot(resolutions, num_inducing_points)
    ax4.plot(resolutions, wasserstein_distances)

    ax1.scatter(resolutions, condition_numbers, s=5)
    ax2.scatter(resolutions, num_inducing_points, s=5)
    ax4.scatter(resolutions, wasserstein_distances, s=5)

    [ax.set_xscale("log") for ax in [ax1, ax2, ax4]]
    [ax.set_yscale("log") for ax in [ax1, ax2, ax4]]
    [ax.set_xlabel("Spatial resolution") for ax in [ax1, ax2, ax4]]

    ax1.set_ylabel("Condition number")
    ax2.set_ylabel("Number of inducing points")
    ax4.set_ylabel("(2-Wasserstein distance)$^2$")

    plt.tight_layout()
    plt.show()
    print()


def stats_conjugate_gradient(
    cg: ConjugateGradient, matrix: Tensor, rhs: Tensor, initial_solution: Optional[Tensor] = None
) -> Tensor:
    rhs = tf.transpose(rhs)

    if initial_solution is None:
        initial_solution = tf.zeros_like(rhs)
    else:
        initial_solution = tf.transpose(initial_solution)

    max_iterations = cg.max_iterations
    if max_iterations is None:
        max_iterations = tf.shape(matrix)[-1]

    max_steps_cycle = cg.max_steps_cycle
    if max_steps_cycle is None:
        max_steps_cycle = max_iterations + 1  # Make sure that we don't run it in the end of CG

    preconditioner = cg.preconditioner
    error_threshold = cg.error_threshold

    solution, stats = conjugate_gradient(
        matrix,
        rhs,
        initial_solution,
        error_threshold,
        preconditioner=preconditioner,
        max_iterations=max_iterations,
        max_steps_cycle=max_steps_cycle,
    )

    solution = tf.transpose(solution)
    return solution, stats


if __name__ == "__main__":
    paper_visualization()
