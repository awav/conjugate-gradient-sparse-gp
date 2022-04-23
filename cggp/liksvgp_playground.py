from typing import Callable, Optional, TypeVar, Type
from functools import partial
from kmeans import (
    kmeans_lloyd,
    kmeans_indices_and_distances,
    create_kernel_distance_fn,
    DistanceType,
)
from models import LpSVGP, ClusterSVGP
from data import snelson1d
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
import numpy as np

from gpflow.config import default_float


ModelClass = TypeVar("ModelClass", type(LpSVGP), type(ClusterSVGP))


def create_model(
    data, num_inducing_points: int, distance_type: DistanceType, model_class: ModelClass
):
    x, y = data
    xt = tf.convert_to_tensor(x, dtype=default_float())
    yt = tf.convert_to_tensor(y, dtype=default_float())
    kernel = gpflow.kernels.SquaredExponential()
    likelihood = gpflow.likelihoods.Gaussian(variance=0.1)

    distance_fn = create_kernel_distance_fn(kernel, distance_type)
    kmeans_fn = tf.function(partial(kmeans_lloyd, distance_fn=distance_fn))

    def clustering_fn():
        iv, _ = kmeans_fn(xt, num_inducing_points)
        return iv

    iv = clustering_fn()

    model = model_class(kernel, likelihood, iv)

    gpflow.utilities.set_trainable(model.inducing_variable, False)
    return (xt, yt), model, clustering_fn, distance_fn


def train_vanilla_using_lbfgs_and_standard_ip_update(
    data,
    model,
    clustering_fn: Callable,
    max_num_iters: int,
):
    lbfgs = gpflow.optimizers.Scipy()
    options = dict(maxiter=max_num_iters)
    loss_fn = model.training_loss_closure(data, compile=False)
    variables = model.trainable_variables

    def step_callback(*args, **kwargs):
        # TODO(awav): This callback is called after every gradient step in L-BFGS
        # Calling clustering every gradient step causes the convergence
        # to a poor local minima.
        new_iv = clustering_fn()
        model.inducing_variable.Z.assign(new_iv)

    use_jit = True
    result = lbfgs.minimize(
        loss_fn,
        variables,
        step_callback=step_callback,
        compile=use_jit,
        options=options,
    )

    return result


def train_vanilla_using_lbfgs(
    data,
    model,
    clustering_fn: Callable,
    max_num_iters: int,
):
    lbfgs = gpflow.optimizers.Scipy()
    options = dict(maxiter=max_num_iters)
    loss_fn = model.training_loss_closure(data, compile=False)
    variables = model.trainable_variables

    use_jit = True
    result = lbfgs.minimize(
        loss_fn,
        variables,
        compile=use_jit,
        options=options,
    )

    return result


def train_using_lbfgs_and_varpar_update(
    data,
    model: ClusterSVGP,
    clustering_fn: Callable,
    max_num_iters: int,
    outer_num_iters: int,
    distance_fn: Optional[Callable] = None,
):
    lbfgs = gpflow.optimizers.Scipy()
    options = dict(maxiter=max_num_iters)
    loss_fn = model.training_loss_closure(data, compile=False)
    variables = model.trainable_variables
    x, y = data

    def update_variational_parameters(*args, **kwargs):
        new_iv = clustering_fn()

        m = new_iv.shape[0]
        indices, _ = kmeans_indices_and_distances(new_iv, x, distance_fn=distance_fn)
        range_indices = tf.range(m, dtype=indices.dtype)
        counting_map = tf.cast(range_indices[:, None] == indices[None, :], tf.int32)
        counts = tf.reduce_sum(counting_map, axis=1, keepdims=True)
        counts = tf.cast(counts, dtype=new_iv.dtype)

        u_init = tf.zeros([m, 1], dtype=new_iv.dtype)
        update_indices = tf.reshape(indices, [-1, 1])
        u = tf.tensor_scatter_nd_add(u_init, update_indices, y) / counts
        sigma2 = model.likelihood.variance
        lambda_diag = sigma2 / counts

        model.inducing_variable.Z.assign(new_iv)
        model.pseudo_u.assign(u)
        model.diag_variance.assign(lambda_diag)

    gpflow.utilities.set_trainable(model.inducing_variable, False)
    use_jit = True  # TODO(awav): resolve the problem with recompiling in Scipy

    # for _ in range(outer_num_iters):
    update_variational_parameters()
    result = lbfgs.minimize(
        loss_fn,
        variables,
        step_callback=update_variational_parameters,
        compile=use_jit,
        options=options,
    )

    return result


if __name__ == "__main__":
    seed = 111
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_data, _ = snelson1d()
    distance_type = "covariance"
    num_inducing_points = 20
    num_iterations = 1000

    x, y = train_data

    xmin = x.min() - 1.0
    xmax = x.max() + 1.0
    num_test_points = 100
    x_test = np.linspace(xmin, xmax, num_test_points).reshape(-1, 1)

    # NOTE:
    # Model setup
    #   `model_class` switches between different models
    #   as well as training procedures.
    #   Available options are LpSVGP and ClusterSVGP.

    # model_class = LpSVGP
    model_class = ClusterSVGP
    data, experimental_model, clustering_fn, distance_fn = create_model(
        (x, y),
        num_inducing_points,
        distance_type,
        model_class,
    )
    xt, _ = data

    if model_class == LpSVGP:
        opt_result = train_vanilla_using_lbfgs(
            data, experimental_model, clustering_fn, num_iterations
        )
    elif model_class == ClusterSVGP:
        outer_num_iters = 100
        opt_result = train_using_lbfgs_and_varpar_update(
            data, experimental_model, clustering_fn, num_iterations, outer_num_iters
        )
    else:
        print("No hyperparameter tuning!")

    print("Optimization results: ")
    print(opt_result)
    
    kernel = experimental_model.kernel
    noise = experimental_model.likelihood.variance.numpy()
    gpr_model = gpflow.models.GPR(train_data, kernel=kernel, noise_variance=noise)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    iv = experimental_model.inducing_variable.Z.numpy()
    indices, _ = kmeans_indices_and_distances(iv, xt, distance_fn=distance_fn)
    indices = indices.numpy()
    color_map = "tab20c"
    colors = plt.get_cmap(color_map)(np.arange(num_inducing_points, dtype=int))

    # Plot #1
    ax1.set_xlim(x_test.min(), x_test.max())
    for i in range(num_inducing_points):
        color = colors[i]
        centroid_mask = indices == i
        x_plot = x[centroid_mask, :]
        y_plot = y[centroid_mask, :]
        scatter = ax1.scatter(x_plot, y_plot, s=8, alpha=0.8, color=color)
        ax1.axvline(x=iv[i][0], color=color, linestyle="--")

    # Plot #2
    ax2.set_xlim(x_test.min(), x_test.max())
    mu_test, var_test = experimental_model.predict_y(x_test)
    gpr_mu_test, gpr_var_test = gpr_model.predict_y(x_test)

    gpr_mu_test = gpr_mu_test.numpy().reshape(-1)
    gpr_std_test = np.sqrt(gpr_var_test.numpy()).reshape(-1)
    gpr_up = gpr_mu_test + gpr_std_test
    gpr_down = gpr_mu_test - gpr_std_test

    std_test = np.sqrt(var_test.numpy())
    mu_test = mu_test.numpy().reshape(-1)
    std_test = std_test.reshape(-1)
    up = mu_test + std_test
    down = mu_test - std_test

    blue = "tab:blue"
    gray = "tab:gray"

    ax2.plot(x_test, mu_test, color=gray)
    ax2.fill_between(x_test.reshape(-1), up, down, color=gray, alpha=0.5)
    ax2.scatter(x, y, color=gray, alpha=0.5, s=8)

    iv = experimental_model.inducing_variable.Z.numpy().reshape(-1)
    variational_mean, variational_variance = experimental_model.q_moments()
    variational_mean = variational_mean.numpy()
    variational_std = np.sqrt(variational_variance)
    scale = 1.96
    variational_upper = variational_mean + scale * variational_std
    variational_lower = variational_mean - scale * variational_std
    for i in range(num_inducing_points):
        color = colors[i]
        x = iv[i]
        ax2.scatter(x, variational_mean[i], color=color, marker="o", s=5)
        ax2.scatter(x, variational_upper[i], color=color, marker="_")
        ax2.scatter(x, variational_lower[i], color=color, marker="_")
        ax2.plot(
            [x, x],
            [variational_lower[i], variational_upper[i]],
            color=color,
            marker="_",
        )
    
    # Plot #3

    ax3.plot(x_test, gpr_mu_test, color=blue, label="GPR")
    ax3.fill_between(x_test.reshape(-1), gpr_up, gpr_down, color=blue, alpha=0.2)

    ax3.plot(x_test, mu_test, color=gray, label="GP with clustering")
    ax3.fill_between(x_test.reshape(-1), up, down, color=gray, alpha=0.2)

    ax3.scatter(x, y, color=gray, alpha=0.5, s=8)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("liksvgp.pdf")
    plt.show()
    print("end")
