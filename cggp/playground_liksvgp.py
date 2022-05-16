from kmeans import kmeans_indices_and_distances
from models import LpSVGP, ClusterGP
from data import snelson1d
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
import numpy as np

from cli_utils import create_model_and_kmeans_update_fn, create_distance_fn
from optimize import (
    train_vanilla_using_lbfgs,
    train_using_lbfgs_and_update,
)


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
    #   Available options are LpSVGP and ClusterGP.

    # model_class = LpSVGP
    model_class = ClusterGP
    experimental_model, update_fn = create_model_and_kmeans_update_fn(
        model_class,
        train_data,
        num_inducing_points,
        distance_type=distance_type,
    )

    distance_fn = create_distance_fn(model.kernel, distance_type)
    distance_fn = jit(use_jit)(distance_fn)

    xt, _ = train_data

    if model_class == LpSVGP:
        opt_result = train_vanilla_using_lbfgs(
            train_data, experimental_model, update_fn, num_iterations
        )
    elif model_class == ClusterGP:
        outer_num_iters = 100
        opt_result = train_using_lbfgs_and_update(
            train_data, experimental_model, update_fn, num_iterations
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
        z = iv[i]
        ax2.scatter(z, variational_mean[i], color=color, marker="o", s=5)
        ax2.scatter(z, variational_upper[i], color=color, marker="_")
        ax2.scatter(z, variational_lower[i], color=color, marker="_")
        ax2.plot(
            [z, z],
            [variational_lower[i], variational_upper[i]],
            color=color,
            marker="_",
        )

    # Plot #3

    ax3.plot(x_test, gpr_mu_test, color=blue, label="GPR")
    ax3.fill_between(x_test.reshape(-1), gpr_up, gpr_down, color=blue, alpha=0.2)

    ax3.plot(x_test, mu_test, color=gray, label="GP clustering")
    ax3.fill_between(x_test.reshape(-1), up, down, color=gray, alpha=0.2)

    ax3.scatter(x, y, color=gray, alpha=0.5, s=8)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("liksvgp.pdf")
    plt.show()
    print("end")
