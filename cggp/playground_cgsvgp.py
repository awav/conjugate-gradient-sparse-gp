from kmeans import kmeans_indices_and_distances
from models import ClusterGP, CGGP
from conjugate_gradient import ConjugateGradient
from data import snelson1d, load_data
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
import numpy as np

from playground_util import create_model

from optimize import (
    kmeans_update_inducing_parameters,
    train_using_lbfgs_and_update,
    train_using_adam_and_update,
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
    # model_class = CGGP

    def model_class(kernel, likelihood, iv):
        error_threshold = 1e-3
        conjugate_gradient = ConjugateGradient(error_threshold)
        return CGGP(kernel, likelihood, iv, conjugate_gradient)

    data, experimental_model, clustering_fn, distance_fn = create_model(
        (x, y),
        num_inducing_points,
        distance_type,
        model_class,
    )

    def update_fn():
        kmeans_update_inducing_parameters(experimental_model, data, distance_fn, clustering_fn)

    update_fn()

    xt, _ = data

    num_iterations = 100
    batch_size = 25
    learning_rate = 0.01
    use_jit = True

    # opt_result = train_using_adam_and_update(
    #     data,
    #     experimental_model,
    #     num_iterations,
    #     batch_size,
    #     learning_rate,
    #     update_fn,
    #     use_jit=use_jit,
    # )

    opt_result = train_using_lbfgs_and_update(
        data,
        experimental_model,
        clustering_fn,
        num_iterations,
        distance_fn=distance_fn,
        use_jit=use_jit,
    )

    print("Optimization results: ")
    print(opt_result)

    kernel = experimental_model.kernel
    noise = experimental_model.likelihood.variance.numpy()

    cluster_model = ClusterGP(
        experimental_model.kernel,
        experimental_model.likelihood,
        experimental_model.inducing_variable,
        pseudo_u=experimental_model.pseudo_u,
        diag_variance=experimental_model.diag_variance,
    )

    gpr_model = gpflow.models.GPR(train_data, kernel=kernel, noise_variance=noise)

    # Plot #0
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
    x_test_flat = x_test.reshape(-1)

    mu_test, var_test = experimental_model.predict_y(x_test)
    # gpr_mu_test, gpr_var_test = gpr_model.predict_y(x_test)
    # cluster_mu_test, cluster_var_test = cluster_model.predict_y(x_test)

    def gen_mean_up_down(mu, var):
        mu = mu.numpy().reshape(-1)
        std = np.sqrt(var.numpy()).reshape(-1)
        up = mu + std
        down = mu - std
        return mu, up, down

    cg_mu, cg_up, cg_down = gen_mean_up_down(mu_test, var_test)
    # gpr_mu, gpr_up, gpr_down = gen_mean_up_down(gpr_mu_test, gpr_var_test)
    # cluster_mu, cluster_up, cluster_down = gen_mean_up_down(cluster_mu_test, cluster_var_test)

    blue = "tab:blue"
    gray = "tab:gray"
    orange = "tab:orange"

    ax2.plot(x_test, mu_test, color=gray)
    ax2.scatter(x, y, color=gray, alpha=0.5, s=8)

    ax2.fill_between(x_test_flat, cg_up, cg_down, color=gray, alpha=0.5)

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

    # ax3.plot(x_test, gpr_mu_test, color=blue, label="GPR")
    # ax3.fill_between(x_test_flat, gpr_up, gpr_down, color=blue, alpha=0.2)

    ax3.plot(x_test, mu_test, color=gray, label="GP clustering (with CG)")
    ax3.fill_between(x_test_flat, cg_up, cg_down, color=gray, alpha=0.2)

    # ax3.plot(x_test, cluster_mu_test, color=orange, label="GP clustering")
    # ax3.fill_between(x_test_flat, cluster_up, cluster_down, color=orange, alpha=0.2)

    ax3.scatter(x, y, color=gray, alpha=0.5, s=8)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("cggp.pdf")
    plt.show()
    print("end")
