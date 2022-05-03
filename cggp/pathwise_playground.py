from pathlib import Path
from models import ClusterSVGP, PathwiseClusterSVGP
from data import snelson1d
import tensorflow as tf
import numpy as np

from playground_util import create_model, train_using_lbfgs_and_varpar_update

import matplotlib.pyplot as plt


Tensor = tf.Tensor


if __name__ == "__main__":
    seed = 111
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_data, _ = snelson1d()
    distance_type = "covariance"
    num_inducing_points = 30
    num_iterations = 1000
    num_bases = 1234

    x, y = train_data

    xmin = x.min() - 1.0
    xmax = x.max() + 1.0
    num_test_points = 100
    x_test = np.linspace(xmin, xmax, num_test_points).reshape(-1, 1)

    model_class = ClusterSVGP
    data, experimental_model, clustering_fn, distance_fn = create_model(
        (x, y),
        num_inducing_points,
        distance_type,
        model_class,
    )

    num_iterations = 100
    opt_res = train_using_lbfgs_and_varpar_update(
        data,
        experimental_model,
        clustering_fn,
        num_iterations,
    )

    pathwise_model = PathwiseClusterSVGP(
        experimental_model.kernel,
        experimental_model.likelihood,
        experimental_model.inducing_variable,
        diag_variance=experimental_model.diag_variance,
        pseudo_u=experimental_model.pseudo_u,
    )

    xt, yt = data

    #######
    ## Test

    num_samples = 1111
    likelihood = pathwise_model.compute_likelihood_term(
        data,
        num_bases=num_bases,
        num_samples=num_samples,
    )

    q_mu_x, q_var_x = experimental_model.predict_f(xt)
    likelihoods = experimental_model.likelihood.variational_expectations(q_mu_x, q_var_x, yt)
    likelihood_ground_truth = tf.reduce_sum(likelihoods)

    print(f"Likelihood term estimated: {likelihood.numpy()}")
    print(f"Likelihood term expected: {likelihood_ground_truth.numpy()}")
    # np.testing.assert_allclose(likelihood, likelihood_ground_truth)

    ###########
    ## Plotting

    num_samples = 2
    samples = pathwise_model.pathwise_samples(x_test, num_bases, num_samples)
    samples_expected = experimental_model.predict_f_samples(x_test, num_samples=num_samples)

    blue = "tab:blue"
    gray = "tab:gray"

    fig, ax = plt.subplots(1, 1)
    samples = samples.numpy().squeeze().T
    samples_expected = samples_expected.numpy().squeeze().T

    ax.set_xlim(x_test.min(), x_test.max())
    ax.plot(x_test, samples, color="tab:blue")
    ax.plot(x_test, samples_expected, color="tab:orange")

    f_mu, f_var = experimental_model.predict_f(x_test)
    f_std = np.sqrt(f_var.numpy())
    f_mu = f_mu.numpy().reshape(-1)
    f_std = f_std.reshape(-1)
    up = f_mu + f_std
    down = f_mu - f_std

    ax.plot(x_test, f_mu, color=gray)
    ax.fill_between(x_test.reshape(-1), up, down, color=gray, alpha=0.5)

    ax.scatter(x, y, color=gray, alpha=0.5, s=8)

    plt.tight_layout()
    plt.show()

    print()
