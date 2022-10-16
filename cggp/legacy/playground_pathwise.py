from pathlib import Path
from models import ClusterGP, PathwiseClusterGP
from data import snelson1d
import tensorflow as tf
import numpy as np
from numpy import newaxis

from cli_utils import create_model_and_kmeans_update_fn
from optimize import train_using_lbfgs_and_update

import matplotlib.pyplot as plt


Tensor = tf.Tensor


def setup():
    seed = 111
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_data, _ = snelson1d()
    distance_type = "covariance"
    num_inducing_points = 30
    num_iterations = 1000
    use_jit = False

    x, y = train_data

    model_class = ClusterGP
    experimental_model, update_fn = create_model_and_kmeans_update_fn(
        model_class,
        train_data,
        num_inducing_points,
        distance_type=distance_type,
        use_jit=use_jit
    )

    num_iterations = 100
    opt_res = train_using_lbfgs_and_update(
        train_data,
        experimental_model,
        update_fn,
        num_iterations,
    )

    pathwise_model = PathwiseClusterGP(
        experimental_model.kernel,
        experimental_model.likelihood,
        experimental_model.inducing_variable,
        cluster_counts=experimental_model.cluster_counts,
        pseudo_u=experimental_model.pseudo_u,
    )

    return train_data, experimental_model, pathwise_model


def test_likelihood_terms():
    num_bases = 3333
    num_samples = 5555

    data, experimental_model, pathwise_model = setup()
    x, y = data

    def compute_expectation_samples(
        samples,
        y,
    ):
        error_squared = tf.square(y[newaxis, ...] - samples)
        expected_error_squared = tf.reduce_mean(error_squared, axis=0)
        return expected_error_squared

    def compute_expectation_analytic(
        q_mu,
        q_var,
        y,
    ):
        expected_f_sq = q_var + q_mu**2
        expected_error_squared = y**2 - (2 * y * q_mu) + expected_f_sq
        return expected_error_squared

    def compute_expected_likelihood_term(model, q_mu, q_var, y):
        N = y.shape[0]
        noise = model.likelihood.variance
        noise_inv = tf.math.reciprocal(noise)
        expected_error_squared = compute_expectation_analytic(q_mu, q_var, y)
        likelihood_term = tf.reduce_sum(0.5 * noise_inv * expected_error_squared)

        constant_term = 0.5 * N * tf.math.log(2 * np.pi * noise)
        return -(likelihood_term + constant_term)

    likelihood = pathwise_model.compute_likelihood_term(
        data,
        num_bases=num_bases,
        num_samples=num_samples,
    )

    q_mu_x, q_var_x = experimental_model.predict_f(x)
    likelihoods = experimental_model.likelihood.variational_expectations(q_mu_x, q_var_x, y)
    likelihood_ground_truth_gpflow = tf.reduce_sum(likelihoods)

    likelihood_ground_truth_sqmean = compute_expected_likelihood_term(
        experimental_model, q_mu_x, q_var_x, y
    )

    print(f"Likelihood term estimated: {likelihood.numpy()}")
    print(f"Likelihood term expected (gpflow): {likelihood_ground_truth_gpflow.numpy()}")
    print(f"Likelihood term expected (sqmean): {likelihood_ground_truth_sqmean.numpy()}")

    ########
    # np.testing.assert_allclose(likelihood, likelihood_ground_truth)

    samples = pathwise_model.pathwise_samples(x, num_bases, num_samples)
    q_mu_x, q_var_x = experimental_model.predict_f(x)

    likelihood_samples = compute_expectation_samples(samples, y)
    likelihood_analytic = compute_expectation_analytic(q_mu_x, q_var_x, y)

    print(f"Expectation estimated: {likelihood_samples.numpy().sum()}")
    print(f"Expectation analytic: {likelihood_analytic.numpy().sum()}")


def plotting_samples():
    data, experimental_model, pathwise_model = setup()
    x, y = data
    xmin = x.min() - 1.0
    xmax = x.max() + 1.0
    num_test_points = 100
    x_test = np.linspace(xmin, xmax, num_test_points).reshape(-1, 1)

    num_bases = 1234
    num_samples = 2
    samples = pathwise_model.pathwise_samples(x_test, num_bases, num_samples)
    samples_expected = experimental_model.predict_f_samples(x_test, num_samples=num_samples)

    blue = "tab:blue"
    gray = "tab:gray"
    orange = "tab:orange"

    fig, ax = plt.subplots(1, 1)
    samples = samples.numpy().squeeze().T
    samples_expected = samples_expected.numpy().squeeze().T

    ax.set_xlim(x_test.min(), x_test.max())
    ax.plot(x_test, samples, color=blue)
    ax.plot(x_test, samples_expected, color=orange)

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
    plt.savefig("pathwise.pdf")
    plt.show()
    pass


if __name__ == "__main__":
    test_likelihood_terms()
