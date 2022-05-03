from pathlib import Path
from models import ClusterSVGP, PathwiseClusterSVGP
from data import snelson1d
import gpflow
import tensorflow as tf
import numpy as np
from numpy import newaxis
from rff import rff_sample

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
    num_bases = 4099*8

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

    np.testing.assert_allclose(experimental_model.likelihood.variance.numpy(), pathwise_model.likelihood.variance.numpy())

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

    num_samples = 4111*2
    samples = pathwise_model.pathwise_samples(x, num_samples, num_bases)
    q_mu_x, q_var_x = experimental_model.predict_f(xt)

    likelihood_samples = compute_expectation_samples(samples, y)
    likelihood_analytic = compute_expectation_analytic(q_mu_x, q_var_x, y)

    print(f"Expectation estimated: {likelihood_samples.numpy().sum()}")
    print(f"Expectation analytic: {likelihood_analytic.numpy().sum()}")





    num_samples = 4111*2
    likelihood = pathwise_model.compute_likelihood_term(
        data,
        num_bases=num_bases,
        num_samples=num_samples,
    )

    def compute_expected_likelihood_term(
        model,
        q_mu,
        q_var,
        y
        ):
        
        N = y.shape[0]
        noise = model.likelihood.variance
        noise_inv = tf.math.reciprocal(noise)
        expected_error_squared = compute_expectation_analytic(q_mu, q_var, y)
        likelihood_term = tf.reduce_sum(0.5 * noise_inv * expected_error_squared)
        constant_term = N * 0.5 * tf.math.log(noise)
        return likelihood_term + constant_term

    q_mu_x, q_var_x = experimental_model.predict_f(xt)
    likelihood_ground_truth = compute_expected_likelihood_term(experimental_model, q_mu_x, q_var_x, yt)

    print(f"Likelihood term estimated: {likelihood.numpy()}")
    print(f"Likelihood term expected: {likelihood_ground_truth.numpy()}")
    # np.testing.assert_allclose(likelihood, likelihood_ground_truth)

    ###########
    ## Plotting

    num_samples = 4127*3
    samples = pathwise_model.pathwise_samples(x_test, num_bases, num_samples)
    samples_expected = experimental_model.predict_f_samples(x_test, num_samples=num_samples)

    blue = "tab:blue"
    gray = "tab:gray"

    fig, ax = plt.subplots(1, 1)
    samples = samples.numpy().squeeze().T
    samples_expected = samples_expected.numpy().squeeze().T

    ax.set_xlim(x_test.min(), x_test.max())
    ax.plot(x_test, samples_expected, alpha=0.1, color="tab:orange")
    ax.plot(x_test, samples, alpha=0.1, color="tab:blue")

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

    print()
