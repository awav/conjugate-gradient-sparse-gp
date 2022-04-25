from models import LpSVGP, ClusterSVGP
from data import snelson1d
import gpflow
import tensorflow as tf
import numpy as np

from rff import rff_sample
from tensorflow_probability import distributions as tfd

from utils import add_diagonal
from playground_util import (
    create_model,
)


if __name__ == "__main__":
    seed = 111
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_data, _ = snelson1d()
    distance_type = "covariance"
    num_inducing_points = 20
    num_iterations = 1000
    num_bases = 100
    num_samples = 10

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

    u = experimental_model.pseudo_u
    iv = experimental_model.inducing_variable.Z
    kernel = experimental_model.kernel
    lambda_diag = experimental_model.diag_variance[:, 0]
    q_mu, q_var = experimental_model.q_moments()
 
    epsilon_mvn = tfd.MultivariateNormalDiag(scale_diag=lambda_diag)
    epsilon = epsilon_mvn.sample()
    kzz = gpflow.covariances.Kuu(iv, kernel, jitter=0.0)
    kzz_lambda = add_diagonal(kzz, lambda_diag[:, 0])
    L = tf.linalg.cholesky(kzz_lambda)
    solve_against = u - q_mu - epsilon
    solution = tf.linalg.cholesky_solve(L, solve_against)

    correction_term = kzz @ solution
    prior_term = rff_sample(iv, kernel, num_bases, num_samples)
    sample = prior_term - correction_term

    # Plotting
    # TODO(awav)