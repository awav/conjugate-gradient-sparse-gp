import numpy as np
import tensorflow as tf
import gpflow

from models import ClusterGP
from data import snelson1d

from cli_utils import create_model_and_kmeans_update_fn
from optimize import train_using_lbfgs_and_update

from utils import add_diagonal
from conjugate_gradient import conjugate_gradient


def logdet_gradient_estimation():
    def gradient_logdet(matrix):
        with tf.GradientTape() as tape:
            tape.watch(matrix)
            logdet = tf.linalg.logdet(matrix)
        gradient = tape.gradient(logdet, matrix)
        return gradient

    def gradient_logdet_cg(matrix):
        threshold = 1e-9
        dtype = matrix.dtype
        identity = tf.linalg.diag(tf.ones(matrix.shape[0], dtype=dtype))
        initial_solution = tf.zeros(matrix.shape, dtype=dtype)
        gradient, stats = conjugate_gradient(matrix, identity, initial_solution, threshold)
        gradient = tf.transpose(gradient)
        return gradient, stats

    def gradient_logdet_solve(matrix):
        dtype = matrix.dtype
        identity = tf.linalg.diag(tf.ones(matrix.shape[0], dtype=dtype))
        gradient = tf.linalg.solve(matrix, identity)
        return gradient

    seed = 111
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_data, _ = snelson1d()
    distance_type = "covariance"
    num_inducing_points = 20

    model_class = ClusterGP
    model, update_fn = create_model_and_kmeans_update_fn(
        model_class,
        train_data,
        num_inducing_points,
        distance_type=distance_type,
    )

    opt_res = train_using_lbfgs_and_update(train_data, model, update_fn, 0)

    iv = model.inducing_variable
    kuu = gpflow.covariances.Kuu(iv, model.kernel)
    lambda_diag = model.diag_variance[:, 0]
    kuu_plus_lambda = add_diagonal(kuu, lambda_diag)

    eig_vals = tf.linalg.eigvalsh(kuu_plus_lambda)
    max_eig_val = np.max(eig_vals)
    min_eig_val = np.min(eig_vals)

    grad_expected = gradient_logdet(kuu_plus_lambda)
    grad_solve = gradient_logdet_solve(kuu_plus_lambda)
    grad_cg, (cg_steps, cg_error) = gradient_logdet_cg(kuu_plus_lambda)

    res_solve = np.allclose(grad_solve, grad_expected)
    res_cg = np.allclose(grad_cg, grad_expected)

    cg_avg_rel_error = np.mean(np.abs(grad_expected - grad_cg) / grad_expected)

    print(f"(Solve) Gradients are {'NOT ' if not res_solve else ''}equal")
    print(f"(CG) Gradients are {'NOT ' if not res_cg else ''}equal")
    print(f"(CG) Gradients average relative error {cg_avg_rel_error}")

    print(f"Conjugate gradient spent {cg_steps} iterations")
    print(f"Conjugate gradient stopped with {cg_error}")
    print(f"Max eigenvalue: {max_eig_val}")
    print(f"Min eigenvalue: {min_eig_val}")
    print(f"Condition number: {max_eig_val / min_eig_val}")

    print()


if __name__ == "__main__":
    logdet_gradient_estimation()
