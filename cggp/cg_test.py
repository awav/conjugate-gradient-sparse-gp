import numpy as np
import tensorflow as tf
import pytest

from gpflow.config import default_float
from gpflow.kernels import SquaredExponential
from models import eval_logdet
from conjugate_gradient import ConjugateGradient
from utils import add_diagonal


@pytest.mark.parametrize(
    "dimension,num_inputs, num_systems, max_error, kernel_class",
    [(2, 100, 5, 1e-12, SquaredExponential)],
)
def test_cg(dimension, num_inputs, num_systems, max_error, kernel_class):
    """
    Test conjugate gradient for solving system of equations
    """
    inputs = np.random.randn(num_inputs, dimension)
    lengthscales = np.random.rand(dimension) ** 2 + 0.5
    variance = 1.3
    noise_variance = 0.1**2
    kernel = kernel_class(lengthscales=lengthscales, variance=variance)

    rhs = np.random.randn(num_inputs, num_systems)

    with tf.GradientTape(persistent=True) as t:
        matrix = kernel(inputs)
        matrix = add_diagonal(
            matrix, noise_variance * tf.ones(matrix.shape[0], dtype=default_float())
        )

        inv_solution = tf.linalg.solve(matrix, rhs)
        inv_sum = tf.reduce_sum(inv_solution)
        cg = ConjugateGradient(max_error)
        cg_solution = tf.transpose(cg(matrix, tf.transpose(rhs)))
        cg_sum = tf.reduce_sum(cg_solution)

    inv_solution_grad = t.gradient(inv_sum, kernel.trainable_variables)
    cg_solution_grad = t.gradient(cg_sum, kernel.trainable_variables)

    np.testing.assert_allclose(cg_solution, inv_solution, rtol=1e-3, atol=1e-4)

    for g1, g2 in zip(cg_solution_grad, inv_solution_grad):
        np.testing.assert_allclose(g1, g2, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "dimension,num_inputs, num_systems, max_error, kernel_class",
    [(2, 100, 5, 1e-12, SquaredExponential)],
)
def test_log_determinant_grad(dimension, num_inputs, num_systems, max_error, kernel_class):
    """
    Test conjugate gradient for solving system of equations
    """
    inputs = np.random.randn(num_inputs, dimension)
    lengthscales = np.random.rand(dimension) ** 2 + 0.5
    variance = 1.3
    noise_variance = 0.1**2
    kernel = kernel_class(lengthscales=lengthscales, variance=variance)

    # Compute log det and gradient directly
    with tf.GradientTape(persistent=True) as t:
        matrix = kernel(inputs)
        matrix = add_diagonal(
            matrix, noise_variance * tf.ones(matrix.shape[0], dtype=default_float())
        )
        logdet = tf.linalg.logdet(matrix)
        logdet_cg = eval_logdet(matrix, ConjugateGradient(max_error))

    logdet_grad = t.gradient(logdet, kernel.trainable_variables)
    logdet_grad_cg = t.gradient(logdet_cg, kernel.trainable_variables)
    np.testing.assert_allclose(0, logdet_cg, rtol=1e-3, atol=1e-4)

    for g1, g2 in zip(logdet_grad, logdet_grad_cg):
        np.testing.assert_allclose(g1, g2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main(args=[__file__])
