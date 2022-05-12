import numpy as np
import tensorflow as tf
import pytest

from gpflow.config import default_float
from gpflow.kernels import SquaredExponential
from models import eval_logdet
from conjugate_gradient import ConjugateGradient
from utils import add_diagonal


@pytest.mark.parametrize("dimension,num_inputs, num_systems, max_error, kernel_class", [(2, 100, 5, 1e-12, SquaredExponential)])
def test_cg(dimension, num_inputs, num_systems, max_error, kernel_class):
    """
    Test conjugate gradient for solving system of equations
    """
    inputs = np.random.randn(num_inputs, dimension)
    lengthscales = np.random.rand(dimension) ** 2 + 0.5
    variance = 1.3
    noise_variance = 0.1 ** 2
    kernel = kernel_class(lengthscales=lengthscales, variance=variance)
    matrix = kernel(inputs)
    matrix = add_diagonal(matrix, noise_variance * tf.ones(matrix.shape[0], dtype=default_float()))
    rhs = np.random.randn(num_systems, num_inputs)
    cg = ConjugateGradient(max_error)
    cg_solution = cg(matrix, rhs)
    cg_solution = tf.transpose(cg_solution)
    inv_soln = tf.linalg.solve(matrix, tf.transpose(rhs))
    np.testing.assert_allclose(cg_solution, inv_soln, rtol=1e-3, atol=1e-4)

@pytest.mark.parametrize("dimension,num_inputs, num_systems, max_error, kernel_class", [(2, 100, 5, 1e-12, SquaredExponential)])
def test_log_determinant_grad(dimension, num_inputs, num_systems, max_error, kernel_class):
    """
    Test conjugate gradient for solving system of equations
    """
    inputs = np.random.randn(num_inputs, dimension)
    lengthscales = np.random.rand(dimension) ** 2 + 0.5
    variance = 1.3
    noise_variance = 0.1 ** 2

    # Compute log det and gradient directly
    with tf.GradientTape() as t:
        kernel = kernel_class(lengthscales=lengthscales, variance=variance)
        matrix = kernel(inputs)
        matrix = add_diagonal(matrix, noise_variance * tf.ones(matrix.shape[0], dtype=default_float()))
        logdet = tf.linalg.logdet(matrix)
        logdet_grad = t.gradient(logdet, kernel.trainable_variables)

    # compute gradient via CG
    with tf.GradientTape() as t:
        kernel = kernel_class(lengthscales=lengthscales, variance=variance)
        matrix = kernel(inputs)
        matrix = add_diagonal(matrix, noise_variance * tf.ones(matrix.shape[0], dtype=default_float()))
        conjugate_gradient = ConjugateGradient(max_error)
        logdet = eval_logdet(matrix, conjugate_gradient)

    logdet_grad_cg = t.gradient(logdet, kernel.trainable_variables)
    
    for g1, g2 in zip(logdet_grad, logdet_grad_cg):
        np.testing.assert_allclose(g1, g2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main()