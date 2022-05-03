import numpy as np
import tensorflow as tf
import pytest

from rff import basis_vectors, basis_theta_parameter, rff_sample
from gpflow.kernels import SquaredExponential, Matern52, Matern32


@pytest.mark.parametrize("dimension,num_inputs,num_bases", [(2, 4, int(1e5))])
@pytest.mark.parametrize("kernel_class", [SquaredExponential, Matern32, Matern52])
def test_rff_kernel(dimension, num_inputs, num_bases, kernel_class):
    """
    Test basis functions for RFF for RBF and Matern kernels.
    Construct basis functions, and check that
    phi phi.T -> Kxx
    """
    inputs = np.random.randn(num_inputs, dimension)
    lengthscales = np.random.rand(dimension) ** 2 + 0.5
    variance = 1.3
    kernel = kernel_class(variance=variance, lengthscales=lengthscales)

    theta = basis_theta_parameter(kernel, num_bases=num_bases)
    basis_vs = basis_vectors(inputs, theta=theta)
    scale_sq = tf.math.truediv(kernel.variance, num_bases)
    rff_approx = scale_sq * tf.matmul(basis_vs, basis_vs, transpose_b=True)

    kxx = kernel(inputs)
    np.testing.assert_allclose(rff_approx, kxx, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("dimension,num_inputs,num_bases,num_samples", [(2, 4, int(1e4), 10000)])
@pytest.mark.parametrize("kernel_class", [SquaredExponential, Matern32, Matern52])
def test_rff_sample(dimension, num_inputs, num_bases, num_samples, kernel_class):
    """
    Test sampling functions for RFF for RBF and Matern kernels.
    Construct basis functions, and check that
    phi phi.T -> Kxx
    """
    inputs = np.random.randn(num_inputs, dimension)
    lengthscales = np.random.rand(dimension) ** 2 + 0.5
    variance = 1.3
    kernel = kernel_class(variance=variance, lengthscales=lengthscales)

    f = rff_sample(inputs, kernel, num_bases, num_samples)
    f_cov = np.cov(f, rowvar=False)

    kxx = kernel(inputs)
    np.testing.assert_allclose(f_cov, kxx, rtol=1e-3, atol=0.04)


if __name__ == "__main__":
    pytest.main()
