import unittest
import numpy as np
import tensorflow as tf

from rff import basis_vectors, basis_theta_parameter
from gpflow.kernels import RBF, Matern52

class TestRFF(unittest.TestCase):
    def test_rbf_basis(self):
        """
        Test basis functions for rff
        """
        dimension = 2
        num_inputs = 4
        num_bases = 50000000
        lengthscales = np.random.rand(dimension) ** 2 + 0.5
        variance = 1.3
        kernel = RBF(variance=variance, lengthscales=lengthscales)
        x_values = np.random.randn(num_inputs, dimension)
        kxx = kernel(x_values)
        theta = basis_theta_parameter(kernel, num_bases=num_bases)
        basis_vs = basis_vectors(x_values, theta=theta)
        scale_sq = tf.math.truediv(kernel.variance, tf.cast(num_bases, dtype=tf.float64)) 
        rff_approx = scale_sq * tf.matmul(basis_vs, basis_vs, transpose_b=True)
        assert np.allclose(rff_approx, kxx, rtol=1e-3, atol=1e-2)

    def test_matern52_basis(self):
        """
        Test basis functions for rff
        """
        dimension = 2
        num_inputs = 4
        num_bases = 50000000
        lengthscales = np.random.rand(dimension) ** 2 + 0.5
        variance = 1.3
        kernel = Matern52(variance=variance, lengthscales=lengthscales)
        x_values = np.random.randn(num_inputs, dimension)
        kxx = kernel(x_values)
        theta = basis_theta_parameter(kernel, num_bases=num_bases)
        basis_vs = basis_vectors(x_values, theta=theta)
        scale_sq = tf.math.truediv(kernel.variance, tf.cast(num_bases, dtype=tf.float64)) 
        rff_approx = scale_sq * tf.matmul(basis_vs, basis_vs, transpose_b=True)
        assert np.allclose(rff_approx, kxx, rtol=1e-3, atol=1e-2)


if __name__ == '__main__':
    unittest.main()