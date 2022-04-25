import unittest
import numpy as np
import tensorflow as tf

from rff import basis_vectors, basis_theta_parameter
from gpflow.kernels import RBF, Matern52, Matern12, Matern32


dtype = tf.float64

def test_rff_kernel(dimension, num_inputs, num_bases, kernel):
    x_values = np.random.randn(num_inputs, dimension)
    kxx = kernel(x_values)
    theta = basis_theta_parameter(kernel, num_bases=num_bases)
    basis_vs = basis_vectors(x_values, theta=theta)
    scale_sq = tf.math.truediv(kernel.variance, tf.cast(num_bases, dtype=dtype)) 
    rff_approx = scale_sq * tf.matmul(basis_vs, basis_vs, transpose_b=True)
    return np.allclose(rff_approx, kxx, rtol=1e-3, atol=1e-2)


class TestRFF(unittest.TestCase):
    def test_rbf_basis(self):
        """
        Test basis functions for RFF for RBF kernel, construct basis functions, and check that 
        phi phi.T -> Kxx
        """
        dimension = 2
        num_inputs = 4
        num_bases = 50000000
        lengthscales = np.random.rand(dimension) ** 2 + 0.5
        variance = 1.3
        kernel = RBF(variance=variance, lengthscales=lengthscales)
        assert test_rff_kernel(dimension, num_inputs, num_bases, kernel)


    def test_matern52_basis(self):
        """
        Test basis functions for RFF for Matern 5/2 kernel, construct basis functions, and check that 
        phi phi.T -> Kxx
        """
        dimension = 2
        num_inputs = 4
        num_bases = 50000000
        lengthscales = np.random.rand(dimension) ** 2 + 0.5
        variance = 1.3
        kernel = Matern52(variance=variance, lengthscales=lengthscales)
        assert test_rff_kernel(dimension, num_inputs, num_bases, kernel)
        


if __name__ == '__main__':
    unittest.main()