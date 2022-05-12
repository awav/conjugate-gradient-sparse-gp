import numpy as np
import tensorflow as tf
import pytest

from gpflow.kernels import SquaredExponential
from gpflow.models import GPR
from gpflow.likelihoods import Gaussian
from models import LpSVGP, ClusterGP


@pytest.mark.parametrize("num_inputs, kernel_class", [(100, 1e-12, SquaredExponential)])
def test_cluster_elbo(num_inputs, kernel_class):
    """
    Test corner case when z=x, should have that ELBO = LML for cluster GP.
    """
    # generate some data
    inputs = np.random.randn(num_inputs, 1)
    lengthscales = np.random.rand() ** 2 + 0.5
    variance = 1.3
    noise_variance = 0.1 ** 2
    outputs = np.sin(inputs) + noise_variance * np.random.randn(num_inputs, 1)
    data = (inputs, outputs)
    kernel = kernel_class(lengthscales=lengthscales, variance=variance)
    
    gpr = GPR(data, kernel, noise_variance=noise_variance)
    lml = gpr.log_marginal_likelihood()
    
    likelihood = Gaussian(variance=noise_variance)
    pseudo_u = outputs
    diag_variance = noise_variance * np.ones((num_inputs, 1))
    cluster_gp = ClusterGP(kernel, likelihood, inducing_variable=inputs, pseudo_u=pseudo_u, diag_variance=diag_variance)
    elbo = cluster_gp.elbo(data)
    np.testing.assert_allclose(elbo, lml, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main()

