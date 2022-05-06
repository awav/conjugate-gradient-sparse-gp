import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt

from rff import rff_sample
from gpflow.utilities import add_noise_cov as add_diag_scalar
from gpflow.config import default_jitter


def samples_rff_plotting():
    seed = 100
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_bases = 100000
    num_data = 1000
    num_samples = 3

    lengthscale = 0.555
    variance = 0.333

    # kernel_class = gpflow.kernels.SquaredExponential
    kernel_class = gpflow.kernels.Matern12
    # kernel_class = gpflow.kernels.Matern32
    # kernel_class = gpflow.kernels.Matern52
    kernel = kernel_class(
        lengthscales=[lengthscale],
        variance=variance,
    )

    inputs = np.linspace(0, 10, num_data).reshape(-1, 1)
    sample = rff_sample(inputs, kernel, num_bases, num_samples=num_samples)
    sample = sample.numpy().T

    kxx = kernel(inputs)
    jitter = tf.convert_to_tensor(default_jitter(), dtype=kxx.dtype)
    kxx_jitter = add_diag_scalar(kxx, jitter)
    kxx_cholesky = tf.linalg.cholesky(kxx_jitter)
    epsilon = tf.random.normal((num_data, num_samples), dtype=sample.dtype)
    gp_prior_sample = kxx_cholesky @ epsilon

    figsize = (7, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
    inputs_flat = inputs.reshape(-1)
    ax1.plot(inputs_flat, sample, label="RFF")
    ax2.plot(inputs_flat, gp_prior_sample, label="Cholesky")

    ax1.set_title("RFF prior samples")
    ax2.set_title("Cholesky prior samples")
    ax1.set_ylabel("$f$")
    ax1.set_xlabel("Inputs")
    ax2.set_xlabel("Inputs")
    plt.tight_layout()
    plt.show()
    print()


if __name__ == "__main__":
    samples_rff_plotting()
