from typing import Dict, Literal, Type, Union, Sequence
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from gpflow.kernels import SquaredExponential, Matern12, Matern32, Matern52

Tensor = tf.Tensor
Kernel = Union[SquaredExponential, Matern12, Matern32, Matern52]
Shape = Sequence
SmoothnessIndex = Literal[1, 3, 5]


__smoothness: Dict[Type, SmoothnessIndex] = {
    Matern12: 1,
    Matern32: 3,
    Matern52: 5,
}


def basis_theta_parameter(kernel: Kernel, num_bases: int) -> Tensor:
    """
    Args:
        kernel: A gpflow kernel. Allowed kernels are squared exponential,
            matern kernels with smoothness parameters 1/2, 3/2, 5/2.
        shape: Shape of the output tensor.
    Result: Tensor with shape defined by `shape` argument.
    """
    matern_classes = [Matern12, Matern32, Matern52]
    kernel_class = kernel.__class__
    lengthscale_inv = tf.math.reciprocal(kernel.lengthscales)
    sample_shape = (num_bases,)
    # dimension = tf.size(lengthscale_inv)

    if kernel_class == SquaredExponential:
        # Spectral density of the SquaredExponential
        return sample_mvn(lengthscale_inv, sample_shape)
        # return tf.random.normal(shape, dtype=dtype)
    elif kernel_class in matern_classes:
        nu = __smoothness[kernel_class]
        nu = tf.convert_to_tensor(nu, dtype=lengthscale_inv.dtype)
        # Spectral density of Matern kernels
        return sample_student_t(nu, lengthscale_inv, sample_shape)

    raise ValueError(f"Not supported kernel class {kernel_class}")


def basis_vectors(inputs: Tensor, theta: Tensor) -> Tensor:
    """
    Args:
        input: Tensor. Shape [N, D]
        theta: Tensor. Shape [L, D]
    Return: Tensor of shape [N, 2L]
    """
    x_theta = tf.matmul(inputs, theta, transpose_b=True)
    features = [tf.math.cos(x_theta), tf.math.sin(x_theta)]
    return tf.concat(features, axis=-1)


def rff_sample(inputs: Tensor, kernel: Kernel, num_bases: int, num_samples: int = 1) -> Tensor:
    dtype = kernel.lengthscales.dtype
    variance = kernel.variance

    theta = basis_theta_parameter(kernel, num_bases)
    bases = basis_vectors(inputs, theta=theta)
    scale = tf.sqrt(tf.math.truediv(variance, num_bases))
    bases *= scale

    weigths_shape = (num_samples, bases.shape[-1])
    weights = tf.random.normal(weigths_shape, dtype=dtype)
    samples = tf.matmul(weights, bases, transpose_b=True)

    return samples


def sample_mvn(scale_diag: Tensor, sample_shape: Shape) -> Tensor:
    mvn = tfd.MultivariateNormalDiag(scale_diag=scale_diag)
    sample = mvn.sample(sample_shape)
    return sample


def sample_student_t(nu: Tensor, scale_diag: Tensor, sample_shape: Shape) -> Tensor:
    mvn_sample = sample_mvn(scale_diag, sample_shape)
    alpha = beta = 0.5 * nu
    gamma_distr = tfd.Gamma(alpha, beta)
    gamma_sample = gamma_distr.sample(sample_shape)[:, None]
    sample = tf.sqrt(tf.truediv(nu, gamma_sample)) * mvn_sample
    return sample
