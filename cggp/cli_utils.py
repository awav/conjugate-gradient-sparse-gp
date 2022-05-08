from typing import Callable

import gpflow
import numpy as np
import tensorflow as tf

from gpflow.config import default_float
from kmeans import kmeans_lloyd, create_kernel_distance_fn
from optimize import kmeans_update_inducing_parameters
from utils import jit


def create_model(
    model_fn: Callable,
    kernel_fn: Callable,
    data,
    num_inducing_points: int,
):
    x = np.array(data[0])
    n = x.shape[0]
    dim = x.shape[-1]
    default_variance = 0.1

    rand_indices = np.random.choice(n, size=num_inducing_points, replace=False)
    iv = tf.convert_to_tensor(x[rand_indices, ...], dtype=default_float())

    likelihood = gpflow.likelihoods.Gaussian(variance=default_variance)
    kernel = kernel_fn(dim)
    model = model_fn(kernel, likelihood, iv, num_data=n)

    return model


def create_update_fn(model, data, num_inducing_points, use_jit: bool = True):
    x, _ = data
    distance_fn = create_kernel_distance_fn(model.kernel, "covariance")
    distance_fn = jit(use_jit)(distance_fn)

    @jit(use_jit)
    def clustering_fn():
        iv, _ = kmeans_lloyd(x, num_inducing_points, distance_fn=distance_fn)
        return iv

    def update_fn():
        kmeans_update_inducing_parameters(model, data, distance_fn, clustering_fn)

    return update_fn
