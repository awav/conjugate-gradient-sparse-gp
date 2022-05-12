from typing import Literal, Callable

import gpflow
import numpy as np
import tensorflow as tf

from gpflow.config import default_float
from kmeans import kmeans_lloyd
from distance import create_kernel_distance_fn
from optimize import kmeans_update_inducing_parameters, covertree_update_inducing_parameters
from utils import jit


ClusteringType = Literal["kmeans", "covertree"]


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


def create_kmeans_update_fn(model, data, num_inducing_points: int, use_jit: bool = True):
    x, _ = data
    distance_fn = create_kernel_distance_fn(model.kernel, "covariance")
    distance_fn = jit(use_jit)(distance_fn)

    @jit(use_jit)
    def clustering_fn():
        iv_init = model.inducing_variable.Z
        iv, _ = kmeans_lloyd(
            x, num_inducing_points, initial_centroids=iv_init, distance_fn=distance_fn
        )
        return iv

    def update_fn():
        return kmeans_update_inducing_parameters(model, data, distance_fn, clustering_fn)

    return update_fn


def create_covertree_update_fn(model, data, use_jit: bool = True):
    distance_fn = create_kernel_distance_fn(model.kernel, "covariance")
    distance_fn = jit(use_jit)(distance_fn)

    def update_fn():
        return covertree_update_inducing_parameters(model, data, distance_fn)

    return update_fn


def create_update_fn(
    clustering_type: ClusteringType,
    model,
    data,
    num_inducing_points,
    use_jit: bool = True,
):
    if clustering_type == "kmeans":
        return create_kmeans_update_fn(model, data, num_inducing_points, use_jit=use_jit)
    elif clustering_type == "covertree":
        return create_covertree_update_fn(model, data, use_jit=use_jit)
    raise ValueError(f"Unknown value for {clustering_type}")
