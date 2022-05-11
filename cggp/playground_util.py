from typing import Callable, Optional, TypeVar
from functools import partial
from distance import create_kernel_distance_fn, DistanceType
from kmeans import (
    kmeans_lloyd,
    kmeans_indices_and_distances,
)
from models import LpSVGP, ClusterGP
import gpflow
import tensorflow as tf

from gpflow.config import default_float


ModelClass = TypeVar("ModelClass", type(LpSVGP), type(ClusterGP))


def create_model(
    data, num_inducing_points: int, distance_type: DistanceType, model_class: ModelClass
):
    x, y = data
    xt = tf.convert_to_tensor(x, dtype=default_float())
    yt = tf.convert_to_tensor(y, dtype=default_float())
    n = x.shape[0]

    lengthscale = [1.0]
    variance = 0.1
    kernel = gpflow.kernels.SquaredExponential(variance=variance, lengthscales=lengthscale)
    likelihood = gpflow.likelihoods.Gaussian(variance=0.1)

    distance_fn = create_kernel_distance_fn(kernel, distance_type)
    kmeans_fn = tf.function(partial(kmeans_lloyd, distance_fn=distance_fn))

    def clustering_fn():
        iv, _ = kmeans_fn(xt, num_inducing_points)
        return iv

    iv = clustering_fn()

    model = model_class(kernel, likelihood, iv, num_data=n)

    gpflow.utilities.set_trainable(model.inducing_variable, False)
    return (xt, yt), model, clustering_fn, distance_fn

