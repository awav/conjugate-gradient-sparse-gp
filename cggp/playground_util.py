from typing import Callable, Optional, TypeVar
from functools import partial
from distance import create_kernel_distance_fn, DistanceType
from kmeans import (
    kmeans_lloyd,
    kmeans_indices_and_distances,
)
from cli_utils import create_model, create_update_fn
from models import LpSVGP, ClusterGP
import gpflow
import tensorflow as tf

from gpflow.config import default_float


ModelClass = TypeVar("ModelClass", type(LpSVGP), type(ClusterGP))


def kernel_fn(dim):
    lengthscale = [1.0] * dim
    variance = 0.1
    kernel = gpflow.kernels.SquaredExponential(variance=variance, lengthscales=lengthscale)
    return kernel


def create_model_and_kmeans_update_fn(
    model_class: ModelClass, data, num_inducing_points: int, use_jit: bool = True
):
    model = create_model(
        model_class,
        kernel_fn,
        data,
        num_inducing_points=num_inducing_points,
    )
    update_fn = create_update_fn(
        "kmeans", model, data, num_inducing_points=num_inducing_points, use_jit=use_jit
    )

    gpflow.utilities.set_trainable(model.inducing_variable, False)
    return model, update_fn


def create_model_and_covertree_update_fn(
    model_class: ModelClass, data, spatial_resolution: float, use_jit: bool = True
):
    model = create_model(model_class, kernel_fn, data)
    update_fn = create_update_fn("covertree", model, data, spatial_resolution=spatial_resolution, use_jit=use_jit)

    gpflow.utilities.set_trainable(model.inducing_variable, False)
    return model, update_fn
