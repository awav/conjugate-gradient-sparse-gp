from typing import Literal, Callable, Optional, List, Optional, TypeVar, Dict
from pathlib import Path

import click
import gpflow
import numpy as np
import tensorflow as tf

from kmeans import kmeans_lloyd
from oips import oips
from distance import create_distance_fn, DistanceType
from optimize import (
    kmeans_update_inducing_parameters,
    covertree_update_inducing_parameters,
    oips_update_inducing_parameters,
)
from data import load_data
from utils import jit, transform_to_dataset
from models import LpSVGP, ClusterGP, CGGP
from conjugate_gradient import ConjugateGradient


ModelClass = TypeVar("ModelClass", type(LpSVGP), type(ClusterGP))
ClusteringType = Literal["kmeans", "covertree", "oips"]


class FloatType(click.ParamType):
    name = "dtype"

    def convert(self, value, param, ctx):
        options = {"fp32": np.float32, "fp64": np.float64}
        try:
            norm_value = value.lower()
            dtype = options[norm_value]
            return dtype
        except Exception as ex:
            self.fail(f"{value} is not a valid float type [fp32, fp64]", param, ctx)


class LogdirPath(click.Path):
    default_logdir = "./logs-default"

    def __init__(self, mkdir: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.mkdir = mkdir

    def convert(self, value, param, ctx):
        logdir: str = super().convert(value, param, ctx)
        logdir_path = Path(logdir).expanduser().resolve()
        if self.mkdir:
            logdir_path.mkdir(parents=True, exist_ok=True)
        return logdir_path


class DatasetType(click.ParamType):
    name = "dataset"
    datasets: List[str] = [
        "snelson1d",
        "elevators",
        "pol",
        "houseelectric",
        "3droad",
        "buzz",
        "keggdirected",
        "keggundirected",
        "song",
        "east_africa",
    ]

    def convert(self, value, param, ctx):
        if value not in self.datasets:
            self.fail(f"{value} dataset is not supported", param, ctx)
        try:
            dataname = value
            data = load_data(dataname, as_tensor=True)
            return data
        except Exception:
            self.fail(f"Error occured during loading {value} dataset", param, ctx)


class KernelType(click.ParamType):
    name = "dataset"
    kernels = {
        "se": gpflow.kernels.SquaredExponential,
        "matern32": gpflow.kernels.Matern32,
    }
    param_keymap = {"var": "variance", "len": "lengthscales"}

    @classmethod
    def parse_kernel_parameters(cls, source: str):
        params = [kv.split("=") for kv in source.split("_")]
        params = {cls.param_keymap[k]: ast.literal_eval(v) for k, v in params}
        return params

    def convert(self, value, param, ctx):
        try:
            kernel_name, *conf = value.split("_", maxsplit=1)
            kernel_class = self.kernels[kernel_name]
            kernel_params = self.parse_kernel_parameters(conf[1]) if conf else {}

            def create_kernel_fn(ndim: int):
                positive = gpflow.utilities.positive(1e-6)
                lengthscale = np.ones(ndim)
                if "lengthscales" in kernel_params:
                    lengthscale_param = kernel_params["lengthscales"]
                    lengthscale = lengthscale * lengthscale_param
                kernel = kernel_class()
                kernel.lengthscales = gpflow.Parameter(lengthscale, transform=positive)
                return kernel

            return create_kernel_fn
        except Exception as ex:
            self.fail(f"{value} is not supported", param, ctx)


def create_model(
    model_fn: Callable,
    kernel_fn: Callable,
    data,
    num_inducing_points: Optional[int] = None,
    **model_kwargs,
):
    x = np.array(data[0])
    n = x.shape[0]
    dim = x.shape[-1]
    default_variance = 0.1
    dtype = x.dtype

    if num_inducing_points is not None:
        rand_indices = np.random.choice(n, size=num_inducing_points, replace=False)
        iv = tf.convert_to_tensor(x[rand_indices, ...], dtype=dtype)
    else:
        rand_num_inducing_points = int(n * 0.1)
        rand_indices = np.random.choice(n, size=rand_num_inducing_points, replace=False)
        iv = tf.Variable(x[rand_indices, ...], dtype=dtype, shape=(None, dim))

    likelihood = gpflow.likelihoods.Gaussian(variance=default_variance)
    kernel = kernel_fn(dim)
    model = model_fn(kernel, likelihood, iv, num_data=n, **model_kwargs)

    return model


def create_kmeans_update_fn(
    model,
    data,
    use_jit: bool = True,
    num_inducing_points: int = 1,
    distance_type: DistanceType = "covariance",
):
    x, _ = data
    distance_fn = create_distance_fn(model.kernel, distance_type)
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


def create_covertree_update_fn(
    model,
    data,
    use_jit: bool = True,
    spatial_resolution: float = 1.0,
    distance_type: DistanceType = "covariance",
):
    distance_fn = create_distance_fn(model.kernel, distance_type)
    distance_fn = jit(use_jit)(distance_fn)

    def update_fn():
        return covertree_update_inducing_parameters(model, data, distance_fn, spatial_resolution)

    return update_fn


def create_oips_update_fn(
    model,
    data,
    rho: float = 0.5,
    use_jit: bool = True,
    max_points: int = -1,
    distance_type: DistanceType = "covariance",
):
    """
    By default method will use half of the size of
    dataset for the number of inducing points.
    """

    distance_fn = create_distance_fn(model.kernel, distance_type)
    distance_fn = jit(use_jit)(distance_fn)
    kernel = model.kernel
    if max_points is None or max_points <= 0:
        max_points = tf.shape(data[0])[0]

    @jit(use_jit)
    def oips_fn(inputs):
        return oips(kernel, inputs, rho, max_points)

    @jit(use_jit)
    def update_fn():
        return oips_update_inducing_parameters(model, data, oips_fn, distance_fn)

    return update_fn


def create_update_fn(
    clustering_type: ClusteringType,
    model,
    data,
    use_jit: bool = True,
    distance_type: DistanceType = "covariance",
    **clustering_kwargs,
):
    if clustering_type == "kmeans":
        return create_kmeans_update_fn(
            model, data, use_jit=use_jit, distance_type=distance_type, **clustering_kwargs
        )
    elif clustering_type == "covertree":
        return create_covertree_update_fn(
            model, data, use_jit=use_jit, distance_type=distance_type, **clustering_kwargs
        )
    elif clustering_type == "oips":
        return create_oips_update_fn(
            model, data, use_jit=use_jit, distance_type=distance_type, **clustering_kwargs
        )
    raise ValueError(f"Unknown value for {clustering_type}")


def kernel_fn(dim):
    lengthscale = [1.0] * dim
    variance = 0.1
    kernel = gpflow.kernels.SquaredExponential(variance=variance, lengthscales=lengthscale)
    return kernel


def create_model_and_update_fn(
    model_class: ModelClass,
    train_data,
    clustering_type: ClusteringType,
    use_jit: bool = True,
    distance_type: DistanceType = "covariance",
    trainable_inducing_points: bool = False,
    model_kwargs: Optional[Dict] = None,
    clustering_kwargs: Optional[Dict] = None,
):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    clustering_kwargs = {} if clustering_kwargs is None else clustering_kwargs

    model = create_model(model_class, kernel_fn, train_data, **model_kwargs)
    internal_update_fn = create_update_fn(
        clustering_type,
        model,
        train_data,
        use_jit=use_jit,
        distance_type=distance_type,
        **clustering_kwargs,
    )

    def update_fn():
        iv, means, count = internal_update_fn()
        if isinstance(model, CGGP):
            model.inducing_variable.Z.assign(iv)
            model.pseudo_u.assign(means)
            model.cluster_counts.assign(count)
        else:
            model.inducing_variable.assign(iv)
        return iv, means, count

    gpflow.utilities.set_trainable(model.inducing_variable, trainable_inducing_points)
    return model, update_fn


def create_predict_fn(model, use_jit: bool = True):
    @jit(use_jit)
    def predict_fn(inputs):
        mu, var = model.predict_f(inputs)
        return mu, var

    return predict_fn


def batch_posterior_computation(predict_fn, data, batch_size):
    data = transform_to_dataset(data, repeat=False, batch_size=batch_size)
    means = []
    variances = []
    for (x, y) in data:
        mean, variance = predict_fn(x)
        means.append(mean.numpy())
        variances.append(variance.numpy())
    means = np.concatenate(means, axis=0)
    variances = np.concatenate(variances, axis=0)
    return means, variances


def cggp_class(kernel, likelihood, iv, error_threshold: float = 1e-6, **kwargs):
    conjugate_gradient = ConjugateGradient(error_threshold)
    return CGGP(kernel, likelihood, iv, conjugate_gradient, **kwargs)


def sgpr_class(train_data, kernel, likelihood, iv, **kwargs):
    noise_variance = likelihood.variance
    return gpflow.models.SGPR(train_data, kernel, iv, noise_variance=noise_variance)
