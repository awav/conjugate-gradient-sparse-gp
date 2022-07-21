from typing import Literal, Callable, Optional, List, Optional, TypeVar
from pathlib import Path

import click
import gpflow
import numpy as np
import tensorflow as tf

from kmeans import kmeans_lloyd
from distance import create_distance_fn, DistanceType
from optimize import kmeans_update_inducing_parameters, covertree_update_inducing_parameters
from data import load_data
from utils import jit
from models import LpSVGP, ClusterGP


ModelClass = TypeVar("ModelClass", type(LpSVGP), type(ClusterGP))
ClusteringType = Literal["kmeans", "covertree"]


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
                positive = gpflow.utilities.positive(1e-5)
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
    raise ValueError(f"Unknown value for {clustering_type}")


def kernel_fn(dim):
    lengthscale = [1.0] * dim
    variance = 0.1
    kernel = gpflow.kernels.SquaredExponential(variance=variance, lengthscales=lengthscale)
    return kernel


def create_model_and_kmeans_update_fn(
    model_class: ModelClass,
    data,
    num_inducing_points: int,
    use_jit: bool = True,
    distance_type: DistanceType = "covariance",
    trainable_inducing_points: bool = False,
    **model_kwargs,
):
    model = create_model(
        model_class,
        kernel_fn,
        data,
        num_inducing_points=num_inducing_points,
        **model_kwargs,
    )
    update_fn = create_update_fn(
        "kmeans",
        model,
        data,
        num_inducing_points=num_inducing_points,
        use_jit=use_jit,
        distance_type=distance_type,
    )

    gpflow.utilities.set_trainable(model.inducing_variable, trainable_inducing_points)
    return model, update_fn


def create_model_and_covertree_update_fn(
    model_class: ModelClass,
    data,
    spatial_resolution: float,
    use_jit: bool = True,
    distance_type: DistanceType = "covariance",
    trainable_inducing_points: bool = False,
    **model_kwargs,
):
    model = create_model(model_class, kernel_fn, data, **model_kwargs)
    update_fn = create_update_fn(
        "covertree",
        model,
        data,
        spatial_resolution=spatial_resolution,
        use_jit=use_jit,
        distance_type=distance_type,
    )

    gpflow.utilities.set_trainable(model.inducing_variable, trainable_inducing_points)
    return model, update_fn
