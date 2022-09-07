from dataclasses import dataclass
from typing import Literal, Union, Callable, Dict
import json
import click
from cli_utils import ModelClass, ModelClassStr
import tensorflow as tf
import numpy as np
import gpflow
from gpflow.utilities import parameter_dict
from utils import store_as_npy, to_numpy
from pathlib import Path

from data import DatasetBundle, Dataset
from distance import DistanceType

from cli_utils import (
    create_model_and_update_fn,
    create_predict_fn,
    batch_posterior_computation,
    cggp_class,
    sgpr_class,
    precision_names,
    DatasetType,
    DatasetCallable,
    KernelType,
    LogdirPath,
    FloatType,
    PrecisionName,
    DistanceChoices,
    ModelClassStr,
    ModelChoices,
)

from optimize import (
    train_using_adam_and_update,
    train_using_lbfgs_and_update,
    create_monitor,
)


@dataclass
class MainContext:
    seed: int
    logdir: Union[Path, str]
    model_class: ModelClassStr 
    dataset: DatasetBundle
    train_data: Dataset
    test_data: Dataset
    kernel_fn: Callable
    model_class_fn: Callable
    jit: bool
    jitter: float
    precision: PrecisionName
    extra_obj: Dict


@click.group()
@click.option("-d", "--dataset", type=DatasetType(), required=True)
@click.option("-mc", "--model-class", type=ModelChoices, required=True)
@click.option("-p", "--precision", type=FloatType(), default="fp64")
@click.option("-j", "--jitter", type=float, default=1e-6)
@click.option("-k", "--kernel", type=KernelType(), default="matern32")
@click.option("-l", "--logdir", type=LogdirPath(), default=LogdirPath.default_logdir)
@click.option("-s", "--seed", type=int, default=0)
@click.option("--jit/--no-jit", type=bool, default=True)
@click.option("-e", "--error-threshold", type=float, default=1e-6)
@click.pass_context
def main(
    ctx: click.Context,
    logdir: Path,
    precision: tf.DType,
    jitter: float,
    kernel: Callable,
    seed: int,
    dataset: DatasetCallable,
    jit: bool,
    model_class: ModelClass,
    error_threshold: float,
):
    """
    This is a core command for all CLI functions.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpflow.config.set_default_float(precision)
    gpflow.config.set_default_jitter(jitter)

    data = dataset(seed)

    train_data = data.train
    test_data = data.test

    def sgpr_class_wrapper(*args, **kwargs):
        return sgpr_class(train_data, *args, **kwargs)

    def cggp_class_wrapper(*args, **kwargs):
        return cggp_class(*args, error_threshold=error_threshold, **kwargs)

    model_classes = dict(sgpr=sgpr_class_wrapper, cggp=cggp_class_wrapper)
    model_class_fn = model_classes[model_class]

    obj = MainContext(
        seed,
        str(logdir),
        model_class,
        data,
        train_data,
        test_data,
        kernel,
        model_class_fn,
        jit,
        jitter,
        precision_names[precision],
        extra_obj=dict(),
    )
    ctx.obj = obj


@main.group("covertree")
@click.option("-s", "--spatial-resolution", type=float, required=True)
@click.option("-d", "--distance-type", type=DistanceChoices, default="euclidean")
@click.pass_context
def covertree(ctx: click.Context, spatial_resolution: float, distance_type: DistanceType):
    obj: MainContext = ctx.obj
    clustering_type = "covertree"
    clustering_kwargs = {"spatial_resolution": spatial_resolution}

    model, update_fn = create_model_and_update_fn(
        obj.model_class_fn,
        obj.train_data,
        clustering_type=clustering_type,
        distance_type=distance_type,
        use_jit=obj.jit,
        clustering_kwargs=clustering_kwargs,
    )

    obj.extra_obj = dict(
        model=model,
        update_fn=update_fn,
        clustering_type=clustering_type,
        distance_type=distance_type,
    )

    ctx.obj = obj


@main.group("kmeans")
@click.option("-m", "--max-num-ip", type=int, required=True)
@click.option("-d", "--distance-type", type=DistanceChoices, default="euclidean")
@click.pass_context
def kmeans(ctx: click.Context, max_num_ip: int, distance_type: DistanceType):
    obj: MainContext = ctx.obj
    clustering_type = "kmeans"
    clustering_kwargs = {"num_inducing_points": max_num_ip}

    model, update_fn = create_model_and_update_fn(
        obj.model_class_fn,
        obj.train_data,
        clustering_type=clustering_type,
        distance_type=distance_type,
        use_jit=obj.jit,
        clustering_kwargs=clustering_kwargs,
    )

    obj.extra_obj = dict(
        model=model,
        update_fn=update_fn,
        clustering_type=clustering_type,
        distance_type=distance_type,
    )

    ctx.obj = obj


@main.group("oips")
@click.option("-r", "--rho", type=float, required=True)
@click.option("-m", "--max-num-ip", type=int, required=True)
@click.option("-d", "--distance-type", type=DistanceChoices, default="euclidean")
@click.pass_context
def oips(ctx: click.Context, rho: float, max_num_ip: int, distance_type: DistanceType):
    obj: MainContext = ctx.obj
    clustering_type = "oips"
    clustering_kwargs = {"rho": rho, "max_points": max_num_ip}

    model, update_fn = create_model_and_update_fn(
        obj.model_class_fn,
        obj.train_data,
        clustering_type=clustering_type,
        distance_type=distance_type,
        use_jit=obj.jit,
        clustering_kwargs=clustering_kwargs,
    )

    obj.extra_obj = dict(
        model=model,
        update_fn=update_fn,
        clustering_type=clustering_type,
        distance_type=distance_type,
    )

    ctx.obj = obj


@click.command("train-adam")
@click.option("-n", "--num-iterations", type=int, required=True)
@click.option("-b", "--batch-size", type=int)
@click.option("-tb", "--test-batch-size", type=int)
@click.option("-l", "--learning-rate", type=float, default=0.01)
@click.option("--tip/--no-tip", type=bool, default=False)
@click.option("--tensorboard/--no-tensorboard", type=bool, default=True)
@click.pass_context
def train_adam(
    ctx: click.Context,
    num_iterations: int,
    batch_size: int,
    test_batch_size: int,
    learning_rate: float,
    tip: bool,
    tensorboard: bool,
):
    obj: MainContext = ctx.obj
    use_tb = tensorboard
    use_jit = obj.jit
    logdir = obj.logdir
    trainable_inducing_points = tip
    train_data = obj.train_data
    test_data = obj.test_data
    model = obj.extra_obj["model"]
    update_fn = obj.extra_obj["update_fn"]
    clustering_type = obj.extra_obj["clustering_type"]
    distance_type = obj.extra_obj["distance_type"]

    gpflow.utilities.set_trainable(model.inducing_variable, trainable_inducing_points)

    if batch_size is None:
        batch_size = train_data[0].shape[0]

    if test_batch_size is None:
        test_batch_size = test_data[0].shape[0]

    monitor = create_monitor(
        model,
        train_data,
        test_data,
        test_batch_size,
        use_tensorboard=use_tb,
        logdir=logdir,
    )

    m = int(model.inducing_variable.num_inducing)
    info = {
        "seed": obj.seed,
        "dataset_name": obj.dataset.name,
        "num_inducing_points": m,
        "num_iterations": num_iterations,
        "use_jit": use_jit,
        "jitter": obj.jitter,
        "precision": obj.precision,
        "learning_rate": learning_rate,
        "logdir": logdir,
        "batch_size": batch_size,
        "train_size": train_data[0].shape[0],
        "test_size": test_data[0].shape[0],
        "input_dimension": train_data[0].shape[-1],
        "clustering_type": clustering_type,
        "distance_type": distance_type,
        "model_class": obj.model_class,
        "trainable_inducing_points": trainable_inducing_points,
    }
    info_str = json.dumps(info, indent=2)
    click.echo(f"-> {info_str}")

    # Run hyperparameter tuning
    click.echo("★★★ Start training ★★★")
    update_fn()
    train_using_adam_and_update(
        train_data,
        model,
        num_iterations,
        batch_size,
        learning_rate,
        update_fn=None,
        update_during_training=None,
        use_jit=use_jit,
        monitor=monitor,
    )
    click.echo("✪✪✪ Training finished ✪✪✪")

    # Post training procedures
    params = parameter_dict(model)
    params_np = to_numpy(params)
    store_as_npy(Path(logdir, "params.npy"), params_np)

    predict_fn = create_predict_fn(model, use_jit=use_jit)

    mean_train, variances_train = batch_posterior_computation(
        predict_fn,
        train_data,
        test_batch_size,
    )

    mean_test, variances_test = batch_posterior_computation(
        predict_fn,
        test_data,
        test_batch_size,
    )

    store_as_npy(Path(logdir, "train_mean.npy"), np.array(mean_train))
    store_as_npy(Path(logdir, "test_mean.npy"), np.array(mean_test))
    store_as_npy(Path(logdir, "train_variances.npy"), np.array(variances_train))
    store_as_npy(Path(logdir, "test_variances.npy"), np.array(variances_test))
    click.echo("⭐⭐⭐ Script finished ⭐⭐⭐")


covertree.add_command(train_adam, "train-adam")
kmeans.add_command(train_adam, "train-adam")
oips.add_command(train_adam, "train-adam")


if __name__ == "__main__":
    main()
