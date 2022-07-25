from dataclasses import dataclass
from typing import Union, Callable
import click
import tensorflow as tf
import numpy as np
import gpflow
from utils import store_logs
from pathlib import Path

from data import DatasetBundle, to_float

from cli_utils import (
    create_model,
    create_model_and_covertree_update_fn,
    create_predict_fn,
    batch_posterior_computation,
    cggp_class,
    sgpr_class,
    DatasetType,
    KernelType,
    LogdirPath,
    FloatType,
)


@dataclass
class EntryContext:
    seed: int
    logdir: Union[Path, str]
    dataset: DatasetBundle
    kernel_fn: Callable
    jit: bool
    jitter: float
    precision: tf.DType


@click.group()
@click.option("-d", "--dataset", type=DatasetType(), required=True)
@click.option("-p", "--precision", type=FloatType(), default="fp64")
@click.option("-j", "--jitter", type=float, default=1e-6)
@click.option("-k", "--kernel", type=KernelType(), default="se")
@click.option("-l", "--logdir", type=LogdirPath(), default=LogdirPath.default_logdir)
@click.option("-s", "--seed", type=int, default=0)
@click.option("--jit/--no-jit", type=bool, default=True)
@click.pass_context
def main(
    ctx: click.Context,
    logdir: Path,
    precision: tf.DType,
    jitter: float,
    kernel: Callable,
    seed: int,
    dataset: DatasetBundle,
    jit: bool,
):
    """
    This is a core command for all CLI functions.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpflow.config.set_default_float(precision)
    gpflow.config.set_default_jitter(jitter)

    obj = EntryContext(
        seed,
        str(logdir),
        dataset,
        kernel,
        jit,
        jitter,
        precision,
    )
    ctx.obj = obj


__model_types = click.Choice(["sgpr", "cdgp"])


@main.command()
@click.option("-mc", "--model-class", type=__model_types, required=True)
@click.option("-s", "--spatial-resolution", type=float, required=True)
@click.option("-b", "--batch-size", type=int, required=True)
@click.option("-tb", "--test-batch-size", type=int)
@click.option("-e", "--error-threshold", type=float, default=1e-6)
@click.option("--tip/--no-tip", type=bool, default=False)
@click.pass_context
def mean_and_variance(
    ctx: click.Context,
    model_class: str,
    batch_size: int,
    test_batch_size: int,
    error_threshold: float,
    spatial_resolution: float,
    tip: bool,
):
    obj: EntryContext = ctx.obj
    use_tb = True
    dataset = obj.dataset
    use_jit = obj.jit
    logdir = obj.logdir
    kernel_fn = obj.kernel_fn
    train_data = (
        to_float(dataset.train[0], as_tensor=True),
        to_float(dataset.train[1], as_tensor=True),
    )
    test_data = (
        to_float(dataset.test[0], as_tensor=True),
        to_float(dataset.test[1], as_tensor=True),
    )

    trainable_inducing_points = tip

    if test_batch_size is None:
        test_batch_size = batch_size

    if model_class == "sgpr":

        def sgpr_class_wrapper(*args, **kwargs):
            return sgpr_class(train_data, *args, **kwargs)

        model = create_model(sgpr_class_wrapper, kernel_fn, train_data)
        gpflow.utilities.set_trainable(model.inducing_variable, trainable_inducing_points)
        _, update_fn = create_model_and_covertree_update_fn(
            cggp_class,
            train_data,
            spatial_resolution,
            use_jit=use_jit,
            distance_type="euclidean",
        )
    elif model_class == "cdgp":
        model, update_fn = create_model_and_covertree_update_fn(
            cggp_class,
            train_data,
            spatial_resolution,
            use_jit=use_jit,
            distance_type="euclidean",
            trainable_inducing_points=trainable_inducing_points,
            error_threshold=error_threshold,
        )
    else:
        raise ValueError(f"Unknown value for model class {model_class}")

    
    params_file = Path(logdir, "params.npy")
    params = np.load(params_file, allow_pickle=False).item()
    gpflow.utilities.multiple_assign(model, params)

    click.echo("✪✪✪ Parameters assignment has finished ✪✪✪")

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

    click.echo("✪✪✪ Mean and variance computation has finished ✪✪✪")

    store_logs(Path(logdir, "train_mean.npy"), np.array(mean_train))
    store_logs(Path(logdir, "test_mean.npy"), np.array(mean_test))
    store_logs(Path(logdir, "train_variances.npy"), np.array(variances_train))
    store_logs(Path(logdir, "test_variances.npy"), np.array(variances_test))
    click.echo("⭐⭐⭐ Script finished ⭐⭐⭐")


if __name__ == "__main__":
    main()
