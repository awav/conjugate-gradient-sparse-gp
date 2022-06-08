from dataclasses import dataclass
from typing import Union, Callable, Optional
import json
import click
from models import ClusterGP, CGGP, LpSVGP
from conjugate_gradient import ConjugateGradient
from data import load_data
import tensorflow as tf
import numpy as np
import gpflow
from gpflow.utilities import parameter_dict
from utils import store_logs, to_numpy, jit
from pathlib import Path

from data import DatasetBundle, to_float

from cli_utils import (
    create_model,
    create_model_and_kmeans_update_fn,
    create_model_and_covertree_update_fn,
    DatasetType,
    KernelType,
    LogdirPath,
    FloatType,
)

from optimize import (
    train_using_adam_and_update,
    train_using_lbfgs_and_update,
    create_monitor,
    transform_to_dataset,
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
@click.option("-n", "--num-iterations", type=int, required=True)
@click.option("-s", "--spatial-resolution", type=float, required=True)
@click.option("-b", "--batch-size", type=int, required=True)
@click.option("-tb", "--test-batch-size", type=int)
@click.option("-l", "--learning-rate", type=float, default=0.01)
@click.option("-e", "--error-threshold", type=float, default=1e-6)
@click.option("--tip/--no-tip", type=bool, default=False)
@click.pass_context
def train_adam_covertree(
    ctx: click.Context,
    num_iterations: int,
    model_class: str,
    batch_size: int,
    test_batch_size: int,
    learning_rate: float,
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

    iv, means, cluster_counts = update_fn()
    model.inducing_variable.Z.assign(iv)
    if isinstance(model, CGGP):
        model.cluster_counts.assign(cluster_counts)
        model.pseudo_u.assign(means)

    m = int(iv.shape[0])

    info = {
        "command": "train_adam_covertree",
        "seed": obj.seed,
        "dataset_name": dataset.name,
        "num_inducing_points": int(m),
        "num_iterations": num_iterations,
        "use_jit": use_jit,
        "jitter": obj.jitter,
        "precision": str(obj.precision),
        "learning_rate": learning_rate,
        "logdir": logdir,
        "batch_size": batch_size,
        "train_size": train_data[0].shape[0],
        "test_size": test_data[0].shape[0],
        "input_dimension": train_data[0].shape[-1],
        "clustering_type": "covertree",
        "model_class": model_class,
        "trainable_inducing_points": trainable_inducing_points,
    }
    info_str = json.dumps(info, indent=2)
    click.echo(f"-> {info_str}")

    monitor = create_monitor(
        model,
        train_data,
        test_data,
        test_batch_size,
        use_tensorboard=use_tb,
        logdir=logdir,
    )

    click.echo("★★★ Start training ★★★")
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
    store_logs(Path(logdir, "params.npy"), params_np)

    predict_fn = create_predict_fn(model, use_jit=use_jit)
    mean_train = batch_posterior_computation(predict_fn, train_data, test_batch_size)
    mean_test = batch_posterior_computation(predict_fn, test_data, test_batch_size)

    store_logs(Path(logdir, "train_mean.npy"), np.array(mean_train))
    store_logs(Path(logdir, "test_mean.npy"), np.array(mean_test))
    click.echo("⭐⭐⭐ Script finished ⭐⭐⭐")


def create_predict_fn(model, use_jit: bool = True):
    @jit(use_jit)
    def predict_fn(inputs):
        mu, _ = model.predict_f(inputs)
        return mu

    return predict_fn


def batch_posterior_computation(predict_fn, data, batch_size):
    data = transform_to_dataset(data, repeat=False, batch_size=batch_size)
    means = []
    for (x, y) in data:
        mean = predict_fn(x)
        means.append(mean.numpy())
    means = np.concatenate(means, axis=0)
    return means


def cggp_class(kernel, likelihood, iv, error_threshold: float = 1e-6, **kwargs):
    conjugate_gradient = ConjugateGradient(error_threshold)
    return CGGP(kernel, likelihood, iv, conjugate_gradient, **kwargs)


def sgpr_class(train_data, kernel, likelihood, iv, **kwargs):
    noise_variance = likelihood.variance
    return gpflow.models.SGPR(train_data, kernel, iv, noise_variance=noise_variance)


if __name__ == "__main__":
    main()
