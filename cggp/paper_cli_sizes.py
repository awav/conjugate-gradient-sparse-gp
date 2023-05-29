import glob
from typing import Union, Dict, Literal
import click
import tensorflow as tf
import numpy as np
import gpflow
from data import DatasetBundle
from utils import store_as_json, load_from_json, load_from_npy
from pathlib import Path

from cli_utils import (
    sgpr_class,
    cdgp_class,
    precision_names,
    ModelClassStr,
    DatasetType,
    DatasetCallable,
    FloatType,
    KernelType,
    LogdirPath,
    ModelChoices,
)

import click_cmds


from optimize import make_metrics_callback


def model_fn_choices(train_data, error_threshold: float = 1e-6):
    def sgpr_class_wrapper(*args, **kwargs):
        return sgpr_class(train_data, *args, **kwargs)

    def cdgp_class_wrapper(*args, **kwargs):
        return cdgp_class(*args, error_threshold=error_threshold, **kwargs)

    return dict(sgpr=sgpr_class_wrapper, cdgp=cdgp_class_wrapper)


@click.group()
@click.option("-mc", "--model-class", type=ModelChoices, required=True)
@click.option("-p", "--precision", type=FloatType(), required=True)
@click.option("-j", "--jitter", type=float, required=True)
@click.option("-c", "--config-dir", type=LogdirPath(mkdir=False))
@click.option("-df", "--data-fraction", type=float, default=1.0)
@click.option("--jit/--no-jit", type=bool, default=True)
@click.pass_context
def main(
    ctx: click.Context,
    config_dir: Union[Path, str],
    model_class: ModelClassStr,
    precision: tf.DType,
    data_fraction: float,
    jitter: float,
    jit: bool,
):
    """
    This is a core command for all CLI functions.
    """
    gpflow.config.set_default_float(precision)
    gpflow.config.set_default_jitter(jitter)

    # NOTE(awav): Some datasets will have values near the boundary and fail at numerics checks.
    gpflow.config.set_default_positive_minimum(1e-9)

    # Reference model
    if config_dir is not None:
        glob_dirs = glob.glob(str(config_dir))

        if len(glob_dirs) > 1:
            raise click.UsageError(
                f"Ambiguous config directory specified using whildcards. Found {glob_dirs}."
            )

        config_dir = glob_dirs[0]
        ref_info = load_from_json(Path(config_dir, "info.json"))
        ref_params = load_from_npy(Path(config_dir, "params.npy"))
        seed: int = ref_info["seed"]
        dataset_name = ref_info["dataset_name"]
    else:
        ref_info = None
        ref_params = None
        seed = 111
        dataset_name = "naval"
        config_dir = "none"

    np.random.seed(seed)
    tf.random.set_seed(seed)

    dataset_fn: DatasetCallable = DatasetType().convert(dataset_name, None, None)
    dataset = dataset_fn(seed)
    size = dataset.train[0].shape[0]
    new_size = np.ceil(size * data_fraction)
    train_x = dataset.train[0][:new_size]
    train_y = dataset.train[1][:new_size]

    train_data = (train_x, train_y)
    new_dataset = DatasetBundle(dataset.name, train_data, dataset.test)

    model_class_fn = model_fn_choices(new_dataset.train)[model_class]

    common_ctx = dict(
        seed=seed,
        dataset_name=dataset_name,
        config_dir=str(config_dir),
        model_class=model_class,
        model_class_fn=model_class_fn,
        ref_info=ref_info,
        ref_params=ref_params,
        dataset=new_dataset,
        jitter=jitter,
        precision=precision_names[precision],
        jit=jit,
        data_fraction=data_fraction,
    )

    ctx.obj = dict(common_ctx=common_ctx)


@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=LogdirPath.default_logdir)
@click.option("-tb", "--test-batch-size", type=int)
@click.pass_context
def compute_metrics(
    ctx: click.Context, logdir: Path, test_batch_size: Union[int, None]
):
    common_ctx = ctx.obj["common_ctx"]
    ip_ctx = ctx.obj["ip_ctx"]

    use_jit = common_ctx["jit"]
    dataset = common_ctx["dataset"]
    params = common_ctx["ref_params"]
    model = ip_ctx["model"]
    update_ip_fn = ip_ctx["update_fn"]
    jitter = common_ctx["jitter"]

    if params is not None:
        gpflow.utilities.multiple_assign(model, params)

    if test_batch_size is None:
        test_batch_size: int = dataset.test[0].shape[0]

    metrics_fn = make_metrics_callback(
        model,
        dataset.train,
        dataset.test,
        batch_size=test_batch_size,
        use_jit=use_jit,
        check_numerics=False,
    )

    update_ip_fn()
    metrics = metrics_fn(-1)
    m = int(model.inducing_variable.num_inducing)
    properties = covariance_properties(model, jitter)

    train_size: int = dataset.train[0].shape[0]
    test_size: int = dataset.test[0].shape[0]
    input_dim: int = dataset.train[0].shape[-1]

    info = {
        "seed": common_ctx["seed"],
        "model": common_ctx["model_class"],
        "dataset": common_ctx["dataset_name"],
        "train_data_size": train_size,
        "test_data_size": test_size,
        "input_dim": input_dim,
        "jitter": common_ctx["jitter"],
        "precision": common_ctx["precision"],
        "jit": common_ctx["jit"],
        "config_dir": common_ctx["config_dir"],
        "clustering_type": ip_ctx["clustering_type"],
        "clustering_args": ip_ctx["clustering_kwargs"],
        "data_fraction": common_ctx["data_fraction"],
        "num_inducing_points": m,
        "jitter": jitter,
    }

    results = {
        **info,
        **metrics,
        **properties,
    }

    store_as_json(Path(logdir, "results.json"), results)
    click.echo("⭐⭐⭐ Script finished ⭐⭐⭐")


def covariance_properties(
    model, jitter
) -> Dict[Literal["condition_number", "eig_min", "eig_max"], float]:
    iv = model.inducing_variable
    kernel = model.kernel
    kuu = gpflow.covariances.Kuu(iv, kernel, jitter=jitter)

    eigvals = tf.linalg.eigvalsh(kuu).numpy()
    eig_min = float(eigvals.min())
    eig_max = float(eigvals.max())
    condition_number = eig_max / eig_min
    return dict(condition_number=condition_number, eig_min=eig_min, eig_max=eig_max)


if __name__ == "__main__":
    cmds = {
        "oips": click_cmds.oips,
        "greedy": click_cmds.greedy,
        "kmeans": click_cmds.kmeans,
        "kmeans2": click_cmds.kmeans2,
        "uniform": click_cmds.uniform,
        "covertree": click_cmds.covertree,
        "grad-ip": click_cmds.grad_ip,
    }

    _ = [
        cmd.add_command(compute_metrics, "compute-metrics")
        for _name, cmd in cmds.items()
    ]
    _ = [main.add_command(cmd, name) for name, cmd in cmds.items()]

    main()
