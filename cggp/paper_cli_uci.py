from dataclasses import dataclass
from typing import Literal, Optional, Union, Callable, Dict
import json
import click
import tensorflow as tf
import numpy as np
import gpflow
from utils import store_as_json, load_from_json, load_from_npy
from pathlib import Path

from cli_utils import (
    sgpr_class,
    cggp_class,
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
        return cggp_class(*args, error_threshold=error_threshold, **kwargs)

    return dict(sgpr=sgpr_class_wrapper, cdgp=cdgp_class_wrapper)


@click.group()
@click.option("-mc", "--model-class", type=ModelChoices, required=True)
@click.option("-p", "--precision", type=FloatType(), required=True)
@click.option("-j", "--jitter", type=float, required=True)
@click.option("-c", "--config-dir", type=LogdirPath(), required=True)
@click.option("--jit/--no-jit", type=bool, default=True)
@click.pass_context
def main(
    ctx: click.Context,
    config_dir: Path,
    model_class: ModelClassStr,
    precision: tf.DType,
    jitter: float,
    jit: bool,
):
    """
    This is a core command for all CLI functions.
    """
    gpflow.config.set_default_float(precision)
    gpflow.config.set_default_jitter(jitter)

    # Reference model
    ref_info = load_from_json(Path(config_dir, "info.json"))
    ref_params = load_from_npy(Path(config_dir, "params.npy"))

    seed: int = ref_info["seed"]

    np.random.seed(seed)
    tf.random.set_seed(seed)

    dataset_name = ref_info["dataset_name"]
    dataset_fn: DatasetCallable = DatasetType().convert(dataset_name, None, None)
    dataset = dataset_fn(seed)

    # kernel_name = ref_info["kernel"]
    # kernel_fn = KernelType().convert(kernel_name, None, None)
    model_class_fn = model_fn_choices(dataset.train)[model_class]

    common_ctx = dict(
        config_dir=config_dir,
        model_class_fn=model_class_fn,
        ref_info=ref_info,
        ref_params=ref_params,
        dataset=dataset,
        jit=jit,
    )

    ctx.obj = dict(common_ctx=common_ctx)


@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=LogdirPath.default_logdir)
@click.option("-tb", "--test-batch-size", type=int)
@click.pass_context
def compute_metrics(ctx: click.Context, logdir: Path, test_batch_size: Union[int, None]):
    common_ctx = ctx.obj["common_ctx"]
    ip_ctx = ctx.obj["ip_ctx"]

    use_jit = common_ctx["jit"]
    dataset = common_ctx["dataset"]
    model = ip_ctx["model"]
    update_ip_fn = ip_ctx["update_fn"]

    if test_batch_size is None:
        test_batch_size: int = dataset.test[0].shape[0]

    metrics_fn = make_metrics_callback(
        model,
        dataset.train,
        dataset.test,
        batch_size=test_batch_size,
        use_jit=use_jit,
    )

    update_ip_fn()
    metrics = metrics_fn(-1)
    m = int(model.inducing_variable.num_inducing)
    results = {**metrics, **{"num_inducing_points": m}}

    store_as_json(Path(logdir, "results.json"), results)
    click.echo("⭐⭐⭐ Script finished ⭐⭐⭐")


if __name__ == "__main__":
    oips_cmd = click_cmds.oips
    kmeans_cmd = click_cmds.kmeans
    uniform_cmd = click_cmds.uniform
    covertree_cmd = click_cmds.covertree

    oips_cmd.add_command(compute_metrics, "compute-metrics")
    kmeans_cmd.add_command(compute_metrics, "compute-metrics")
    uniform_cmd.add_command(compute_metrics, "compute-metrics")
    covertree_cmd.add_command(compute_metrics, "compute-metrics")

    main.add_command(oips_cmd, "oips")
    main.add_command(kmeans_cmd, "kmeans")
    main.add_command(uniform_cmd, "uniform")
    main.add_command(covertree_cmd, "covertree")

    main()
