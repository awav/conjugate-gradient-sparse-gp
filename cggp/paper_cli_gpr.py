from typing import Optional, Callable
import json
import click
import tensorflow as tf
import numpy as np
from gpflow.utilities import parameter_dict
from utils import store_as_npy, to_numpy, store_as_json
from pathlib import Path

from cli_utils import (
    create_predict_fn,
    batch_posterior_computation,
    create_gpr_model,
    kernel_to_name,
    DatasetType,
    DatasetCallable,
    KernelType,
    LogdirPath,
)

from optimize import (
    train_using_lbfgs_and_update,
    create_monitor,
)


@click.command()
@click.option("-s", "--seed", type=int, default=0)
@click.option("-l", "--logdir", type=LogdirPath(), default=LogdirPath.default_logdir)
@click.option("-n", "--num-iterations", type=int, required=True)
@click.option("-d", "--dataset", type=DatasetType(), required=True)
@click.option("-k", "--kernel", type=KernelType(), default="se")
@click.option("-tb", "--test-batch-size", type=int)
@click.option("-r", "--record-step", type=int)
@click.option("--jit/--no-jit", type=bool, default=True)
@click.option("--tensorboard/--no-tensorboard", type=bool, default=False)
def main(
    logdir: Path,
    seed: int,
    dataset: DatasetCallable,
    kernel: Callable,
    test_batch_size: Optional[int],
    num_iterations: int,
    record_step: Optional[int],
    jit: bool,
    tensorboard: bool,
):
    """
    This is a core command for all CLI functions.
    """
    use_jit = jit
    use_tb = tensorboard
    np.random.seed(seed)
    tf.random.set_seed(seed)

    data = dataset(seed)
    train_data = data.train
    test_data = data.test

    model = create_gpr_model(train_data, kernel)

    monitor = create_monitor(
        model,
        train_data,
        test_data,
        test_batch_size,
        record_step=record_step,
        use_tensorboard=use_tb,
        logdir=logdir,
    )

    if test_batch_size is None:
        test_batch_size = test_data[0].shape[0]

    info = {
        "seed": seed,
        "dataset_name": dataset.name,
        "num_iterations": num_iterations,
        "kernel": kernel_to_name(model.kernel),
        "use_jit": use_jit,
        "logdir": str(logdir),
        "train_size": train_data[0].shape[0],
        "test_size": test_data[0].shape[0],
        "input_dimension": train_data[0].shape[-1],
        "model_class": "gpr",
    }
    info_str = json.dumps(info, indent=2)
    click.echo(f"-> {info_str}")

    click.echo("★★★ Start training ★★★")
    train_using_lbfgs_and_update(
        train_data,
        model,
        num_iterations,
        update_fn=None,
        update_during_training=None,
        monitor=monitor,
        use_jit=use_jit,
    )
    click.echo("✪✪✪ Training finished ✪✪✪")

    params = parameter_dict(model)
    params_np = to_numpy(params)

    store_as_json(Path(logdir, "info.json"), info)
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


if __name__ == "__main__":
    main()
