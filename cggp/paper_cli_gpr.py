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
    make_metrics_callback,
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
    np.random.seed(seed)
    tf.random.set_seed(seed)

    size_limit = 10000

    data = dataset(seed)
    train_data = data.train
    test_data = data.test

    train_data_slice = tuple(map(lambda d: d[:size_limit], train_data))

    model = create_gpr_model(train_data_slice, kernel)

    monitor = create_monitor(
        model,
        train_data,
        test_data,
        test_batch_size,
        record_step=record_step,
        use_tensorboard=tensorboard,
        logdir=logdir,
    )

    if test_batch_size is None:
        test_batch_size = test_data[0].shape[0]

    info = {
        "seed": seed,
        "dataset_name": data.name,
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

    for _ in range(2):
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

    metrics_fn = make_metrics_callback(
        model,
        train_data,
        test_data,
        batch_size=test_batch_size,
        use_jit=use_jit,
        print_on=True,
    )

    metrics = metrics_fn(0)
    store_as_json(Path(logdir, "results.json"), metrics)

    click.echo("⭐⭐⭐ Script finished ⭐⭐⭐")


if __name__ == "__main__":
    main()
