import ast
import json
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, Union, List

import click
import gpflow
import numpy as np
import tensorflow as tf

from gpflow.kernels import Matern32, SquaredExponential

from cli_utils import create_model, create_update_fn
from data import load_data, DatasetBundle
from monitor import Monitor
from optimize import train_using_adam_and_update, create_monitor
from models import CGGP, LpSVGP, ClusterGP
from conjugate_gradient import ConjugateGradient


Tensor = tf.Tensor
__default_logdir = "./logs-default"


class LogdirPath(click.Path):
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
    ]

    def convert(self, value, param, ctx):
        if value not in self.datasets:
            self.fail(f"{value} dataset is not supported", param, ctx)
        try:
            dataname = value
            data = load_data(dataname, as_tensor=True)
            return data
        except Exception as ex:
            self.fail(f"{value} dataset is not supported", param, ctx)


class KernelType(click.ParamType):
    name = "dataset"
    kernels = {"se": SquaredExponential, "matern32": Matern32}
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


@dataclass
class EntryContext:
    seed: int
    logdir: Union[Path, str]
    dataset: DatasetBundle
    kernel_fn: Callable
    jit: bool


@click.group()
@click.option("-d", "--dataset", type=DatasetType(), required=True)
@click.option("-k", "--kernel", type=KernelType(), default="se")
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_logdir)
@click.option("-s", "--seed", type=int, default=0)
@click.option("--jit/--no-jit", type=bool, default=True)
@click.pass_context
def main(
    ctx: click.Context,
    logdir: Path,
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

    obj = EntryContext(
        seed,
        str(logdir),
        dataset,
        kernel,
        jit,
    )
    ctx.obj = obj


@main.command()
@click.option("-n", "--num-iterations", type=int, required=True)
@click.option("-m", "--num-inducing-points", type=int, required=True)
@click.option("-b", "--batch-size", type=int, required=True)
@click.option("-l", "--learning-rate", type=float, default=0.01)
@click.option("-e", "--error-threshold", type=float, default=1e-5)
@click.pass_context
def train_cggp_adam(
    ctx: click.Context,
    num_iterations: int,
    num_inducing_points: int,
    batch_size: int,
    learning_rate: float,
    error_threshold: float,
):
    """
    This command `train-cggp-adam' run Adam training on CGGP model.
    """

    obj: EntryContext = ctx.obj
    dataset = obj.dataset
    jit = obj.jit
    kernel_fn = obj.kernel_fn
    train_data = dataset.train
    test_data = dataset.test

    info = {
        "command": "train_cggp_adam",
        "seed": obj.seed,
        "dataset_name": dataset.name,
        "num_inducing_points": num_inducing_points,
        "num_iterations": num_iterations,
        "use_jit": jit,
        "learning_rate": learning_rate,
        "logdir": obj.logdir,
        "batch_size": batch_size,
        "train_size": train_data[0].shape[0],
        "test_size": test_data[0].shape[0],
        "input_dimension": train_data[0].shape[-1],
    }

    info_str = json.dumps(info, indent=2)
    click.echo(f"-> {info_str}")

    def model_fn(kernel, likelihood, iv, **kwargs) -> CGGP:
        conjugate_gradient = ConjugateGradient(error_threshold)
        return CGGP(kernel, likelihood, iv, conjugate_gradient, **kwargs)

    clustering_type = "kmeans"
    model = create_model(model_fn, kernel_fn, train_data, num_inducing_points)
    update_fn = create_update_fn(
        clustering_type, model, train_data, num_inducing_points, use_jit=jit
    )
    monitor_batch_size = batch_size * 5
    monitor = create_monitor(model, test_data, monitor_batch_size, use_jit=jit)

    train_using_adam_and_update(
        train_data,
        model,
        num_iterations,
        batch_size,
        learning_rate,
        update_fn,
        monitor=monitor,
        use_jit=jit,
    )


if __name__ == "__main__":
    # Switch between testing and real CLI script mode
    testing = False
    # testing = True

    if not testing:
        main()
    else:
        from click.testing import CliRunner

        runner = CliRunner(echo_stdin=True, mix_stderr=True)
        args = [
            "--dataset",
            "elevators",
            "--jit",
            "train-cggp-adam",
            "--num-iterations",
            "100",
            "--num-inducing-points",
            "1000",
            "--batch-size",
            "1000",
        ]
        result = runner.invoke(main, args)
        print()
