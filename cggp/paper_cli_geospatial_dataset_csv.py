import click
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path

from cli_utils import (
    DatasetType,
    DatasetCallable,
    LogdirPath,
)


@click.command()
@click.option("-d", "--dataset", type=DatasetType(), required=True)
@click.option("-l", "--logdir", type=LogdirPath(), default=LogdirPath.default_logdir)
@click.option("-s", "--seed", type=int, default=0)
def main(
    logdir: Path,
    seed: int,
    dataset: DatasetCallable,
):
    """
    This is a core command for all CLI functions.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    data = dataset(seed, as_tensor=False)
    train_data = data.train
    test_data = data.test

    merged_train_data = np.concatenate([train_data[0], train_data[1]], axis=-1)
    merged_test_data = np.concatenate([test_data[0], test_data[1]], axis=-1)

    logdir = Path(logdir)

    df_train = pd.DataFrame(merged_train_data)
    df_train.to_csv(logdir / "train_data.csv")

    df_test = pd.DataFrame(merged_test_data)
    df_test.to_csv(logdir / "test_data.csv")


if __name__ == "__main__":
    main()
