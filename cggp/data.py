from typing import Tuple, NamedTuple, Optional
from collections import namedtuple
from pathlib import Path
import numpy as np
import shutil
from zipfile import ZipFile
from io import BytesIO
import urllib.request

import bayesian_benchmarks.data as bbd
import tensorflow as tf
from gpflow.config import default_float

Dataset = Tuple[np.ndarray, np.ndarray]
DatasetBundle = namedtuple("DatasetBundle", "name, train, test")


def download_and_upzip_file(outpath, url):
    """Taken from https://stackoverflow.com/a/61195974 (modified)."""

    with urllib.request.urlopen(url) as response:
        with ZipFile(BytesIO(response.read())) as zip_file:
            zip_file.extractall(outpath)


def snelson1d(target_dir: str = ".datasets/snelson1d"):
    """
    Load Edward Snelson's 1d regression data set [@snelson2006fitc].
    It contains 200 examples of a few oscillations of an example function. It has
    seen extensive use as a toy dataset for illustrating qualitative behaviour of
    Gaussian process approximations.
    Args:
      target_dir: str.
        Path to directory which either stores file or otherwise file will be
        downloaded and extracted there. Filenames are `snelson_train_*`.
    Returns:
      Tuple of two np.darray `inputs` and `outputs` with 200 rows and 1 column.
    """

    # Contains all source as well. We just need the data.
    data_url = "http://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip"

    target_dir_path = Path(target_dir).expanduser().resolve()
    inputs_path = Path(target_dir_path, "snelson_train_inputs")
    outputs_path = Path(target_dir_path, "snelson_train_outputs")

    if not (inputs_path.exists() and outputs_path.exists()):
        download_and_upzip_file(target_dir_path, data_url)

        dist_dir = Path(target_dir_path, "SPGP_dist")
        shutil.copy(Path(dist_dir, "train_inputs"), inputs_path)
        shutil.copy(Path(dist_dir, "train_outputs"), outputs_path)

        # Clean up everything else
        shutil.rmtree(dist_dir)

    X = np.loadtxt(inputs_path)[:, None]
    Y = np.loadtxt(outputs_path)[:, None]

    return (X, Y), (X, Y)


def east_africa(dirpath: str, train_proportion: int = 0.7, seed: int = 0):
    import pandas as pd

    test_filename = "east_africa_test.csv"
    test_filepath = Path(dirpath, test_filename)
    test = np.array(pd.read_csv(test_filepath))
    test_x, test_y = test[:, :-1], test[:, -1:]
    test_data = test_x, test_y

    train_filename = "east_africa_train.csv"
    train_filepath = Path(dirpath, train_filename)
    train = np.array(pd.read_csv(train_filepath))
    train_x, train_y = train[:, :-1], train[:, -1:]
    train_data = train_x, train_y

    x = np.concatenate([train_x, test_x], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)

    n = x.shape[0]
    ind = np.arange(x.shape[0])
    rng = np.random.RandomState(seed)
    rng.shuffle(ind)

    n_train = int(np.floor(train_proportion * n))
    train_ind = ind[:n_train]
    x_train = x[train_ind]
    y_train = y[train_ind]

    test_ind = ind[n_train:]
    x_test = x[test_ind]
    y_test = y[test_ind]

    train_data = x_train, y_train
    test_data = x_test, y_test

    return train_data, test_data


def norm(x: np.ndarray) -> np.ndarray:
    """Normalise array with mean and variance."""
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-6
    return (x - mu) / std, mu, std


def norm_dataset(data: Dataset) -> Dataset:
    """Normalise dataset tuple."""
    return norm(data[0]), norm(data[1])


def load_data(
    name: str, as_tensor: bool = True, normalise: bool = True, seed: int = 0
) -> DatasetBundle:
    split_proportion = 0.67
    if name == "snelson1d":
        train, test = snelson1d("~/.dataset/snelson1d/")
    elif name == "east_africa":
        train, test = east_africa(
            "~/.datasets/east_africa", train_proportion=split_proportion, seed=seed
        )
    elif name == "naval":
        dat = getattr(bbd, "Naval")(split=seed, prop=split_proportion)
        train, test = (dat.X_train, dat.Y_train), (dat.X_test, dat.Y_test)
    else:
        uci_name = name
        if not name.startswith("Wilson_"):
            uci_name = f"Wilson_{name}"

        dat = getattr(bbd, uci_name)(split=seed, prop=split_proportion)
        train, test = (dat.X_train, dat.Y_train), (dat.X_test, dat.Y_test)

    if normalise:
        (x_train, x_mu, x_std), (y_train, y_mu, y_std) = norm_dataset(train)
        x_test = (test[0] - x_mu) / x_std
        y_test = (test[1] - y_mu) / y_std
    else:
        x_train, y_train = train
        x_test, y_test = test

    x_train = to_float(x_train, as_tensor=as_tensor)
    y_train = to_float(y_train, as_tensor=as_tensor)
    x_test = to_float(x_test, as_tensor=as_tensor)
    y_test = to_float(y_test, as_tensor=as_tensor)

    return DatasetBundle(name, (x_train, y_train), (x_test, y_test))


def to_float(arr: np.ndarray, as_tensor: bool, dtype: Optional[tf.DType] = None):
    if dtype is None:
        dtype = default_float()
    if as_tensor and not isinstance(arr, tf.Tensor):
        return tf.convert_to_tensor(arr, dtype=dtype)
    elif as_tensor:
        return tf.cast(arr, dtype=dtype)
    return arr.astype(dtype)
