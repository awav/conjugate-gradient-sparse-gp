from typing import Tuple, NamedTuple
from collections import namedtuple
from pathlib import Path
import numpy as np
import shutil
from zipfile import ZipFile
from io import BytesIO
import urllib.request

import bayesian_benchmarks.data as bbd
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


def norm(x: np.ndarray) -> np.ndarray:
    """Normalise array with mean and variance."""
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-6
    return (x - mu) / std, mu, std


def norm_dataset(data: Dataset) -> Dataset:
    """Normalise dataset tuple."""
    return norm(data[0]), norm(data[1])


def load_data(name: str) -> DatasetBundle:
    if name == "snelson1d":
        train, test = snelson1d("~/.dataset/snelson1d/")
    else:
        uci_name = name
        if not name.startswith("Wilson_"):
            uci_name = f"Wilson_{name}"

        dat = getattr(bbd, uci_name)(prop=0.67)
        train, test = (dat.X_train, dat.Y_train), (dat.X_test, dat.Y_test)

    (x_train, x_mu, x_std), (y_train, y_mu, y_std) = norm_dataset(train)
    x_test = (test[0] - x_mu) / x_std
    y_test = (test[1] - y_mu) / y_std
    return DatasetBundle(name,
        (_to_float(x_train), _to_float(y_train)),
        (_to_float(x_test), _to_float(y_test)),
    )


def _to_float(arr: np.ndarray):
    return arr.astype(default_float())
