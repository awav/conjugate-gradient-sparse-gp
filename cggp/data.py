from pathlib import Path
import numpy as np
import shutil
from zipfile import ZipFile
from io import BytesIO
import urllib.request


def download_zip_file(outpath, url):
    """Taken from https://stackoverflow.com/a/61195974 (modified)."""

    with urllib.request.urlopen(url) as response:
        with ZipFile(BytesIO(response.read())) as zip_file:
            zip_file.extractall(outpath)


def snelson1d(target_dir: str = "~/.datasets/snelson1d"):
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
        download_zip_file(target_dir_path, data_url)

        dist_dir = Path(target_dir_path, "SPGP_dist")
        shutil.copy(Path(dist_dir, "train_inputs"), inputs_path)
        shutil.copy(Path(dist_dir, "train_outputs"), outputs_path)

        # Clean up everything else
        shutil.rmtree(dist_dir)
        Path(target_dir_path, "SPGP_dist.zip").unlink()

    X = np.loadtxt(inputs_path)[:, None]
    Y = np.loadtxt(outputs_path)[:, None]

    return (X, Y), (X, Y) 
