from typing import Callable, Optional, Union
import tensorflow as tf
from pathlib import Path
from typing import Dict
import numpy as np
import json

Tensor = tf.Tensor


def add_diagonal(matrix: Tensor, diagonal: Tensor):
    """
    Returns `matrix + diagional`, where `diagonal` is a vector of size math::`n`,
    and `matrix` has shape math::`[n, n]`.
    """
    matrix_diag = tf.linalg.diag_part(matrix)
    return tf.linalg.set_diag(matrix, matrix_diag + diagonal)


def jit(apply: bool = True, **function_kwargs):
    def inner(func: Callable) -> Callable:
        if apply:
            return tf.function(func, **function_kwargs)
        return func

    return inner


def store_as_npy(path: Path, logs: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, logs, allow_pickle=True)


def load_from_npy(path: Path) -> Dict:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    return data.item()


def to_numpy(logs: Dict):
    return {key: np.array(val) for key, val in logs.items()}


def transform_to_dataset(
    data, batch_size, repeat: bool = True, shuffle: Optional[int] = None
) -> tf.data.Dataset:
    data = tf.data.Dataset.from_tensor_slices(data)
    if shuffle is not None:
        data = data.shuffle(shuffle)

    data = data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    if repeat:
        data = data.repeat()
    return data


def store_as_json(filename: Union[Path, str], obj):
    with open(filename, "w") as fp:
        json.dump(obj, fp)


def load_from_json(filename: Union[Path, str]):
    with open(filename, "r") as fp:
        return json.load(fp)
