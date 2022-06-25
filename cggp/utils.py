from typing import Callable
import tensorflow as tf
from pathlib import Path
from typing import Dict
import numpy as np

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


def store_logs(path: Path, logs: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, logs, allow_pickle=True)


def to_numpy(logs: Dict):
    return {key: np.array(val) for key, val in logs.items()}
