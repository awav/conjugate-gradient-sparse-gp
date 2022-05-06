from typing import Callable
import tensorflow as tf

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