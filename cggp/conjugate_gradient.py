import abc
from typing import Callable, NamedTuple, Optional, Tuple
import tensorflow as tf

Tensor = tf.Tensor


def conjugate_gradient(
    matrix: Tensor,
    rhs: Tensor,
    initial_solution: Tensor,
    error_threshold: float,
    max_iterations: Optional[int] = None,
    max_steps_cycle: int = 100,
    preconditioner: Optional[Callable] = None,
):
    """
    Conjugate gradient for solving system of linear equations math:`Av = b`.

    Args:
        matrix: Matrix `A` to backsolve from, [n, n].
        rhs: Vector `b` to backsolve, [m, n] where `m` is a batch size.
        initial_solution: Initialization for approximate solution vector `v`, [m, n].
        preconditioner: Precondition function. Default is `EyePreconditioner`.
        error_threshold:
    """

    if max_iterations is None:
        max_iterations = tf.shape(matrix)[0]

    preconditioner = EyePreconditioner() if preconditioner is None else preconditioner


    A = matrix
    v = initial_solution
    b = rhs

    class CGState(NamedTuple):
        """
        Args:
            i: Iteration index
            v: Solution vector(s)
        """

        i: Tensor
        v: Tensor
        r: Tensor
        p: Tensor
        rz: Tensor

    def stopping_condition(state):
        return (tf.reduce_any(0.5 * state.rz > error_threshold)) and (state.i < max_iterations)

    def cg_step(state):
        pA = state.p @ A
        denom = tf.reduce_sum(state.p * pA, axis=-1, keepdims=True)
        gamma = state.rz / denom
        v = state.v + gamma * state.p
        i = state.i + 1
        r = tf.cond(
            state.i % max_steps_cycle == max_steps_cycle - 1,
            lambda: b - v @ A,
            lambda: state.r - gamma * pA,
        )
        z, new_rz = preconditioner(r)
        p = tf.cond(
            state.i % max_steps_cycle == max_steps_cycle - 1,
            lambda: z,
            lambda: z + state.p * new_rz / state.rz,
        )
        return [CGState(i, v, r, p, new_rz)]

    vA = v @ A
    r = b - vA
    z, rz = preconditioner(r)
    p = z
    i = tf.constant(0, dtype=tf.int32)
    initial_state = CGState(i, v, r, p, rz)
    final_state = tf.while_loop(stopping_condition, cg_step, [initial_state])
    final_state = tf.nest.map_structure(tf.stop_gradient, final_state)[0]
    stats_steps = final_state.i
    stats_error = 0.5 * final_state.rz
    solution = final_state.v
    return solution, (stats_steps, stats_error)


class CGPreconditioner:
    @abc.abstractmethod
    def __call__(self, vec: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class EyePreconditioner:
    @abc.abstractmethod
    def __call__(self, vec: Tensor) -> Tuple[Tensor, Tensor]:
        return vec, tf.reduce_sum(tf.square(vec), axis=-1, keepdims=True)
