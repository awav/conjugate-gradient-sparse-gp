import abc
from typing import Union
from distutils.log import error
from typing import Callable, NamedTuple, Optional, Tuple
import tensorflow as tf

Tensor = tf.Tensor


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


def conjugate_gradient(
    matrix: Tensor,
    rhs: Tensor,
    initial_solution: Tensor,
    error_threshold: float,
    preconditioner: Optional[Callable] = None,
    max_iterations: Optional[int] = None,
    max_steps_cycle: int = 100,
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

    if preconditioner is None:
        preconditioner = EyePreconditioner()

    if max_iterations is None:
        max_iterations = tf.shape(matrix)[0]

    A = matrix
    v = initial_solution
    b = rhs

    def stopping_condition(state):
        over_threshold = tf.reduce_any(0.5 * state.rz > error_threshold)
        return tf.logical_and(over_threshold, (state.i < max_iterations))

    def cg_step(state):
        pA = state.p @ A
        min_float = 1e-10
        denom = tf.reduce_sum(state.p * pA, axis=-1, keepdims=True)
        gamma = state.rz / denom
        gamma = tf.where(denom <= min_float, 0.0, gamma)
        v = state.v + gamma * state.p
        i = state.i + 1
        reset = state.i % max_steps_cycle == max_steps_cycle - 1
        r = tf.cond(
            reset,
            lambda: b - v @ A,
            lambda: state.r - gamma * pA,
        )
        z, new_rz = preconditioner(r)
        p = tf.cond(
            reset,
            lambda: z,
            lambda: z + state.p * new_rz / state.rz,
        )
        p = tf.where(state.rz <= min_float, 0.0, p)
        return [CGState(i, v, r, p, new_rz)]

    vA = v @ A
    r = b - vA
    z, rz = preconditioner(r)
    p = z
    i = tf.convert_to_tensor(0, dtype=tf.int32)
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


class EyePreconditioner(CGPreconditioner):
    @abc.abstractmethod
    def __call__(self, vec: Tensor) -> Tuple[Tensor, Tensor]:
        return vec, tf.reduce_sum(tf.square(vec), axis=-1, keepdims=True)


class ConjugateGradient:
    preconditioner: CGPreconditioner
    error_threshold: Union[Tensor, float]
    max_iterations: Optional[int]
    max_steps_cycle: Optional[int]

    def __init__(
        self,
        error_threshold: Union[Tensor, float],
        preconditioner: Optional[CGPreconditioner] = None,
        max_iterations: Optional[int] = None,
        max_steps_cycle: Optional[int] = None,
    ):
        self.error_threshold = error_threshold
        if preconditioner is None:
            preconditioner = EyePreconditioner()
        self.preconditioner = preconditioner
        self.max_iterations = max_iterations
        self.max_steps_cycle = max_steps_cycle

    def __call__(
        self, matrix: Tensor, rhs: Tensor, initial_solution: Optional[Tensor] = None
    ) -> Tensor:
        if initial_solution is None:
            initial_solution = tf.zeros_like(rhs)

        max_iterations = self.max_iterations
        if max_iterations is None:
            max_iterations = tf.shape(matrix)[-1]

        max_steps_cycle = self.max_steps_cycle
        if max_steps_cycle is None:
            max_steps_cycle = max_iterations + 1  # Make sure that we don't run it in the end of CG

        preconditioner = self.preconditioner
        error_threshold = self.error_threshold

        solution, stats = conjugate_gradient(
            matrix,
            rhs,
            initial_solution,
            error_threshold,
            preconditioner=preconditioner,
            max_iterations=max_iterations,
            max_steps_cycle=max_steps_cycle,
        )

        return solution
