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

    min_float = tf.convert_to_tensor(1e-16, dtype=initial_solution.dtype)
    zero = tf.constant(0.0, dtype=initial_solution.dtype)

    @tf.custom_gradient
    def _conjugate_gradient(A, b, v):
        """
        Solve x = A^{-1} b, where v is the initial solution
        """

        def stopping_condition(state):
            norm_r_sq = tf.reduce_sum(tf.square(state.r), axis=-1, keepdims=True)
            over_threshold = tf.reduce_any(0.5 * norm_r_sq > error_threshold)
            return tf.logical_and(over_threshold, (state.i < max_iterations))

        def cg_step(state, b):
            pA = state.p @ A
            denom = tf.reduce_sum(state.p * pA, axis=-1, keepdims=True)
            gamma = state.rz / denom
            gamma = tf.where(denom <= min_float, zero, gamma)
            v = state.v + gamma * state.p
            i = state.i + 1
            reset = state.i % max_steps_cycle == max_steps_cycle - 1
            r = tf.cond(
                reset,
                lambda: b - v @ A,
                lambda: state.r - gamma * pA,
            )
            z, new_rz = preconditioner(r, A)
            z_update = state.p * new_rz / state.rz
            z_update = tf.where(state.rz <= min_float, zero, z_update)
            p = tf.cond(
                reset,
                lambda: z,
                lambda: z + z_update,
            )
            return [CGState(i, v, r, p, new_rz)]

        vA = v @ A
        r = b - vA
        z, rz = preconditioner(r, A)
        p = z
        i = tf.convert_to_tensor(0, dtype=tf.int32)
        initial_state = CGState(i, v, r, p, rz)
        final_state = tf.while_loop(
            stopping_condition, lambda state: cg_step(state, b), [initial_state]
        )[0]
        stats_steps = final_state.i
        stats_error = 0.5 * final_state.rz
        solution = final_state.v

        def grad_conjugate_gradient(
            dx: Tensor, d_stats_steps, d_stats_error
        ) -> Tuple[Tensor, Tensor]:
            """
            Given sensitivity dx for Ax = b, compute db = A^{-1} dx and dA = -db dx^T
            """
            grad_v = tf.zeros_like(dx)
            grad_vA = grad_v @ A
            grad_r = dx - grad_vA
            grad_z, grad_rz = preconditioner(grad_r, A)
            grad_p = grad_z
            grad_i = tf.convert_to_tensor(0, dtype=tf.int32)
            grad_initial_state = CGState(grad_i, grad_v, grad_r, grad_p, grad_rz)
            grad_final_state = tf.while_loop(
                stopping_condition, lambda state: cg_step(state, dx), [grad_initial_state]
            )[0]
            db = grad_final_state.v
            dA = -tf.transpose(solution) @ db
            return (dA, db, None)

        return (solution, (stats_steps, stats_error)), grad_conjugate_gradient

    return _conjugate_gradient(matrix, rhs, initial_solution)


class CGPreconditioner:
    @abc.abstractmethod
    def __call__(self, vec: Tensor, mat: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class EyePreconditioner(CGPreconditioner):
    @abc.abstractmethod
    def __call__(self, vec: Tensor, mat: Tensor) -> Tuple[Tensor, Tensor]:
        return vec, tf.reduce_sum(tf.square(vec), axis=-1, keepdims=True)

class BlockPreconditioner(CGPreconditioner):
    def __init__(self, block_indices) -> None:
        super().__init__()
        self.block_indices = block_indices
    
    def __call__(self, vec: Tensor, mat: Tensor) -> Tuple[Tensor, Tensor]:
        def cholesky_solve_linear_system(indices: Tensor):
            b = tf.gather(vec, indices)
            A_indices = tf.concat((indices[:,None,None] + tf.zeros_like(indices[None,:,None]), indices[None,:,None] + tf.zeros_like(indices[:,None,None])), axis=2)
            A = tf.gather_nd(mat, A_indices)
            L = tf.linalg.cholesky(A)
            return tf.linalg.cholesky_solve(L,b)
        
        new_vec = tf.vectorized_map(cholesky_solve_linear_system, self.block_indices)
        return new_vec, tf.reduce_sum(new_vec * vec, axis=-1, keepdims=True)

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
        rhs = tf.transpose(rhs)

        if initial_solution is None:
            initial_solution = tf.zeros_like(rhs)
        else:
            initial_solution = tf.transpose(initial_solution)

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

        solution = tf.transpose(solution)
        return solution
