import numpy as np
from collections import namedtuple
import tensorflow as tf
import gpflow

Tensor = tf.Tensor


def oips(kernel: gpflow.kernels.Kernel, inputs: Tensor, rho: float, max_points: int) -> Tensor:
    n = inputs.shape[0]
    kxx = kernel(inputs, full_cov=False)
    i = tf.math.argmax(kxx)
    inducing_points = tf.convert_to_tensor([inputs[i]])
    inputs = tf.concat([inputs[:i], inputs[i:]], axis=0)

    def cond(i, j, _):
        return i < n and j < max_points

    def body(i, j, inducing_points):
        point = inputs[i : i + 1]
        kix = kernel(point, inducing_points)
        weight = tf.math.reduce_max(kix)
        if weight < rho:
            inducing_points = tf.concat([inducing_points, point], axis=0)
            return [i + 1, j + 1, inducing_points]
        return [i + 1, j, inducing_points]

    i0 = tf.constant(1)
    j0 = tf.constant(1)
    initial_state = [i0, j0, inducing_points]
    shapes = [i0.shape, j0.shape, tf.TensorShape([None, inputs.shape[-1]])]
    result = tf.while_loop(cond, body, initial_state, shape_invariants=shapes)
    return result[-1]


# if __name__ == "__main__":
#     n = 100
#     d = 1
#     m = 50
#     rho = 0.7
#     k = gpflow.kernels.SquaredExponential()
#     inputs = tf.random.normal([n, d], dtype=gpflow.config.default_float())
#     oips_fn = lambda x: oips(k, x, m, rho)
#     oips_jit = tf.function(oips_fn)
#     inducing_points_v1 = oips(k, inputs, m, rho)
#     inducing_points_v2 = oips_jit(inputs)
#     print()
