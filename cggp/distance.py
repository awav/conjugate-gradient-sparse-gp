from typing import Literal
import tensorflow as tf
import gpflow

Tensor = tf.Tensor
DistanceType = Literal["euclidean", "covariance", "correlation"]


def euclid_distance(args):
    x, y = args
    return tf.linalg.norm(x - y, axis=-1)


def create_distance_fn(kernel: gpflow.kernels.Kernel, distance_type: DistanceType):
    def cov(args):
        x, y = args
        x_dist = kernel(x, full_cov=False)
        # TODO(awav): apparently, gpflow kernel works inconsistently for different shapes with full_cov=False.
        y_dist = kernel(y, full_cov=False)
        xy_dist = kernel(x, y)
        distance = x_dist + y_dist - 2 * xy_dist
        return distance

    def cor(args):
        x, y = args
        x_dist = kernel(x, full_cov=False)
        # TODO(awav): apparently, gpflow kernel works inconsistently for different shapes with full_cov=False.
        y_dist = kernel(y, full_cov=False)
        xy_dist = kernel(x, y)
        return 1.0 - xy_dist / tf.sqrt(x_dist * y_dist)

    functions = {"covariance": cov, "correlation": cor, "euclidean": euclid_distance}
    func = functions[distance_type]
    return func
