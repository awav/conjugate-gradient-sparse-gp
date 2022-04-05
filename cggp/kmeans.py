import gpflow
from typing import Callable, Literal, Optional, Tuple
import numpy as np
import tensorflow as tf

Tensor = tf.Tensor


DistanceType = Literal["covariance", "correlation"]


def create_kernel_distance_fn(kernel: gpflow.kernels.Kernel, distance_type: DistanceType):
    def cov(args):
        x, y = args
        x_dist = kernel(x, full_cov=False)
        y_dist = kernel(y, y)  # TODO(awav): apparently, gpflow kernel works inconsistently for different shapes with full_cov=False.
        xy_dist = kernel(x, y)
        distance = x_dist + y_dist - 2 * xy_dist
        return distance
    
    def cor(args):
        x, y = args
        x_dist = kernel(x, full_cov=False)
        y_dist = kernel(y, y)  # TODO(awav): apparently, gpflow kernel works inconsistently for different shapes with full_cov=False.
        xy_dist = kernel(x, y)
        return 1.0 - xy_dist / tf.sqrt(x_dist * y_dist)
    
    functions = {"covariance": cov, "correlation": cor}
    func = functions[distance_type]
    return func


def euclid_distance(args):
    x, y = args
    return tf.linalg.norm(x - y, axis=-1)


def kmeans_indices_and_distances(
    centroids: Tensor, points: Tensor, distance_fn: Optional[Callable] = None
) -> Tuple[Tensor, Tensor]:
    vmap = tf.vectorized_map

    if distance_fn is None:
        distance_fn = euclid_distance

    def pointwise_centroid_indices(point):
        dist = distance_fn((centroids, point))
        min_centroid_dist = tf.math.argmin(dist, axis=-1)
        return min_centroid_dist

    centroid_indices = vmap(pointwise_centroid_indices, points)
    chosen_centroids = tf.gather(centroids, centroid_indices)
    chosen_distances = vmap(distance_fn, (chosen_centroids, points))
    return centroid_indices, chosen_distances


def kmeans_lloyd(
    points, k_centroids: int, threshold: float = 1e-5, distance_fn: Optional[Callable] = None
):
    newaxis = None

    def cond(_, mean_distance, prev_mean_distance):
        return prev_mean_distance - mean_distance > threshold

    def body(centroids, mean_distance, prev_mean_distance):
        indices, distances = kmeans_indices_and_distances(
            centroids, points, distance_fn=distance_fn
        )

        range_indices = tf.range(k_centroids, dtype=indices.dtype)
        counting_map = tf.cast(indices[newaxis, :] == range_indices[:, newaxis], tf.int32)
        counts = tf.reduce_sum(counting_map, axis=1, keepdims=True)
        counts = tf.clip_by_value(counts, 1, counts.dtype.max)
        counts = tf.cast(counts, points.dtype)

        prox = tf.where(
            indices[:, newaxis, newaxis] == range_indices[newaxis, :, newaxis],
            points[:, newaxis, :],
            0,
        )
        new_centroids = tf.reduce_sum(prox, axis=0) / counts
        return new_centroids, tf.reduce_mean(distances), mean_distance

    initial_centroid_indices = tf.random.shuffle(tf.range(points.shape[0]))[:k_centroids]
    initial_centroids = tf.gather(points, initial_centroid_indices)
    inf = tf.convert_to_tensor(np.inf, dtype=points.dtype)
    initial_args = body(initial_centroids, inf, None)
    centroids, mean_distance, _ = tf.while_loop(cond, body, initial_args)
    return centroids, mean_distance

