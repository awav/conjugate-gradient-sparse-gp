from collections import namedtuple
import gpflow
from typing import Callable, Optional, Tuple
import numpy as np
import tensorflow as tf
from distance import euclid_distance
import numpy as np
import tensorflow as tf
import gpflow

Tensor = tf.Tensor


def kmeans_indices_and_distances(
    centroids: Tensor,
    points: Tensor,
    distance_fn: Optional[Callable] = None,
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
    points,
    k_centroids: int,
    threshold: float = 1e-5,
    initial_centroids: Optional[Tensor] = None,
    distance_fn: Optional[Callable] = None,
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

    if initial_centroids is None:
        initial_centroid_indices = tf.random.shuffle(tf.range(points.shape[0]))[:k_centroids]
        initial_centroids = tf.gather(points, initial_centroid_indices)

    inf = tf.convert_to_tensor(np.inf, dtype=points.dtype)
    initial_args = body(initial_centroids, inf, None)
    centroids, mean_distance, _ = tf.while_loop(cond, body, initial_args)
    return centroids, mean_distance


def oips(kernel: gpflow.kernels.Kernel, inputs: Tensor, rho: float, max_points: int) -> Tensor:
    n = inputs.shape[0]
    kxx = kernel(inputs, full_cov=False)
    i = tf.math.argmax(kxx)
    inducing_points = tf.convert_to_tensor([inputs[i]])
    indices = tf.convert_to_tensor([i])
    inputs = tf.concat([inputs[:i], inputs[i:]], axis=0)

    def cond(i, j, _inducing_points, _indices):
        return i < n and j < max_points

    def body(i, j, inducing_points, indices):
        point = inputs[i : i + 1]
        kix = kernel(point, inducing_points)
        weight = tf.math.reduce_max(kix)
        if weight < rho:
            new_inducing_points = tf.concat([inducing_points, point], axis=0)
            new_indices = tf.concat([indices, [i]], axis=0)
            return [i + 1, j + 1, new_inducing_points, new_indices]
        return [i + 1, j, inducing_points, indices]

    i0 = tf.constant(1)
    j0 = tf.constant(1)
    initial_state = [i0, j0, inducing_points]
    shapes = [i0.shape, j0.shape, tf.TensorShape([None, inputs.shape[-1]]), tf.TensorShape([None])]
    result = tf.while_loop(cond, body, initial_state, shape_invariants=shapes)
    return result[-1]


def uniform(inputs: Tensor, max_points: int) -> Tuple[Tensor, Tensor]:
    max_value = tf.shape(inputs)[0]
    indices = tf.random.uniform([max_points], maxval=max_value, dtype=max_value.dtype)
    sample = tf.gather(inputs, indices, axis=0)
    return sample, indices


def greedy_selection(kernel, inputs: Tensor, max_points: int) -> Tuple[Tensor, Tensor]:
    n = tf.shape(inputs)[0]
    m = n if max_points > n else max_points

    perm = tf.random.shuffle(tf.range(n))
    X = tf.gather(inputs, perm)  # shuffle training data
    di = kernel(X, full_cov=False)  #  diagonal entries of kernel
    inds = tf.math.argmax(di)[None]  # select first point, point with highest variance
    ci = tf.zeros([1, n], dtype=inputs.dtype)

    State = namedtuple("State", "m, inds, di, ci")
    size = tf.cast(n, di.dtype)

    def stopping_criterion(state):
        return state.m < m

    def loop_body(state):
        j = gpflow.utilities.to_default_int(state.inds[-1])  # index in X of last point chose
        new_Z = X[j : j + 1]  # input value of last point chosen
        dj = tf.math.sqrt(state.di[j])  # conditional standard deviation of point chosen
        cj = state.ci[: state.m, j : j + 1]  # [m, 1]
        K = kernel(X, new_Z, full_cov=True)  # [n, 1],  covariance between new point and all points
        ei = (K - tf.matmul(state.ci, cj, transpose_a=True)) / dj  # [n,1]
        new_ci = tf.concat([state.ci, tf.transpose(ei)], axis=0)
        new_di = state.di - tf.square(ei)[:, 0]
        new_inds = tf.concat([state.inds, [tf.math.argmax(new_di)]], axis=0)
        new_state = State(state.m + 1, new_inds, new_di, new_ci)
        return [new_state]

    init_state = [State(1, inds, di, ci)]
    tshape = tf.TensorShape

    shape_invariants = [State(tshape([]), tshape([None]), tshape([None]), tshape([None, None]))]

    final_state = tf.while_loop(
        stopping_criterion, loop_body, init_state, shape_invariants=shape_invariants
    )
    final_state = tf.nest.map_structure(tf.stop_gradient, final_state)
    perm_inds = tf.gather(perm, final_state[0].inds)
    Z = tf.gather(inputs, perm_inds)
    return Z, perm_inds
