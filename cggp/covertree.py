from typing import Tuple, List
import gpflow
from typing import Callable, Literal, Optional
import tensorflow as tf
import matplotlib.pyplot as plt
import math


Tensor = tf.Tensor


class CoverTreeNode:
    def __init__(
        self,
        point,
        radius,
        parent,
        data,
        plotting: bool = False,
    ):
        self.point = point
        self.radius = radius
        self.parent = parent
        self.data = data
        self.children = []
        if plotting:
            self.original_data = data


class ModifiedCoverTree:
    def __init__(
        self,
        distance: Callable,
        data,
        spatial_resolution: Optional[float] = None,
        num_levels: Optional[int] = 1,
        plotting: bool = False,
    ):
        self.distance = distance

        x, y = data
        x = tf.convert_to_tensor(x, dtype=x.dtype)
        y = tf.convert_to_tensor(y, dtype=y.dtype)
        data = (x, y)

        root_mean = tf.reduce_mean(x, axis=-2)
        root_distances = self.distance((root_mean, x))
        max_radius = tf.reduce_max(root_distances)

        if spatial_resolution is not None:
            num_levels = math.ceil(math.log2(max_radius / spatial_resolution)) + 1
            max_radius = spatial_resolution * (2 ** (num_levels - 1))

        node = CoverTreeNode(root_mean, max_radius, None, data, plotting=plotting)
        self.levels = [[] for _ in range(num_levels)]
        self.levels[0].append(node)

        for level in range(1, num_levels):
            radius = max_radius / (2**level)
            for node in self.levels[level - 1]:
                active_x, active_y = node.data
                balance_active_x, balance_active_y = node.data
                while tf.shape(active_x)[0] > 0:
                    point = (2 * active_x[0, ...] / 3) + (node.point / 3)
                    distances = self.distance((point, active_x))
                    indices = distances <= radius
                    active_x = tf.boolean_mask(active_x, ~indices)
                    active_y = tf.boolean_mask(active_y, ~indices)
                    child = CoverTreeNode(point, radius, node, None, plotting=plotting)
                    self.levels[level].append(child)
                    node.children.append(child)
                children = tf.stack([child.point for child in node.children])
                if tf.size(balance_active_x) > 0:
                    child_distances = tf.linalg.norm(
                        children[:, None, ...] - balance_active_x[None, :, ...], axis=-1
                    )
                    nearest_child = tf.math.argmin(child_distances, axis=0)
                    for idx, child in enumerate(node.children):
                        child_indices = tf.equal(nearest_child, idx)
                        neighborhood_x = tf.boolean_mask(balance_active_x, child_indices)
                        neighborhood_y = tf.boolean_mask(balance_active_y, child_indices)
                        balance_active_x = tf.boolean_mask(balance_active_x, ~child_indices)
                        balance_active_y = tf.boolean_mask(balance_active_y, ~child_indices)
                        nearest_child = tf.boolean_mask(nearest_child, ~child_indices)
                        child.data = (neighborhood_x, neighborhood_y)
                        if plotting:
                            child.original_data = (neighborhood_x, neighborhood_y)

        return None

    @property
    def centroids(self):
        return tf.stack([node.point for node in self.levels[-1]])

    @property
    def cluster_ys(self) -> List[Tensor]:
        ys = [node.data[1] for node in self.levels[-1]]
        return ys

    @property
    def cluster_mean_and_counts(self) -> Tuple[Tensor, Tensor]:
        means_and_counts = [
            (tf.reduce_mean(node.data[1]), tf.shape(node.data[1])[0]) for node in self.levels[-1]
        ]
        dtype = self.levels[-1][0].data[1].dtype
        means, counts = zip(*means_and_counts)
        ctt = tf.convert_to_tensor
        return ctt(means, dtype=dtype)[..., None], ctt(counts, dtype=dtype)[..., None]
