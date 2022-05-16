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
    ):
        self.point = point
        self.radius = radius
        self.parent = parent
        self.data = data
        self.original_data = data
        self.children = []

    def print(self, original_data):
        plt.scatter(original_data[:, 0], original_data[:, 1], c="C1", marker="o")
        plt.scatter(self.original_data[:, 0], self.original_data[:, 1], c="C2", marker="x")
        plt.scatter(self.point[0], self.point[1], c="C3", marker="+", s=40)
        circle = plt.Circle((self.point[0], self.point[1]), self.radius, color="blue", alpha=0.2)
        ax = plt.gca()
        plt.xlim([-0.5, 1.5])
        plt.ylim([-0.5, 1.5])
        ax.add_patch(circle)
        plt.show()


class ModifiedCoverTree:
    def __init__(
        self,
        distance: Callable,
        data,
        spatial_resolution: Optional[float] = None,
        num_levels: Optional[int] = 1,
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
            num_levels = math.ceil(math.log2(max_radius / spatial_resolution)) + 2
            max_radius = spatial_resolution * (2 ** (num_levels - 1))

        node = CoverTreeNode(root_mean, max_radius, None, data)
        self.levels = [[] for _ in range(num_levels)]
        self.levels[0].append(node)

        for level in range(1, num_levels):
            radius = max_radius / (2**level)
            for node in self.levels[level - 1]:
                active_x, active_y = node.data
                while tf.shape(active_x)[0] > 0:
                    point = (0.75 * active_x[0, ...]) + (0.25 * node.point)
                    distances = self.distance((point, active_x))
                    indices = distances <= radius
                    neighborhood_x = tf.boolean_mask(active_x, indices)
                    neighborhood_y = tf.boolean_mask(active_y, indices)
                    neighborhood = (neighborhood_x, neighborhood_y)
                    child = CoverTreeNode(point, radius, node, neighborhood)
                    self.levels[level].append(child)
                    node.children.append(child)
                    active_x = tf.boolean_mask(active_x, ~indices)
                    active_y = tf.boolean_mask(active_y, ~indices)

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
