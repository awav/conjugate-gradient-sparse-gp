import gpflow
from typing import Callable, Literal, Optional
import tensorflow as tf
import matplotlib.pyplot as plt
import math

Tensor = tf.Tensor
DistanceType = Literal["Euclidean", "covariance", "correlation"]


def create_kernel_distance_fn(kernel: gpflow.kernels.Kernel, distance_type: DistanceType):
    def cov(args):
        x, y = args
        x_dist = kernel(x, full_cov=False)
        y_dist = kernel(
            y, y
        )  # TODO(awav): apparently, gpflow kernel works inconsistently for different shapes with full_cov=False.
        xy_dist = kernel(x, y)
        distance = x_dist + y_dist - 2 * xy_dist
        return distance

    def cor(args):
        x, y = args
        x_dist = kernel(x, full_cov=False)
        y_dist = kernel(
            y, y
        )  # TODO(awav): apparently, gpflow kernel works inconsistently for different shapes with full_cov=False.
        xy_dist = kernel(x, y)
        return 1.0 - xy_dist / tf.sqrt(x_dist * y_dist)

    functions = {"covariance": cov, "correlation": cor}
    func = functions[distance_type]
    return func


def euclid_distance(args):
    x, y = args
    return tf.linalg.norm(x - y, axis=-1)


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
        self.original_data = data.numpy()
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
        min_radius: Optional[float] = None,
        num_levels: Optional[int] = 1,
    ):
        self.distance = distance

        data = tf.convert_to_tensor(data)

        root_mean = tf.math.reduce_mean(data, axis=-2)
        root_distances = self.distance((root_mean, data))
        max_radius = tf.math.reduce_mean(root_distances)

        print(max_radius)
        print(max_radius / min_radius)
        print(tf.math.log(max_radius / min_radius) / math.log(2))

        if min_radius is not None:
            num_levels = math.ceil(math.log2(max_radius / min_radius)) + 2
            max_radius = min_radius * (2**(num_levels-1))

        print(num_levels)
        print(max_radius)

        node = CoverTreeNode(root_mean, max_radius, None, data)
        self.levels = [[] for _ in range(num_levels)]
        self.levels[0].append(node)

        for level in range(1, num_levels):
            radius = max_radius / (2**level)
            for node in self.levels[level - 1]:
                active_data = node.data
                while len(active_data) > 0:
                    point = (0.75 * active_data[0]) + (0.25 * node.point)
                    distances = self.distance((point, active_data))
                    indices = distances <= radius
                    neighborhood = tf.boolean_mask(active_data, indices)
                    child = CoverTreeNode(point, radius, node, neighborhood)
                    self.levels[level].append(child)
                    node.children.append(child)
                    active_data = tf.boolean_mask(active_data, ~indices)
    
    def inducing_points(self):
        return tf.stack([node.point for node in self.levels[-1]])
