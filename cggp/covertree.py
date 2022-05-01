import gpflow
from typing import Callable, Literal, Optional, Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

Tensor = tf.Tensor
DistanceType = Literal["Euclidean","covariance", "correlation"]


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
        self.data = data.copy()
        self.original_data = data.copy()
        self.children = []

    def print(self, original_data):
        plt.scatter(original_data[:, 0], original_data[:, 1], c='C1', marker='o')
        plt.scatter(self.original_data[:, 0], self.original_data[:, 1], c= 'C2', marker = 'x')
        plt.scatter(self.point[0], self.point[1], c='C3', marker = '+', s = 40)
        circle = plt.Circle((self.point[0], self.point[1]), self.radius, color="blue", alpha=0.2)
        ax = plt.gca()
        plt.xlim([-0.5,1.5])
        plt.ylim([-0.5,1.5])
        ax.add_patch(circle)
        plt.show()

class OriginalCoverTree:
    def __init__(
        self,
        distance: Callable,
        data,
        minimum_radius: float,
    ):
        self.distance = distance
        self.root_node = None
        self.leaf_nodes = []
        self.nodes = []
        original_data = data.copy()
        
        point = data[0]
        distances = self.distance((point, data))
        radius = np.max(distances)

        node = CoverTreeNode(point, radius, None, data)
        self.root_node = node
        self.nodes.append(node)
        
        while len(node.data) > 0 or node.parent is not None:
            node.print(original_data)
            if len(node.data) == 0:
                self.leaf_nodes.append(node)
                node = node.parent
            else:
                point = node.data[0]
                distances = self.distance((point, node.data))
                radius = node.radius / 2
                data = node.data[distances <= radius, :]
                if len(data) <= 1: # or radius <= minimum_radius:
                    self.leaf_nodes.append(node)
                    node = node.parent
                else:
                    child = CoverTreeNode(point, radius, node, data)
                    self.nodes.append(child)
                    node.children.append(child)
                    node.data = node.data[distances > radius, :].copy()
                    node = child

class ModifiedCoverTree:
    def __init__(
        self,
        distance: Callable,
        data,
        num_levels: int,
    ):
        self.distance = distance
        self.levels = [[] for _ in range(num_levels)]
        original_data = data.copy()
        
        root_mean = data.mean(axis = -2)
        root_distances = self.distance((root_mean, data))
        max_radius = np.max(root_distances)

        node = CoverTreeNode(root_mean, max_radius, None, data)
        self.levels[0].append(node)

        for level in range(1, num_levels):
            radius = max_radius / (2 ** level)
            for node in self.levels[level - 1]:
                active_data = node.data.copy()
                while len(active_data) > 0:
                    point = active_data[0]
                    point_distances = self.distance((point, active_data))
                    point_neighborhood = active_data[point_distances <= radius, :]
                    mean = (0.75 * point_neighborhood.mean(axis = -2)) + (0.25 * node.point)
                    mean_distances = self.distance((mean, active_data))
                    mean_idxs = mean_distances <= radius
                    mean_neighborhood = active_data[mean_idxs, :]
                    child = CoverTreeNode(mean, radius, node, mean_neighborhood)
                    self.levels[level].append(child)
                    node.children.append(child)
                    active_data = active_data[~mean_idxs, :]
        
        self.nodes = [node for level in self.levels for node in level]
                


