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



class OriginalCoverTreeNode:
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
        original_data = data
        
        point = data[0]
        distances = self.distance((point, data))
        radius = np.max(distances)

        node = OriginalCoverTreeNode(point, radius, None, data)
        self.root_node = node
        self.nodes.append(node)
        
        while len(node.data) > 0:
            plt.scatter(original_data[:, 0], original_data[:, 1], c='C1', marker='o')
            plt.scatter(node.original_data[:, 0], node.original_data[:, 1], c= 'C2', marker = 'x')
            plt.scatter(node.point[0], node.point[1], c='C3', marker = '+', s = 40)
            circle = plt.Circle((node.point[0], node.point[1]), node.radius, color="blue", alpha=0.2)
            ax = plt.gca()
            plt.xlim([-0.5,1.5])
            plt.ylim([-0.5,1.5])
            ax.add_patch(circle)
            plt.show()

            point = node.data[0]
            distances = self.distance((point, node.data))
            radius = node.radius / 2
            # import pdb; pdb.set_trace()
            data = node.data[distances <= radius, :]
            if len(data) <= 1: # or radius <= minimum_radius:
                self.leaf_nodes.append(node)
                node = node.parent
            else:
                child = OriginalCoverTreeNode(point, radius, node, data)
                self.nodes.append(child)
                node.children.append(child)
                node.data = node.data[distances > radius, :].copy()
                node = child
