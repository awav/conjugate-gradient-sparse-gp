import gpflow
from typing import Callable, Literal, Optional, Tuple
import numpy as np
import tensorflow as tf

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
        self.children = []

class OriginalCoverTree:
    def __init__(
        self,
        distance: Callable,
        data,
        minimum_radius: int,
    ):
        self.distance = distance
        self.root_node = None
        self.leaf_nodes = []
        
        point = data[0]
        distances = self.distance(point, data)
        radius = np.max(distances)

        node = OriginalCoverTreeNode(point, radius, None, data)
        self.root_node = node
        
        while len(node.data) > 0:
            point = node.data[0]
            distances = self.distance(point, node.data)
            radius = node.radius / 2
            data = node.data[distances <= radius]
            if len(data) <= 1 or radius <= minimum_radius:
                self.leaf_nodes.append(node)
                node = node.parent
            else:
                child = OriginalCoverTreeNode(point, radius, node, data)
                node.children.append(child)
                node.data = node.data[distances > radius]
                node = child
