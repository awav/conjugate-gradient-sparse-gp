from typing import Tuple, List
from typing import Callable, Literal, Optional
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np


Tensor = tf.Tensor


class CoverTreeNode:
    def __init__(
        self,
        point,
        radius,
        parent,
        data,
        siblings: Optional[list] = [],
        plotting: bool = False,
    ):
        self.point = point
        self.radius = radius
        self.parent = parent
        self.data = data
        self.original_data = (data[0].copy(), data[1].copy())
        self.children = []
        self.siblings = siblings
        if plotting:
            pass


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
                    child_distances = tf.linalg.norm(children[:,None, ...] - balance_active_x[None,:, ...], axis=-1)
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


class SiblingAwareCoverTree:
    def __init__(
        self,
        distance: Callable,
        data,
        spatial_resolution: Optional[float] = None,
        num_levels: Optional[int] = 1,
        plotting: bool = False,
    ):
        def distance_fn(args):
            result = distance(args)
            return np.array(result, dtype=result.dtype.as_numpy_dtype)

        self.distance = distance_fn
        (x, y) = data

        root_mean = x.mean(axis=-2)
        root_distances = self.distance((root_mean, x))
        max_radius = np.max(root_distances)

        if spatial_resolution is not None:
            num_levels = math.ceil(math.log2(max_radius / spatial_resolution)) + 1
            max_radius = spatial_resolution * (2 ** (num_levels - 1))

        node = CoverTreeNode(root_mean, max_radius, None, data, [], plotting=plotting)
        self.levels = [[] for _ in range(num_levels)]
        self.levels[0].append(node)

        for level in range(1, num_levels):
            radius = max_radius / (2**level)
            for node in self.levels[level - 1]:
                (active_x, active_y) = node.data
                while len(active_x) > 0:
                    initial_point = active_x[0]
                    initial_distances = self.distance((initial_point, active_x))
                    initial_neighborhood = active_x[initial_distances <= radius, :]
                    point = initial_neighborhood.mean(axis = -2)
                    distances = self.distance((point, active_x))
                    indices = distances <= radius
                    neighborhood_x = active_x[indices, :]
                    neighborhood_y = active_y[indices, :]
                    active_x = active_x[~indices, :]
                    active_y = active_y[~indices, :]
                    node.data = (active_x, active_y)
                    for sibling in node.siblings:
                        (sibling_x, sibling_y) = sibling.data
                        sibling_distances = self.distance((point, sibling_x))
                        sibling_indices = sibling_distances <= radius
                        sibling_neighborhood_x = sibling_x[sibling_indices, :]
                        sibling_neighborhood_y = sibling_y[sibling_indices, :]
                        neighborhood_x = np.concatenate((neighborhood_x, sibling_neighborhood_x), axis=-2)
                        neighborhood_y = np.concatenate((neighborhood_y, sibling_neighborhood_y), axis=-2)
                        sibling_x = sibling_x[~sibling_indices, :]
                        sibling_y = sibling_y[~sibling_indices, :]
                        sibling.data = (sibling_x, sibling_y)
                    child = CoverTreeNode(point, radius, node, (neighborhood_x, neighborhood_y), plotting=plotting)
                    self.levels[level].append(child)
                    node.children.append(child)
            for node in self.levels[level-1]:
                potential_child_siblings = node.children + [child for sibling in node.siblings for child in sibling.children]
                for child in node.children:    
                    child.siblings = [sibling for sibling in potential_child_siblings if np.linalg.norm(sibling.point - child.point) <= 4*radius]
                    child.data = (child.data[0], child.data[1])

            # for node in self.levels[level-1]:
            #     (voronoi_x, voronoi_y) = node.original_data
            #     if voronoi_x.size > 0:
            #         potential_child_siblings = node.children + [child for sibling in node.siblings for child in sibling.children]
            #         potential_points = np.stack([child.point for child in potential_child_siblings])
            #         potential_distances = np.linalg.norm(potential_points[:,None,...] - voronoi_x[None,:,...], axis=-1)
            #         nearest_potential_child = np.argmin(potential_distances, axis=0)
            #         for (idx, child) in enumerate(potential_child_siblings):
            #             child_indices = nearest_potential_child == idx
            #             node_neighborhood_x = voronoi_x[child_indices, :]
            #             node_neighborhood_y = voronoi_y[child_indices, :]
            #             neighborhood_x = np.concatenate((child.data[0], node_neighborhood_x))
            #             neighborhood_y = np.concatenate((child.data[1], node_neighborhood_y))
            #             voronoi_x = voronoi_x[~child_indices, :]
            #             voronoi_y = voronoi_y[~child_indices, :]
            #             nearest_potential_child = nearest_potential_child[~child_indices]
            #             child.data = (neighborhood_x, neighborhood_y)
            #             child.original_data = (child.data[0].copy(), child.data[1].copy())

        self.nodes = [node for level in self.levels for node in level]
    
    @property
    def centroids(self):
        return np.stack([node.point for node in self.levels[-1]])

    @property
    def cluster_ys(self) -> List[Tensor]:
        ys = [node.data[1] for node in self.levels[-1]]
        return ys

    @property
    def cluster_mean_and_counts(self) -> Tuple[Tensor, Tensor]:
        means_and_counts = [
            (np.mean(node.data[1]), node.data[1].shape[0]) for node in self.levels[-1]
        ]
        dtype = self.levels[-1][0].data[1].dtype
        means, counts = zip(*means_and_counts)
        return means.astype(dtype)[..., None], counts.astype(dtype)[..., None]


    # def rebalance(self):
    #     for level in self.levels:
    #         for node in level:
    #             node.rebalanced_children = []
    #     for level in self.levels:
    #         for node in level:
    #             potential_parents = [node] + node.siblings
    #             potential_points = np.stack([parent.point for parent in potential_parents])
    #             for child in node.children:
    #                 distances = np.linalg.norm(potential_points - child.point[None,:], axis=0)
    #                 min_idx = np.argmin(distances)
    #                 child.parent = potential_parents[min_idx]
    #                 child.parent.rebalanced_children.append(child)
    #     for level in self.levels:
    #         for node in level:
    #             node.children = node.rebalanced_children
    #             del node.rebalanced_children
    #     for level in reversed(self.levels):

                    
                    
                    
                    
