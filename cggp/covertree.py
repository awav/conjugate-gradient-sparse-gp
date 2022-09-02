from typing import Tuple, List
from typing import Callable, Literal, Optional
import warnings
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
        r_neighbors: Optional[list] = None
    ):
        self.point = point
        self.radius = radius
        self.parent = parent
        self.data = data
        self.children = []
        if r_neighbors is None:
            r_neighbors = [self]
        self.r_neighbors = r_neighbors


class CoverTree:
    def __init__(
        self,
        distance: Callable,
        data,
        spatial_resolution: Optional[float] = None,
        num_levels: Optional[int] = 1,
        lloyds = True,
        voronoi = True,
        plotting = False,
    ):
        warnings.warn("Distance function will be ignored and instead `numpy.linalg.norm` will be used.")

        def distance_fn(args):
            x, y = args
            result = np.linalg.norm(x - y, axis=-1)
            return result
            # result = distance(args)
            # return np.array(result, dtype=result.dtype.as_numpy_dtype)

        self.distance = distance_fn
        (x, y) = data

        root_mean = x.mean(axis=-2)
        root_distances = self.distance((root_mean, x))
        max_radius = np.max(root_distances)

        if spatial_resolution is not None:
            num_levels = math.ceil(math.log2(max_radius / spatial_resolution)) + 1
            max_radius = spatial_resolution * (2 ** (num_levels - 1))

        root = CoverTreeNode(root_mean, max_radius, None, data, None)
        if voronoi: root.voronoi_data = (root.data[0].copy(), root.data[1].copy())
        if plotting: root.plotting_data = (root.data[0].copy(), root.data[1].copy())
        self.levels = [[] for _ in range(num_levels)]
        self.levels[0].append(root)
        neighbor_factor = 4 * (1 - 1/2**np.arange(num_levels, -1, -1)) 

        for level in range(1, num_levels):
            radius = max_radius / (2**level)
            for parent in self.levels[level - 1]:
                while len(parent.data[0]) > 0:
                    initial_point = parent.data[0][0]
                    if lloyds:
                        initial_r_neighbor_x = parent.data[0]
                        initial_distances = self.distance((initial_point, initial_r_neighbor_x))
                        initial_neighborhood = initial_r_neighbor_x[initial_distances <= radius, :]
                        point = initial_neighborhood.mean(axis = -2)
                        for r_neighbor in parent.r_neighbors:
                            for child in r_neighbor.children:
                                if np.linalg.norm(point - child.point) < radius:
                                    point = initial_point
                                    break
                            else:
                                continue
                            break
                    else:
                        point = initial_point
                    neighborhood_x = np.empty((0,parent.data[0].shape[-1]))
                    neighborhood_y = np.empty((0,parent.data[1].shape[-1]))
                    for r_neighbor in parent.r_neighbors:
                        (r_neighbor_x, r_neighbor_y) = r_neighbor.data
                        distances = self.distance((point, r_neighbor_x))
                        indices = distances <= radius
                        neighborhood_x = np.concatenate((neighborhood_x, r_neighbor_x[indices, :]), axis=-2)
                        neighborhood_y = np.concatenate((neighborhood_y, r_neighbor_y[indices, :]), axis=-2)
                        r_neighbor.data = (r_neighbor_x[~indices, :], r_neighbor_y[~indices, :])
                    child = CoverTreeNode(point, radius, parent, (neighborhood_x, neighborhood_y))
                    self.levels[level].append(child)
                    parent.children.append(child)
            for parent in self.levels[level-1]:
                potential_child_r_neighbors = [child for r_neighbor in parent.r_neighbors for child in r_neighbor.children]
                # children = [child.point for child in parent.children]
                # r_neighbors = [child.point for child in potential_child_r_neighbors]
                for child in parent.children:
                    child.r_neighbors = [r_neighbor for r_neighbor in potential_child_r_neighbors if self.distance((r_neighbor.point, child.point)) <= neighbor_factor[level]*radius]
                    if plotting:
                        child.plotting_data = (child.data[0].copy(), child.data[1].copy())
            if voronoi:
                for parent in self.levels[level-1]:
                    (voronoi_x, voronoi_y) = parent.voronoi_data
                    if voronoi_x.size > 0:
                        potential_child_r_neighbors = [child for r_neighbor in parent.r_neighbors for child in r_neighbor.children]
                        potential_points = np.stack([child.point for child in potential_child_r_neighbors])
                        potential_distances = self.distance((potential_points[:,None,...], voronoi_x[None,:,...]))
                        nearest_potential_child = np.argmin(potential_distances, axis=0)
                        for (idx, child) in enumerate(potential_child_r_neighbors):
                            if not hasattr(child, "voronoi_data"):
                                child.voronoi_data = (np.empty((0,parent.voronoi_data[0].shape[-1])), np.empty((0,parent.voronoi_data[1].shape[-1])))
                            child_indices = nearest_potential_child == idx
                            node_neighborhood_x = voronoi_x[child_indices, :]
                            node_neighborhood_y = voronoi_y[child_indices, :]
                            neighborhood_x = np.concatenate((child.voronoi_data[0], node_neighborhood_x))
                            neighborhood_y = np.concatenate((child.voronoi_data[1], node_neighborhood_y))
                            voronoi_x = voronoi_x[~child_indices, :]
                            voronoi_y = voronoi_y[~child_indices, :]
                            nearest_potential_child = nearest_potential_child[~child_indices]
                            child.voronoi_data = (neighborhood_x, neighborhood_y)
                            child.data = (child.voronoi_data[0].copy(), child.voronoi_data[1].copy())

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
        return np.array(means, dtype=dtype)[..., None], np.array(counts, dtype=dtype)[..., None]