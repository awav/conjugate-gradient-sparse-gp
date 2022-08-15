from typing import Tuple, List
from typing import Callable, Literal, Optional
# import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np


# Tensor = tf.Tensor


class CoverTreeNode:
    def __init__(
        self,
        point,
        radius,
        parent,
        data,
        R_neighbors: Optional[list] = None
    ):
        self.point = point
        self.radius = radius
        self.parent = parent
        self.data = data
        self.original_data = (data[0].copy(), data[1].copy())
        self.children = []
        if R_neighbors is None:
            R_neighbors = [self]
        self.R_neighbors = R_neighbors



class CoverTree:
    def __init__(
        self,
        distance: Callable,
        data,
        spatial_resolution: Optional[float] = None,
        num_levels: Optional[int] = 1,
        lloyds = True,
        voronoi = False,
    ):
        self.distance = distance
        (x, y) = data

        root_mean = x.mean(axis=-2)
        root_distances = self.distance((root_mean, x))
        max_radius = np.max(root_distances)

        if spatial_resolution is not None:
            num_levels = math.ceil(math.log2(max_radius / spatial_resolution)) + 1
            max_radius = spatial_resolution * (2 ** (num_levels - 1))

        root = CoverTreeNode(root_mean, max_radius, None, data, None)
        self.levels = [[] for _ in range(num_levels)]
        self.levels[0].append(root)


        for level in range(1, num_levels):
            radius = max_radius / (2**level)
            for parent in self.levels[level - 1]:
                while len(parent.data[0]) > 0:
                    initial_point = parent.data[0][0]
                    if lloyds:
                        initial_R_neighbor_x = np.concatenate([r.data[0] for r in parent.R_neighbors], axis=-2)
                        initial_distances = self.distance((initial_point, initial_R_neighbor_x))
                        initial_neighborhood = initial_R_neighbor_x[initial_distances <= radius, :]
                        point = initial_neighborhood.mean(axis = -2)
                    else:
                        point = initial_point
                    neighborhood_x = np.empty((0,parent.data[0].shape[-1]))
                    neighborhood_y = np.empty((0,parent.data[1].shape[-1]))
                    for R_neighbor in parent.R_neighbors:
                        (R_neighbor_x, R_neighbor_y) = R_neighbor.data
                        distances = self.distance((point, R_neighbor_x))
                        indices = distances <= radius
                        neighborhood_x = np.concatenate((neighborhood_x, R_neighbor_x[indices, :]), axis=-2)
                        neighborhood_y = np.concatenate((neighborhood_y, R_neighbor_y[indices, :]), axis=-2)
                        R_neighbor.data = (R_neighbor_x[~indices, :], R_neighbor_y[~indices, :])
                    child = CoverTreeNode(point, radius, parent, (neighborhood_x, neighborhood_y))
                    self.levels[level].append(child)
                    parent.children.append(child)
            for parent in self.levels[level-1]:
                potential_child_R_neighbors = [child for R_neighbor in parent.R_neighbors for child in R_neighbor.children]
                for child in parent.children:
                    child.R_neighbors = [R_neighbor for R_neighbor in potential_child_R_neighbors if np.linalg.norm(R_neighbor.point - child.point) <= 4*radius]
            if voronoi:
                for parent in self.levels[level-1]:
                    (voronoi_x, voronoi_y) = parent.original_data
                    if voronoi_x.size > 0:
                        potential_child_R_neighbors = [child for R_neighbor in parent.R_neighbors for child in R_neighbor.children]
                        potential_points = np.stack([child.point for child in potential_child_R_neighbors])
                        potential_distances = np.linalg.norm(potential_points[:,None,...] - voronoi_x[None,:,...], axis=-1)
                        nearest_potential_child = np.argmin(potential_distances, axis=0)
                        for (idx, child) in enumerate(potential_child_R_neighbors):
                            child_indices = nearest_potential_child == idx
                            node_neighborhood_x = voronoi_x[child_indices, :]
                            node_neighborhood_y = voronoi_y[child_indices, :]
                            neighborhood_x = np.concatenate((child.data[0], node_neighborhood_x))
                            neighborhood_y = np.concatenate((child.data[1], node_neighborhood_y))
                            voronoi_x = voronoi_x[~child_indices, :]
                            voronoi_y = voronoi_y[~child_indices, :]
                            nearest_potential_child = nearest_potential_child[~child_indices]
                            child.data = (neighborhood_x, neighborhood_y)
                            child.original_data = (child.data[0].copy(), child.data[1].copy())

        self.nodes = [node for level in self.levels for node in level]
