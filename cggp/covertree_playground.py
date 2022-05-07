import numpy as np
from covertree import ModifiedCoverTree, euclid_distance
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.random.rand(200, 2)
    minimum_radius = 0.1
    num_levels = 5
    tree = ModifiedCoverTree(euclid_distance, data, num_levels)
    fig, axes = plt.subplots(1, num_levels, figsize=((6*num_levels, 6)))
    for level in range(len(tree.levels)):
        ax = axes[level]
        for node in tree.levels[level]:
            point = node.point
            radius = node.radius
            node_data = node.original_data
            ax.scatter(node_data[:, 0], node_data[:, 1], alpha=0.75)
            ax.scatter(point[0], point[1], c='white', marker = 'o', edgecolors= "black", s = 100)
            circle = plt.Circle((point[0], point[1]), radius, color="C0", alpha=0.1)
            ax.add_patch(circle)
            ax.set_xlim([-0.125,1.125])
            ax.set_ylim([-0.125,1.125])
    
    plt.savefig("covertree.pdf")
    plt.show()