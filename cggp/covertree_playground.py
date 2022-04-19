import numpy as np
from covertree import OriginalCoverTree, euclid_distance
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.random.rand(20, 2)
    minimum_radius = 0.1
    tree = OriginalCoverTree(euclid_distance, data, minimum_radius)
    # for node in tree.nodes:
    #     point = node.point
    #     radius = node.radius
    #     node_data = node.original_data
    #     plt.scatter(data[:, 0], data[:, 1], c='C1', marker='o')
    #     plt.scatter(node_data[:, 0], node_data[:, 1], c= 'C2', marker = 'x')
    #     plt.scatter(point[0], point[1], c='C3', marker = '+', s = 40)
    #     circle = plt.Circle((point[0], point[1]), radius, color="blue", alpha=0.2)
    #     ax = plt.gca()
    #     plt.xlim([-0.5,1.5])
    #     plt.ylim([-0.5,1.5])
    #     ax.add_patch(circle)
    # 
    # plt.show()