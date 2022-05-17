import numpy as np
from covertree import ModifiedCoverTree
from distance import euclid_distance
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.random.rand(128, 2)
    y = np.random.rand(128, 1)
    data = x, y
    np.savetxt(f"ct_x.csv", x, delimiter=",")
    tree = ModifiedCoverTree(euclid_distance, data, spatial_resolution=0.18, plotting=True)
    fig, axes = plt.subplots(1, len(tree.levels), figsize=((6 * len(tree.levels), 6)))
    for level in range(len(tree.levels)):
        ax = axes[level]
        points = np.array([node.point.numpy() for node in tree.levels[level]])
        radius = tree.levels[level][0].radius
        distances = np.linalg.norm(points[:,None,:] - points[None,:,:], axis=-1)
        separation = distances[np.triu_indices_from(distances,1)].min() if level > 0 else 0
        ratio = radius/separation if separation > 0 else 0
        ax.set_xlim([-0.125, 1.125])
        ax.set_ylim([-0.125, 1.125])
        ax.axis('equal')
        ax.set_title(f"$R_{level} = {radius}$, $\delta_{level} = {separation:.02g}$, $\\frac{{R_{level}}}{{\delta_{level}}} = {ratio:.02f}$")
        for node in tree.levels[level]:
            point = node.point
            node_data, _ = node.original_data
            ax.scatter(node_data[:, 0], node_data[:, 1], alpha=0.75)
            ax.scatter(point[0], point[1], c="white", marker="o", edgecolors="black", s=10)
            circle = plt.Circle((point[0], point[1]), radius, color="C0", alpha=0.1)
            ax.add_patch(circle)
        out = np.array([np.concatenate((node.point.numpy(),np.array([node.radius]),node.parent.point.numpy() if node.parent is not None else ())) for node in tree.levels[level]])
        if level > 0:
            np.savetxt(f"ct_L{level}.csv", out, delimiter=",")

    if x.size < 512:
        plt.savefig("covertree.pdf")
        plt.show()
    else:
        plt.savefig("covertree.png")
    print()
