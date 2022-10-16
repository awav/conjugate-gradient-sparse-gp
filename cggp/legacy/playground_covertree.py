import numpy as np
from covertree import CoverTree
import matplotlib.pyplot as plt
import time

def euclid_distance(args):
    x, y = args
    return np.linalg.norm(x - y, axis=-1)

if __name__ == "__main__":
    x = np.random.rand(1024*16, 2)
    y = np.random.rand(1024*16, 1)
    data = x, y
    start = time.time()
    tree = CoverTree(euclid_distance, data, spatial_resolution=0.18/16, plotting=True)
    print(f"Total time: {time.time() - start}")
    fig, axes = plt.subplots(1, len(tree.levels), figsize=((6 * len(tree.levels), 6)))
    for level in range(len(tree.levels)):
        ax = axes[level]
        points = np.array([node.point for node in tree.levels[level]])
        radius = tree.levels[level][0].radius
        distances = np.linalg.norm(points[:,None,:] - points[None,:,:], axis=-1)
        separation = distances[np.triu_indices_from(distances,1)].min() if level > 0 and distances.size > 1 else 0
        ratio = radius/separation if separation > 0 else 0
        ax.set_xlim([-0.125, 1.125])
        ax.set_ylim([-0.125, 1.125])
        ax.axis('equal')
        ax.set_title(f"$R_{level} = {radius}$, $\delta_{level} = {separation:.02g}$, $\\frac{{R_{level}}}{{\delta_{level}}} = {ratio:.02f}$")
        for (idx,node) in enumerate(tree.levels[level]):
            point = node.point
            node_data, _ = node.plotting_data
            if level > 0: np.savetxt(f"ct_L{level}_X{idx}.csv", node.plotting_data[0], delimiter=",")
            if level > 0: np.savetxt(f"ct_L{level}_V{idx}.csv", node.voronoi_data[0], delimiter=",")
            ax.scatter(node_data[:, 0], node_data[:, 1], alpha=0.1)
            ax.scatter(point[0], point[1], c="white", marker="o", edgecolors="black", s=10)
            circle = plt.Circle((point[0], point[1]), radius, color="C0", alpha=0.1)
            ax.add_patch(circle)
        if level > 0: np.savetxt(f"ct_L{level}.csv", np.array([node.point for node in tree.levels[level]]), delimiter=",")
    if level > 0: np.savetxt("ct_r.csv", np.array([nodes[0].radius for nodes in tree.levels])[1:])

    for (idx,nodes) in enumerate(tree.levels):
        print(f"L{idx}X: ", np.concatenate([node.plotting_data[0] for node in nodes], axis=-2).size)
        print(f"L{idx}V: ", np.concatenate([node.voronoi_data[0] for node in nodes], axis=-2).size)

    print(f"Saving, total time: {time.time() - start}")
    if x.size < 512:
        plt.savefig("covertree.pdf")
        plt.show()
    else:
        plt.savefig("covertree.png")
    print(f"Saved, total time: {time.time() - start}")
