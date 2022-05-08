import matplotlib.pyplot as plt
from functools import partial
from gpflow.kernels import SquaredExponential
import tensorflow as tf

from kmeans import create_kernel_distance_fn, kmeans_lloyd, kmeans_indices_and_distances


if __name__ == "__main__":
    tf.random.set_seed(999)
    n = 1000
    d = 2
    k_centroids = 4
    kernel = SquaredExponential(lengthscales=10.0)
    # distance_type = "correlation"
    distance_type = "covariance"
    distance_fn = create_kernel_distance_fn(kernel, distance_type=distance_type)
    points = tf.random.normal((n, d), dtype=tf.float64)
    points_numpy = points.numpy()

    kmeans_lloyd_jit = tf.function(partial(kmeans_lloyd, distance_fn=distance_fn))

    def compute_centroids_and_indices(lengthscale):
        kernel.lengthscales.assign(lengthscale)
        centroids, mean_distance = kmeans_lloyd_jit(points, k_centroids)
        indices, _ = kmeans_indices_and_distances(centroids, points, distance_fn)
        return centroids, indices

    def cluster_plotting(lengthscale, ax):
        centroids, indices = compute_centroids_and_indices(lengthscale)
        for centroid_id in range(k_centroids):
            centroid_mask = indices.numpy() == centroid_id
            x, y = points_numpy[centroid_mask, :].T
            x_centroid, y_centroid = centroids[centroid_id]
            scatter = ax.scatter(x, y, s=8, alpha=0.5)
            color = scatter.get_facecolor()
            ax.scatter([x_centroid], [y_centroid], s=15, marker="x", color=color)
            ax.set_title(f"Lengthscale: {lengthscale:.2f}")
    
    figsize = (10, 4)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    cluster_plotting(10.0, ax1)
    cluster_plotting(1.0, ax2)
    cluster_plotting(0.1, ax3)

    plt.tight_layout()
    plt.show()

