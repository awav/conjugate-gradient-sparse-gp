from typing import Callable
from functools import partial
from kmeans import kmeans_lloyd, kmeans_indices_and_distances, create_kernel_distance_fn, DistanceType
from models import LpSVGP
from data import snelson1d
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
import numpy as np

from gpflow.config import default_float


def create_model(data, num_inducing_points: int, distance_type: DistanceType):
    x, y = data
    xt = tf.convert_to_tensor(x, dtype=default_float())
    yt = tf.convert_to_tensor(y, dtype=default_float())
    kernel = gpflow.kernels.SquaredExponential()
    likelihood = gpflow.likelihoods.Gaussian(variance=0.1)

    distance_fn = create_kernel_distance_fn(kernel, distance_type)
    kmeans_fn = tf.function(partial(kmeans_lloyd, distance_fn=distance_fn))

    def clustering_fn():
        iv, _ = kmeans_fn(xt, num_inducing_points)
        return iv

    iv = clustering_fn()

    model = LpSVGP(kernel, likelihood, iv)
    gpflow.utilities.set_trainable(model.inducing_variable, False)
    return (xt, yt), model, clustering_fn, distance_fn


def train_using_lbfgs(data, model, clustering_fn: Callable, max_num_iters: int):
    lbfgs = gpflow.optimizers.Scipy()
    options = dict(maxiter=max_num_iters)
    loss_fn = model.training_loss_closure(data, compile=False)
    variables = model.trainable_variables

    def step_callback(*args, **kwargs):
        new_iv = clustering_fn()
        model.inducing_variable.Z.assign(new_iv)

    result = lbfgs.minimize(
        loss_fn,
        variables,
        step_callback=step_callback,
        compile=True,
        options=options,
    )

    return result


if __name__ == "__main__":
    seed = 111
    np.random.seed(seed)
    tf.random.set_seed(seed)

    (x, y), _ = snelson1d()
    distance_type = "covariance"
    num_inducing_points = 10
    num_iterations = 1000

    xmin = x.min() - 1.0
    xmax = x.max() + 1.0
    num_test_points = 100
    x_test = np.linspace(xmin, xmax, num_test_points).reshape(-1, 1)

    # Model setup
    data, model, clustering_fn, distance_fn = create_model((x, y), num_inducing_points, distance_type)
    xt, _ = data
    result = train_using_lbfgs(data, model, clustering_fn, num_iterations)

    # Plotting
    fig, (top_ax, bottom_ax) = plt.subplots(2, 1)

    iv = model.inducing_variable.Z.numpy()
    indices, _ = kmeans_indices_and_distances(iv, xt, distance_fn=distance_fn)
    indices = indices.numpy()
    color_map = "tab20c"
    colors = plt.get_cmap(color_map)(np.arange(num_inducing_points, dtype=int))

    # Top plot
    for i in range(num_inducing_points):
        color = colors[i]
        centroid_mask = indices == i
        x_plot = x[centroid_mask, :]
        y_plot = y[centroid_mask, :]
        scatter = top_ax.scatter(x_plot, y_plot, s=8, alpha=0.5, color=color)
        top_ax.axvline(x=iv[i][0], color=color, linestyle="--")
    
    # Bottom plot
    mu_test, var_test = model.predict_y(x_test)
    std_test = np.sqrt(var_test.numpy())
    mu_test = mu_test.numpy().reshape(-1)
    std_test = std_test.reshape(-1)
    up = mu_test + std_test
    down = mu_test - std_test

    line = bottom_ax.plot(x_test, mu_test)[0]
    bottom_color = line.get_color()
    bottom_ax.fill_between(x_test.reshape(-1), up, down, color=bottom_color, alpha=0.3)
    bottom_ax.scatter(x, y, color=bottom_color, alpha=0.5)

    plt.tight_layout()
    plt.show()

