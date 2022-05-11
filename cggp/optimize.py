from pathlib import Path
from typing import Callable, Optional, Union
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import parameter_dict
from kmeans import kmeans_indices_and_distances
from covertree import ModifiedCoverTree
from models import ClusterGP
from utils import jit
from monitor import Monitor


def covertree_update_inducing_parameters(model, data, distance_fn):
    covertree = ModifiedCoverTree(distance_fn, data)
    new_iv = covertree.centroids
    means, counts = covertree.cluster_mean_and_counts
    sigma2 = model.likelihood.variance
    lambda_diag = sigma2 / counts

    model.inducing_variable.Z.assign(new_iv)
    model.pseudo_u.assign(means)
    model.diag_variance.assign(lambda_diag)


def kmeans_update_inducing_parameters(
    model, data, distance_fn: Optional[Callable], clustering_fn: Callable
) -> None:
    x, y = data
    new_iv = clustering_fn()

    m = new_iv.shape[0]
    indices, _ = kmeans_indices_and_distances(new_iv, x, distance_fn=distance_fn)
    range_indices = tf.range(m, dtype=indices.dtype)
    counting_map = tf.cast(range_indices[:, None] == indices[None, :], tf.int32)
    counts = tf.reduce_sum(counting_map, axis=1, keepdims=True)
    counts = tf.cast(counts, dtype=new_iv.dtype)

    u_init = tf.zeros([m, 1], dtype=new_iv.dtype)
    update_indices = tf.reshape(indices, [-1, 1])
    u = tf.tensor_scatter_nd_add(u_init, update_indices, y) / counts
    sigma2 = model.likelihood.variance
    lambda_diag = sigma2 / counts

    model.inducing_variable.Z.assign(new_iv)
    model.pseudo_u.assign(u)
    model.diag_variance.assign(lambda_diag)


def train_vanilla_using_lbfgs_and_standard_ip_update(
    data,
    model,
    clustering_fn: Callable,
    max_num_iters: int,
):
    lbfgs = gpflow.optimizers.Scipy()
    options = dict(maxiter=max_num_iters)
    loss_fn = model.training_loss_closure(data, compile=False)
    variables = model.trainable_variables

    def step_callback(*args, **kwargs):
        # TODO(awav): This callback is called after every gradient step in L-BFGS
        # Calling clustering every gradient step causes the convergence
        # to a poor local minima.
        new_iv = clustering_fn()
        model.inducing_variable.Z.assign(new_iv)

    use_jit = True
    result = lbfgs.minimize(
        loss_fn,
        variables,
        step_callback=step_callback,
        compile=use_jit,
        options=options,
    )

    return result


def train_vanilla_using_lbfgs(
    data,
    model,
    clustering_fn: Callable,
    max_num_iters: int,
):
    lbfgs = gpflow.optimizers.Scipy()
    options = dict(maxiter=max_num_iters)
    loss_fn = model.training_loss_closure(data, compile=False)
    variables = model.trainable_variables

    use_jit = True
    result = lbfgs.minimize(
        loss_fn,
        variables,
        compile=use_jit,
        options=options,
    )

    return result


def train_using_lbfgs_and_update(
    data,
    model: ClusterGP,
    clustering_fn: Callable,
    max_num_iters: int,
    distance_fn: Optional[Callable] = None,
    use_jit: bool = True,
):
    lbfgs = gpflow.optimizers.Scipy()
    options = dict(maxiter=max_num_iters)
    loss_fn = model.training_loss_closure(data, compile=False)
    variables = model.trainable_variables

    def update_variational_parameters(*args, **kwargs):
        kmeans_update_inducing_parameters(model, data, distance_fn, clustering_fn)

    gpflow.utilities.set_trainable(model.inducing_variable, False)

    # for _ in range(outer_num_iters):
    update_variational_parameters()
    if max_num_iters > 0:
        result = lbfgs.minimize(
            loss_fn,
            variables,
            step_callback=update_variational_parameters,
            compile=use_jit,
            options=options,
        )
        return result
    return None


def train_using_adam_and_update(
    data,
    model: ClusterGP,
    iterations: int,
    batch_size: int,
    learning_rate: float,
    update_fn: Callable,
    update_during_training: Optional[int] = None,
    monitor: Optional[Monitor] = None,
    use_jit: bool = True,
):
    n = data[0].shape[0]
    dataset = transform_to_dataset(data, batch_size, shuffle=n)
    data_iter = iter(dataset)

    loss_fn = model.training_loss_closure(data_iter, compile=False)
    variables = model.trainable_variables

    dtype = variables[0].dtype
    learning_rate = tf.convert_to_tensor(learning_rate, dtype=dtype)
    opt = tf.optimizers.Adam(learning_rate)

    variables = model.trainable_variables

    @jit(use_jit)
    def optimize_step():
        opt.minimize(loss_fn, variables)

    def monitor_wrapper(step):
        if monitor is None:
            return
        return monitor(step)

    iteration = 0
    update_fn()

    print("Run monitor")
    monitor_wrapper(iteration)

    for iteration in range(iterations):
        optimize_step()
        # TODO(@awav): uncomment this if clustering should run at each iteration!
        # update_fn()
        monitor_wrapper(iteration)

        if monitor is not None:
            monitor.flush()


def make_print_callback():
    import click

    def print_callback(step: int, *args, **kwargs):
        click.echo(f"Step: {step}")
        return {}

    return print_callback


def make_param_callback(model):
    """
    Callback for tracking parameters in TensorBoard
    """

    def _callback(*args, **kwargs):
        ks = {
            f"kernel/{k.strip('.')}": v.numpy() for (k, v) in parameter_dict(model.kernel).items()
        }
        ls = {
            f"likelihood/{k.strip('.')}": v.numpy()
            for (k, v) in parameter_dict(model.likelihood).items()
        }
        return {**ks, **ls}

    return _callback


def make_metrics_callback(model, train_data, test_data, batch_size: int, use_jit: bool = True):
    """
    Callback for computing test metrics (RMSE and NLPD)
    """
    test_dataset = transform_to_dataset(test_data, batch_size, repeat=False)
    train_dataset = transform_to_dataset(train_data, batch_size, repeat=False)

    @jit(use_jit)
    def test_metrics_fn(data):
        x, y = data
        mu, var = model.predict_f(x)
        lpd = model.likelihood.predict_log_density(mu, var, y)
        lpd = tf.reduce_sum(lpd)
        error = y - mu
        return error, lpd

    @jit(use_jit)
    def train_metrics_fn(data):
        return model.elbo(data)

    def step_callback(step, *args, **kwargs):
        error = np.array([]).reshape(-1, 1)
        lpd = 0.0
        elbo = 0.0

        for batch in test_dataset:
            batch_error, batch_log_density = test_metrics_fn(batch)
            lpd += batch_log_density.numpy()
            error = np.concatenate([error, batch_error.numpy()], axis=0)

        for batch in train_dataset:
            batch_elbo = train_metrics_fn(batch)
            elbo += batch_elbo.numpy()

        rmse = np.sqrt(np.mean(error**2))
        nlpd = -lpd
        return {"train/elbo": elbo, "test/rmse": rmse, "test/nlpd": nlpd}

    return step_callback


def transform_to_dataset(
    data, batch_size, repeat: bool = True, shuffle: Optional[int] = None
) -> tf.data.Dataset:
    data = tf.data.Dataset.from_tensor_slices(data)
    if shuffle is not None:
        data = data.shuffle(shuffle)

    data = data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    if repeat:
        data = data.repeat()
    return data


def create_monitor(
    model,
    train_data,
    test_data,
    batch_size,
    logdir: Union[str, Path] = "./logs-default/",
    use_jit: bool = True,
) -> Monitor:
    monitor = Monitor(logdir)
    print_callback = make_print_callback()
    param_callback = make_param_callback(model)
    metric_callback = make_metrics_callback(model, train_data, test_data, batch_size, use_jit=use_jit)
    monitor.add_callback("print", print_callback)
    monitor.add_callback("params", param_callback)
    monitor.add_callback("metrics", metric_callback, record_step=5)
    return monitor
