from pathlib import Path
from typing import Callable, Optional, Union, Tuple
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import parameter_dict
import tensorflow as tf
from selection import kmeans_indices_and_distances

from covertree import ModifiedCoverTree, SiblingAwareCoverTree
from models import ClusterGP, LpSVGP
from utils import jit, transform_to_dataset
from monitor import Monitor


Tensor = tf.Tensor


def covertree_update_inducing_parameters(
    model,
    data,
    distance_fn,
    spatial_resolution: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    # covertree = ModifiedCoverTree(distance_fn, data, spatial_resolution=spatial_resolution)
    data = data[0].numpy(), data[1].numpy()
    covertree = SiblingAwareCoverTree(distance_fn, data, spatial_resolution=spatial_resolution)
    new_iv = covertree.centroids
    means, counts = covertree.cluster_mean_and_counts
    new_iv = tf.convert_to_tensor(new_iv)
    means = tf.convert_to_tensor(means)
    counts = tf.convert_to_tensor(counts)

    filter_empty_clusters = tf.reshape(counts != 0.0, -1)
    new_iv = tf.boolean_mask(new_iv, filter_empty_clusters)
    means = tf.boolean_mask(means, filter_empty_clusters)
    counts = tf.boolean_mask(counts, filter_empty_clusters)

    return new_iv, means, counts


def oips_update_inducing_parameters(
    model,
    data,
    oips_fn,
    distance_fn,
) -> Tuple[Tensor, Tensor, Tensor]:
    inputs, outputs = data
    iv = oips_fn(inputs)
    m = tf.shape(iv)[0]
    cross_distances = model.kernel(iv, inputs)
    max_distance_indices = tf.argmax(cross_distances, axis=0)

    def mean_and_count_fn(label: int) -> Tuple[Tensor, Tensor]:
        mask = max_distance_indices == label
        int_dtype = max_distance_indices.dtype
        neighbours = tf.boolean_mask(outputs, mask, axis=0)
        count = tf.cast(tf.shape(neighbours)[0], int_dtype)
        mean = tf.reduce_mean(neighbours)
        return mean, count

    labels = tf.range(m, dtype=tf.int64)
    means, counts = tf.map_fn(
        mean_and_count_fn,
        labels,
        fn_output_signature=(inputs.dtype, labels.dtype),
    )

    nonempty_clusters = counts != 0
    new_means = tf.boolean_mask(means, nonempty_clusters)
    new_counts = tf.boolean_mask(counts, nonempty_clusters)
    new_iv = tf.boolean_mask(iv, nonempty_clusters)

    return new_iv, new_means, new_counts


def kmeans_update_inducing_parameters(
    model, data, distance_fn: Optional[Callable], clustering_fn: Callable
) -> Tuple[Tensor, Tensor, Tensor]:
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

    return new_iv, u, counts


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
    model: Union[ClusterGP, gpflow.models.SGPR, gpflow.models.GPR],
    max_num_iters: int,
    update_fn: Optional[Callable] = None,
    update_during_training: Optional[int] = None,
    monitor: Optional[Monitor] = None,
    use_jit: bool = True,
):
    lbfgs = gpflow.optimizers.Scipy()
    options = dict(maxiter=max_num_iters)

    if isinstance(model, gpflow.models.InternalDataTrainingLossMixin):
        loss_fn = model.training_loss_closure(compile=False)
    else:
        loss_fn = model.training_loss_closure(data, compile=False)

    variables = model.trainable_variables

    def internal_update_fn(iteration: int, *args, **kwargs):
        if update_during_training and (update_fn is not None):
            update_fn()
        if monitor is not None:
            monitor(iteration)

    internal_update_fn(0)

    if max_num_iters > 0:
        result = lbfgs.minimize(
            loss_fn,
            variables,
            step_callback=internal_update_fn,
            compile=use_jit,
            options=options,
        )
        return result

    internal_update_fn(-1)

    if monitor is not None:
        monitor.close()

    return None


def train_using_adam_and_update(
    data,
    model: Union[gpflow.models.SGPR, LpSVGP],
    iterations: int,
    batch_size: int,
    learning_rate: float,
    update_fn: Optional[Callable] = None,
    update_during_training: Optional[int] = None,
    monitor: Optional[Monitor] = None,
    use_jit: bool = True,
):
    update_during_training = update_during_training and (update_fn is not None)

    def internal_update_fn():
        if update_fn is not None:
            update_fn()

    if isinstance(model, gpflow.models.InternalDataTrainingLossMixin):
        loss_fn = model.training_loss_closure(compile=False)
    else:
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
    internal_update_fn()

    print("Run monitor")
    monitor_wrapper(iteration)

    for iteration in range(iterations):
        optimize_step()

        if update_during_training is not None:
            internal_update_fn()

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


def make_metrics_callback(
    model,
    train_data,
    test_data,
    batch_size: int,
    use_jit: bool = True,
    print_on: bool = True,
):
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
    def train_metrics_batch_fn(data):
        return model.elbo(data)

    @jit(use_jit)
    def train_metrics_full_fn():
        if isinstance(model, gpflow.models.GPR):
            return -model.maximum_log_likelihood_objective()
        else:
            return model.elbo()

    import click
    import json

    def step_callback(step, *args, **kwargs):
        error = np.array([]).reshape(-1, 1)
        lpd = 0.0
        elbo = 0.0

        for batch in test_dataset:
            batch_error, batch_log_density = test_metrics_fn(batch)
            lpd += batch_log_density.numpy()
            error = np.concatenate([error, batch_error.numpy()], axis=0)

        if isinstance(model, gpflow.models.InternalDataTrainingLossMixin):
            elbo = train_metrics_full_fn().numpy()
        else:
            for batch in train_dataset:
                batch_elbo = train_metrics_batch_fn(batch)
                elbo += batch_elbo.numpy()

        rmse = np.sqrt(np.mean(error**2))
        nlpd = -lpd
        metrics = {"train/elbo": elbo, "test/rmse": rmse, "test/nlpd": nlpd}

        if print_on:
            metrics_fmt = {
                k: np.format_float_scientific(v, precision=4) for k, v in metrics.items()
            }
            metrics_str = json.dumps(metrics_fmt)
            click.echo(f"Step [{step}], metrics: {metrics_str}")

        tf.debugging.check_numerics(elbo, f"The training ELBO has got an undefined value {elbo}")
        return metrics

    return step_callback


def create_monitor(
    model,
    train_data,
    test_data,
    batch_size,
    logdir: Union[str, Path] = "./logs-default/",
    record_step: Optional[int] = 5,
    use_jit: bool = True,
    use_tensorboard: bool = True,
) -> Monitor:
    monitor = Monitor(logdir, use_tensorboard=use_tensorboard)
    param_callback = make_param_callback(model)
    metric_callback = make_metrics_callback(
        model,
        train_data,
        test_data,
        batch_size,
        use_jit=use_jit,
        print_on=True,
    )
    monitor.add_callback("params", param_callback)
    monitor.add_callback("metrics", metric_callback, record_step=record_step)
    return monitor
