from typing import Callable, Optional
import tensorflow as tf
import gpflow
from kmeans import kmeans_indices_and_distances
from models import ClusterGP


def update_inducing_parameters(
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
        update_inducing_parameters(model, data, distance_fn, clustering_fn)

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
    clustering_fn: Callable,
    iterations: int,
    batch_size: int,
    learning_rate: float,
    distance_fn: Optional[Callable] = None,
    update_at_step: Optional[int] = None,
    use_jit: bool = True,
):
    n = data[0].shape[0]
    data_iter = iter(
        tf.data.Dataset.from_tensor_slices(data)
        .shuffle(n)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .repeat()
    )

    loss_fn = model.training_loss_closure(data_iter, compile=False)
    variables = model.trainable_variables

    dtype = variables[0].dtype
    learning_rate = tf.convert_to_tensor(learning_rate, dtype=dtype)
    opt = tf.keras.optimizers.Adam(learning_rate)

    def update_variational_parameters(*args, **kwargs):
        update_inducing_parameters(model, data, distance_fn, clustering_fn)

    gpflow.utilities.set_trainable(model.inducing_variable, False)

    variables = model.trainable_variables

    def optimize_step():
        opt.minimize(loss_fn, variables)

    update_variational_parameters()

    iteration = 0
    for iteration in range(iterations):
        optimize_step()
        update_variational_parameters()

    return None