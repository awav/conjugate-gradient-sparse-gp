from typing import Callable, Optional, TypeVar
from functools import partial
from kmeans import (
    kmeans_lloyd,
    kmeans_indices_and_distances,
    create_kernel_distance_fn,
    DistanceType,
)
from models import LpSVGP, ClusterSVGP
import gpflow
import tensorflow as tf

from gpflow.config import default_float


ModelClass = TypeVar("ModelClass", type(LpSVGP), type(ClusterSVGP))


def create_model(
    data, num_inducing_points: int, distance_type: DistanceType, model_class: ModelClass
):
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

    model = model_class(kernel, likelihood, iv)

    gpflow.utilities.set_trainable(model.inducing_variable, False)
    return (xt, yt), model, clustering_fn, distance_fn


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


def train_using_lbfgs_and_varpar_update(
    data,
    model: ClusterSVGP,
    clustering_fn: Callable,
    max_num_iters: int,
    outer_num_iters: int,
    distance_fn: Optional[Callable] = None,
):
    lbfgs = gpflow.optimizers.Scipy()
    options = dict(maxiter=max_num_iters)
    loss_fn = model.training_loss_closure(data, compile=False)
    variables = model.trainable_variables
    x, y = data

    def update_variational_parameters(*args, **kwargs):
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

    gpflow.utilities.set_trainable(model.inducing_variable, False)
    use_jit = True  # TODO(awav): resolve the problem with recompiling in Scipy

    # for _ in range(outer_num_iters):
    update_variational_parameters()
    result = lbfgs.minimize(
        loss_fn,
        variables,
        step_callback=update_variational_parameters,
        compile=use_jit,
        options=options,
    )

    return result
