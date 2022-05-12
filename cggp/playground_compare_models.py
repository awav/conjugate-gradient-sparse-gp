from email.policy import default
from kmeans import kmeans_indices_and_distances
from models import ClusterGP, CGGP, LpSVGP
from conjugate_gradient import ConjugateGradient
from data import snelson1d, load_data
import matplotlib.pyplot as plt
import gpflow
from gpflow.config import default_float
import tensorflow as tf
import numpy as np

from playground_util import create_model_and_update_fn

from optimize import (
    kmeans_update_inducing_parameters,
    train_using_lbfgs_and_update,
    train_using_adam_and_update,
    create_monitor,
)


if __name__ == "__main__":
    seed = 111
    np.random.seed(seed)
    tf.random.set_seed(seed)

    _, train_data, test_data = load_data("elevators")
    distance_type = "covariance"
    num_inducing_points = 500
    num_iterations = 1000

    slice_size = 5000
    x, y = train_data
    xt, yt = test_data
    x = tf.convert_to_tensor(x[:slice_size], dtype=default_float())
    y = tf.convert_to_tensor(y[:slice_size], dtype=default_float())
    xt = tf.convert_to_tensor(xt[:slice_size], dtype=default_float())
    yt = tf.convert_to_tensor(yt[:slice_size], dtype=default_float())

    train_data = (x, y)
    test_data = (xt, yt)

    def model_class(kernel, likelihood, iv, **kwargs):
        error_threshold = 1e-6
        conjugate_gradient = ConjugateGradient(error_threshold)
        return CGGP(kernel, likelihood, iv, conjugate_gradient, **kwargs)

    cggp, cggp_update_fn = create_model_and_update_fn(
        model_class,
        train_data,
        num_inducing_points,
    )

    clustergp, clustergp_update_fn = create_model_and_update_fn(
        ClusterGP,
        train_data,
        num_inducing_points,
    )

    lpsvgp, _ = create_model_and_update_fn(
        LpSVGP,
        train_data,
        num_inducing_points,
    )

    iv, means, lambda_diag = cggp_update_fn()

    lpsvgp.inducing_variable.Z.assign(iv)

    cggp.inducing_variable.Z.assign(iv)
    cggp.diag_variance.assign(lambda_diag)
    cggp.pseudo_u.assign(lambda_diag)

    clustergp.inducing_variable.Z.assign(iv)
    clustergp.diag_variance.assign(lambda_diag)
    clustergp.pseudo_u.assign(lambda_diag)

    num_iterations = 1000
    batch_size = 500
    monitor_batch_size = 2000
    learning_rate = 0.01
    use_jit = True
    # use_jit = False
    use_tb = True
    logdir = "./logs-compare-playground"

    logdir_cggp = f"{logdir}/cggp"
    monitor_cggp = create_monitor(
        cggp,
        train_data,
        test_data,
        monitor_batch_size,
        use_tensorboard=use_tb,
        logdir=logdir_cggp,
    )
    train_using_adam_and_update(
        train_data,
        cggp,
        num_iterations,
        batch_size,
        learning_rate,
        update_fn=cggp_update_fn,
        update_during_training=True,
        use_jit=use_jit,
        monitor=monitor_cggp,
    )

    logdir_clustergp = f"{logdir}/clustergp"
    monitor_clustergp = create_monitor(
        clustergp,
        train_data,
        test_data,
        monitor_batch_size,
        use_tensorboard=use_tb,
        logdir=logdir_clustergp,
    )
    train_using_adam_and_update(
        train_data,
        clustergp,
        num_iterations,
        batch_size,
        learning_rate,
        update_fn=clustergp_update_fn,
        update_during_training=True,
        use_jit=use_jit,
        monitor=monitor_clustergp,
    )

    logdir_lpsvgp = f"{logdir}/lpsvgp"
    monitor_lpsvgp = create_monitor(
        lpsvgp,
        train_data,
        test_data,
        monitor_batch_size,
        use_tensorboard=use_tb,
        logdir=logdir_lpsvgp,
    )
    train_using_adam_and_update(
        train_data,
        lpsvgp,
        num_iterations,
        batch_size,
        learning_rate,
        update_fn=None,
        update_during_training=False,
        use_jit=use_jit,
        monitor=monitor_lpsvgp,
    )

    print(f"End. Check tensorboard logdir {logdir}")
