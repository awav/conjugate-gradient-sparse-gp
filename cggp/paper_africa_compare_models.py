from models import ClusterGP, CGGP, LpSVGP
from conjugate_gradient import ConjugateGradient
from data import load_data
from gpflow.config import default_float
import tensorflow as tf
import numpy as np

from cli_utils import create_model_and_kmeans_update_fn
from optimize import train_using_adam_and_update, create_monitor


if __name__ == "__main__":
    seed = 111
    np.random.seed(seed)
    tf.random.set_seed(seed)

    _, train_data, test_data = load_data("africa")
    distance_type = "covariance"
    num_inducing_points = 500
    num_iterations = 1000

    batch_size = 500
    monitor_batch_size = 2000
    learning_rate = 0.01
    use_jit = True
    # use_jit = False
    use_tb = True
    logdir = "./logs-compare-playground"
    update_during_training = True

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

    create_fn = lambda fn: create_model_and_kmeans_update_fn(
        fn, train_data, num_inducing_points, use_jit=use_jit, distance_type=distance_type
    )

    cggp, cggp_update_fn = create_fn(model_class)
    clustergp, clustergp_update_fn = create_fn(ClusterGP)
    lpsvgp, _ = create_fn(LpSVGP)

    iv, means, cluster_counts = cggp_update_fn()

    lpsvgp.inducing_variable.Z.assign(iv)

    cggp.inducing_variable.Z.assign(iv)
    cggp.cluster_counts.assign(cluster_counts)
    cggp.pseudo_u.assign(means)

    clustergp.inducing_variable.Z.assign(iv)
    clustergp.cluster_counts.assign(cluster_counts)
    clustergp.pseudo_u.assign(means)

    logdir_cggp = f"{logdir}/cggp-repeat"
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
        update_during_training=update_during_training,
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
        update_during_training=update_during_training,
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
        update_during_training=update_during_training,
        use_jit=use_jit,
        monitor=monitor_lpsvgp,
    )

    print(f"End. Check tensorboard logdir {logdir}")
