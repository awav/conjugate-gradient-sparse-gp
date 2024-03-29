from models import ClusterGP, CGGP, LpSVGP
from conjugate_gradient import ConjugateGradient
from data import load_data
import tensorflow as tf
import numpy as np
import gpflow
from gpflow.utilities import parameter_dict
from utils import store_as_npy, to_numpy, jit
from pathlib import Path

from cli_utils import create_model_and_covertree_update_fn
from optimize import (
    train_using_adam_and_update,
    create_monitor,
    transform_to_dataset,
)


def create_predict_fn(model, use_jit: bool = True):
    @jit(use_jit)
    def predict_fn(inputs):
        mu, _ = model.predict_f(inputs)
        return mu

    return predict_fn


def batch_posterior_computation(predict_fn, data, batch_size):
    data = transform_to_dataset(data, repeat=False, batch_size=batch_size)
    means = []
    for (x, y) in data:
        mean = predict_fn(x)
        means.append(mean.numpy())
    means = np.concatenate(means, axis=0)
    return means


if __name__ == "__main__":
    seed = 333
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpflow.config.set_default_float(tf.float32)
    gpflow.config.set_default_jitter(1e-5)

    distance_type = "euclidean"
    num_inducing_points = 2000
    num_iterations = 1000
    batch_size = 2000
    monitor_batch_size = 4000
    learning_rate = 0.01
    use_jit = True
    # use_jit = False
    use_tb = True
    logdir = "./logs-africa"
    update_during_training = None
    spatial_resolution = 0.07  # Use in practice
    # spatial_resolution = 0.5
    as_tensor = True

    _, train_data, test_data = load_data("east_africa", as_tensor=as_tensor)

    def cggp_class(kernel, likelihood, iv, **kwargs):
        error_threshold = 1e-6
        conjugate_gradient = ConjugateGradient(error_threshold)
        return CGGP(kernel, likelihood, iv, conjugate_gradient, **kwargs)

    def sgpr_class(kernel, likelihood, iv, **kwargs):
        noise_variance = likelihood.variance
        return gpflow.models.SGPR(train_data, kernel, iv, noise_variance=noise_variance)

    def create_fn(cls_fn, trainable_inducing_points: bool = False):
        return create_model_and_covertree_update_fn(
            cls_fn,
            train_data,
            spatial_resolution,
            use_jit=use_jit,
            distance_type=distance_type,
            trainable_inducing_points=trainable_inducing_points,
        )

    cggp, cggp_update_fn = create_fn(cggp_class)
    clustergp, clustergp_update_fn = create_fn(ClusterGP)
    lpsvgp, _ = create_fn(LpSVGP)
    sgpr, _ = create_fn(sgpr_class)

    iv, means, cluster_counts = cggp_update_fn()
    m = int(iv.shape[0])

    sgpr.inducing_variable.Z.assign(iv)
    lpsvgp.inducing_variable.Z.assign(iv)

    cggp.inducing_variable.Z.assign(iv)
    cggp.cluster_counts.assign(cluster_counts)
    cggp.pseudo_u.assign(means)

    clustergp.inducing_variable.Z.assign(iv)
    clustergp.cluster_counts.assign(cluster_counts)
    clustergp.pseudo_u.assign(means)

    # SGPR
    #
    logdir_sgpr = f"{logdir}/sgpr-{m}"
    monitor_sgpr = create_monitor(
        sgpr,
        train_data,
        test_data,
        monitor_batch_size,
        use_tensorboard=use_tb,
        logdir=logdir_sgpr,
    )
    train_using_adam_and_update(
        train_data,
        sgpr,
        num_iterations,
        batch_size,
        learning_rate,
        update_fn=None,
        update_during_training=None,
        use_jit=use_jit,
        monitor=monitor_sgpr,
    )
    sgpr_params = parameter_dict(sgpr)
    sgpr_params_np = to_numpy(sgpr_params)
    store_as_npy(Path(logdir_sgpr, "params.npy"), sgpr_params_np)

    sgpr_predict_fn = create_predict_fn(sgpr, use_jit=use_jit)
    sgpr_mean_train = batch_posterior_computation(sgpr_predict_fn, train_data, monitor_batch_size)
    sgpr_mean_test = batch_posterior_computation(sgpr_predict_fn, test_data, monitor_batch_size)

    store_as_npy(Path(logdir_sgpr, "train_mean.npy"), np.array(sgpr_mean_train))
    store_as_npy(Path(logdir_sgpr, "test_mean.npy"), np.array(sgpr_mean_test))
    # CGGP
    #
    logdir_cggp = f"{logdir}/cggp-{m}"
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
    cggp_params = parameter_dict(cggp)
    cggp_params_np = to_numpy(cggp_params)
    store_as_npy(Path(logdir_cggp, "params.npy"), cggp_params_np)

    cggp_predict_fn = create_predict_fn(cggp, use_jit=use_jit)
    cggp_mean_train, cggp_variance_train = batch_posterior_computation(
        cggp_predict_fn,
        train_data,
        monitor_batch_size,
    )
    cggp_mean_test, cggp_variance_test = batch_posterior_computation(
        cggp_predict_fn,
        test_data,
        monitor_batch_size,
    )

    store_as_npy(Path(logdir_cggp, "train_mean.npy"), np.array(cggp_mean_train))
    store_as_npy(Path(logdir_cggp, "train_variance.npy"), np.array(cggp_variance_train))
    store_as_npy(Path(logdir_cggp, "test_mean.npy"), np.array(cggp_mean_test))
    store_as_npy(Path(logdir_cggp, "test_variance.npy"), np.array(cggp_variance_test))

    # # ClusterGP
    # #
    # logdir_clustergp = f"{logdir}/clustergp"
    # monitor_clustergp = create_monitor(
    #     clustergp,
    #     train_data,
    #     test_data,
    #     monitor_batch_size,
    #     use_tensorboard=use_tb,
    #     logdir=logdir_clustergp,
    # )
    # train_using_adam_and_update(
    #     train_data,
    #     clustergp,
    #     num_iterations,
    #     batch_size,
    #     learning_rate,
    #     update_fn=clustergp_update_fn,
    #     update_during_training=update_during_training,
    #     use_jit=use_jit,
    #     monitor=monitor_clustergp,
    # )

    # # LpSVGP
    # #
    # logdir_lpsvgp = f"{logdir}/lpsvgp"
    # monitor_lpsvgp = create_monitor(
    #     lpsvgp,
    #     train_data,
    #     test_data,
    #     monitor_batch_size,
    #     use_tensorboard=use_tb,
    #     logdir=logdir_lpsvgp,
    # )
    # train_using_adam_and_update(
    #     train_data,
    #     lpsvgp,
    #     num_iterations,
    #     batch_size,
    #     learning_rate,
    #     update_fn=None,
    #     update_during_training=update_during_training,
    #     use_jit=use_jit,
    #     monitor=monitor_lpsvgp,
    # )

    print(f"End. Check tensorboard logdir {logdir}")
