from models import CGGP
from conjugate_gradient import ConjugateGradient
from data import load_data
import tensorflow as tf
import numpy as np
import gpflow
from utils import store_logs, jit
from pathlib import Path

from cli_utils import (
    create_model_and_covertree_update_fn,
    batch_posterior_computation,
    create_predict_fn,
)


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

    files_to_load = ["path_to.npy"]
    for f in files_to_load:
        cggp, cggp_update_fn = create_fn(cggp_class)
        sgpr, _ = create_fn(sgpr_class)

        params = np.load(f, allow_pickle=False).item()
        gpflow.utilities.multiple_assign(cggp, params)

        predict_fn = create_predict_fn(sgpr, use_jit=use_jit)
        mean_train, variance_train = batch_posterior_computation(
            predict_fn,
            train_data,
            monitor_batch_size,
        )
        mean_test, variance_test = batch_posterior_computation(
            predict_fn,
            test_data,
            monitor_batch_size,
        )

        store_logs(Path(Path(f).parent, "train_mean.npy"), np.array(mean_train))
        store_logs(Path(Path(f).parent, "test_mean.npy"), np.array(mean_test))
        store_logs(Path(Path(f).parent, "train_variance.npy"), np.array(variance_train))
        store_logs(Path(Path(f).parent, "test_variance.npy"), np.array(variance_test))

    print(f"End. Check tensorboard logdir {logdir}")
