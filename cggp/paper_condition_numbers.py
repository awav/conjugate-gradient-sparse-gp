import numpy as np
import tensorflow as tf
import gpflow

from models import ClusterGP
from cli_utils import create_model_and_covertree_update_fn
from utils import add_diagonal


def gen_data(seed: int, noise_level: float = 0.001):
    rng = np.random.RandomState(seed)
    n = 1000
    l = 2 * np.pi
    x = 2 * l * rng.randn(n, 1) - l

    def fn(x):
        return np.sin(x) ** 2 + np.cos(x)

    y = fn(x) + rng.randn(n, 1) * noise_level
    return x, y


def resolution_vs_condition_number():
    seed = 999
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data = gen_data(seed)

    model_cls = ClusterGP
    resolutions = np.linspace(1.0, 10.0, 5)

    def covertree_setup(resolution):
        return create_model_and_covertree_update_fn(model_cls, data, resolution)

    models_and_update_fns = [covertree_setup(res) for res in resolutions]
    for model, update_fn in models_and_update_fns:
        model: ClusterGP = model
        update_fn()
        kernel = model.kernel
        iv = model.inducing_variable
        diag_lambda = model.diag_variance

        kuu = gpflow.covariances.Kuu(iv, kernel)
        kuu_lambda = add_diagonal(kuu, diag_lambda)

        eigvals = tf.linalg.eigvalsh(kuu_lambda).numpy()
        eig_min = eigvals.min()
        eig_max = eigvals.max()
        condition_number = eig_max / eig_min
