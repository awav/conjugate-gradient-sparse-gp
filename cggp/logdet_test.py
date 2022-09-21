import gpflow
from data import load_data
import tensorflow as tf
from gpflow.kernels import SquaredExponential
import matplotlib.pyplot as plt


def main():
    data = load_data("snelson1d")
    X, y = data.train[0], data.train[1]
    N, D = X.shape
    sn2 = 1e-3
    k = SquaredExponential()
    K = k(X) + sn2 * tf.eye(N, dtype=X.dtype)
    L = tf.linalg.cholesky(K)
    exact_log_det = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

    # naive implementation for demonstration purposes
    # actually, the argument for the lower bound can also be made via Gershgorin circles
    sub_log_dets = tf.cumsum(2 * tf.math.log(tf.linalg.diag_part(L)))
    exact_log_det = sub_log_dets[-1]
    bounds = []
    titsias_bounds = []
    for n in range(1, N):
        T = tf.linalg.triangular_solve(L[:n,:n], K[:n, n:], lower=True)
        covar = K[n:, n:] - tf.linalg.matmul(T, T, transpose_a=True)
        covar_diag = tf.linalg.diag_part(covar)
        element_wise_bound = covar_diag - tf.reduce_sum(tf.square(covar - covar_diag), axis=0) / sn2
        assert(element_wise_bound.shape[0] == N-n)
        bound = sub_log_dets[n-1] + tf.reduce_sum(tf.math.log(tf.math.maximum(element_wise_bound, sn2)))
        bounds.append(bound.numpy())

        titsias_bounds.append(tf.linalg.trace(covar).numpy() / sn2)
    
    plt.plot(tf.range(1, N).numpy(), bounds, color='blue');
    # plt.plot(tf.range(1, N).numpy(), titsias_bounds, color='red');
    plt.plot([1, N], [exact_log_det, exact_log_det], color='black')
    plt.tight_layout()
    plt.show()
    print()


if __name__ == "__main__":
    main()
