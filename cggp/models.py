import numpy as np
import tensorflow as tf
import gpflow
from gpflow import Parameter
from gpflow.utilities import positive
from gpflow.config import default_float


class LpSVGP(gpflow.models.GPModel, gpflow.models.ExternalDataTrainingLossMixin):
    """
    SVGP from

    ::
        @article{panos2018fully,
            title={Fully scalable gaussian processes using subspace inducing inputs},
            author={Panos, Aristeidis and Dellaportas, Petros and Titsias, Michalis K},
            journal={arXiv preprint arXiv:1807.02537},
            year={2018}
        }
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        q_mu=None,
        q_var=None,
        num_data=None,
    ):
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        inducing_variable = gpflow.models.util.inducingpoint_wrapper(inducing_variable)
        self.inducing_variable = inducing_variable

        # init variational parameters
        m = inducing_variable.num_inducing
        self.num_latent_gps = num_latent_gps

        mu = np.zeros((m, num_latent_gps)) if q_mu is None else q_mu
        var = np.ones((m, num_latent_gps)) * 1e-4 if q_var is None else q_var

        self.q_mu = Parameter(mu, dtype=default_float())
        self.q_var = Parameter(var, dtype=default_float(), transform=positive())

    def prior_kl(self) -> tf.Tensor:
        kernel = self.kernel
        iv = self.inducing_variable
        mu = self.q_mu
        var = self.q_var
        Kmm = gpflow.covariances.Kuu(iv, kernel, jitter=0.0)
        quad = tf.reduce_sum(mu * tf.matmul(Kmm, mu))
        K = Kmm + tf.linalg.diag(tf.transpose(var))[0, :, :]
        L = tf.linalg.cholesky(K)
        trace = tf.linalg.trace(tf.linalg.cholesky_solve(L, Kmm))
        logdet = tf.reduce_sum(2.0 * tf.math.log(tf.linalg.diag_part(L))) - tf.reduce_sum(
            tf.math.log(var)
        )
        return 0.5 * (quad - trace + logdet)

    def maximum_log_likelihood_objective(self, data: gpflow.base.RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: gpflow.base.RegressionData) -> tf.Tensor:
        """
        This gives a variational bound on the model likelihood.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        scale = self.scale(X.shape[0], kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        assert not full_output_cov

        iv = self.inducing_variable
        kernel = self.kernel
        Kmm = gpflow.covariances.Kuu(iv, kernel, jitter=0.0)  # M x M
        Kmn = gpflow.covariances.Kuf(iv, kernel, Xnew)  # M x N
        Knn = kernel.K(Xnew) if full_cov else kernel.K_diag(Xnew)

        mu = self.q_mu
        var = self.q_var
        K = Kmm + tf.linalg.diag(tf.transpose(var))[0, :, :]

        L = tf.linalg.cholesky(K)
        A = tf.linalg.triangular_solve(L, Kmn)

        if not full_cov:
            fvar = (Knn - tf.reduce_sum(tf.square(A), axis=0))[:, None]
        else:
            fvar = Knn - tf.matmul(A, A, transpose_a=True)
        
        fmu = tf.matmul(Kmn, mu, transpose_a=True)

        predict_mu = fmu + self.mean_function(Xnew)
        predict_var = fvar
        return predict_mu, predict_var

    def scale(self, batchsize, dtype):
        if self.num_data is not None:
            num_data = tf.convert_to_tensor(self.num_data, dtype=dtype)
            return num_data / tf.cast(batchsize, dtype)

        scale = tf.cast(1.0, dtype)
        return scale
