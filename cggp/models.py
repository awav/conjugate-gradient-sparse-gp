from typing import Tuple
from xml.sax.handler import property_declaration_handler
import numpy as np
import tensorflow as tf
import gpflow
from tensorflow_probability import distributions as tfd
from numpy import newaxis
from gpflow import Parameter
from gpflow.utilities import positive
from gpflow.config import default_float

from utils import add_diagonal
from rff import rff_sample
from conjugate_gradient import ConjugateGradient, conjugate_gradient


Tensor = tf.Tensor
Moments = Tuple[Tensor, Tensor]


def eval_logdet(matrix, cg):
    @tf.custom_gradient
    def _eval_logdet(matrix):
        dtype = matrix.dtype

        def grad_logdet(df: Tensor) -> Tensor:
            n = tf.shape(matrix)[-1]
            eye = tf.linalg.eye(n, dtype=dtype)
            KmmLambdaInv = cg(matrix, eye)
            KmmLambdaInv = tf.transpose(KmmLambdaInv)
            return df * KmmLambdaInv

        return tf.constant(0.0, dtype=dtype), grad_logdet

    return _eval_logdet(matrix)


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
        nu=None,
        diag_variance=None,
        num_data=None,
    ):
        assert num_latent_gps == 1, "One latent GP is allowed"
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        inducing_variable = gpflow.models.util.inducingpoint_wrapper(inducing_variable)
        self.inducing_variable = inducing_variable

        # init variational parameters
        m = inducing_variable.num_inducing
        self.num_latent_gps = num_latent_gps

        nu = np.zeros((m, num_latent_gps)) if nu is None else nu
        var = np.ones((m, num_latent_gps)) * 1e-4 if diag_variance is None else diag_variance

        self._nu = Parameter(nu, dtype=default_float())
        self._diag_variance = Parameter(var, dtype=default_float(), transform=positive())

    @property
    def nu(self):
        return self._nu

    @property
    def diag_variance(self):
        return self._diag_variance

    def prior_kl(self) -> Tensor:
        kernel = self.kernel
        iv = self.inducing_variable
        nu = self.nu
        var = self.diag_variance
        Kmm = gpflow.covariances.Kuu(iv, kernel, jitter=0.0)
        quad = tf.reduce_sum(nu * tf.matmul(Kmm, nu))
        K = add_diagonal(Kmm, var[:, 0])
        L = tf.linalg.cholesky(K)
        trace = tf.linalg.trace(tf.linalg.cholesky_solve(L, Kmm))
        logdet = tf.reduce_sum(2.0 * tf.math.log(tf.linalg.diag_part(L))) - tf.reduce_sum(
            tf.math.log(var)
        )
        return 0.5 * (quad - trace + logdet)

    def maximum_log_likelihood_objective(self, data: gpflow.base.RegressionData) -> Tensor:
        return self.elbo(data)

    def elbo(self, data: gpflow.base.RegressionData) -> Tensor:
        """
        This gives a variational bound on the model likelihood.
        """
        x, y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(x, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, y)
        scale = self.scale(tf.shape(x)[0], kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False) -> Moments:
        assert not full_output_cov

        iv = self.inducing_variable
        kernel = self.kernel
        Kmm = gpflow.covariances.Kuu(iv, kernel, jitter=0.0)  # M x M
        Kmn = gpflow.covariances.Kuf(iv, kernel, Xnew)  # M x N
        Knn = kernel.K(Xnew) if full_cov else kernel.K_diag(Xnew)

        nu = self.nu
        var = self.diag_variance
        K = add_diagonal(Kmm, var[:, 0])
        L = tf.linalg.cholesky(K)
        A = tf.linalg.triangular_solve(L, Kmn)

        if not full_cov:
            fvar = (Knn - tf.reduce_sum(tf.square(A), axis=0))[:, None]
        else:
            fvar = Knn - tf.matmul(A, A, transpose_a=True)
            fvar = fvar[None, ...]

        fmu = tf.matmul(Kmn, nu, transpose_a=True)

        predict_mu = fmu + self.mean_function(Xnew)
        predict_var = fvar
        return predict_mu, predict_var

    def scale(self, batch_size, dtype):
        if self.num_data is not None:
            num_data = tf.convert_to_tensor(self.num_data, dtype=dtype)
            return num_data / tf.cast(batch_size, dtype)

        scale = tf.cast(1.0, dtype)
        return scale

    def q_moments(self, full_cov: bool = False) -> Moments:
        ip = self.inducing_variable.Z
        return self.predict_f(ip, full_cov=full_cov)


class ClusterGP(LpSVGP):
    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        cluster_counts=None,
        num_data=None,
        pseudo_u=None,
    ):
        assert num_latent_gps == 1, "One latent GP is allowed"
        super().__init__(
            kernel,
            likelihood,
            inducing_variable=inducing_variable,
            num_latent_gps=num_latent_gps,
            mean_function=mean_function,
            num_data=num_data,
        )
        self.pseudo_u = self._nu  # Copy ν parameter into another name

        del self._nu
        del self._diag_variance

        counts = tf.ones_like(self.pseudo_u)
        self.cluster_counts = tf.Variable(counts, dtype=self.pseudo_u.dtype, trainable=False)
        if cluster_counts is not None:
            self.cluster_counts.assign(cluster_counts)

        if pseudo_u is not None:
            self.pseudo_u.assign(pseudo_u)

        gpflow.utilities.set_trainable(self.inducing_variable, False)
        gpflow.utilities.set_trainable(self.pseudo_u, False)

    @property
    def nu(self):
        raise NotImplementedError(f"This property is not supported in {self.__class__}")

    @property
    def diag_variance(self) -> Tensor:
        return self.likelihood.variance / self.cluster_counts

    def prior_kl(self) -> Tensor:
        kernel = self.kernel
        iv = self.inducing_variable
        pseudo_u = self.pseudo_u
        var = self.diag_variance

        Kmm = gpflow.covariances.Kuu(iv, kernel, jitter=0.0)
        K = add_diagonal(Kmm, var[:, 0])

        L = tf.linalg.cholesky(K)
        KzzLambdaInv_u = tf.linalg.cholesky_solve(L, pseudo_u)

        quad = tf.reduce_sum(tf.matmul(Kmm, KzzLambdaInv_u) * KzzLambdaInv_u)

        trace = tf.linalg.trace(tf.linalg.cholesky_solve(L, Kmm))
        logdet = tf.reduce_sum(2.0 * tf.math.log(tf.linalg.diag_part(L)))
        const = tf.reduce_sum(tf.math.log(var))

        return 0.5 * (quad - trace + logdet - const)

    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False) -> Moments:
        assert not full_output_cov

        iv = self.inducing_variable
        kernel = self.kernel
        Kmm = gpflow.covariances.Kuu(iv, kernel, jitter=0.0)  # M x M
        Kmn = gpflow.covariances.Kuf(iv, kernel, Xnew)  # M x N
        Knn = kernel.K(Xnew) if full_cov else kernel.K_diag(Xnew)

        pseudo_u = self.pseudo_u
        var = self.diag_variance
        K = add_diagonal(Kmm, var[:, 0])

        L = tf.linalg.cholesky(K)
        KuuInv_u = tf.linalg.cholesky_solve(L, pseudo_u)
        A = tf.linalg.triangular_solve(L, Kmn)

        if not full_cov:
            fvar = (Knn - tf.reduce_sum(tf.square(A), axis=0))[:, None]
        else:
            fvar = Knn - tf.matmul(A, A, transpose_a=True)
            fvar = fvar[None, ...]

        fmu = tf.matmul(Kmn, KuuInv_u, transpose_a=True)
        predict_mu = fmu + self.mean_function(Xnew)
        predict_var = fvar
        return predict_mu, predict_var


class CGGP(ClusterGP):
    def __init__(
        self, kernel, likelihood, inducing_variable, conjugate_gradient: ConjugateGradient, **kwargs
    ):
        super().__init__(kernel, likelihood, inducing_variable, **kwargs)
        self.conjugate_gradient = conjugate_gradient

    def prior_kl(self) -> Tensor:
        kernel = self.kernel
        iv = self.inducing_variable
        pseudo_u = self.pseudo_u
        var = self.diag_variance

        zero = tf.convert_to_tensor(0.0, dtype=pseudo_u.dtype)
        Kmm = gpflow.covariances.Kuu(iv, kernel, jitter=zero)
        KmmLambda = add_diagonal(Kmm, var[:, 0])

        KmmLambdaInv_u = self.conjugate_gradient(KmmLambda, pseudo_u)
        KmmLambdaInv_Kmm = self.conjugate_gradient(KmmLambda, Kmm)

        quad_Kmm_KmmLambdaInv_u = tf.matmul(Kmm, KmmLambdaInv_u) * KmmLambdaInv_u
        quad = tf.reduce_sum(quad_Kmm_KmmLambdaInv_u)

        logdet = eval_logdet(KmmLambda, self.conjugate_gradient)
        trace = tf.linalg.trace(KmmLambdaInv_Kmm)
        const = tf.reduce_sum(tf.math.log(var))
        return 0.5 * (quad - trace + logdet - const)

    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False) -> Moments:
        assert not full_output_cov

        iv = self.inducing_variable
        kernel = self.kernel
        pseudo_u = self.pseudo_u
        var = self.diag_variance

        zero = tf.convert_to_tensor(0.0, dtype=pseudo_u.dtype)
        Kmm = gpflow.covariances.Kuu(iv, kernel, jitter=zero)  # M x M
        Kmn = gpflow.covariances.Kuf(iv, kernel, Xnew)  # M x N
        Knn = kernel.K(Xnew) if full_cov else kernel.K_diag(Xnew)

        KmmLambda = add_diagonal(Kmm, var[:, 0])

        KmmLambdaInv_u = self.conjugate_gradient(KmmLambda, pseudo_u)
        KmmLambdaInv_Kmn = self.conjugate_gradient(KmmLambda, Kmn)
        Knm_KmmLambdaInv_Kmn = tf.matmul(Kmn, KmmLambdaInv_Kmn, transpose_a=True)

        if not full_cov:
            fvar = Knn - tf.linalg.diag_part(Knm_KmmLambdaInv_Kmn)
            fvar = fvar[:, None]
        else:
            fvar = Knn - Knm_KmmLambdaInv_Kmn
            fvar = fvar[None, ...]

        fmu = tf.matmul(Kmn, KmmLambdaInv_u, transpose_a=True)
        predict_mu = fmu + self.mean_function(Xnew)
        predict_var = fvar
        return predict_mu, predict_var


class PathwiseClusterGP(ClusterGP):
    def elbo(
        self,
        data: gpflow.base.RegressionData,
        *,
        num_bases: int = 1,
        num_samples: int = 1,
    ) -> Tensor:
        """
        This gives a variational bound on the model likelihood.
        """
        kl = self.prior_kl()
        likelihood = self.compute_likelihood_term(data, num_bases, num_samples)
        x, _ = data
        scale = self.scale(tf.shape(x)[0], kl.dtype)
        return likelihood * scale - kl

    def compute_likelihood_term(
        self,
        data,
        num_bases: int,
        num_samples: int,
    ) -> Tensor:
        x, y = data
        num_data = tf.cast(tf.shape(y)[0], y.dtype)
        samples = self.pathwise_samples(x, num_bases, num_samples)
        noise = self.likelihood.variance
        noise_inv = tf.math.reciprocal(noise)

        error_squared = tf.square(y[newaxis, ...] - samples)
        likelihood = noise_inv * tf.reduce_sum(error_squared) / num_samples
        constant_term = num_data * tf.math.log(2.0 * np.pi * noise)
        return -0.5 * (likelihood + constant_term)

    def pathwise_samples(self, sample_at: Tensor, num_bases: int, num_samples: int) -> Tensor:
        u = self.pseudo_u
        iv = self.inducing_variable
        kernel = self.kernel
        lambda_diag = self.diag_variance[:, 0]

        prior_at = tf.concat([sample_at, iv.Z], axis=0)
        n = tf.shape(sample_at)[0]
        prior_samples = rff_sample(prior_at, kernel, num_bases, num_samples)  # [S, N]
        prior_samples = prior_samples[..., newaxis]  # [S, N + M, 1]
        prior_fx = prior_samples[:, :n]  # [S, N, 1]
        prior_fz = prior_samples[:, n:]  # [S, M, 1]

        epsilon_mvn = tfd.MultivariateNormalDiag(scale_diag=lambda_diag)

        ## \epsilon per sample or the same epsilon for all samples?
        epsilon = epsilon_mvn.sample((num_samples,))[..., newaxis]
        # epsilon = epsilon_mvn.sample((1, ))[..., newaxis]

        kzz = gpflow.covariances.Kuu(iv, kernel, jitter=0.0)
        kzx = gpflow.covariances.Kuf(iv, kernel, sample_at)
        kzz_lambda = add_diagonal(kzz, lambda_diag)

        solve_against = u[newaxis, ...] - prior_fz - epsilon
        L = tf.linalg.cholesky(kzz_lambda)
        weights = tf.linalg.cholesky_solve(L, solve_against)

        correction_term = tf.matmul(kzx, weights, transpose_a=True)
        samples = prior_fx + correction_term  # [S, N, 1]
        return samples
