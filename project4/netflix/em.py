"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    n = X.shape[0]
    d = X.shape[1]
    K = mixture.var.shape[0]

    table = np.ndarray(shape=(n, K))
    soft_cnt = np.ndarray(shape=(n, K))

    for i in range(0, n):
        for k in range(0, K):
            #x = X[i, :].reshape((1, d))
            #x_minus_u = np.zeros((1, d))
            #for j in range(0, d):
            #    if abs(x[0, j]) > 0.00000001:
            #        x_minus_u[0, j] = x[0, j] - mixture.mu[k, j] 
            #x_minus_u = x_minus_u[abs(x_minus_u) > 0.00000001]

            #cu = x_minus_u.shape[0]
            #x_minus_u = x_minus_u.reshape((1, cu))

            x = X[i, :]
            mask = (abs(x) > 0.00001)
            #print("mask = {}".format(mask))
            x_minus_u = x[mask] - mixture.mu[k, mask]
            cu = mask.sum()

            cov = mixture.var[k] * np.identity(cu)
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)

            exp_arg = -(x_minus_u @ cov_inv @ np.transpose(x_minus_u)) / 2.0
            coeff1 = (2.0*np.pi) ** (cu / 2.0)
            coeff2 = cov_det ** 0.5
            table[i, k] = mixture.p[k] * (1.0 / (coeff1 * coeff2)) * np.exp(exp_arg)

    prob_table = np.ndarray(shape=(n, 1))

    for i in range(0, n):
        total_prob = 0
        for k in range(0, K):
            total_prob = total_prob + table[i, k]
        
        prob_table[i, 0] = np.log(total_prob)

        for k in range(0, K):
            soft_cnt[i, k] = table[i, k] / total_prob

    l = 0
    for i in range(0, n):
        l = l + prob_table[i, 0]

    #print("l = {}".format(l))

    return [soft_cnt, l]

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n = X.shape[0]
    d = X.shape[1]
    K = post.shape[1]
    mixture = GaussianMixture(np.zeros((K, d)), np.zeros(K), np.zeros(K))
    for k in range(0, K):
        gamma_k_sum = 0
        hat_mu_k = np.zeros((1, d))
        for i in range(0, n):
            hat_mu_k = hat_mu_k + post[i, k] * X[i, :]
            gamma_k_sum = gamma_k_sum + post[i, k]

        mixture.mu[k, :] = hat_mu_k / gamma_k_sum
        mixture.p[k] = gamma_k_sum / n

    for k in range(0, K):
        gamma_k_sum = 0
        mixture.var[k] = 0
        for i in range(0, n):
            x_minus_u = X[i, :] - mixture.mu[k, :]
            mixture.var[k] = mixture.var[k] + \
                post[i, k] * (x_minus_u @ np.transpose(x_minus_u))
            gamma_k_sum = gamma_k_sum + post[i, k]
        
        mixture.var[k] = mixture.var[k] / (d * gamma_k_sum)

    return mixture

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_l = 0 
    while True:
        [post, new_l] = estep(X, mixture)
        mixture = mstep(X, post, mixture)

        if abs(new_l - old_l) <= 1e-6 * abs(new_l):
            break

        old_l = new_l

    return [mixture, post, new_l]

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
