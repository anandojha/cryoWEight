"""Reweight an ensemble to a target distribution in CV space (RMSD, or RMSD+Rg).

Self-contained. The EM weight update is the same as the cryo-EM method; only the
likelihood is built from CV distances instead of image distances.

    weights = cv_reweight(cv_struct, cv_target, sigma)
"""
import numpy as np
from scipy.special import logsumexp


def em_weights(log_Pij, num_iterations=2000):
    """EM over a likelihood matrix log_Pij (N target rows, M structure cols) -> weights (M,)."""
    N, M = log_Pij.shape
    logw = np.full(M, -np.log(M))
    for _ in range(num_iterations):
        lw = log_Pij + logw
        logpost = lw - logsumexp(lw, axis=1, keepdims=True)
        logw = logsumexp(logpost - np.log(N), axis=0)
    w = np.exp(logw - logsumexp(logw))
    return w / w.sum()


def cv_reweight(cv_struct, cv_target, sigma, dims=None, num_iterations=2000):
    """cv_struct (M,d): CV of the ensemble structures.
       cv_target (N,d): CV samples from the target distribution.
       sigma: CV-space bandwidth. dims: columns to use, e.g. [0] for RMSD alone.
       Returns weights (M,) summing to 1."""
    cv_struct = np.atleast_2d(np.asarray(cv_struct, float))
    cv_target = np.atleast_2d(np.asarray(cv_target, float))
    if dims is not None:
        cv_struct, cv_target = cv_struct[:, dims], cv_target[:, dims]
    sq = ((cv_target[:, None, :] - cv_struct[None, :, :]) ** 2).sum(-1)   # (N target, M struct)
    log_Pij = -sq / (2 * sigma ** 2)
    return em_weights(log_Pij, num_iterations)
