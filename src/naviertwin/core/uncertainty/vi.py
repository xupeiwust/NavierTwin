"""Mean-field VI — diagonal Gaussian q(z) = N(μ, diag σ²)."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def mean_field_vi(
    log_p: Callable[[NDArray[np.float64]], float],
    dim: int,
    *, n_iter: int = 1000, lr: float = 0.05, mc_samples: int = 16,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """diag Gaussian q 의 μ, log σ 반환 (FD gradient + Adam ascent on ELBO)."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(dim)
    log_sigma = np.zeros(dim)

    m_mu = np.zeros(dim)
    v_mu = np.zeros(dim)
    m_ls = np.zeros(dim)
    v_ls = np.zeros(dim)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def elbo(mu_, log_s_, n_mc=mc_samples):
        sigma = np.exp(log_s_)
        total = 0.0
        for _ in range(n_mc):
            eps_s = rng.standard_normal(dim)
            z = mu_ + sigma * eps_s
            total += log_p(z)
        total /= n_mc
        entropy = float(np.sum(log_s_) + 0.5 * dim * (1 + np.log(2 * np.pi)))
        return total + entropy

    for t in range(1, n_iter + 1):
        fd_eps = 1e-3
        e0 = elbo(mu, log_sigma)
        g_mu = np.zeros(dim)
        g_ls = np.zeros(dim)
        for i in range(dim):
            mu_p = mu.copy()
            mu_p[i] += fd_eps
            g_mu[i] = (elbo(mu_p, log_sigma) - e0) / fd_eps
            ls_p = log_sigma.copy()
            ls_p[i] += fd_eps
            g_ls[i] = (elbo(mu, ls_p) - e0) / fd_eps

        m_mu = b1 * m_mu + (1 - b1) * g_mu
        v_mu = b2 * v_mu + (1 - b2) * (g_mu * g_mu)
        mh = m_mu / (1 - b1 ** t)
        vh = v_mu / (1 - b2 ** t)
        mu = mu + lr * mh / (np.sqrt(vh) + eps)

        m_ls = b1 * m_ls + (1 - b1) * g_ls
        v_ls = b2 * v_ls + (1 - b2) * (g_ls * g_ls)
        mh2 = m_ls / (1 - b1 ** t)
        vh2 = v_ls / (1 - b2 ** t)
        log_sigma = log_sigma + lr * mh2 / (np.sqrt(vh2) + eps)

    return mu, log_sigma


__all__ = ["mean_field_vi"]
