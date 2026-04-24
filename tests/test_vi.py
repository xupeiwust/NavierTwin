"""Round 212 — VI."""

from __future__ import annotations

import numpy as np


class TestVI:
    def test_gaussian_posterior(self) -> None:
        from naviertwin.core.uncertainty.vi import mean_field_vi

        def logp(z):
            return -0.5 * float(((z[0] - 3.0) / 0.4) ** 2)

        mu, log_sigma = mean_field_vi(logp, dim=1, n_iter=300, lr=0.05, seed=0)
        assert abs(float(mu[0]) - 3.0) < 0.3
        sigma = float(np.exp(log_sigma[0]))
        assert 0.2 < sigma < 0.8
