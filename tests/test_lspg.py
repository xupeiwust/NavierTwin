"""Round 271 — LSPG."""

from __future__ import annotations

import numpy as np


class TestLSPG:
    def test_full_basis_recovers(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.lspg import lspg_solve

        rng = np.random.default_rng(0)
        A = rng.standard_normal((5, 5)) + 5 * np.eye(5)
        b = rng.standard_normal(5)
        x = lspg_solve(A, b, np.eye(5))
        assert np.allclose(x, np.linalg.solve(A, b), atol=1e-8)

    def test_truncated_basis_lower_residual_than_random(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.lspg import (
            lspg_residual_norm,
        )

        rng = np.random.default_rng(1)
        A = np.diag([100.0, 50.0, 1.0, 0.5, 0.1])
        b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        # Phi spanning the dominant directions
        Phi = np.eye(5)[:, :2]
        r_good = lspg_residual_norm(A, b, Phi)
        # random Phi
        Q, _ = np.linalg.qr(rng.standard_normal((5, 2)))
        r_rand = lspg_residual_norm(A, b, Q)
        # exact subspace must be ≤ random
        assert r_good <= r_rand + 1e-8
