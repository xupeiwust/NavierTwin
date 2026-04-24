"""Round 410 — O category milestone: nonlinear ROM (R401-R409) + OpInf e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneO:
    def test_imports(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear import (  # noqa: F401
            adaptive_enrich,
            closure_rom,
            latent_linearize,
            lift_learn,
            manifold_tangent,
            opinf,
            reduced_barycentric,
        )
        from naviertwin.core.system_id import sindyc  # noqa: F401

    def test_opinf_predict_e2e(self) -> None:
        """fit linear ROM in latent → predict trajectory."""
        from naviertwin.core.dimensionality_reduction.nonlinear.latent_linearize import (
            predict_latent,
        )
        from naviertwin.core.dimensionality_reduction.nonlinear.opinf import opinf_fit

        # synthesize: latent ż = -0.5 z; samples = z_{k+1} - z_k = -0.5 z dt
        rng = np.random.default_rng(0)
        z = [rng.standard_normal(2)]
        dt = 0.1
        for _ in range(100):
            z.append(z[-1] + dt * (-0.5 * z[-1]))
        Z = np.asarray(z)
        Zdot = np.diff(Z, axis=0) / dt
        ops = opinf_fit(Z[:-1], Zdot)
        # A ≈ -0.5 I
        assert np.allclose(ops["A"], -0.5 * np.eye(2), atol=0.05)
        # predict
        # discrete A = I + dt * A_cont
        A_disc = np.eye(2) + dt * ops["A"]
        traj = predict_latent(A_disc, Z[0], n_steps=10)
        assert traj.shape == (11, 2)
