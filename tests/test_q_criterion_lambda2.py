"""Round 586 — Q-criterion + lambda2 happy-path coverage (was 45%)."""

from __future__ import annotations

import numpy as np
import pytest


def _make_simple_mesh():
    pv = pytest.importorskip("pyvista")
    # tiny structured grid with vortex-like velocity (rotation about z)
    grid = pv.ImageData(dimensions=(8, 8, 8), spacing=(0.1, 0.1, 0.1))
    pts = grid.points
    # u = (-y, x, 0) → solid-body rotation
    U = np.zeros_like(pts)
    U[:, 0] = -pts[:, 1]
    U[:, 1] = pts[:, 0]
    grid.point_data["U"] = U
    return grid.cast_to_unstructured_grid()


class TestQLambda:
    def test_compute_q_pyvista_path(self) -> None:
        pv = pytest.importorskip("pyvista")
        from naviertwin.core.flow_analysis.vortex.q_criterion import (
            compute_q_criterion,
        )

        mesh = _make_simple_mesh()
        out = compute_q_criterion(mesh, velocity_name="U")
        assert "Q-criterion" in out.point_data
        # solid-body rotation → Omega dominates → Q > 0
        q = np.asarray(out.point_data["Q-criterion"])
        assert (q > 0).any()
        assert "vorticity" in out.point_data
        _ = pv

    def test_compute_lambda2(self) -> None:
        pytest.importorskip("pyvista")
        from naviertwin.core.flow_analysis.vortex.q_criterion import compute_lambda2

        mesh = _make_simple_mesh()
        out = compute_lambda2(mesh, velocity_name="U")
        assert "lambda2" in out.point_data
        l2 = np.asarray(out.point_data["lambda2"])
        # rotation core → λ₂ negative
        assert (l2 < 0).any()
