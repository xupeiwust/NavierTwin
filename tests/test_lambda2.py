"""Round 169 — Λ₂."""

from __future__ import annotations

import numpy as np


class TestLambda2:
    def test_shape(self) -> None:
        from naviertwin.core.analysis.lambda2 import lambda2_2d

        u = np.zeros((16, 16))
        v = np.zeros((16, 16))
        L = lambda2_2d(u, v)
        assert L.shape == (16, 16)

    def test_solid_rotation_negative(self) -> None:
        """u=-y, v=x → solid body rotation, Λ₂ < 0."""
        from naviertwin.core.analysis.lambda2 import lambda2_2d

        n = 32
        x = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        u = -Y
        v = X
        L = lambda2_2d(u, v, x[1] - x[0], x[1] - x[0])
        # 내부 평균이 음수
        inner = L[5:-5, 5:-5]
        assert inner.mean() < 0

    def test_shear_non_vortex(self) -> None:
        """단순 shear 는 vortex 아님 → Λ₂ 평균이 0 근처."""
        from naviertwin.core.analysis.lambda2 import lambda2_2d

        n = 32
        y = np.linspace(0, 1, n)
        Y, _ = np.meshgrid(y, y, indexing="ij")
        u = Y  # linear shear u=y, v=0
        v = np.zeros_like(Y)
        L = lambda2_2d(u, v)
        # rotation 케이스보다 덜 음수
        assert abs(L[5:-5, 5:-5].mean()) < 1.0
