"""Round 584 — coverage uplift for DMDAnalyzer (was 29%)."""

from __future__ import annotations

import numpy as np
import pytest


class TestDMDAnalyzer:
    def test_invalid_method(self) -> None:
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        with pytest.raises(ValueError, match="지원되지 않는"):
            DMDAnalyzer(method="bogus")

    def test_supported_methods_init(self) -> None:
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        for m in ["dmd", "fbdmd", "hodmd", "spdmd"]:
            a = DMDAnalyzer(method=m, dt=0.05, n_modes=3)
            assert a.method == m
            assert a.dt == 0.05
            assert a.n_modes == 3

    def test_fit_or_skip(self) -> None:
        pytest.importorskip("pydmd")
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        # build a low-rank oscillating snapshot matrix
        t = np.linspace(0, 2, 100)
        x = np.linspace(0, 1, 30)
        X = np.outer(np.sin(2 * np.pi * x), np.cos(2 * np.pi * t)) + \
            0.5 * np.outer(np.cos(3 * np.pi * x), np.sin(3 * np.pi * t))
        a = DMDAnalyzer(method="dmd", dt=t[1] - t[0])
        a.fit(X)
        assert a._is_fitted
