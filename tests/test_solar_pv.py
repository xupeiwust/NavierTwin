"""Round 458 — solar PV."""

from __future__ import annotations


class TestPV:
    def test_iv_decreasing(self) -> None:
        from naviertwin.core.applied.solar_pv import iv_curve

        V, I = iv_curve(I_ph=8.0, I_0=1e-9, V_T=0.0257, n=1.0, V_max=0.7, n_pts=50)
        # I monotone non-increasing
        assert (I[:-1] >= I[1:] - 1e-9).all()

    def test_mppt(self) -> None:
        from naviertwin.core.applied.solar_pv import iv_curve, mppt

        V, I = iv_curve()
        Vm, Im, Pm = mppt(V, I)
        # power = V*I positive
        assert Pm > 0
        # MPP voltage somewhere in middle (not at 0 or Voc)
        assert 0.3 < Vm < 0.7
