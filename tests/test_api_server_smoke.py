"""Customer-facing FastAPI endpoint smoke tests."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="FastAPI is required for REST API smoke tests")


def test_advertised_rest_endpoints_return_json() -> None:
    import numpy as np

    from naviertwin.api.server import CouetteReq, LBMReq, PODReq, create_app

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    assert route_map["/health"]() == {"status": "ok", "service": "naviertwin"}

    couette_payload = route_map["/analytic/couette"](
        CouetteReq(U_top=1.0, H=1.0, n_points=5)
    )
    assert len(couette_payload["coords"]) == 5
    assert couette_payload["velocity"][-1] == pytest.approx(1.0)

    snapshots = np.eye(4).tolist()
    reduce_payload = route_map["/reduce"](
        PODReq(snapshots=snapshots, n_modes=2, reducer_kind="pod")
    )
    assert reduce_payload["reducer_kind"] == "pod"
    assert reduce_payload["n_modes"] == 2
    assert len(reduce_payload["singular_values"]) >= 2

    lbm_payload = route_map["/simulate/lbm_cavity"](
        LBMReq(nx=6, ny=6, tau=0.8, u_top=0.05, n_steps=2, record_every=1)
    )
    assert lbm_payload["n_snapshots"] == 2
    assert lbm_payload["shape"] == [2, 6, 6, 3]
    assert abs(lbm_payload["ux_max"]) < 1.0
