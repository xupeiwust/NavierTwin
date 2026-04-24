"""FastAPI REST 서버 — NavierTwin 기능을 HTTP 엔드포인트로 노출.

엔드포인트:
    - GET  /health                        : 헬스 체크
    - POST /reduce/pod                    : POD 수행, 모드/에너지 반환
    - POST /analytic/couette              : Couette 해석해 샘플
    - POST /analytic/poiseuille_2d        : Poiseuille 2D 해석해 샘플
    - POST /optimize/bayesian             : BO 최소화 (간단 quadratic)

Usage:
    uvicorn naviertwin.api.server:app --host 0.0.0.0 --port 8000
"""

from typing import Any, List, Optional  # noqa: UP035 — pydantic v1 호환

_HAS_FASTAPI = True
try:
    import fastapi
    from fastapi import Body, FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False
    fastapi = None
    Body = None  # type: ignore[assignment]
    FastAPI = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc,assignment]


if _HAS_FASTAPI:

    class CouetteReq(BaseModel):
        U_top: float = 1.0
        H: float = 1.0
        n_points: int = 50

    class PoiseuilleReq(BaseModel):
        dpdx: float = -1.0
        mu: float = 1.0
        H: float = 1.0
        n_points: int = 50

    class PODReq(BaseModel):
        snapshots: List[List[float]]  # noqa: UP006
        n_modes: int = 5

    class BayesianOptReq(BaseModel):
        bounds: List[List[float]]  # noqa: UP006
        n_initial: int = 5
        max_iter: int = 10
        problem: str = "quadratic"

    class LBMReq(BaseModel):
        nx: int = 32
        ny: int = 32
        tau: float = 0.8
        u_top: float = 0.05
        n_steps: int = 200
        record_every: int = 200


def create_app() -> Any:
    """FastAPI app 팩토리."""
    if not _HAS_FASTAPI:
        raise RuntimeError(
            "fastapi 설치 필요: pip install fastapi uvicorn"
        )
    import numpy as np

    app = FastAPI(title="NavierTwin API", version="1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "naviertwin"}

    @app.post("/analytic/couette")
    def couette(req: CouetteReq = Body(...)) -> dict[str, list]:
        from naviertwin.core.validation.analytic_solutions import couette_flow

        y = np.linspace(0.0, req.H, req.n_points)
        sol = couette_flow(U_top=req.U_top, H=req.H, y=y)
        return {"coords": y.tolist(), "velocity": sol.velocity.tolist()}

    @app.post("/analytic/poiseuille_2d")
    def poiseuille(req: PoiseuilleReq = Body(...)) -> dict[str, list]:
        from naviertwin.core.validation.analytic_solutions import poiseuille_flow_2d

        y = np.linspace(0.0, req.H, req.n_points)
        sol = poiseuille_flow_2d(dpdx=req.dpdx, mu=req.mu, H=req.H, y=y)
        return {"coords": y.tolist(), "velocity": sol.velocity.tolist()}

    @app.post("/reduce/pod")
    def pod(req: PODReq = Body(...)) -> dict[str, Any]:
        from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD

        X = np.asarray(req.snapshots, dtype=np.float64)
        pod_obj = SnapshotPOD(n_modes=req.n_modes)
        pod_obj.fit(X)
        return {
            "n_modes": pod_obj.n_components,
            "singular_values": pod_obj.singular_values_.tolist(),
            "cumulative_energy": pod_obj.energy_ratio_.tolist(),
        }

    @app.post("/simulate/lbm_cavity")
    def lbm_cavity(req: LBMReq = Body(...)) -> dict[str, Any]:
        from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

        lbm = LBMD2Q9(nx=req.nx, ny=req.ny, tau=req.tau, u_top=req.u_top)
        snaps = lbm.run(n_steps=req.n_steps, record_every=req.record_every)
        last = snaps[-1]
        return {
            "n_snapshots": int(snaps.shape[0]),
            "shape": list(snaps.shape),
            "ux_max": float(last[..., 1].max()),
            "ux_min": float(last[..., 1].min()),
            "uy_max": float(last[..., 2].max()),
            "rho_mean": float(last[..., 0].mean()),
        }

    @app.post("/optimize/bayesian")
    def bayesian(req: BayesianOptReq = Body(...)) -> dict[str, Any]:
        from naviertwin.core.optimization.bayesian_opt import BayesianOptimizer

        bounds = np.array(req.bounds, dtype=np.float64)
        if req.problem == "quadratic":
            def obj(x: np.ndarray) -> float:
                return float(np.sum(x ** 2))
        else:
            def obj(x: np.ndarray) -> float:
                return float(np.sum(np.sin(x)))

        opt = BayesianOptimizer(
            bounds=bounds, n_initial=req.n_initial, max_iter=req.max_iter, seed=0
        )
        x_best, f_best = opt.minimize(obj)
        return {"x_best": x_best.tolist(), "f_best": f_best}

    return app


# 모듈 레벨 app
try:
    app: Optional[Any] = create_app() if _HAS_FASTAPI else None
except RuntimeError:
    app = None
