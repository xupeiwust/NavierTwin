"""검증 지표(Validation Metrics) 모듈.

공개 API:
    - :func:`rmse`: Root Mean Squared Error
    - :func:`r2_score`: 결정계수 R²
    - :func:`relative_l2_error`: 상대 L2 노름 오차
    - :func:`max_error`: 최대 절대 오차
    - :func:`compute_all_metrics`: 모든 지표 일괄 계산
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "rmse": "naviertwin.core.validation.metrics",
    "r2_score": "naviertwin.core.validation.metrics",
    "relative_l2_error": "naviertwin.core.validation.metrics",
    "max_error": "naviertwin.core.validation.metrics",
    "compute_all_metrics": "naviertwin.core.validation.metrics",
    "AnalyticSolution": "naviertwin.core.validation.analytic_solutions",
    "couette_flow": "naviertwin.core.validation.analytic_solutions",
    "poiseuille_flow_2d": "naviertwin.core.validation.analytic_solutions",
    "poiseuille_pipe": "naviertwin.core.validation.analytic_solutions",
    "spectral_poiseuille": "naviertwin.core.validation.analytic_solutions",
    "compare_against_analytic": "naviertwin.core.validation.analytic_solutions",
}


def __getattr__(name: str) -> Any:
    """Lazily expose validation helpers without importing numeric stacks eagerly."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "rmse",
    "r2_score",
    "relative_l2_error",
    "max_error",
    "compute_all_metrics",
    "AnalyticSolution",
    "couette_flow",
    "poiseuille_flow_2d",
    "poiseuille_pipe",
    "spectral_poiseuille",
    "compare_against_analytic",
]
