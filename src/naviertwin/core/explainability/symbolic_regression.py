"""심볼릭 회귀 — PySR 선호 + 자체 탐색적 polynomial fallback.

PySR 이 설치되어 있지 않으면 1..3 차 다항식을 격자 검색해 최적 표현식을
문자열로 반환한다 (교육/테스트용).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.explainability.symbolic_regression import SymbolicRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-1, 1, (100, 2))
    >>> y = 2 * X[:, 0] + 0.5 * X[:, 1] ** 2
    >>> sr = SymbolicRegressor(max_degree=2)
    >>> sr.fit(X, y)
    >>> expr = sr.expression_
    >>> isinstance(expr, str)
    True
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class SymbolicRegressor:
    """PySR 래퍼 + 다항식 폴백."""

    def __init__(
        self,
        max_degree: int = 3,
        threshold: float = 1e-3,
    ) -> None:
        self.max_degree = max_degree
        self.threshold = threshold
        self.coef_: dict[str, float] = {}
        self.expression_: str = ""
        self.is_fitted: bool = False
        self._use_pysr: bool = False
        self._pysr_model: object | None = None

    def _poly_basis(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[str]]:
        N, d = X.shape
        feats = [np.ones(N)]
        names = ["1"]
        for deg in range(1, self.max_degree + 1):
            for combo in combinations_with_replacement(range(d), deg):
                col = np.ones(N)
                nm: list[str] = []
                for idx in combo:
                    col = col * X[:, idx]
                    nm.append(f"x{idx}")
                feats.append(col)
                names.append("*".join(nm))
        return np.stack(feats, axis=1), names

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError("X 2D 필요")

        try:
            from pysr import PySRRegressor

            model = PySRRegressor(
                niterations=20,
                binary_operators=["+", "-", "*"],
                unary_operators=["cos", "sin", "exp"],
                progress=False,
                verbosity=0,
            )
            model.fit(X, y)
            self._pysr_model = model
            self._use_pysr = True
            self.expression_ = str(model.sympy())
            self.is_fitted = True
            logger.info("SymbolicRegressor(PySR): %s", self.expression_)
            return
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("PySR 실패 → polynomial fallback: %s", e)

        # Fallback: polynomial least-squares
        Phi, names = self._poly_basis(X)
        coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        # threshold 아래는 0 으로
        coef = np.where(np.abs(coef) < self.threshold, 0.0, coef)
        self.coef_ = dict(zip(names, coef.tolist()))
        terms = [
            f"{c:.4g}*{n}" if n != "1" else f"{c:.4g}"
            for n, c in self.coef_.items()
            if c != 0
        ]
        self.expression_ = " + ".join(terms) if terms else "0"
        self.is_fitted = True
        logger.info("SymbolicRegressor(poly): %s", self.expression_)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        if self._use_pysr and self._pysr_model is not None:
            return np.asarray(self._pysr_model.predict(X), dtype=np.float64)
        Phi, names = self._poly_basis(np.asarray(X, dtype=np.float64))
        coef_vec = np.array([self.coef_.get(n, 0.0) for n in names])
        return Phi @ coef_vec


__all__ = ["SymbolicRegressor"]
