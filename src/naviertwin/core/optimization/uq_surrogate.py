"""UQ surrogate — Polynomial Chaos Expansion (PCE).

    y(ξ) ≈ Σ_α c_α · Ψ_α(ξ)

ξ 는 균일 ([-1,1]^d) 분포 가정 → Legendre 다항식 기저.
tensor-product 기저 + 최소제곱 계수 피팅.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.uq_surrogate import PolynomialChaos
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-1, 1, (100, 2))
    >>> y = X[:, 0] ** 2 + 0.5 * X[:, 1]
    >>> pce = PolynomialChaos(degree=3)
    >>> pce.fit(X, y)
    >>> float(pce.mean_), float(pce.variance_)  # 통계 추정
    (...)
"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _legendre(x: NDArray[np.float64], n: int) -> list[NDArray[np.float64]]:
    """0..n 차 Legendre 다항식 값 리스트."""
    P = [np.ones_like(x), x.copy()]
    for k in range(1, n):
        P.append(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1))
    return P[: n + 1]


class PolynomialChaos:
    """PCE — Legendre tensor-product 기저 + least-squares."""

    def __init__(self, degree: int = 3) -> None:
        self.degree = degree
        self.multi_indices_: list[tuple[int, ...]] = []
        self.coef_: NDArray[np.float64] | None = None
        self.mean_: float = 0.0
        self.variance_: float = 0.0
        self.is_fitted: bool = False
        self.d_: int = 0

    def _make_indices(self, d: int) -> list[tuple[int, ...]]:
        """total-degree ≤ self.degree 인 multi-indices."""
        idx = []
        for combo in product(range(self.degree + 1), repeat=d):
            if sum(combo) <= self.degree:
                idx.append(combo)
        return idx

    def _design_matrix(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        d = X.shape[1]
        bases = [_legendre(X[:, j], self.degree) for j in range(d)]
        cols = []
        for alpha in self.multi_indices_:
            col = np.ones(X.shape[0])
            for j, a in enumerate(alpha):
                col *= bases[j][a]
            cols.append(col)
        return np.column_stack(cols)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError("X (N, d) 2D 필요")
        self.d_ = X.shape[1]
        self.multi_indices_ = self._make_indices(self.d_)
        Psi = self._design_matrix(X)
        coef, *_ = np.linalg.lstsq(Psi, y, rcond=None)
        self.coef_ = coef

        # 평균 = constant term (α=0)
        zero_idx = self.multi_indices_.index(tuple([0] * self.d_))
        self.mean_ = float(coef[zero_idx])
        # 분산 = Σ_{α≠0} c_α² · <Ψ_α²>_ξ
        # Legendre 정규화: <P_n²> = 1/(2n+1) on [-1,1] (weight 1/2)
        var = 0.0
        for i, alpha in enumerate(self.multi_indices_):
            if sum(alpha) == 0:
                continue
            norm = 1.0
            for a in alpha:
                norm *= 1.0 / (2 * a + 1)
            var += coef[i] ** 2 * norm
        self.variance_ = float(var)

        self.is_fitted = True
        logger.info(
            "PCE 학습 완료: d=%d, terms=%d, mean=%.4g, var=%.4g",
            self.d_, len(self.multi_indices_), self.mean_, self.variance_,
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError("fit() 먼저 호출")
        return self._design_matrix(X) @ self.coef_

    def sobol_indices(self) -> dict[str, NDArray[np.float64]]:
        """PCE 계수로부터 First-order / Total Sobol 지수 산출."""
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError("fit() 먼저 호출")
        d = self.d_
        total_var = max(self.variance_, 1e-30)

        # 각 축 j 에 대해 alpha_j > 0 이고 다른 축은 0 인 항이 first-order
        S1 = np.zeros(d)
        ST = np.zeros(d)
        for i, alpha in enumerate(self.multi_indices_):
            if sum(alpha) == 0:
                continue
            norm = 1.0
            for a in alpha:
                norm *= 1.0 / (2 * a + 1)
            contrib = self.coef_[i] ** 2 * norm / total_var
            nonzero_axes = [j for j, a in enumerate(alpha) if a > 0]
            if len(nonzero_axes) == 1:
                S1[nonzero_axes[0]] += contrib
            for j in nonzero_axes:
                ST[j] += contrib
        return {"S1": S1, "ST": ST}


__all__ = ["PolynomialChaos"]
