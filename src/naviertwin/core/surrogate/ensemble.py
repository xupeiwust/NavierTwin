"""Ensemble / Mixture of Experts (MoE) surrogate.

단순 평균 Ensemble:
    ŷ = (1/M) Σ_m f_m(x)

MoE (soft gating):
    ŷ = Σ_m g_m(x) · f_m(x)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.surrogate.ensemble import EnsembleSurrogate
    >>> from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-1, 1, (30, 2))
    >>> y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    >>> ens = EnsembleSurrogate([RBFSurrogate() for _ in range(3)])
    >>> ens.fit(X, y)
    >>> ens.predict(X[:3]).shape
    (3, 1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleSurrogate:
    """균등 평균 앙상블."""

    def __init__(self, models: list[object]) -> None:
        if not models:
            raise ValueError("models 가 비었습니다")
        self.models = models
        self.is_fitted: bool = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        # 배경 부트스트랩
        rng = np.random.default_rng(0)
        for m in self.models:
            idx = rng.choice(len(X), size=len(X), replace=True)
            m.fit(X[idx], y[idx])
        self.is_fitted = True
        logger.info("EnsembleSurrogate 학습 완료: %d 모델", len(self.models))

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        preds = [np.asarray(m.predict(X)) for m in self.models]
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0)

    def predict_with_std(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        preds = [np.asarray(m.predict(X)) for m in self.models]
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)


class MixtureOfExperts:
    """각 expert 에 k-means gating 으로 입력 영역별 특화."""

    def __init__(
        self,
        experts: list[object],
        n_clusters: int | None = None,
        seed: int | None = 0,
    ) -> None:
        self.experts = experts
        self.n_clusters = n_clusters or len(experts)
        self.seed = seed
        self._centroids: NDArray[np.float64] | None = None
        self.is_fitted: bool = False

    def _assign(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        assert self._centroids is not None
        # (N, M, d) - (1, M, d) broadcast → (N, M) 거리
        dif = X[:, None, :] - self._centroids[None, :, :]
        d2 = np.sum(dif ** 2, axis=-1)
        return np.argmin(d2, axis=1)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        rng = np.random.default_rng(self.seed)
        # 간단 k-means++ 초기화
        idx = rng.integers(0, len(X))
        centroids = [X[idx]]
        for _ in range(self.n_clusters - 1):
            dists = np.min(
                np.sum((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2, axis=-1),
                axis=1,
            )
            prob = dists / max(dists.sum(), 1e-30)
            next_idx = rng.choice(len(X), p=prob)
            centroids.append(X[next_idx])
        self._centroids = np.array(centroids)
        # Lloyd 몇 번
        for _ in range(10):
            labels = self._assign(X)
            for k in range(self.n_clusters):
                sel = labels == k
                if sel.any():
                    self._centroids[k] = X[sel].mean(axis=0)

        # 각 cluster 에 expert 학습
        labels = self._assign(X)
        for k, ex in enumerate(self.experts[: self.n_clusters]):
            sel = labels == k
            if sel.sum() < 2:
                # 부족하면 전체로 학습
                ex.fit(X, y)
            else:
                ex.fit(X[sel], y[sel])
        self.is_fitted = True
        logger.info("MoE 학습 완료: %d experts", self.n_clusters)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        X = np.asarray(X, dtype=np.float64)
        labels = self._assign(X)
        # 각 cluster 별 예측을 모음
        preds: list[NDArray[np.float64]] = []
        for i, lab in enumerate(labels):
            p = self.experts[int(lab)].predict(X[i : i + 1])
            preds.append(np.asarray(p).ravel())
        return np.stack(preds)


__all__ = ["EnsembleSurrogate", "MixtureOfExperts"]
