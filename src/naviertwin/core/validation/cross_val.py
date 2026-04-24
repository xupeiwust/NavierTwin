"""Cross-validation + grid search (no sklearn dependency).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.validation.cross_val import kfold_scores
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


def kfold_scores(
    X: NDArray[np.float64], y: NDArray[np.float64],
    fit_predict: Callable[[NDArray, NDArray, NDArray], NDArray],
    score_fn: Callable[[NDArray, NDArray], float],
    *, k: int = 5, seed: int | None = 0,
) -> list[float]:
    """fit_predict(X_tr, y_tr, X_val) → y_pred.  반환: k개 fold 점수."""
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    scores: list[float] = []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        y_pred = fit_predict(X[tr_idx], y[tr_idx], X[val_idx])
        scores.append(float(score_fn(y[val_idx], y_pred)))
    return scores


def grid_search(
    X: NDArray[np.float64], y: NDArray[np.float64],
    fit_predict_factory: Callable[[dict], Callable],
    param_grid: dict[str, list[Any]],
    score_fn: Callable[[NDArray, NDArray], float],
    *, k: int = 3, seed: int | None = 0, higher_better: bool = False,
) -> dict:
    keys = list(param_grid.keys())
    best: dict = {"score": np.inf if not higher_better else -np.inf, "params": None}
    history = []
    for vals in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        fp = fit_predict_factory(params)
        scores = kfold_scores(X, y, fp, score_fn, k=k, seed=seed)
        mean_s = float(np.mean(scores))
        history.append({"params": params, "mean_score": mean_s, "scores": scores})
        if (higher_better and mean_s > best["score"]) or \
           (not higher_better and mean_s < best["score"]):
            best = {"score": mean_s, "params": params}
    return {"best": best, "history": history}


__all__ = ["kfold_scores", "grid_search"]
