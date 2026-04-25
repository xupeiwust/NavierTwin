"""Class-imbalance reweighting — inverse frequency or balanced.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.class_weights import balanced_weights
    >>> balanced_weights(np.array([0, 0, 0, 1]))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def balanced_weights(y: NDArray) -> dict[int, float]:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_classes = len(classes)
    return {int(c): float(n / (n_classes * cnt)) for c, cnt in zip(classes, counts, strict=True)}


def per_sample_weights(y: NDArray) -> NDArray[np.float64]:
    w = balanced_weights(y)
    y = np.asarray(y)
    return np.array([w[int(yi)] for yi in y])


__all__ = ["balanced_weights", "per_sample_weights"]
