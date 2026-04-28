"""차원 축소 모듈.

선형(POD, SVD 등)과 비선형(오토인코더, VAE 등) 차원 축소 기법을 제공한다.

공개 API:
    - :class:`BaseReducer`: 차원 축소기 추상 기반 클래스
    - :class:`SnapshotPOD`: SVD 기반 스냅샷 POD
    - :class:`RandomizedPOD`: Randomized SVD 기반 고속 POD
    - :class:`IncrementalPOD`: 스트리밍(온라인) POD
    - :class:`MRPOD`: 다중 해상도 POD
"""

from naviertwin.core.dimensionality_reduction.base import BaseReducer
from naviertwin.core.dimensionality_reduction.linear import (
    MRPOD,
    IncrementalPOD,
    RandomizedPOD,
    SnapshotPOD,
)

__all__ = [
    "BaseReducer",
    "SnapshotPOD",
    "RandomizedPOD",
    "IncrementalPOD",
    "MRPOD",
]


def __getattr__(name: str) -> object:
    """Lazy import 비선형 reducer — PyTorch 없는 환경에서도 패키지 import 성공."""
    if name == "Autoencoder":
        from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import Autoencoder

        return Autoencoder
    if name == "VAE":
        from naviertwin.core.dimensionality_reduction.nonlinear.vae import VAE

        return VAE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
