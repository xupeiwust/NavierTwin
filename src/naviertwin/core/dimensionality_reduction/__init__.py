"""차원 축소 모듈.

선형(POD, SVD 등)과 비선형(오토인코더, VAE 등) 차원 축소 기법을 제공한다.

공개 API:
    - :class:`BaseReducer`: 차원 축소기 추상 기반 클래스
"""

from naviertwin.core.dimensionality_reduction.base import BaseReducer

__all__ = ["BaseReducer"]


def __getattr__(name: str) -> object:
    """Lazy import 비선형 reducer — PyTorch 없는 환경에서도 패키지 import 성공."""
    if name == "Autoencoder":
        from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import Autoencoder

        return Autoencoder
    if name == "VAE":
        from naviertwin.core.dimensionality_reduction.nonlinear.vae import VAE

        return VAE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
