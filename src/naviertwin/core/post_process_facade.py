"""Post-processor 통합 Facade — 신규 모듈에 대한 단일 API 표면.

R591-647에서 추가한 ~30개 후처리/AI/ROM 모듈의 핵심 기능을 단일 클래스로
노출. GUI나 CLI에서 파라미터 dict 하나로 모든 분석을 호출 가능.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> u = np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.1 * rng.standard_normal(200)
    >>> from naviertwin.core.post_process_facade import PostProcessFacade
    >>> facade = PostProcessFacade()
    >>> result = facade.run("psd_welch", signal=u, fs=100.0)
    >>> "frequency" in result and "psd" in result
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class PostProcessFacade:
    """모든 신규 후처리 도구의 단일 호출 표면.

    각 메서드는 dict로 결과를 반환 (GUI 표시용 + JSON 직렬화 친화).

    사용법::

        facade = PostProcessFacade()
        ops = facade.list_operations()  # 사용 가능한 op 이름 list
        info = facade.describe(op_name)  # 파라미터 설명
        result = facade.run(op_name, **kwargs)
    """

    def list_operations(self) -> list[str]:
        """지원하는 모든 op 이름 반환."""
        return sorted(_OPERATIONS.keys())

    def describe(self, op_name: str) -> dict[str, Any]:
        """op의 인자 / 반환 명세를 dict로 반환."""
        if op_name not in _OPERATIONS:
            raise KeyError(f"unknown op '{op_name}'")
        spec = _OPERATIONS[op_name]
        return {
            "name": op_name,
            "category": spec["category"],
            "description": spec["description"],
            "params": spec["params"],
            "returns": spec["returns"],
        }

    def run(self, op_name: str, **kwargs: Any) -> dict[str, Any]:
        """op 실행. **kwargs는 op_spec의 params와 일치해야 한다."""
        if op_name not in _OPERATIONS:
            raise KeyError(f"unknown op '{op_name}'")
        try:
            return _OPERATIONS[op_name]["fn"](**kwargs)
        except TypeError as e:
            raise ValueError(
                f"invalid parameters for '{op_name}': {e}"
            ) from e


# ---------------------------------------------------------------------------
# Op 정의 — 신규 모듈을 dict로 등록
# ---------------------------------------------------------------------------


def _op_psd_welch(
    signal: NDArray[np.float64],
    fs: float = 1.0,
    nperseg: int | None = None,
    window: str = "hann",
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.psd import welch_psd

    f, P = welch_psd(signal, fs=fs, nperseg=nperseg, window=window)
    return {"frequency": f, "psd": P}


def _op_reynolds_stats(
    u: NDArray[np.float64],
    v: NDArray[np.float64] | None = None,
    w: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.reynolds_stats import (
        mean_field,
        rms,
        turbulence_intensity,
        turbulent_kinetic_energy,
    )

    out: dict[str, Any] = {
        "mean": mean_field(u, axis=0),
        "rms": rms(u, axis=0),
    }
    if v is not None:
        out["tke"] = turbulent_kinetic_energy(u, v, w, axis=0)
        out["intensity"] = turbulence_intensity(u, v, w, axis=0)
    return out


def _op_quadrant_analysis(
    up: NDArray[np.float64],
    vp: NDArray[np.float64],
    hole: float = 0.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.quadrant_pdf import quadrant_split

    quads = quadrant_split(up, vp, hole=hole)
    return {"quadrants": quads}


def _op_kolmogorov_slope(
    signal: NDArray[np.float64],
    dx: float = 1.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.spectral_energy import (
        energy_spectrum_1d,
        kolmogorov_slope,
    )

    k, E = energy_spectrum_1d(signal, dx=dx)
    slope, r2 = kolmogorov_slope(k, E)
    return {"k": k, "E": E, "slope": slope, "r2": r2}


def _op_box_stats(
    x: NDArray[np.float64],
    whisker_factor: float = 1.5,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.quantile_stats import box_stats

    return {"box": box_stats(x, whisker_factor=whisker_factor)}


def _op_anomaly_mahalanobis(
    X: NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.anomaly_score import mahalanobis_score

    return {"scores": mahalanobis_score(X)}


def _op_ts_features(
    signal: NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.ts_features import extract_features

    return {"features": extract_features(signal)}


def _op_change_points(
    signal: NDArray[np.float64],
    n_changepoints: int = 1,
    method: str = "binary",
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.change_point import (
        binary_segmentation,
        pelt,
        segment_means,
    )

    if method == "binary":
        cps = binary_segmentation(signal, n_changepoints=n_changepoints)
    elif method == "pelt":
        cps = pelt(signal)
    else:
        raise ValueError(f"method must be 'binary' or 'pelt', got '{method}'")
    means = segment_means(signal, cps)
    return {"changepoints": cps, "segment_means": means}


def _op_denoise(
    signal: NDArray[np.float64],
    window_length: int = 11,
    polyorder: int = 3,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.denoise import savgol_filter

    return {"smoothed": savgol_filter(signal, window_length=window_length, polyorder=polyorder)}


def _op_phase_average(
    t: NDArray[np.float64],
    signal: NDArray[np.float64],
    period: float,
    n_bins: int = 36,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.phase_lock import phase_average

    phases, mean, rms = phase_average(t, signal, period=period, n_bins=n_bins)
    return {"phases": phases, "mean": mean, "rms": rms}


def _op_eof(
    X: NDArray[np.float64],
    n_modes: int = 5,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

    eofs, pcs, var = eof_decomposition(X, n_modes=n_modes)
    return {"eofs": eofs, "pcs": pcs, "var_explained": var}


def _op_safe_eval(
    expression: str,
    variables: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.expression_eval import safe_eval

    result = safe_eval(expression, variables)
    return {"result": np.asarray(result)}


def _op_two_point_acf(
    u: NDArray[np.float64],
    dx: float = 1.0,
    max_lag: int | None = None,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.two_point import (
        integral_length_scale_from_R,
        spatial_autocorrelation,
    )

    r, R = spatial_autocorrelation(u, dx=dx, max_lag=max_lag)
    L = integral_length_scale_from_R(r, R)
    return {"r": r, "R": R, "L_int": L}


def _op_running_moments(
    samples: list[NDArray[np.float64]] | NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.running_moments import RunningMoments

    arr = (
        np.asarray(samples, dtype=np.float64)
        if not isinstance(samples, list)
        else np.stack(samples)
    )
    rm = RunningMoments(shape=arr.shape[1:] if arr.ndim > 1 else ())
    for s in arr:
        rm.update(s)
    return {
        "mean": rm.mean,
        "std": rm.std,
        "n": rm.n,
    }


def _op_pod_truncation(
    singular_values: NDArray[np.float64],
    fraction: float = 0.99,
) -> dict[str, Any]:
    from naviertwin.core.dimensionality_reduction.truncation_criteria import (
        cumulative_energy_curve,
        truncate_by_energy,
    )

    r = truncate_by_energy(singular_values, fraction=fraction)
    curve = cumulative_energy_curve(singular_values)
    return {"n_modes": r, "cumulative_energy": curve}


def _op_quantile(
    x: NDArray[np.float64],
    q: float = 50.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.quantile_stats import percentile

    return {"value": percentile(x, q)}


def _op_critical_points(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.critical_points import find_critical_points

    cps = find_critical_points(u, v, dx=dx, dy=dy)
    return {"critical_points": cps, "count": len(cps)}


_OPERATIONS: dict[str, dict[str, Any]] = {
    "psd_welch": {
        "fn": _op_psd_welch,
        "category": "spectral",
        "description": "Welch 파워 스펙트럼 밀도",
        "params": ["signal", "fs", "nperseg", "window"],
        "returns": ["frequency", "psd"],
    },
    "reynolds_stats": {
        "fn": _op_reynolds_stats,
        "category": "statistics",
        "description": "Reynolds 분해 + TKE + intensity",
        "params": ["u", "v?", "w?"],
        "returns": ["mean", "rms", "tke?", "intensity?"],
    },
    "quadrant_analysis": {
        "fn": _op_quadrant_analysis,
        "category": "statistics",
        "description": "u'v' 사분면 분석 (Q1-Q4)",
        "params": ["up", "vp", "hole"],
        "returns": ["quadrants"],
    },
    "kolmogorov_slope": {
        "fn": _op_kolmogorov_slope,
        "category": "spectral",
        "description": "에너지 스펙트럼 + Kolmogorov -5/3 적합",
        "params": ["signal", "dx"],
        "returns": ["k", "E", "slope", "r2"],
    },
    "box_stats": {
        "fn": _op_box_stats,
        "category": "statistics",
        "description": "Tukey 박스플롯 통계",
        "params": ["x", "whisker_factor"],
        "returns": ["box"],
    },
    "anomaly_mahalanobis": {
        "fn": _op_anomaly_mahalanobis,
        "category": "anomaly",
        "description": "Mahalanobis 다변량 이상치 점수",
        "params": ["X"],
        "returns": ["scores"],
    },
    "ts_features": {
        "fn": _op_ts_features,
        "category": "features",
        "description": "시계열 18 특성 추출",
        "params": ["signal"],
        "returns": ["features"],
    },
    "change_points": {
        "fn": _op_change_points,
        "category": "anomaly",
        "description": "변화점 검출 (binary/PELT)",
        "params": ["signal", "n_changepoints", "method"],
        "returns": ["changepoints", "segment_means"],
    },
    "denoise": {
        "fn": _op_denoise,
        "category": "preprocessing",
        "description": "Savitzky-Golay 평활",
        "params": ["signal", "window_length", "polyorder"],
        "returns": ["smoothed"],
    },
    "phase_average": {
        "fn": _op_phase_average,
        "category": "statistics",
        "description": "위상잠금 평균",
        "params": ["t", "signal", "period", "n_bins"],
        "returns": ["phases", "mean", "rms"],
    },
    "eof": {
        "fn": _op_eof,
        "category": "rom",
        "description": "Empirical Orthogonal Functions 분해",
        "params": ["X", "n_modes"],
        "returns": ["eofs", "pcs", "var_explained"],
    },
    "safe_eval": {
        "fn": _op_safe_eval,
        "category": "preprocessing",
        "description": "사용자 표현식 평가 (AST sandbox)",
        "params": ["expression", "variables"],
        "returns": ["result"],
    },
    "two_point_acf": {
        "fn": _op_two_point_acf,
        "category": "statistics",
        "description": "공간 두-점 상관 + 적분 길이",
        "params": ["u", "dx", "max_lag"],
        "returns": ["r", "R", "L_int"],
    },
    "running_moments": {
        "fn": _op_running_moments,
        "category": "statistics",
        "description": "Welford 누적 평균/분산",
        "params": ["samples"],
        "returns": ["mean", "std", "n"],
    },
    "pod_truncation": {
        "fn": _op_pod_truncation,
        "category": "rom",
        "description": "에너지 기반 POD 절단 차수",
        "params": ["singular_values", "fraction"],
        "returns": ["n_modes", "cumulative_energy"],
    },
    "quantile": {
        "fn": _op_quantile,
        "category": "statistics",
        "description": "분위수 계산",
        "params": ["x", "q"],
        "returns": ["value"],
    },
    "critical_points": {
        "fn": _op_critical_points,
        "category": "topology",
        "description": "벡터장 임계점 검출 + 분류",
        "params": ["u", "v", "dx", "dy"],
        "returns": ["critical_points", "count"],
    },
}


__all__ = ["PostProcessFacade"]
