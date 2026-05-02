"""GUI 검증용 합성 CFD 결과 파일 생성기.

합성 cavity flow 비슷한 데이터를 VTU + ntwin (HDF5) 형식으로 저장.
GUI에서 ① Import → 이 파일을 로드해 모든 탭 검증.

Usage:
    python3 scripts/make_test_dataset.py [out_dir]

Output (default: /tmp/naviertwin_demo/):
    cavity.vtu       — pyvista로 작성된 unstructured grid (U/p/T fields)
    cavity_time.npz  — 시간-공간 필드 행렬 (n_t=50, n_x=400)
    cavity_probe.csv — 단일 프로브 시계열 (CSV)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def make_cavity_vtu(out_path: Path, nx: int = 20, ny: int = 20) -> None:
    """Lid-driven cavity 비슷한 정상 상태 필드를 VTU로 저장."""
    import pyvista as pv

    grid_cls = getattr(pv, "ImageData", None) or pv.UniformGrid
    grid = grid_cls(
        dimensions=(nx, ny, 1),
        spacing=(1.0 / (nx - 1), 1.0 / (ny - 1), 1.0),
    )
    pts = grid.points
    x, y = pts[:, 0], pts[:, 1]
    # 회전 와류 비슷한 패턴
    U = np.column_stack([
        np.sin(np.pi * x) * np.cos(np.pi * y),
        -np.cos(np.pi * x) * np.sin(np.pi * y),
        np.zeros_like(x),
    ]).astype(np.float32)
    p = (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) * 0.5
    T = 300.0 + 10.0 * np.sin(np.pi * x) * np.sin(np.pi * y)
    grid.point_data["U"] = U
    grid.point_data["p"] = p.astype(np.float32)
    grid.point_data["T"] = T.astype(np.float32)
    grid.point_data["wallShearStress"] = np.column_stack([
        0.5 * np.cos(np.pi * x), 0.0 * x, 0.0 * x,
    ]).astype(np.float32)
    ug = grid.cast_to_unstructured_grid()
    ug.save(str(out_path))


def make_time_series_npz(out_path: Path, n_t: int = 50, n_x: int = 400) -> None:
    """(n_t, n_x) 시공간 행렬 + 시간 배열."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, n_x)
    t = np.linspace(0, 5, n_t)
    # rank 5 진짜 모드 + 노이즈
    spatial = np.column_stack([
        np.sin(np.pi * x), np.sin(2 * np.pi * x),
        np.sin(3 * np.pi * x), np.sin(5 * np.pi * x),
        np.cos(np.pi * x),
    ])
    temporal = np.column_stack([
        np.sin(2 * np.pi * t), np.cos(2 * np.pi * t),
        np.sin(4 * np.pi * t), np.exp(-t / 2),
        np.sin(6 * np.pi * t),
    ])
    X = temporal @ spatial.T + 0.05 * rng.standard_normal((n_t, n_x))
    np.savez_compressed(str(out_path), times=t, X=X.astype(np.float32))


def make_probe_csv(out_path: Path, n: int = 2000) -> None:
    """단일 probe 시계열 (PSD/change-points/anomaly 검증용)."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 20, n)
    sig = (
        np.sin(2 * np.pi * 5 * t)
        + 0.3 * np.sin(2 * np.pi * 17 * t)
        + 0.1 * rng.standard_normal(n)
    )
    # 두 번째 절반에 평균 변화 + spike
    sig[n // 2:] += 1.5
    sig[1500] = 10.0
    out_path.write_text(
        "time,signal\n" + "\n".join(
            f"{ti:.6f},{si:.6f}" for ti, si in zip(t, sig)
        ),
        encoding="utf-8",
    )


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/naviertwin_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"출력 디렉토리: {out_dir}")
    try:
        vtu_path = out_dir / "cavity.vtu"
        make_cavity_vtu(vtu_path)
        print(f"  ✅ {vtu_path}")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ cavity.vtu: {e} (pyvista 필요)")

    npz_path = out_dir / "cavity_time.npz"
    make_time_series_npz(npz_path)
    print(f"  ✅ {npz_path}")

    csv_path = out_dir / "cavity_probe.csv"
    make_probe_csv(csv_path)
    print(f"  ✅ {csv_path}")

    print("\n사용법:")
    print(f"  GUI 실행: PYTHONPATH=src python3 -m naviertwin --gui")
    print(f"  ① Import 탭에서 {vtu_path} 로드")
    return 0


if __name__ == "__main__":
    sys.exit(main())
