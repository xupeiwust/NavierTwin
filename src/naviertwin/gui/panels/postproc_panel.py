"""Post-Processor Tools 패널 — Facade를 GUI에 노출.

Facade의 list_operations / describe / run을 호출해 사용자가 어떤 op든
선택해 실행 가능. 결과는 텍스트로 표시, 큰 배열은 요약 (mean/std/shape).

Signals:
    operation_done(str, object): op 실행 완료 시 (op_name, result_dict).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from naviertwin.core.post_process_facade import PostProcessFacade


class PostProcessPanel(QWidget):
    """후처리 도구 통합 패널 (R591-647 신규 모듈)."""

    operation_done = Signal(str, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._facade = PostProcessFacade()
        self._dataset = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 좌측: 카테고리 + op 리스트
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)

        title = QLabel("Post-Processor Tools")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        # 카테고리 필터
        cat_group = QGroupBox("카테고리")
        cat_layout = QVBoxLayout(cat_group)
        self._category_combo = QComboBox()
        self._category_combo.addItem("전체")
        cats = sorted({
            self._facade.describe(op)["category"]
            for op in self._facade.list_operations()
        })
        for c in cats:
            self._category_combo.addItem(c)
        self._category_combo.currentIndexChanged.connect(
            self._refresh_op_list,
        )
        cat_layout.addWidget(self._category_combo)
        left_layout.addWidget(cat_group)

        # op 리스트
        op_group = QGroupBox("연산")
        op_layout = QVBoxLayout(op_group)
        self._op_list = QListWidget()
        self._op_list.currentTextChanged.connect(self._on_op_selected)
        op_layout.addWidget(self._op_list)
        left_layout.addWidget(op_group)

        # 실행 버튼
        self._run_btn = QPushButton("Smoke 실행 (랜덤 데이터)")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.clicked.connect(self._on_run_clicked)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        layout.addWidget(left)

        # 우측: 설명 + 결과
        right_split = QSplitter()
        right_split.setOrientation(Qt.Orientation.Vertical)

        desc_group = QGroupBox("설명")
        desc_layout = QFormLayout(desc_group)
        self._desc_label = QLabel("연산을 선택하세요.")
        self._desc_label.setWordWrap(True)
        self._params_label = QLabel("-")
        self._params_label.setWordWrap(True)
        self._returns_label = QLabel("-")
        self._returns_label.setWordWrap(True)
        desc_layout.addRow("Description:", self._desc_label)
        desc_layout.addRow("Parameters:", self._params_label)
        desc_layout.addRow("Returns:", self._returns_label)
        right_split.addWidget(desc_group)

        result_group = QGroupBox("결과")
        result_layout = QVBoxLayout(result_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        result_layout.addWidget(self._result_text)
        right_split.addWidget(result_group)

        right_split.setSizes([200, 400])
        layout.addWidget(right_split, stretch=1)

        # 초기화
        self._refresh_op_list()

    def _refresh_op_list(self) -> None:
        """카테고리 필터에 맞춰 op 리스트 갱신."""
        cat = self._category_combo.currentText()
        self._op_list.clear()
        for op in self._facade.list_operations():
            info = self._facade.describe(op)
            if cat == "전체" or info["category"] == cat:
                self._op_list.addItem(op)

    def _on_op_selected(self, op_name: str) -> None:
        if not op_name:
            return
        info = self._facade.describe(op_name)
        self._desc_label.setText(
            f"[{info['category']}] {info['description']}"
        )
        self._params_label.setText(", ".join(info["params"]))
        self._returns_label.setText(", ".join(info["returns"]))

    def _on_run_clicked(self) -> None:
        op_name = self._op_list.currentItem()
        if op_name is None:
            self._result_text.setPlainText("연산을 선택하세요.")
            return
        op_name = op_name.text()
        try:
            kwargs = self._build_smoke_kwargs(op_name)
            result = self._facade.run(op_name, **kwargs)
            self._result_text.setPlainText(self._summarize_result(result))
            self.operation_done.emit(op_name, result)
        except Exception as e:
            self._result_text.setPlainText(f"실행 실패: {e}")

    @staticmethod
    def _build_smoke_kwargs(op_name: str) -> dict[str, Any]:
        """각 op에 대한 합리적인 smoke 데이터 생성."""
        rng = np.random.default_rng(0)
        base_signal = np.sin(np.linspace(0, 4 * np.pi, 500)) + 0.1 * rng.standard_normal(500)
        unit_square = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ], dtype=np.float64)
        xyz = rng.uniform(0.2, 1.2, (80, 3))
        grid_x = np.linspace(-1.0, 1.0, 32)
        grid_y = np.linspace(-1.0, 1.0, 32)
        X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")
        kwargs_map: dict[str, dict[str, Any]] = {
            "psd_welch": {"signal": base_signal, "fs": 100.0, "nperseg": 128, "window": "hann"},
            "reynolds_stats": {"u": rng.standard_normal((100, 5))},
            "surface_forces": {
                "triangles": unit_square,
                "pressure": np.array([1.0, 1.0]),
                "shear_traction": np.tile([0.1, 0.0, 0.0], (2, 1)),
                "rho": 1.225,
                "u_inf": 10.0,
                "area_ref": 1.0,
            },
            "plane_flux": {
                "triangles": unit_square,
                "velocity": np.tile([0.0, 0.0, 1.0], (2, 1)),
                "scalar": np.array([300.0, 320.0]),
                "density": np.array([1.0, 1.1]),
            },
            "stat_convergence": {
                "signal": 1.0 + 0.05 * rng.standard_normal(2000),
                "n_batches": 20,
                "window": 100,
            },
            "quadrant_analysis": {
                "up": rng.standard_normal(2000),
                "vp": rng.standard_normal(2000),
                "hole": 0.0,
            },
            "kolmogorov_slope": {"signal": base_signal, "dx": 1.0},
            "box_stats": {"x": rng.standard_normal(500), "whisker_factor": 1.5},
            "anomaly_mahalanobis": {"X": rng.standard_normal((100, 3))},
            "ts_features": {"signal": base_signal},
            "change_points": {
                "signal": np.concatenate([
                    rng.standard_normal(60), 5 + rng.standard_normal(60),
                ]),
                "n_changepoints": 1,
                "method": "binary",
            },
            "denoise": {"signal": base_signal, "window_length": 21, "polyorder": 3},
            "phase_average": {
                "t": np.linspace(0, 50, 2000),
                "signal": np.sin(2 * np.pi * np.linspace(0, 50, 2000)),
                "period": 1.0,
                "n_bins": 20,
            },
            "eof": {"X": rng.standard_normal((100, 30)), "n_modes": 5},
            "safe_eval": {
                "expression": "sqrt(u**2 + v**2)",
                "variables": {"u": np.array([1.0, 2.0, 3.0]), "v": np.array([0.0, 1.0, 0.0])},
            },
            "two_point_acf": {"u": rng.standard_normal((100, 50)), "dx": 0.1, "max_lag": 20},
            "running_moments": {"samples": rng.standard_normal((100, 4))},
            "pod_truncation": {
                "singular_values": np.array([10.0, 5.0, 1.0, 0.1]),
                "fraction": 0.99,
            },
            "quantile": {"x": np.arange(101.0), "q": 50.0},
            "critical_points": {
                "u": -np.outer(np.linspace(-1, 1, 21), np.ones(21)).T,
                "v": np.outer(np.linspace(-1, 1, 21), np.ones(21)),
                "dx": 0.1,
                "dy": 0.1,
            },
            "time_interp": {
                "snapshots": np.stack([
                    np.array([t, t * t, np.sin(t)], dtype=np.float64)
                    for t in np.linspace(0.0, 1.0, 6)
                ]),
                "times": np.linspace(0.0, 1.0, 6),
                "t_query": 0.45,
                "n_uniform": 8,
            },
            "coord_transform": {
                "xyz": xyz[:12],
                "vectors": rng.standard_normal((12, 3)),
            },
            "line_probe": {
                "points": xyz,
                "field": xyz[:, 0] + 0.5 * xyz[:, 1],
                "start": np.array([0.2, 0.2, 0.2]),
                "end": np.array([1.2, 1.2, 1.2]),
                "n_samples": 16,
            },
            "gof_normality": {"x": rng.standard_normal(300)},
            "conditional_sampling": {
                "signal": base_signal,
                "threshold": 0.0,
                "half_window": 8,
            },
            "grid_derivatives": {
                "field_2d": X ** 2 + Y ** 2,
                "dx": grid_x[1] - grid_x[0],
                "dy": grid_y[1] - grid_y[0],
            },
            "anisotropy_state": {
                "reynolds_stress": np.diag([1.2, 0.8, 0.6]),
            },
            "morphology_components": {
                "field": ((X ** 2 + Y ** 2) < 0.45).astype(float),
                "threshold": 0.5,
                "min_size": 4,
            },
            "cell_volume_integrals": {
                "vertices": np.array([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 0.0, 1.0],
                ], dtype=np.float64),
                "connectivity": np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
                "field": np.array([2.0, 4.0]),
            },
        }
        if op_name not in kwargs_map:
            raise ValueError(f"smoke 데이터 미정의: {op_name}")
        return kwargs_map[op_name]

    @staticmethod
    def _summarize_result(result: dict[str, Any]) -> str:
        """결과 dict를 사람이 읽기 쉬운 텍스트로 요약."""
        lines = []
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                if v.size <= 6:
                    lines.append(f"{k}: {v.tolist()}")
                else:
                    lines.append(
                        f"{k}: shape={v.shape}, mean={float(v.mean()):.4g}, "
                        f"std={float(v.std()):.4g}"
                    )
            elif isinstance(v, (int, float)):
                lines.append(f"{k}: {v}")
            elif isinstance(v, list):
                if len(v) <= 6:
                    lines.append(f"{k}: {v}")
                else:
                    lines.append(f"{k}: list len={len(v)}")
            elif isinstance(v, dict):
                lines.append(f"{k}: dict keys={list(v.keys())[:5]}")
            else:
                lines.append(f"{k}: {type(v).__name__}")
        return "\n".join(lines)


__all__ = ["PostProcessPanel"]
