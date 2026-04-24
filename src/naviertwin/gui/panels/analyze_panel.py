"""유동 분석 패널 — Q-criterion, FFT/PSD, y+ 계산 및 결과 시각화.

Signals:
    analysis_done(str, object): 분석 완료 시 (분석 이름, 결과 메쉬) 발생.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


class AnalyzePanel(QWidget):
    """유동 분석 탭 패널.

    Q-criterion / λ₂, FFT/PSD, y+ 계산 기능을 제공한다.

    Signals:
        analysis_done: 분석 완료 시 결과 메쉬와 함께 발생.
    """

    analysis_done = Signal(str, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 좌측: 분석 목록 + 파라미터
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title = QLabel("Flow Analysis")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        # 분석 선택
        method_group = QGroupBox("분석 방법")
        method_layout = QVBoxLayout(method_group)
        self._method_list = QListWidget()
        self._method_list.addItems([
            "Q-criterion",
            "λ₂ Criterion",
            "FFT / PSD",
            "y+ (Wall Units)",
            "해석해 비교 (Analytic)",
        ])
        self._method_list.currentRowChanged.connect(self._on_method_selected)
        method_layout.addWidget(self._method_list)
        left_layout.addWidget(method_group)

        # 파라미터 스택
        param_group = QGroupBox("파라미터")
        param_layout = QVBoxLayout(param_group)
        self._param_stack = QStackedWidget()

        # Q-criterion 파라미터
        self._param_stack.addWidget(self._build_qcrit_params())
        # λ₂ 파라미터 (동일)
        self._param_stack.addWidget(self._build_qcrit_params())
        # FFT 파라미터
        self._param_stack.addWidget(self._build_fft_params())
        # y+ 파라미터
        self._param_stack.addWidget(self._build_yplus_params())
        # 해석해 비교 파라미터
        self._param_stack.addWidget(self._build_analytic_params())

        param_layout.addWidget(self._param_stack)
        left_layout.addWidget(param_group)

        # 실행 버튼
        self._run_btn = QPushButton("분석 실행")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_analysis)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        layout.addWidget(left)

        # 우측: 결과
        right_splitter = QSplitter()
        right_splitter.setOrientation(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Orientation.Vertical)

        # 결과 로그
        log_group = QGroupBox("결과 / 로그")
        log_layout = QVBoxLayout(log_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        log_layout.addWidget(self._result_text)
        right_splitter.addWidget(log_group)

        # 해석해 비교 시각화 위젯
        from naviertwin.gui.widgets.analytic_compare_widget import (
            AnalyticCompareWidget,
        )
        self._compare_widget = AnalyticCompareWidget()
        right_splitter.addWidget(self._compare_widget)

        # 상태 레이블
        self._status_label = QLabel("데이터를 먼저 가져오세요.")
        self._status_label.setObjectName("subtitleLabel")
        right_splitter.addWidget(self._status_label)

        right_splitter.setSizes([200, 300, 40])
        layout.addWidget(right_splitter, stretch=1)

        self._method_list.setCurrentRow(0)

    def _build_qcrit_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        combo = QComboBox()
        combo.setObjectName("velocity_combo")
        combo.addItems(["U", "velocity", "u"])
        form.addRow("속도 필드:", combo)
        return w

    def _build_fft_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        dt_spin = QDoubleSpinBox()
        dt_spin.setRange(1e-6, 1e3)
        dt_spin.setValue(0.01)
        dt_spin.setDecimals(6)
        form.addRow("Δt (s):", dt_spin)
        combo = QComboBox()
        combo.addItems(["U", "p", "T"])
        form.addRow("필드:", combo)
        return w

    def _build_analytic_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)

        flow_combo = QComboBox()
        flow_combo.setObjectName("analytic_flow_combo")
        flow_combo.addItems(["Couette", "Poiseuille 2D", "Poiseuille Pipe"])
        form.addRow("유동 유형:", flow_combo)

        field_combo = QComboBox()
        field_combo.setObjectName("analytic_field_combo")
        field_combo.addItems(["U"])
        form.addRow("비교 필드:", field_combo)

        axis_combo = QComboBox()
        axis_combo.setObjectName("analytic_axis_combo")
        axis_combo.addItems(["y", "x", "z"])
        form.addRow("샘플 축:", axis_combo)

        # 공용 파라미터
        param1 = QDoubleSpinBox()
        param1.setObjectName("analytic_param1")
        param1.setRange(-1e9, 1e9)
        param1.setDecimals(6)
        param1.setValue(1.0)
        form.addRow("U_top / dp/dx:", param1)

        mu_spin = QDoubleSpinBox()
        mu_spin.setObjectName("analytic_mu")
        mu_spin.setRange(1e-10, 1e6)
        mu_spin.setDecimals(8)
        mu_spin.setValue(1.0)
        form.addRow("μ (Pa·s):", mu_spin)

        h_spin = QDoubleSpinBox()
        h_spin.setObjectName("analytic_h")
        h_spin.setRange(1e-6, 1e6)
        h_spin.setDecimals(6)
        h_spin.setValue(1.0)
        form.addRow("H / R (m):", h_spin)

        n_spin = QDoubleSpinBox()
        n_spin.setObjectName("analytic_n")
        n_spin.setRange(5, 1000)
        n_spin.setDecimals(0)
        n_spin.setValue(50)
        form.addRow("샘플 수:", n_spin)

        return w

    def _build_yplus_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)

        wss_combo = QComboBox()
        wss_combo.addItems(["wallShearStress", "tau_w"])
        form.addRow("벽면 전단응력 필드:", wss_combo)

        rho_spin = QDoubleSpinBox()
        rho_spin.setRange(0.001, 1e5)
        rho_spin.setValue(1.225)
        rho_spin.setDecimals(4)
        form.addRow("밀도 ρ (kg/m³):", rho_spin)

        nu_spin = QDoubleSpinBox()
        nu_spin.setRange(1e-10, 1.0)
        nu_spin.setValue(1.5e-5)
        nu_spin.setDecimals(8)
        nu_spin.setSingleStep(1e-6)
        form.addRow("동점성계수 ν (m²/s):", nu_spin)

        y_spin = QDoubleSpinBox()
        y_spin.setRange(1e-10, 1.0)
        y_spin.setValue(1e-4)
        y_spin.setDecimals(8)
        y_spin.setSingleStep(1e-5)
        form.addRow("첫 번째 셀 높이 y (m):", y_spin)

        return w

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_dataset(self, dataset: CFDDataset) -> None:
        """분석할 CFDDataset을 설정한다."""
        self._dataset = dataset
        self._run_btn.setEnabled(True)
        self._status_label.setText(
            f"데이터셋 준비 완료 ({dataset.n_points} points, {dataset.n_time_steps} steps)"
        )
        # 속도 필드 콤보 업데이트
        for page_idx in [0, 1]:
            page = self._param_stack.widget(page_idx)
            combo = page.findChild(QComboBox, "velocity_combo")
            if combo is not None:
                combo.clear()
                combo.addItems(dataset.field_names)

        # 해석해 비교 필드 콤보 업데이트
        analytic_page = self._param_stack.widget(4)
        if analytic_page is not None:
            field_combo = analytic_page.findChild(QComboBox, "analytic_field_combo")
            if field_combo is not None and dataset.field_names:
                field_combo.clear()
                field_combo.addItems(dataset.field_names)

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _on_method_selected(self, row: int) -> None:
        self._param_stack.setCurrentIndex(row)

    def _run_analysis(self) -> None:
        if self._dataset is None:
            return
        row = self._method_list.currentRow()
        methods = ["q_criterion", "lambda2", "fft_psd", "yplus", "analytic"]
        method = methods[row] if row >= 0 else "q_criterion"
        try:
            result = self._dispatch(method)
            self._result_text.append(f"[{method}] 완료\n{result}\n")
            self.analysis_done.emit(method, result)
            if method == "analytic" and isinstance(result, dict):
                self._compare_widget.update_result(result)
        except Exception as exc:
            self._result_text.append(f"[ERROR] {method}: {exc}\n")

    def _dispatch(self, method: str) -> object:
        from naviertwin.core.flow_analysis.vortex.q_criterion import (
            compute_lambda2,
            compute_q_criterion,
        )

        mesh = self._dataset.mesh  # type: ignore[union-attr]

        if method == "q_criterion":
            page = self._param_stack.widget(0)
            combo = page.findChild(QComboBox, "velocity_combo")
            vel = combo.currentText() if combo else "U"
            result_mesh = compute_q_criterion(mesh, vel)
            vals = result_mesh.point_data.get("Q-criterion")
            if vals is not None:
                return f"Q range: [{vals.min():.4g}, {vals.max():.4g}]"
            return result_mesh

        elif method == "lambda2":
            page = self._param_stack.widget(1)
            combo = page.findChild(QComboBox, "velocity_combo")
            vel = combo.currentText() if combo else "U"
            result_mesh = compute_lambda2(mesh, vel)
            vals = result_mesh.point_data.get("lambda2")
            if vals is not None:
                return f"λ₂ range: [{vals.min():.4g}, {vals.max():.4g}]"
            return result_mesh

        elif method == "fft_psd":
            import numpy as np

            from naviertwin.core.flow_analysis.statistics.fft_psd import (
                compute_fft,
                find_dominant_frequencies,
            )

            n_steps = int(self._dataset.n_time_steps)  # type: ignore[union-attr]
            if n_steps <= 1:
                return "FFT: time-series 데이터가 없어 실행할 수 없습니다."

            page = self._param_stack.widget(2)
            spins = page.findChildren(QDoubleSpinBox)
            dt = spins[0].value() if spins else 0.01
            field = self._dataset.field_names[0] if self._dataset.field_names else None  # type: ignore[union-attr]
            if field:
                snapshots = self._dataset.extract_field_snapshots(field)  # type: ignore[union-attr]
                if snapshots.shape[1] <= 1:
                    return "FFT: 현재 데이터 구조에서는 시계열을 복원할 수 없습니다."
                signal = snapshots.mean(axis=0).astype(float)

                freqs, amps = compute_fft(signal, dt)
                peaks = find_dominant_frequencies(freqs, amps, n_peaks=3)
                peak_text = [f"{p['frequency']:.4g} Hz" for p in peaks]
                return f"Top frequencies: {peak_text}"
            return "FFT: 필드 없음"

        elif method == "yplus":
            import numpy as np

            from naviertwin.core.flow_analysis.boundary_layer.yplus import compute_yplus

            page = self._param_stack.widget(3)
            spins = page.findChildren(QDoubleSpinBox)
            rho = spins[0].value() if len(spins) > 0 else 1.225
            nu = spins[1].value() if len(spins) > 1 else 1.5e-5
            y_wall = spins[2].value() if len(spins) > 2 else 1e-4

            # 벽면 전단응력 필드 탐색
            wss_combo = page.findChild(QComboBox)
            wss_field = wss_combo.currentText() if wss_combo else "wallShearStress"
            if wss_field in mesh.point_data:
                tau = mesh.point_data[wss_field]
                tau_arr = np.asarray(tau, dtype=float)
                if tau_arr.ndim == 1:
                    # 크기만 있는 경우에도 각 포인트별 전단응력으로 계산되도록 2D로 맞춘다.
                    tau_arr = tau_arr[:, np.newaxis]
                yplus = compute_yplus(
                    tau_arr,
                    rho,
                    nu,
                    np.full(tau_arr.shape[0], y_wall, dtype=float),
                )
                return f"y+ range: [{yplus.min():.4g}, {yplus.max():.4g}], mean={yplus.mean():.4g}"
            return "y+: 벽면 전단응력 필드 없음"

        elif method == "analytic":
            return self._run_analytic_compare(mesh)

        return "알 수 없는 분석 방법"

    def _run_analytic_compare(self, mesh: object) -> object:
        """해석해와 수치해를 비교하고 결과 dict 를 반환한다."""
        import numpy as np

        from naviertwin.core.validation.analytic_solutions import (
            compare_against_analytic,
            couette_flow,
            poiseuille_flow_2d,
            poiseuille_pipe,
        )

        page = self._param_stack.widget(4)
        flow_combo: QComboBox = page.findChild(QComboBox, "analytic_flow_combo")
        field_combo: QComboBox = page.findChild(QComboBox, "analytic_field_combo")
        axis_combo: QComboBox = page.findChild(QComboBox, "analytic_axis_combo")
        p1: QDoubleSpinBox = page.findChild(QDoubleSpinBox, "analytic_param1")
        mu: QDoubleSpinBox = page.findChild(QDoubleSpinBox, "analytic_mu")
        h: QDoubleSpinBox = page.findChild(QDoubleSpinBox, "analytic_h")
        n: QDoubleSpinBox = page.findChild(QDoubleSpinBox, "analytic_n")

        flow = flow_combo.currentText() if flow_combo else "Couette"
        field = field_combo.currentText() if field_combo else "U"
        axis = axis_combo.currentText() if axis_combo else "y"
        param1_val = p1.value() if p1 else 1.0
        mu_val = mu.value() if mu else 1.0
        h_val = h.value() if h else 1.0
        n_val = int(n.value()) if n else 50

        coords = np.linspace(0.0, h_val, n_val)
        if flow == "Couette":
            sol = couette_flow(U_top=param1_val, H=h_val, y=coords)
        elif flow == "Poiseuille 2D":
            sol = poiseuille_flow_2d(dpdx=param1_val, mu=mu_val, H=h_val, y=coords)
        else:  # Poiseuille Pipe
            sol = poiseuille_pipe(dpdx=param1_val, mu=mu_val, R=h_val, r=coords)

        return compare_against_analytic(mesh, sol, field_name=field, axis=axis)
