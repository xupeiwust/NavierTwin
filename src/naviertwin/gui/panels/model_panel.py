"""모델 학습/평가 패널 — Surrogate 모델(RBF/Kriging) 학습 및 검증 지표 표시.

Signals:
    model_trained(str, object): 모델 학습 완료 시 (모델 타입, surrogate) 발생.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


class ModelPanel(QWidget):
    """Surrogate 모델 학습 패널.

    Signals:
        model_trained: 모델 학습 완료 시 surrogate 객체와 함께 발생.
    """

    model_trained = Signal(str, object)
    active_learning_done = Signal(object)
    online_learning_done = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._reducer: Optional[object] = None
        self._surrogate: Optional[object] = None
        self._reduction_artifact: Optional[dict[str, object]] = None
        self._loaded_metadata: dict[str, object] = {}
        self._loss_series: dict[str, list[float]] = {}
        self._online_learner: Optional[object] = None
        self._last_active_candidates: Optional[np.ndarray] = None
        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 좌측
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title = QLabel("Surrogate Model")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        # 모델 선택
        model_group = QGroupBox("모델 선택")
        model_form = QFormLayout(model_group)

        self._model_combo = QComboBox()
        self._model_combo.addItems(["Kriging (RBF default)", "RBF", "Kriging"])
        model_form.addRow("Surrogate 타입:", self._model_combo)

        left_layout.addWidget(model_group)

        # 학습 데이터 설정
        data_group = QGroupBox("학습 데이터 생성 (데모)")
        data_form = QFormLayout(data_group)

        self._n_samples_spin = QSpinBox()
        self._n_samples_spin.setRange(2, 1000)
        self._n_samples_spin.setValue(20)
        data_form.addRow("샘플 수:", self._n_samples_spin)

        self._n_params_spin = QSpinBox()
        self._n_params_spin.setRange(1, 20)
        self._n_params_spin.setValue(2)
        data_form.addRow("파라미터 차원:", self._n_params_spin)

        self._n_outputs_spin = QSpinBox()
        self._n_outputs_spin.setRange(1, 100)
        self._n_outputs_spin.setValue(5)
        data_form.addRow("출력 차원:", self._n_outputs_spin)

        self._train_ratio_spin = QDoubleSpinBox()
        self._train_ratio_spin.setRange(0.5, 0.95)
        self._train_ratio_spin.setValue(0.8)
        self._train_ratio_spin.setSingleStep(0.05)
        data_form.addRow("학습 비율:", self._train_ratio_spin)

        left_layout.addWidget(data_group)

        self._train_btn = QPushButton("모델 학습")
        self._train_btn.setObjectName("primaryButton")
        self._train_btn.clicked.connect(self._train_model)
        left_layout.addWidget(self._train_btn)

        # Active learning candidate selection
        active_group = QGroupBox("Active Learning")
        active_form = QFormLayout(active_group)

        self._active_strategy_combo = QComboBox()
        self._active_strategy_combo.addItems(["variance", "random"])
        active_form.addRow("Strategy:", self._active_strategy_combo)

        self._active_pool_spin = QSpinBox()
        self._active_pool_spin.setRange(10, 10000)
        self._active_pool_spin.setValue(128)
        active_form.addRow("Candidate pool:", self._active_pool_spin)

        self._active_k_spin = QSpinBox()
        self._active_k_spin.setRange(1, 32)
        self._active_k_spin.setValue(3)
        active_form.addRow("Query count:", self._active_k_spin)

        self._active_low_spin = QDoubleSpinBox()
        self._active_low_spin.setRange(-1e9, 1e9)
        self._active_low_spin.setDecimals(4)
        self._active_low_spin.setValue(0.0)
        active_form.addRow("Lower bound:", self._active_low_spin)

        self._active_high_spin = QDoubleSpinBox()
        self._active_high_spin.setRange(-1e9, 1e9)
        self._active_high_spin.setDecimals(4)
        self._active_high_spin.setValue(1.0)
        active_form.addRow("Upper bound:", self._active_high_spin)

        self._active_btn = QPushButton("후보 추천")
        self._active_btn.clicked.connect(self._run_active_learning)
        active_form.addRow(self._active_btn)
        left_layout.addWidget(active_group)

        online_group = QGroupBox("Online Update")
        online_form = QFormLayout(online_group)

        self._online_buffer_spin = QSpinBox()
        self._online_buffer_spin.setRange(4, 10000)
        self._online_buffer_spin.setValue(100)
        online_form.addRow("Buffer size:", self._online_buffer_spin)

        self._online_refit_spin = QSpinBox()
        self._online_refit_spin.setRange(1, 1000)
        self._online_refit_spin.setValue(1)
        online_form.addRow("Refit every:", self._online_refit_spin)

        self._online_x_edit = QLineEdit()
        self._online_x_edit.setPlaceholderText("예: 0.1, 0.2 (공백이면 추천 후보)")
        online_form.addRow("New x:", self._online_x_edit)

        self._online_y_spin = QDoubleSpinBox()
        self._online_y_spin.setRange(-1e12, 1e12)
        self._online_y_spin.setDecimals(6)
        self._online_y_spin.setValue(0.0)
        online_form.addRow("Observed y:", self._online_y_spin)

        self._online_update_btn = QPushButton("온라인 업데이트")
        self._online_update_btn.clicked.connect(self._run_online_update)
        online_form.addRow(self._online_update_btn)
        left_layout.addWidget(online_group)

        # ─── 신경 연산자 (v2.0+) ───
        op_group = QGroupBox("신경 연산자 (Operator Learning)")
        op_form = QFormLayout(op_group)

        self._op_combo = QComboBox()
        self._op_combo.addItems(["FNO1D", "FNO2D", "TFNO2D", "DeepONet", "UNet2D", "WNO1D"])
        op_form.addRow("연산자 타입:", self._op_combo)

        self._op_epochs_spin = QSpinBox()
        self._op_epochs_spin.setRange(1, 1000)
        self._op_epochs_spin.setValue(10)
        op_form.addRow("Epoch:", self._op_epochs_spin)

        self._op_samples_spin = QSpinBox()
        self._op_samples_spin.setRange(4, 1000)
        self._op_samples_spin.setValue(20)
        op_form.addRow("데모 샘플 수:", self._op_samples_spin)

        self._op_train_btn = QPushButton("연산자 학습")
        self._op_train_btn.clicked.connect(self._train_operator)
        op_form.addRow(self._op_train_btn)

        left_layout.addWidget(op_group)

        left_layout.addStretch()
        self._left_scroll = QScrollArea()
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setFixedWidth(300)
        self._left_scroll.setWidget(left)
        layout.addWidget(self._left_scroll)

        # 우측
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 검증 지표
        metrics_group = QGroupBox("검증 지표")
        metrics_layout = QVBoxLayout(metrics_group)
        self._metrics_list = QListWidget()
        self._metrics_list.setAlternatingRowColors(True)
        self._metrics_list.setMaximumHeight(160)
        metrics_layout.addWidget(self._metrics_list)
        right_layout.addWidget(metrics_group)

        active_result_group = QGroupBox("Active Learning 후보")
        active_result_layout = QVBoxLayout(active_result_group)
        self._active_table = QTableWidget(0, 3)
        self._active_table.setHorizontalHeaderLabels(["Rank", "Parameters", "Score"])
        self._active_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        active_result_layout.addWidget(self._active_table)
        right_layout.addWidget(active_result_group)

        online_result_group = QGroupBox("Online Update 상태")
        online_result_layout = QVBoxLayout(online_result_group)
        self._online_table = QTableWidget(0, 3)
        self._online_table.setHorizontalHeaderLabels(["Metric", "Value", "Notes"])
        self._online_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        online_result_layout.addWidget(self._online_table)
        right_layout.addWidget(online_result_group)

        # 학습 loss curve
        loss_group = QGroupBox("학습 Loss Curve")
        loss_layout = QVBoxLayout(loss_group)
        from naviertwin.gui.widgets.loss_curve_widget import LossCurveWidget

        self._loss_curve = LossCurveWidget()
        loss_layout.addWidget(self._loss_curve)
        right_layout.addWidget(loss_group)

        # 로그
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        log_layout.addWidget(self._log_text)
        right_layout.addWidget(log_group)

        layout.addWidget(right, stretch=1)

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_reducer(self, reducer: object) -> None:
        """차원 축소 결과를 설정한다."""
        self._reducer = reducer
        self._log(f"Reducer 설정: {type(reducer).__name__}")

    def set_dataset(self, dataset: CFDDataset) -> None:
        """학습에 사용할 원본 데이터셋을 설정한다."""
        self._dataset = dataset
        self._reduction_artifact = None
        self._log(
            f"Dataset 설정: {dataset.n_points} pts, {dataset.n_cells} cells, "
            f"{dataset.n_time_steps} steps"
        )

    def set_reduction_artifact(self, artifact: dict[str, object]) -> None:
        """Reduce 단계 산출물을 설정한다."""
        self._reduction_artifact = artifact
        method = str(artifact.get("method", "unknown"))
        field_name = str(artifact.get("field_name", ""))
        self._log(f"Reduction artifact 설정: method={method}, field={field_name}")

    def set_loaded_metadata(self, meta: dict[str, object]) -> None:
        """프로젝트 로드 시 저장된 메타데이터로 UI 상태를 복원한다."""
        self._loaded_metadata = dict(meta)

        engine_meta = meta.get("engine")
        surrogate_meta: dict[str, object] = {}
        if isinstance(engine_meta, dict):
            candidate = engine_meta.get("surrogate")
            if isinstance(candidate, dict):
                surrogate_meta = candidate

        reducer_meta: dict[str, object] = {}
        if isinstance(engine_meta, dict):
            candidate = engine_meta.get("reducer")
            if isinstance(candidate, dict):
                reducer_meta = candidate

        restored: list[str] = []

        n_modes = (
            self._as_positive_int(meta.get("n_modes"))
            or self._as_positive_int(engine_meta.get("n_modes") if isinstance(engine_meta, dict) else None)
            or self._as_positive_int(surrogate_meta.get("n_modes"))
            or self._as_positive_int(reducer_meta.get("n_modes"))
        )
        if n_modes is not None:
            self._n_outputs_spin.setValue(n_modes)
            restored.append(f"n_modes={n_modes}")

        n_params = (
            self._as_positive_int(meta.get("n_params"))
            or self._as_positive_int(surrogate_meta.get("n_params"))
        )
        if n_params is not None:
            self._n_params_spin.setValue(n_params)
            restored.append(f"n_params={n_params}")

        n_outputs = (
            self._as_positive_int(meta.get("n_outputs"))
            or self._as_positive_int(surrogate_meta.get("n_outputs"))
            or self._as_positive_int(surrogate_meta.get("n_modes"))
            or self._as_positive_int(engine_meta.get("n_modes") if isinstance(engine_meta, dict) else None)
        )
        if n_outputs is not None:
            self._n_outputs_spin.setValue(n_outputs)
            restored.append(f"n_outputs={n_outputs}")

        n_samples = (
            self._as_positive_int(meta.get("n_samples"))
            or self._as_positive_int(surrogate_meta.get("n_samples"))
        )
        if n_samples is not None:
            self._n_samples_spin.setValue(n_samples)
            restored.append(f"n_samples={n_samples}")

        if restored:
            self._log("Loaded metadata restored: " + ", ".join(restored))
        else:
            self._log("Loaded metadata restored: no numeric fields found")

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _train_model(self) -> None:
        model_idx = self._model_combo.currentIndex()
        n_samples = self._n_samples_spin.value()
        n_params = self._n_params_spin.value()
        n_outputs = self._n_outputs_spin.value()
        ratio = self._train_ratio_spin.value()

        try:
            X: np.ndarray
            Y: np.ndarray
            used_real_data = False

            # 최우선: Reduce 단계 산출물(coeffs/params) 사용
            if self._reduction_artifact is not None:
                coeffs_raw = self._reduction_artifact.get("coeffs")
                params_raw = self._reduction_artifact.get("params")
                if coeffs_raw is not None and params_raw is not None:
                    coeffs = np.asarray(coeffs_raw, dtype=float)
                    params = np.asarray(params_raw, dtype=float)
                    if params.ndim == 1:
                        params = params.reshape(-1, 1)
                    if coeffs.ndim == 1:
                        coeffs = coeffs.reshape(-1, 1)
                    if coeffs.shape[0] == params.shape[0] and coeffs.shape[0] >= 2:
                        X = params
                        Y = coeffs
                        used_real_data = True
                        n_samples = X.shape[0]
                        n_params = X.shape[1]
                        n_outputs = Y.shape[1]
                        self._n_params_spin.setValue(n_params)
                        self._n_outputs_spin.setValue(n_outputs)
                        self._log(
                            f"artifact 기반 학습 사용: samples={n_samples}, "
                            f"params={n_params}, outputs={n_outputs}"
                        )

            # 우선: reducer + dataset이 있으면 실제 스냅샷 기반 학습 데이터 사용
            if (
                not used_real_data
                and self._reducer is not None
                and self._dataset is not None
                and self._dataset.field_names
            ):
                try:
                    field_name = self._dataset.field_names[0]
                    reducer_meta = getattr(self._reducer, "training_metadata", None)
                    if isinstance(reducer_meta, dict):
                        meta_field = reducer_meta.get("field_name")
                        if (
                            isinstance(meta_field, str)
                            and meta_field in self._dataset.field_names
                        ):
                            field_name = meta_field
                    snapshots = self._extract_snapshots_from_dataset(field_name)
                    if snapshots.shape[1] >= 2:
                        coeffs = self._reducer.encode(snapshots)
                        # 기본 파라미터는 time step 사용
                        ts = np.asarray(self._dataset.time_steps, dtype=float)
                        if ts.size == coeffs.shape[0]:
                            params_1d = ts
                        else:
                            params_1d = np.linspace(0.0, 1.0, coeffs.shape[0])
                        X = params_1d.reshape(-1, 1)
                        Y = coeffs
                        used_real_data = True
                        n_samples = X.shape[0]
                        n_params = X.shape[1]
                        n_outputs = Y.shape[1]
                        self._n_params_spin.setValue(n_params)
                        self._n_outputs_spin.setValue(n_outputs)
                        self._log(
                            f"실데이터 학습 사용: field={field_name}, samples={n_samples}, "
                            f"params={n_params}, outputs={n_outputs}"
                        )
                except Exception as exc:
                    self._log(f"[WARN] 실데이터 학습 데이터 구성 실패, 데모로 폴백: {exc}")

            if not used_real_data:
                if self._reducer is not None:
                    reducer_modes = int(getattr(self._reducer, "n_components", n_outputs))
                    if reducer_modes > 0:
                        n_outputs = reducer_modes
                        self._n_outputs_spin.setValue(n_outputs)
                        self._log(f"Reducer 모드 수에 맞춰 출력 차원을 {n_outputs}로 설정")

                # 데모: 랜덤 데이터 생성
                rng = np.random.default_rng(42)
                X = rng.random((n_samples, n_params))
                Y = np.sin(X.sum(axis=1, keepdims=True) * np.pi) * rng.random(
                    (n_samples, n_outputs)
                )
                self._log("데모 데이터로 학습 수행")

            n_train = max(2, int(n_samples * ratio))
            X_train, X_test = X[:n_train], X[n_train:]
            Y_train, Y_test = Y[:n_train], Y[n_train:]

            if model_idx <= 0 or model_idx == 2:
                from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate
                surrogate = KrigingSurrogate()
            else:
                from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate
                surrogate = RBFSurrogate()

            surrogate.fit(X_train, Y_train)
            self._surrogate = surrogate
            self._online_learner = None
            self._last_active_candidates = None
            model_name = type(surrogate).__name__
            surrogate.training_metadata = {
                "dataset_id": id(self._dataset) if self._dataset is not None else None,
                "field_name": (
                    self._dataset.field_names[0]
                    if self._dataset and self._dataset.field_names
                    else ""
                ),
                "n_modes": int(Y.shape[1]),
                "n_outputs": int(Y.shape[1]),
                "n_params": int(X.shape[1]),
                "n_samples": int(Y.shape[0]),
                "source": "real_data" if used_real_data else "demo",
            }
            surrogate.training_metadata["explainability"] = (
                self._build_explainability_metadata(X_train)
            )

            # 검증
            metrics_text = ""
            validation_metrics: dict[str, float] = {}
            if len(X_test) > 0:
                from naviertwin.core.validation.metrics import compute_all_metrics
                Y_pred = surrogate.predict(X_test)
                metrics = compute_all_metrics(Y_test, Y_pred)
                validation_metrics = {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))
                }
                self._metrics_list.clear()
                for k, v in metrics.items():
                    self._metrics_list.addItem(f"{k}: {v:.6f}")
                metrics_text = f"  RMSE={metrics.get('rmse', 0):.4g}, R²={metrics.get('r2', 0):.4g}"
            else:
                self._metrics_list.clear()
                self._metrics_list.addItem("테스트 셋 없음 (샘플 수 증가 필요)")
            surrogate.training_metadata["validation_metrics"] = validation_metrics

            self._log(f"{model_name} 학습 완료 (n_train={n_train}){metrics_text}")
            self.model_trained.emit(model_name.lower(), surrogate)

        except Exception as exc:
            self._log(f"[ERROR] {exc}")

    def _run_active_learning(self) -> None:
        """학습된 surrogate로 다음 CFD 평가 후보를 추천한다."""
        if self._surrogate is None:
            self._log("[WARN] Active Learning을 실행할 학습된 surrogate가 없습니다.")
            return

        low = float(self._active_low_spin.value())
        high = float(self._active_high_spin.value())
        if not low < high:
            self._log("[WARN] Active Learning bounds는 lower < upper 이어야 합니다.")
            return

        n_params = int(getattr(self._surrogate, "input_dim", 0) or self._n_params_spin.value())
        if n_params <= 0:
            self._log("[WARN] surrogate 입력 차원을 확인할 수 없습니다.")
            return

        n_pool = int(self._active_pool_spin.value())
        k = min(int(self._active_k_spin.value()), n_pool)
        rng = np.random.default_rng(0)
        pool = rng.uniform(low, high, size=(n_pool, n_params))
        strategy = self._active_strategy_combo.currentText()

        try:
            model = self._active_learning_adapter(self._surrogate)
            idx = self._select_next_samples(model, pool, k=k, strategy=strategy)
            selected = pool[np.asarray(idx, dtype=int)]
            self._last_active_candidates = selected.copy()
            scores = self._active_learning_scores(self._surrogate, selected, strategy)
            self._render_active_candidates(selected, scores)
            result = {
                "strategy": strategy,
                "selected": selected,
                "scores": scores,
                "bounds": np.array([[low, high]], dtype=float),
            }
            self._log(
                f"Active Learning 후보 추천 완료: strategy={strategy}, "
                f"k={len(selected)}, dim={n_params}"
            )
            self.active_learning_done.emit(result)
        except Exception as exc:
            self._log(f"[ERROR] Active Learning 실패: {exc}")

    def _run_online_update(self) -> None:
        """새 관측값을 OnlineKriging 버퍼에 반영한다."""
        if self._surrogate is None:
            self._log("[WARN] Online Update를 실행할 학습된 surrogate가 없습니다.")
            return

        try:
            background = self._online_background(self._surrogate)
            if background is None:
                self._log("[WARN] Online Update 초기화용 background가 없습니다.")
                return
            y_init = self._predict_scalar(self._surrogate, background)
            x_new = self._online_update_x(background.shape[1])
            y_obs = float(self._online_y_spin.value())
            learner = self._ensure_online_learner(background, y_init)
            before = float(np.ravel(learner.predict(x_new[None, :]))[0])
            learner.update(x_new, y_obs)
            after = float(np.ravel(learner.predict(x_new[None, :]))[0])
            self._surrogate = learner
            learner.training_metadata = self._online_training_metadata(
                background,
                y_init,
                x_new,
                y_obs,
            )
            self._render_online_update(
                x_new=x_new,
                y_obs=y_obs,
                pred_before=before,
                pred_after=after,
                learner=learner,
            )
            result = {
                "x": x_new,
                "y": y_obs,
                "prediction_before": before,
                "prediction_after": after,
                "buffer_size": len(getattr(learner, "_X", [])),
            }
            self._log(
                f"OnlineKriging 업데이트 완료: y={y_obs:.6g}, "
                f"pred_before={before:.6g}, pred_after={after:.6g}"
            )
            self.online_learning_done.emit(result)
            self.model_trained.emit("online_kriging", learner)
        except Exception as exc:
            self._log(f"[ERROR] Online Update 실패: {exc}")

    def _online_background(self, surrogate: object) -> np.ndarray | None:
        metadata = getattr(surrogate, "training_metadata", None)
        if not isinstance(metadata, Mapping):
            return None
        explain_meta = metadata.get("explainability")
        if not isinstance(explain_meta, Mapping):
            return None
        background = np.asarray(explain_meta.get("background"), dtype=float)
        if background.ndim != 2 or background.shape[0] < 2 or background.shape[1] == 0:
            return None
        return background

    def _predict_scalar(self, surrogate: object, X: np.ndarray) -> np.ndarray:
        raw = surrogate.predict(np.asarray(X, dtype=float))  # type: ignore[attr-defined]
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            return np.full((X.shape[0],), float(arr))
        if arr.ndim == 1:
            return arr
        flat = arr.reshape(arr.shape[0], -1)
        idx = min(self._online_output_index(surrogate), flat.shape[1] - 1)
        return flat[:, idx]

    @staticmethod
    def _online_output_index(surrogate: object) -> int:
        metadata = getattr(surrogate, "training_metadata", None)
        if not isinstance(metadata, Mapping):
            return 0
        explain_meta = metadata.get("explainability")
        if not isinstance(explain_meta, Mapping):
            return 0
        try:
            return max(0, int(explain_meta.get("output_index", 0)))
        except (TypeError, ValueError):
            return 0

    def _online_update_x(self, n_params: int) -> np.ndarray:
        text = self._online_x_edit.text().strip()
        if text:
            parts = [item for item in text.replace(",", " ").split() if item]
            values = np.array([float(item) for item in parts], dtype=float)
            if values.size != n_params:
                raise ValueError(
                    f"New x 차원 불일치: expected={n_params}, got={values.size}"
                )
            return values

        if (
            self._last_active_candidates is not None
            and self._last_active_candidates.ndim == 2
            and self._last_active_candidates.shape[1] == n_params
            and self._last_active_candidates.shape[0] > 0
        ):
            return self._last_active_candidates[0].astype(float, copy=True)

        self._log("[INFO] 추천 후보가 없어 background 평균점을 온라인 업데이트에 사용합니다.")
        assert self._surrogate is not None
        background = self._online_background(self._surrogate)
        if background is None:
            raise RuntimeError("online background unavailable")
        return np.mean(background, axis=0)

    def _ensure_online_learner(
        self, X_init: np.ndarray, y_init: np.ndarray,
    ) -> object:
        from naviertwin.core.online_learning.online_learning import OnlineKriging

        if (
            isinstance(self._surrogate, OnlineKriging)
            and self._surrogate.is_initialized
            and self._surrogate.input_dim == X_init.shape[1]
        ):
            self._online_learner = self._surrogate
            return self._surrogate
        if (
            isinstance(self._online_learner, OnlineKriging)
            and self._online_learner.is_initialized
            and self._online_learner.input_dim == X_init.shape[1]
        ):
            return self._online_learner

        learner = OnlineKriging(
            buffer_size=int(self._online_buffer_spin.value()),
            refit_every=int(self._online_refit_spin.value()),
        )
        learner.initialize(X_init, y_init)
        self._online_learner = learner
        return learner

    def _online_training_metadata(
        self,
        X_init: np.ndarray,
        y_init: np.ndarray,
        x_new: np.ndarray,
        y_obs: float,
    ) -> dict[str, object]:
        background = np.vstack([X_init, x_new[None, :]])
        y_all = np.append(y_init, y_obs)
        n_features = background.shape[1]
        return {
            "source": "online_update",
            "n_params": int(n_features),
            "n_outputs": 1,
            "n_samples": int(background.shape[0]),
            "online_y_mean": float(np.mean(y_all)),
            "explainability": {
                "background": background[:32].copy(),
                "feature_names": [f"param_{i}" for i in range(n_features)],
                "output_index": 0,
            },
        }

    def _render_online_update(
        self,
        *,
        x_new: np.ndarray,
        y_obs: float,
        pred_before: float,
        pred_after: float,
        learner: object,
    ) -> None:
        rows = [
            ("x", np.array2string(x_new, precision=4), "new observation"),
            ("observed_y", f"{y_obs:.6g}", "customer/plant value"),
            ("prediction_before", f"{pred_before:.6g}", "before update"),
            ("prediction_after", f"{pred_after:.6g}", "after update"),
            ("abs_error_after", f"{abs(pred_after - y_obs):.6g}", "post-update fit"),
            ("buffer_count", str(len(getattr(learner, "_X", []))), "OnlineKriging"),
        ]
        self._online_table.setRowCount(len(rows))
        for row, (metric, value, note) in enumerate(rows):
            self._online_table.setItem(row, 0, QTableWidgetItem(metric))
            self._online_table.setItem(row, 1, QTableWidgetItem(value))
            self._online_table.setItem(row, 2, QTableWidgetItem(note))

    def _active_learning_adapter(self, surrogate: object) -> object:
        """select_next_samples가 기대하는 predict(return_std=True) adapter."""

        class _Adapter:
            def __init__(self, model: object) -> None:
                self._model = model

            def predict(
                self,
                X: np.ndarray,
                return_std: bool = False,
            ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
                X_arr = np.asarray(X, dtype=float)
                if return_std:
                    predict_var = getattr(self._model, "predict_with_variance", None)
                    if predict_var is None:
                        raise TypeError("surrogate does not support return_std")
                    pred, var = predict_var(X_arr)
                    pred_arr = np.asarray(pred, dtype=float).reshape(X_arr.shape[0], -1)
                    var_arr = np.asarray(var, dtype=float).reshape(X_arr.shape[0], -1)
                    mean = np.mean(pred_arr, axis=1)
                    std = np.sqrt(np.maximum(np.mean(var_arr, axis=1), 0.0))
                    return mean, std
                pred = self._model.predict(X_arr)  # type: ignore[attr-defined]
                pred_arr = np.asarray(pred, dtype=float).reshape(X_arr.shape[0], -1)
                return np.mean(pred_arr, axis=1)

        return _Adapter(surrogate)

    def _select_next_samples(
        self,
        model: object,
        pool: np.ndarray,
        k: int,
        strategy: str,
    ) -> np.ndarray:
        from naviertwin.core.online_learning.active_learning import select_next_samples

        return select_next_samples(model, pool, k=k, strategy=strategy, seed=0)

    def _active_learning_scores(
        self,
        surrogate: object,
        selected: np.ndarray,
        strategy: str,
    ) -> np.ndarray:
        if strategy != "variance":
            return np.full((selected.shape[0],), np.nan)
        predict_var = getattr(surrogate, "predict_with_variance", None)
        if predict_var is None:
            return np.full((selected.shape[0],), np.nan)
        _, var = predict_var(selected)
        var_arr = np.asarray(var, dtype=float).reshape(selected.shape[0], -1)
        return np.mean(var_arr, axis=1)

    def _render_active_candidates(
        self,
        selected: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        self._active_table.setRowCount(selected.shape[0])
        for row, params in enumerate(selected):
            score = scores[row] if row < len(scores) else np.nan
            self._active_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self._active_table.setItem(
                row,
                1,
                QTableWidgetItem(np.array2string(params, precision=4)),
            )
            score_text = "n/a" if not np.isfinite(score) else f"{float(score):.6g}"
            self._active_table.setItem(row, 2, QTableWidgetItem(score_text))

    def _train_operator(self) -> None:
        """선택한 신경 연산자를 합성 데이터로 빠르게 학습/검증한다."""
        import numpy as np

        op_type = self._op_combo.currentText()
        epochs = int(self._op_epochs_spin.value())
        n = int(self._op_samples_spin.value())
        rng = np.random.default_rng(0)

        try:
            if op_type == "FNO1D":
                from naviertwin.core.operator_learning.fno.fno import FNO1D

                X = rng.standard_normal((n, 32, 1)).astype(np.float32)
                Y = np.sin(X).astype(np.float32)
                op = FNO1D(
                    in_channels=1, out_channels=1, modes=4, width=8,
                    n_layers=2, max_epochs=epochs,
                )
                op.fit({"inputs": X, "outputs": Y})
                y_hat = op.predict({"x": X[:2]})
                msg = f"FNO1D: 예측 shape={y_hat.shape}, 마지막 loss={op.train_losses_[-1]:.4g}"

            elif op_type == "FNO2D":
                from naviertwin.core.operator_learning.fno.fno import FNO2D

                X = rng.standard_normal((max(n // 2, 4), 16, 16, 1)).astype(np.float32)
                Y = X ** 2
                op = FNO2D(
                    in_channels=1, out_channels=1, modes1=4, modes2=4,
                    width=8, n_layers=2, max_epochs=epochs,
                )
                op.fit({"inputs": X, "outputs": Y})
                y_hat = op.predict({"x": X[:2]})
                msg = f"FNO2D: shape={y_hat.shape}, loss={op.train_losses_[-1]:.4g}"

            elif op_type == "TFNO2D":
                from naviertwin.core.operator_learning.fno.tfno import TFNO2D

                X = rng.standard_normal((max(n // 2, 4), 16, 16, 1)).astype(np.float32)
                Y = X ** 2
                op = TFNO2D(
                    in_channels=1, out_channels=1, modes1=4, modes2=4,
                    width=8, rank=4, n_layers=2, max_epochs=epochs,
                )
                op.fit({"inputs": X, "outputs": Y})
                y_hat = op.predict({"x": X[:2]})
                msg = (
                    f"TFNO2D: shape={y_hat.shape}, params={op.param_count()}, "
                    f"loss={op.train_losses_[-1]:.4g}"
                )

            elif op_type == "DeepONet":
                from naviertwin.core.operator_learning.deeponet.deeponet import DeepONet

                m, q = 16, 10
                Bx = rng.standard_normal((n, m)).astype(np.float32)
                Tx = rng.standard_normal((q, 2)).astype(np.float32)
                Y = np.tanh(Bx @ rng.standard_normal((m, q)).astype(np.float32))
                op = DeepONet(
                    branch_in=m, trunk_in=2, hidden=16, latent=8, max_epochs=epochs,
                )
                op.fit({"branch_inputs": Bx, "trunk_inputs": Tx, "outputs": Y})
                y_hat = op.predict({"branch_inputs": Bx[:2]})
                msg = f"DeepONet: shape={y_hat.shape}, loss={op.train_losses_[-1]:.4g}"

            elif op_type == "UNet2D":
                from naviertwin.core.operator_learning.unet.unet import UNet2D

                X = rng.standard_normal((max(n // 2, 4), 16, 16, 1)).astype(np.float32)
                Y = X ** 2
                op = UNet2D(
                    in_channels=1, out_channels=1, base_ch=8, max_epochs=epochs,
                )
                op.fit({"inputs": X, "outputs": Y})
                y_hat = op.predict({"x": X[:2]})
                msg = f"UNet2D: shape={y_hat.shape}, loss={op.train_losses_[-1]:.4g}"

            elif op_type == "WNO1D":
                try:
                    from naviertwin.core.operator_learning.fno.wno import WNO1D
                except Exception as exc:  # noqa: BLE001
                    self._log(f"[WARN] WNO1D import 실패: {exc}")
                    return

                X = rng.standard_normal((n, 64, 1)).astype(np.float32)
                Y = X ** 2
                op = WNO1D(
                    in_channels=1, out_channels=1, width=8,
                    wavelet="db2", level=2, n_layers=2, max_epochs=epochs,
                )
                try:
                    op.fit({"inputs": X, "outputs": Y})
                except RuntimeError as exc:
                    self._log(f"[WARN] WNO1D 학습 실패 (pywt 필요): {exc}")
                    return
                y_hat = op.predict({"x": X[:2]})
                msg = f"WNO1D: shape={y_hat.shape}, loss={op.train_losses_[-1]:.4g}"

            else:
                self._log(f"[WARN] 알 수 없는 연산자: {op_type}")
                return

            self._log(f"[{op_type}] 학습 완료 — {msg}")
            self._update_loss_curve(op_type, op)
            self.model_trained.emit(op_type.lower(), op)

        except Exception as exc:  # noqa: BLE001
            self._log(f"[ERROR] {op_type} 학습 실패: {exc}")

    def _extract_snapshots_from_dataset(self, field: str) -> np.ndarray:
        """CFDDataset에서 (n_features, n_samples) 스냅샷 행렬을 구성한다."""
        if self._dataset is None:
            raise RuntimeError("dataset이 설정되지 않았습니다.")
        return self._dataset.extract_field_snapshots(field)

    @staticmethod
    def _build_explainability_metadata(X_train: np.ndarray) -> dict[str, object]:
        """SHAP GUI가 사용할 bounded background metadata를 구성한다."""
        X_arr = np.asarray(X_train, dtype=float)
        if X_arr.ndim != 2:
            X_arr = X_arr.reshape(len(X_arr), -1)
        n_background = min(32, X_arr.shape[0])
        n_features = X_arr.shape[1] if X_arr.ndim == 2 else 0
        return {
            "background": X_arr[:n_background].copy(),
            "feature_names": [f"param_{i}" for i in range(n_features)],
            "output_index": 0,
        }

    @staticmethod
    def _as_positive_int(value: object) -> int | None:
        """양의 정수로 해석 가능한 값을 반환한다."""
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    def _update_loss_curve(self, label: str, model: object) -> None:
        """모델의 train_losses_를 LossCurveWidget에 반영한다."""
        raw_losses = getattr(model, "train_losses_", None)
        if raw_losses is None:
            return
        try:
            losses = [
                float(value)
                for value in raw_losses
                if np.isfinite(float(value))
            ]
        except (TypeError, ValueError):
            return
        if not losses:
            return
        self._loss_series[label] = losses
        self._loss_curve.set_losses(self._loss_series)
        self._log(f"Loss curve 업데이트: {label} ({len(losses)} epochs)")

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
