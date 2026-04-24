"""내보내기 패널 — .ntwin HDF5, VTK, CSV 내보내기 및 프로젝트 저장/복원.

Signals:
    export_done(str): 내보내기 완료 시 저장된 파일 경로 발생.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


class ExportPanel(QWidget):
    """내보내기 탭 패널.

    Signals:
        export_done: 내보내기 완료 시 파일 경로와 함께 발생.
    """

    export_done = Signal(str)
    project_loaded = Signal(object, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._engine: Optional[object] = None
        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Export")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        subtitle = QLabel("CFD 데이터, 모델, 프로젝트를 다양한 포맷으로 내보냅니다.")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)

        # 출력 경로
        path_group = QGroupBox("출력 경로")
        path_layout = QHBoxLayout(path_group)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("저장할 파일 경로...")
        path_layout.addWidget(self._path_edit)
        browse_btn = QPushButton("찾기")
        browse_btn.setFixedWidth(60)
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(browse_btn)
        layout.addWidget(path_group)

        # 내보내기 옵션
        options_group = QGroupBox("내보내기 옵션")
        options_form = QFormLayout(options_group)

        self._format_combo = QComboBox()
        self._format_combo.addItems([
            ".ntwin (HDF5 — 프로젝트 전체)",
            ".vtu (VTK UnstructuredGrid)",
            ".vtk (레거시 VTK)",
            ".csv (점 데이터)",
        ])
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        options_form.addRow("포맷:", self._format_combo)

        self._include_mesh_cb = QCheckBox("메쉬 포함")
        self._include_mesh_cb.setChecked(True)
        options_form.addRow("", self._include_mesh_cb)

        self._include_model_cb = QCheckBox("모델(TwinEngine) 포함")
        self._include_model_cb.setChecked(True)
        options_form.addRow("", self._include_model_cb)

        self._compress_cb = QCheckBox("HDF5 압축 (gzip level 4)")
        self._compress_cb.setChecked(True)
        options_form.addRow("", self._compress_cb)

        layout.addWidget(options_group)

        # 프로젝트 저장/복원
        project_group = QGroupBox("프로젝트 (.ntwin)")
        project_layout = QHBoxLayout(project_group)

        self._save_project_btn = QPushButton("프로젝트 저장")
        self._save_project_btn.clicked.connect(self._save_project)
        project_layout.addWidget(self._save_project_btn)

        self._load_project_btn = QPushButton("프로젝트 열기")
        self._load_project_btn.clicked.connect(self._load_project)
        project_layout.addWidget(self._load_project_btn)

        layout.addWidget(project_group)

        # 내보내기 버튼
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._export_btn = QPushButton("내보내기")
        self._export_btn.setObjectName("primaryButton")
        self._export_btn.clicked.connect(self._export)
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

        # 로그
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group, stretch=1)

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_dataset(self, dataset: CFDDataset) -> None:
        """내보낼 CFDDataset을 설정한다."""
        self._dataset = dataset
        # dataset이 바뀌면 기존 engine은 같은 데이터 기준이 아닐 수 있으므로 초기화한다.
        self._engine = None
        self._log(f"Dataset 설정: {dataset.n_points} pts, {dataset.n_cells} cells")
        self._log("TwinEngine 상태 초기화")

    def set_engine(self, engine: object) -> None:
        """내보낼 TwinEngine을 설정한다."""
        self._engine = engine
        self._log(f"TwinEngine 설정: {type(engine).__name__}")

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _browse_path(self) -> None:
        fmt_idx = self._format_combo.currentIndex()
        filters = [
            "NavierTwin Project (*.ntwin)",
            "VTK UnstructuredGrid (*.vtu)",
            "VTK Legacy (*.vtk)",
            "CSV (*.csv)",
        ]
        path, _ = QFileDialog.getSaveFileName(self, "저장 경로 선택", "", filters[fmt_idx])
        if path:
            self._path_edit.setText(path)

    def _on_format_changed(self, idx: int) -> None:
        is_hdf5 = idx == 0
        self._include_model_cb.setEnabled(is_hdf5)
        self._compress_cb.setEnabled(is_hdf5)

    def _export(self) -> None:
        path_str = self._path_edit.text().strip()
        if not path_str:
            self._log("[WARN] 출력 경로를 입력하세요.")
            return

        path = Path(path_str)
        fmt_idx = self._format_combo.currentIndex()

        try:
            if fmt_idx == 0:
                self._export_ntwin(path)
            elif fmt_idx in (1, 2):
                self._export_vtk(path)
            else:
                self._export_csv(path)
        except Exception as exc:
            self._log(f"[ERROR] 내보내기 실패: {exc}")

    def _export_ntwin(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        from naviertwin.core.export.ntwin_format import save_dataset

        compression = "gzip" if self._compress_cb.isChecked() else None
        save_dataset(self._dataset, path, compression=compression)
        self._log(f"✓ .ntwin 저장: {path}")
        metadata = self._build_project_metadata(path)
        self._write_metadata_sidecar(path, metadata)
        self._attach_project_metadata(metadata)
        if self._include_model_cb.isChecked():
            if self._engine is None:
                self._log("[WARN] 모델 포함이 선택됐지만 TwinEngine이 없습니다.")
            elif hasattr(self._engine, "save"):
                model_path = path.with_suffix(".engine.pkl")
                self._engine.save(model_path)  # type: ignore[union-attr]
                self._log(f"✓ TwinEngine 저장: {model_path}")
                metadata["has_engine"] = True
                metadata["engine_path"] = str(model_path)
                self._write_metadata_sidecar(path, metadata)
                self._attach_project_metadata(metadata)
            else:
                self._log("[WARN] 현재 엔진은 save()를 지원하지 않습니다.")
        self.export_done.emit(str(path))

    def _export_vtk(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return

        mesh = self._dataset.mesh
        mesh.save(str(path))
        self._log(f"✓ VTK 저장: {path}")
        self.export_done.emit(str(path))

    def _export_csv(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        import numpy as np

        mesh = self._dataset.mesh
        pts = mesh.points
        header = "x,y,z"
        rows = [pts]
        for name in self._dataset.field_names:
            if name in mesh.point_data:
                arr = np.array(mesh.point_data[name])
                if arr.ndim == 1:
                    header += f",{name}"
                    rows.append(arr.reshape(-1, 1))
                else:
                    for i in range(arr.shape[1]):
                        header += f",{name}_{i}"
                    rows.append(arr)
        data = np.hstack([r.reshape(len(pts), -1) for r in rows])
        np.savetxt(str(path), data, delimiter=",", header=header, comments="")
        self._log(f"✓ CSV 저장: {path} ({len(pts)} rows)")
        self.export_done.emit(str(path))

    def _save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "프로젝트 저장", "project.ntwin", "NavierTwin Project (*.ntwin)"
        )
        if not path:
            return
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        try:
            self._path_edit.setText(path)
            self._export_ntwin(Path(path))
        except Exception as exc:
            self._log(f"[ERROR] 프로젝트 저장 실패: {exc}")

    def _load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "", "NavierTwin Project (*.ntwin)"
        )
        if not path:
            return
        try:
            from naviertwin.core.export.ntwin_format import load_dataset
            self._engine = None
            dataset = load_dataset(Path(path))
            self._dataset = dataset
            self._log(f"✓ 프로젝트 로드: {path} ({dataset.n_points} pts)")

            metadata = self._read_metadata_sidecar(Path(path))
            if metadata is not None:
                self._attach_project_metadata(metadata)
                self._apply_loaded_metadata_to_dataset(metadata)

            model_path = Path(path).with_suffix(".engine.pkl")
            if model_path.exists():
                from naviertwin.core.digital_twin.twin_engine import TwinEngine

                self._engine = TwinEngine.load(model_path)
                self._log(f"✓ TwinEngine 로드: {model_path}")
            else:
                self._log("ℹ TwinEngine 파일 없음 (.engine.pkl)")
            if metadata is not None and self._engine is not None:
                self._attach_project_metadata(metadata)
            self.project_loaded.emit(self._dataset, self._engine)
        except Exception as exc:
            self._log(f"[ERROR] 프로젝트 로드 실패: {exc}")

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _build_project_metadata(self, path: Path) -> dict[str, Any]:
        """현재 프로젝트 상태를 JSON 직렬화 가능한 메타데이터로 구성한다."""
        assert self._dataset is not None
        metadata: dict[str, Any] = {
            "project_path": str(path),
            "include_model": self._include_model_cb.isChecked(),
            "has_engine": self._engine is not None and self._include_model_cb.isChecked(),
            "engine_path": None,
            "dataset": {
                "n_points": int(self._dataset.n_points),
                "n_cells": int(self._dataset.n_cells),
                "n_time_steps": int(self._dataset.n_time_steps),
                "field_names": list(self._dataset.field_names),
            },
        }
        if self._engine is not None:
            metadata["engine"] = self._extract_engine_metadata(self._engine)
        return metadata

    def _extract_engine_metadata(self, engine: object) -> dict[str, Any]:
        """TwinEngine의 안전한 메타데이터를 추출한다."""
        reducer = getattr(engine, "reducer", None)
        surrogate = getattr(engine, "surrogate", None)
        return {
            "reducer_type": getattr(engine, "reducer_type", None),
            "surrogate_type": getattr(engine, "surrogate_type", None),
            "n_modes": int(getattr(engine, "n_modes", 0) or 0),
            "reducer": getattr(reducer, "training_metadata", None),
            "surrogate": getattr(surrogate, "training_metadata", None),
        }

    def _write_metadata_sidecar(self, path: Path, metadata: dict[str, Any]) -> None:
        """프로젝트 메타데이터 sidecar JSON을 기록한다."""
        sidecar = path.with_suffix(".meta.json")
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log(f"✓ 메타데이터 저장: {sidecar}")

    def _read_metadata_sidecar(self, path: Path) -> dict[str, Any] | None:
        """프로젝트 메타데이터 sidecar JSON을 읽는다."""
        sidecar = path.with_suffix(".meta.json")
        if not sidecar.exists():
            return None
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._log(f"✓ 메타데이터 로드: {sidecar}")
                return data
        except Exception as exc:
            self._log(f"[WARN] 메타데이터 로드 실패: {exc}")
        return None

    def _attach_project_metadata(self, metadata: dict[str, Any]) -> None:
        """로드/저장된 메타데이터를 dataset/engine 객체에 부착한다."""
        if self._dataset is not None:
            try:
                self._dataset.metadata["project_metadata"] = metadata
            except Exception:
                pass
        if self._engine is not None:
            try:
                setattr(self._engine, "project_metadata", metadata)
            except Exception:
                pass

    def _apply_loaded_metadata_to_dataset(self, metadata: dict[str, Any]) -> None:
        """sidecar 메타데이터를 dataset 메타데이터에 반영한다."""
        if self._dataset is not None:
            try:
                self._dataset.metadata["project_metadata"] = metadata
            except Exception:
                pass
