"""GUI project-open path tests for customer-facing .ntwin workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pyvista = pytest.importorskip("pyvista", reason="pyvista is required for .ntwin GUI tests")
pytest.importorskip("h5py", reason="h5py is required for .ntwin GUI tests")


def _make_dataset() -> object:
    """Create a minimal CFDDataset that round-trips through .ntwin."""
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    cell_types = np.array([10], dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.point_data["p"] = np.arange(mesh.n_points, dtype=np.float32)
    return CFDDataset(mesh=mesh, time_steps=[0.0], field_names=["p"])


def _write_ntwin(tmp_path: Path) -> Path:
    from naviertwin.core.export.ntwin_format import save_dataset

    path = tmp_path / "customer_project.ntwin"
    save_dataset(_make_dataset(), path)
    return path


def test_export_panel_load_project_path_emits_project_loaded(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    path = _write_ntwin(tmp_path)
    panel = ExportPanel()
    qtbot.addWidget(panel)
    emitted: list[tuple[object, object | None]] = []
    panel.project_loaded.connect(lambda dataset, engine: emitted.append((dataset, engine)))

    assert panel.load_project_path(path) is True

    assert panel._dataset is not None
    assert panel._path_edit.text() == str(path)
    assert len(emitted) == 1
    assert emitted[0][0].n_points == 4  # type: ignore[union-attr]
    assert emitted[0][1] is None


def test_main_window_open_selected_ntwin_routes_to_project_loader(
    qtbot, tmp_path: Path
) -> None:
    from naviertwin.gui.main_window import MainWindow

    path = _write_ntwin(tmp_path)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._open_selected_path(path)

    assert win._latest_dataset is not None
    assert win._export_panel._dataset is not None
    assert win._export_panel._path_edit.text() == str(path)
    assert win._import_panel._path_edit.text() == ""
    assert "프로젝트 로드 완료" in win._status_label.text()


def test_main_window_open_selected_cfd_routes_to_import_panel(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.main_window import MainWindow

    path = tmp_path / "case.su2"
    path.write_text("% minimal placeholder\n", encoding="utf-8")
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._open_selected_path(path)

    assert win._import_panel._path_edit.text() == str(path)
    assert win._tabs.currentWidget() is win._import_panel
