"""GUI smoke tests for customer report export from ExportPanel."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("jinja2")


@dataclass
class DummyDataset:
    """Minimal dataset shape needed for report generation."""

    n_points: int = 12
    n_cells: int = 5
    n_time_steps: int = 3
    field_names: list[str] = field(default_factory=lambda: ["U", "p"])
    metadata: dict[str, Any] = field(default_factory=dict)


def _dataset_with_metrics() -> DummyDataset:
    return DummyDataset(
        metadata={
            "project_metadata": {
                "engine": {
                    "surrogate": {
                        "validation_metrics": {
                            "rmse": 0.0123,
                            "r2": 0.9876,
                        }
                    }
                }
            }
        }
    )


def test_export_panel_report_format_visible(qtbot) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)

    formats = [
        panel._format_combo.itemText(i)
        for i in range(panel._format_combo.count())
    ]
    assert any("보고서" in item for item in formats)


def test_export_panel_exports_html_report(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)
    emitted: list[str] = []
    panel.export_done.connect(emitted.append)
    panel.set_dataset(_dataset_with_metrics())  # type: ignore[arg-type]
    panel._format_combo.setCurrentIndex(4)
    out = tmp_path / "customer_report.html"
    panel._path_edit.setText(str(out))

    panel._export()

    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "NavierTwin CFD digital-twin report" in html
    assert "0.0123" in html
    assert emitted == [str(out)]


def test_export_panel_report_adds_html_suffix(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)
    panel.set_dataset(DummyDataset())  # type: ignore[arg-type]
    panel._format_combo.setCurrentIndex(4)
    panel._path_edit.setText(str(tmp_path / "handoff"))

    panel._export()

    assert (tmp_path / "handoff.html").exists()
