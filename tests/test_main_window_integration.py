"""Round 17 — MainWindow i18n + 비교 탭 통합 (import-only)."""

from __future__ import annotations

import pytest


class TestMainWindowAssembly:
    """QApplication 필요하므로 import/attribute 수준 검증만."""

    def test_language_switch_method_exists(self) -> None:
        pytest.importorskip("PySide6")
        from naviertwin.gui.main_window import MainWindow

        assert hasattr(MainWindow, "set_language")
        assert hasattr(MainWindow, "update_compare_dashboard")

    def test_translator_inlined(self) -> None:
        from naviertwin.utils.i18n import Translator

        t = Translator(lang="en")
        assert t("panel.import") == "Import"
        t.set_language("ko")
        assert "불러오기" in t("panel.import")
