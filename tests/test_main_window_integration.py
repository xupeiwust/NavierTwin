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

    def test_startup_applies_language_and_theme_config(
        self, qtbot, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        import naviertwin.gui.main_window as main_window
        from naviertwin.utils.config import NavierTwinConfig, save_config

        config_path = tmp_path / "config.json"
        save_config(NavierTwinConfig(language="en", theme="light"), config_path)
        loaded_themes: list[str] = []

        def capture_stylesheet(theme: str = "dark") -> str:
            loaded_themes.append(theme)
            return f"/* {theme} theme */"

        app = QApplication.instance()
        original_stylesheet = app.styleSheet()  # type: ignore[union-attr]
        monkeypatch.setattr(main_window, "_load_stylesheet", capture_stylesheet)
        try:
            win = main_window.MainWindow(
                confirm_on_close=False,
                config_path=config_path,
            )
            qtbot.addWidget(win)

            assert loaded_themes == ["light"]
            assert win._t.lang == "en"
            assert win.windowTitle() == "NavierTwin — CFD Digital Twin"
            assert win._tabs.tabText(0) == "① Import"
            assert app.styleSheet() == "/* light theme */"  # type: ignore[union-attr]
        finally:
            app.setStyleSheet(original_stylesheet)  # type: ignore[union-attr]
