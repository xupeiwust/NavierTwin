"""R648 — PostProcessPanel GUI smoke tests."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def app(qtbot):
    """qtbot fixture for Qt event loop."""
    return qtbot


class TestPanelConstruction:
    def test_panel_creates(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # op 리스트가 비어 있지 않음
        assert panel._op_list.count() > 0

    def test_category_filter(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        n_total = panel._op_list.count()
        # 특정 카테고리로 변경
        panel._category_combo.setCurrentText("statistics")
        n_filtered = panel._op_list.count()
        assert 0 < n_filtered <= n_total

    def test_op_selection_updates_description(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # 첫 op 선택
        panel._op_list.setCurrentRow(0)
        desc = panel._desc_label.text()
        assert len(desc) > 0
        assert "[" in desc

    def test_smoke_run_succeeds(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)

        # 모든 op에 대해 smoke 실행 시도
        for i in range(panel._op_list.count()):
            panel._op_list.setCurrentRow(i)
            op_name = panel._op_list.currentItem().text()
            panel._on_run_clicked()
            result_text = panel._result_text.toPlainText()
            assert "실행 실패" not in result_text, f"{op_name} failed: {result_text}"

    def test_signal_emitted(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)

        emitted: list = []
        panel.operation_done.connect(lambda name, res: emitted.append((name, res)))

        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()

        assert len(emitted) == 1
        name, result = emitted[0]
        assert isinstance(result, dict)


class TestSummarizeResult:
    def test_array_short(self) -> None:
        import numpy as np

        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        text = PostProcessPanel._summarize_result({"x": np.array([1.0, 2.0])})
        assert "1.0" in text

    def test_array_long_summarized(self) -> None:
        import numpy as np

        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        text = PostProcessPanel._summarize_result(
            {"x": np.zeros(100)},
        )
        assert "shape" in text and "mean" in text

    def test_dict_value(self) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        text = PostProcessPanel._summarize_result(
            {"box": {"median": 0.0, "Q1": -1.0, "Q3": 1.0}},
        )
        assert "keys" in text

    def test_scalar_value(self) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        text = PostProcessPanel._summarize_result({"r2": 0.95})
        assert "0.95" in text


class TestSmokeKwargs:
    def test_all_facade_ops_have_smoke_kwargs(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        facade = PostProcessFacade()
        for op_name in facade.list_operations():
            kwargs = PostProcessPanel._build_smoke_kwargs(op_name)
            assert isinstance(kwargs, dict)

    def test_undefined_op_raises(self) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        with pytest.raises(ValueError, match="smoke 데이터"):
            PostProcessPanel._build_smoke_kwargs("undefined_op_xyz")
