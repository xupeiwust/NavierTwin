"""Round 149 — device 유틸."""

from __future__ import annotations

import pytest


class TestDevice:
    def test_auto(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.utils.device import get_device

        d = get_device("auto")
        assert str(d) in ("cpu", "cuda", "mps", "cuda:0")

    def test_cpu_force(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.utils.device import get_device

        assert str(get_device("cpu")) == "cpu"

    def test_move_nested(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.utils.device import get_device, move_to

        d = get_device("cpu")
        obj = {"x": torch.zeros(3), "y": [torch.ones(2), (torch.zeros(1),)]}
        out = move_to(obj, d)
        assert out["x"].device.type == "cpu"

    def test_memory_info(self) -> None:
        from naviertwin.utils.device import available_memory_mb

        info = available_memory_mb()
        assert "cuda" in info
