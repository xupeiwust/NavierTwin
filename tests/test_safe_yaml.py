"""Round 247 — safe YAML."""

from __future__ import annotations

from pathlib import Path


class TestYAML:
    def test_load_basic(self) -> None:
        from naviertwin.utils.safe_yaml import safe_yaml_load

        d = safe_yaml_load("a: 1\nb: 2.5\nc: hello\n")
        assert d["a"] == 1
        assert abs(d["b"] - 2.5) < 1e-12
        assert d["c"] == "hello"

    def test_file(self, tmp_path: Path) -> None:
        from naviertwin.utils.safe_yaml import (
            safe_yaml_dump,
            safe_yaml_load_file,
        )

        p = tmp_path / "c.yaml"
        p.write_text(safe_yaml_dump({"x": 1, "y": True}))
        d = safe_yaml_load_file(p)
        assert d["x"] == 1
        assert d["y"] is True

    def test_null_and_bool(self) -> None:
        from naviertwin.utils.safe_yaml import safe_yaml_load

        d = safe_yaml_load("a: null\nb: true\nc: false\n")
        assert d["a"] is None
        assert d["b"] is True
        assert d["c"] is False
