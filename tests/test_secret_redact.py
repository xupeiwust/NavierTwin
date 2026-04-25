"""Round 445 — secret redaction."""

from __future__ import annotations


class TestRedact:
    def test_token(self) -> None:
        from naviertwin.utils.secret_redact import redact

        assert "abcdef" not in redact("token=abcdef123")
        assert "REDACTED" in redact("token=abcdef123")

    def test_bearer(self) -> None:
        from naviertwin.utils.secret_redact import redact

        out = redact("Authorization: Bearer abc.def.ghi")
        assert "abc.def.ghi" not in out

    def test_clean_passes_through(self) -> None:
        from naviertwin.utils.secret_redact import redact

        assert redact("hello world") == "hello world"
