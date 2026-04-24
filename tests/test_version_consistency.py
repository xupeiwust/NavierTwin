"""버전 문자열 정합성 테스트."""

from __future__ import annotations

from argparse import _VersionAction

from naviertwin import __version__
from naviertwin.main import _build_parser


def test_cli_help_contains_package_version() -> None:
    """CLI 도움말에 패키지 버전이 포함되는지 확인한다."""
    parser = _build_parser()
    help_text = parser.format_help()
    assert __version__ in help_text


def test_version_action_uses_package_version() -> None:
    """--version 액션이 패키지 버전을 사용하는지 확인한다."""
    parser = _build_parser()
    version_actions = [a for a in parser._actions if isinstance(a, _VersionAction)]
    assert version_actions, "버전 액션(--version)을 찾을 수 없습니다."
    assert any(__version__ in (a.version or "") for a in version_actions)
