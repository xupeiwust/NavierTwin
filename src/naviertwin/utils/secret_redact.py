"""Secret redaction — mask common patterns (api keys, tokens) in logs.

Examples:
    >>> from naviertwin.utils.secret_redact import redact
    >>> redact('token=abcdef123456')
    'token=***REDACTED***'
"""

from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"(token|api[_-]?key|secret|password|passwd|aws_secret_access_key)\s*[=:]\s*[\w\-./]+",
               re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----"),
]


def redact(text: str) -> str:
    out = text
    for pat in _PATTERNS:
        out = pat.sub(lambda m: m.group(0).split("=")[0].split(":")[0].rstrip()
                       + "=***REDACTED***" if "=" in m.group(0) or ":" in m.group(0)
                       else "***REDACTED***", out)
    return out


__all__ = ["redact"]
