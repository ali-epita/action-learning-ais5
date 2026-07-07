"""Logging configured once via Rich for nice console output.

Priority for the level: explicit setup_logging(level=...) argument, then the
AIS5_LOG_LEVEL env var, then INFO. Many modules call get_logger() at import
time, which configures logging implicitly; a later EXPLICIT setup_logging call
(e.g. PointCast's --log-level/--log-file flags) must still win, so implicit
configuration never locks out an explicit one.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED: str | None = None  # None | "implicit" | "explicit"


def _coerce_level(value: int | str) -> int:
    """Accept 'debug', 'DEBUG', '10', or 10; fall back to INFO on garbage so a
    malformed AIS5_LOG_LEVEL can never crash every command at import time."""
    if isinstance(value, int):
        return value
    s = str(value).strip().upper()
    if s.isdigit():
        return int(s)
    level = getattr(logging, s, None)
    if isinstance(level, int):
        return level
    return logging.INFO


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Idempotent logging setup. Safe to call from notebooks and scripts."""
    global _CONFIGURED
    explicit = level != logging.INFO or log_file is not None
    if _CONFIGURED == "explicit" or (_CONFIGURED == "implicit" and not explicit):
        return

    if explicit:
        resolved = _coerce_level(level)
    else:
        resolved = _coerce_level(os.environ.get("AIS5_LOG_LEVEL", level))

    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, show_path=False)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=10_485_760, backupCount=3)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        handlers.append(fh)

    logging.basicConfig(
        level=resolved,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )
    _CONFIGURED = "explicit" if explicit else "implicit"


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
