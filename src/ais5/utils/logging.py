"""Logging configured once via Rich for nice console output."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Idempotent logging setup. Safe to call from notebooks and scripts."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, show_path=False)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=10_485_760, backupCount=3)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        handlers.append(fh)

    logging.basicConfig(
        level=os.environ.get("AIS5_LOG_LEVEL", level),
        format="%(message)s",
        handlers=handlers,
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
