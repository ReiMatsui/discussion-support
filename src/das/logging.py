"""構造化ログの初期化。

エージェントごとのトレースを残せるよう structlog を使う。
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """structlog を初期化する。複数回呼ばれても安全。"""

    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stderr,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """名前付きロガーを返す。"""

    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
