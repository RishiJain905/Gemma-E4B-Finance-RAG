"""
src/utils/logging.py
Structured logging utilities for the ingestion pipeline.

Each log line carries the data source plus arbitrary structured fields
(ticker, operation, duration_ms, status, error, ...) appended as
key=value pairs, so log output stays greppable and dependency-free.

Usage:
    from src.utils.logging import IngestionLogger
    log = IngestionLogger("fred")
    log.info("Fetched GDP", series_id="GDP", value=28.5)
    log.error("FRED API error", series_id="GDP", error="HTTP 429")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any


def setup_file_logging(log_dir: str = "logs", level: int = logging.INFO):
    """Configure rotating file logging for the RAG system.

    Adds two handlers to the root logger:
      - ``rag-system.log`` (10 MB x 5 backups) at ``level``
      - ``rag-errors.log`` (10 MB x 3 backups) at WARNING and above

    Sensitive values (API keys, full filing text) should never be passed to
    log calls — these handlers persist whatever is logged.

    Returns:
        (main_handler, error_handler)
    """
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = RotatingFileHandler(
        os.path.join(log_dir, "rag-system.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)

    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "rag-errors.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.WARNING)

    root = logging.getLogger()
    root.addHandler(handler)
    root.addHandler(error_handler)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)

    return handler, error_handler


class IngestionLogger:
    """Structured logger that tags every line with its data source."""

    def __init__(self, source: str):
        self.source = source
        self._logger = logging.getLogger(f"ingestion.{source}")

    # ── Formatting ─────────────────────────────────────

    def _format(self, message: str, fields: dict[str, Any]) -> str:
        """Render `message [source=... key=value ...]`."""
        parts = [f"source={self.source}"]
        for key, value in fields.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return f"{message} [{' '.join(parts)}]"

    # ── Levels ─────────────────────────────────────────

    def info(self, message: str, **fields: Any) -> None:
        self._logger.info(self._format(message, fields))

    def warning(self, message: str, **fields: Any) -> None:
        self._logger.warning(self._format(message, fields))

    def error(self, message: str, **fields: Any) -> None:
        self._logger.error(self._format(message, fields))
