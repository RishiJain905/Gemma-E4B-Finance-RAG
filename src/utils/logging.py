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
from typing import Any


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
