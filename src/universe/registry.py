"""src/universe/registry.py
Validated coordinator for atomic security-universe reconciliation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from src.storage.store import Store

from .models import (
    SnapshotValidationError,
    UniverseRecord,
    UniverseRefreshResult,
    normalize_symbol,
)

logger = logging.getLogger(__name__)


class UniverseRegistry:
    """Validate provider snapshots and apply them through the Store transaction seam."""

    CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "universe.yaml"

    def __init__(
        self,
        store: Store,
        *,
        config_path: Optional[Path] = None,
        minimums: Optional[dict[str, int]] = None,
    ) -> None:
        self.store = store
        self.config_path = config_path or self.CONFIG_PATH
        self.minimums = self._load_minimums()
        if minimums:
            self.minimums.update({str(key): int(value) for key, value in minimums.items()})

    def _load_minimums(self) -> dict[str, int]:
        try:
            with open(self.config_path, encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file) or {}
        except FileNotFoundError:
            logger.warning("Universe configuration not found: %s", self.config_path)
            return {}
        minimums: dict[str, int] = {}
        for provider in (config.get("providers") or {}).values():
            if not isinstance(provider, dict):
                continue
            key = provider.get("index_code") or provider.get("source")
            minimum = provider.get("min_constituents", provider.get("min_records"))
            if key and minimum is not None:
                minimums[str(key)] = int(minimum)
        return minimums

    @staticmethod
    def _coerce_rows(rows: list[UniverseRecord | dict]) -> list[UniverseRecord]:
        records = []
        for row in rows:
            if isinstance(row, UniverseRecord):
                records.append(row)
                continue
            if isinstance(row, dict):
                try:
                    records.append(UniverseRecord(**row))
                except TypeError as exc:
                    raise SnapshotValidationError(f"invalid universe row: {exc}") from exc
                continue
            raise SnapshotValidationError("universe rows must be records or dictionaries")
        return records

    def refresh(
        self,
        source: str,
        observed_at: str,
        rows: list[UniverseRecord | dict],
    ) -> UniverseRefreshResult:
        """Validate and reconcile a complete provider snapshot in one transaction."""
        source = str(source or "").strip().lower()
        if not source:
            raise SnapshotValidationError("source is required")
        records = self._coerce_rows(rows)
        if not records:
            raise SnapshotValidationError(f"{source} snapshot is empty")
        if any(not record.symbol.strip() or not record.company_name.strip() for record in records):
            raise SnapshotValidationError(
                f"{source} snapshot contains an empty symbol or company name"
            )
        symbols = [normalize_symbol(record.symbol) for record in records]
        duplicates = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
        if duplicates:
            raise SnapshotValidationError(
                f"{source} snapshot contains duplicate symbols: {', '.join(duplicates)}"
            )
        index_codes = {record.index_code for record in records if record.index_code}
        if not index_codes <= {"sp500", "nasdaq100"}:
            raise SnapshotValidationError("snapshot contains an unsupported index code")
        if len(index_codes) > 1:
            raise SnapshotValidationError("one provider snapshot cannot mix index codes")

        minimum_key = next(iter(index_codes), source)
        minimum = int(self.minimums.get(minimum_key, 1))
        if len(records) < minimum:
            raise SnapshotValidationError(
                f"{source} snapshot must contain at least {minimum} valid records; "
                f"got {len(records)}"
            )

        result = self.store.upsert_universe_snapshot(source, observed_at, records)
        return UniverseRefreshResult.from_dict(result)
