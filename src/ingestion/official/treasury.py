"""src/ingestion/official/treasury.py
U.S. Treasury daily nominal, real-yield, and bill-rate ingestion.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import requests

from src.storage.store import Store

from . import (
    OfficialProviderError,
    as_timestamp,
    catalog_entries,
    finish,
    make_observation,
    metadata_values,
    parse_number,
    persist_records,
    request_payload,
    result,
    row_value,
    source_url,
    status_for_error,
    utc_now,
)

logger = logging.getLogger(__name__)


class TreasuryIngestor:
    """Fetch and normalize only the Treasury series listed in the catalog."""

    SOURCE_NAME = "treasury"

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        coverage_resolver: Optional[object] = None,
        api_key: Optional[str] = None,
        base_url: str = "",
        http_get: Optional[Callable[..., Any]] = None,
        timeout: float = 30.0,
        now_fn: Callable[[], object] = utc_now,
    ) -> None:
        self.store = store or Store()
        self.coverage = coverage_resolver
        self.base_url = str(base_url).rstrip("/")
        self.http_get = http_get or requests.get
        self.timeout = max(float(timeout), 0.1)
        self.now_fn = now_fn

    def parse(
        self,
        payload: object,
        *,
        entry_name: Optional[str] = None,
        accessed_at: Optional[str] = None,
    ) -> list[object]:
        """Parse Treasury CSV/API rows into observations for selected series."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        rows = self._rows(payload)
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for row in rows:
            observation_date = row_value(row, "Date", "record_date", "observation_date")
            if not observation_date:
                raise ValueError("Treasury row requires Date")
            vintage = row_value(row, "vintage_at", "vintage_date", "retrieved_at") or accessed
            source_id = str(row_value(row, "source_id", "id") or f"treasury-{observation_date}")
            for entry in entries:
                value = parse_number(row_value(row, str(entry["field"])))
                if value is None:
                    continue
                endpoint = source_url(str(entry["endpoint"]), str(entry["endpoint"]))
                records.append(
                    make_observation(
                        source_name=self.SOURCE_NAME,
                        source_category=str(entry["source_category"]),
                        metric_id=str(entry["name"]),
                        series_id=str(entry["series_id"]),
                        value=value,
                        unit=str(entry["unit"]),
                        frequency=str(entry["frequency"]),
                        observation_period=observation_date,
                        vintage=vintage,
                        provider_record_id=source_id,
                        source_url_value=endpoint,
                        accessed_at=accessed,
                        published_at=vintage,
                        period_start_value=observation_date,
                        metadata=metadata_values(
                            dataset_id=str(entry["dataset_id"]),
                            series_id=str(entry["series_id"]),
                            frequency=str(entry["frequency"]),
                        ),
                    )
                )
        return records

    def ingest(
        self,
        payload: object | None = None,
        *,
        entry_names: Optional[list[str]] = None,
    ) -> dict[str, object]:
        """Ingest cataloged Treasury feeds, with fixture payload injection for offline runs."""
        output = result(self.SOURCE_NAME, "official_rates")
        entries = catalog_entries(self.SOURCE_NAME, entry_names)
        accessed = self._now_timestamp()
        try:
            if payload is not None:
                persist_records(self.store, self.parse(payload, accessed_at=accessed), output)
                output["pages"] = 1
            else:
                for entry in entries:
                    fetched = request_payload(self.http_get, str(entry["endpoint"]), timeout=self.timeout)
                    output["requests"] = int(output["requests"]) + 1
                    persist_records(
                        self.store,
                        self.parse(fetched, entry_name=str(entry["name"]), accessed_at=accessed),
                        output,
                    )
                    output["pages"] = int(output["pages"]) + 1
        except OfficialProviderError as exc:
            output["status"] = status_for_error(exc)
            output["error_class"] = exc.error_class
            output["retry_after"] = exc.retry_after
            output["errors"].append(str(exc))
        except (TypeError, ValueError) as exc:
            output["malformed"] = int(output["malformed"]) + 1
            output["errors"].append(str(exc))
        return finish(self.store, output)

    @staticmethod
    def _rows(payload: object) -> list[dict[str, object]]:
        from . import rows_from_payload

        return rows_from_payload(payload)

    def _now_timestamp(self) -> str:
        return as_timestamp(self.now_fn(), "now")


TreasuryAdapter = TreasuryIngestor
