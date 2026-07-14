"""src/ingestion/official/bea.py
BEA GDP, corporate-profit, income, and trade observation ingestion.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

import requests

from src.storage.store import Store

from . import (
    OfficialProviderError,
    api_key as resolve_api_key,
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


class BEAIngestor:
    """Fetch only the cataloged BEA tables and retain release provenance."""

    SOURCE_NAME = "bea"
    API_KEY_ENV = "BEA_API_KEY"

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
        self.api_key = resolve_api_key(api_key, self.API_KEY_ENV)
        self.base_url = str(base_url).rstrip("/")
        self.http_get = http_get or requests.get
        self.timeout = max(float(timeout), 0.1)
        self.now_fn = now_fn

    def parse(
        self,
        payload: Mapping[str, object],
        *,
        entry_name: Optional[str] = None,
        accessed_at: Optional[str] = None,
    ) -> list[object]:
        """Parse BEA API data rows into typed observations."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        bea_api = payload.get("BEAAPI", payload) if isinstance(payload, Mapping) else {}
        results = bea_api.get("Results", {}) if isinstance(bea_api, Mapping) else {}
        rows = results.get("Data", []) if isinstance(results, Mapping) else []
        release_date = row_value(results, "ReleaseDate", "release_date") if isinstance(results, Mapping) else None
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            table_name = str(row_value(row, "TableName", "table") or "")
            line_description = str(row_value(row, "LineDescription", "description") or "")
            for entry in entries:
                if table_name != str(entry["dataset_id"]):
                    continue
                expected = str(entry.get("line_contains") or "").lower()
                if expected and expected not in line_description.lower():
                    continue
                value = parse_number(row_value(row, "DataValue", "value"))
                time_period = row_value(row, "TimePeriod", "period")
                if value is None or not time_period:
                    continue
                provider_id = str(
                    row_value(row, "RecordID", "record_id")
                    or f"{table_name}:{row_value(row, 'LineNumber', 'line_number')}:{time_period}"
                )
                published = release_date or accessed
                records.append(
                    make_observation(
                        source_name=self.SOURCE_NAME,
                        source_category=str(entry["source_category"]),
                        metric_id=str(entry["name"]),
                        series_id=table_name,
                        value=value,
                        unit=str(entry["unit"]),
                        frequency=str(entry["frequency"]),
                        observation_period=time_period,
                        vintage=published,
                        provider_record_id=provider_id,
                        source_url_value=source_url(str(entry["endpoint"]), str(entry["endpoint"])),
                        accessed_at=accessed,
                        published_at=published,
                        metadata=metadata_values(
                            dataset_id=table_name,
                            series_id=table_name,
                            report_period=str(time_period),
                            provider_revision=row_value(results, "Revision", "revision") if isinstance(results, Mapping) else None,
                        ),
                    )
                )
        return records

    def ingest(
        self,
        payload: Mapping[str, object] | None = None,
        *,
        entry_names: Optional[list[str]] = None,
    ) -> dict[str, object]:
        """Ingest BEA data; missing credentials disable only this agency."""
        output = result(self.SOURCE_NAME, "official_national_accounts")
        if not self.api_key:
            from . import disabled

            return disabled(self.store, output, self.API_KEY_ENV)
        entries = catalog_entries(self.SOURCE_NAME, entry_names)
        accessed = self._now_timestamp()
        try:
            if payload is not None:
                persist_records(self.store, self.parse(payload, accessed_at=accessed), output)
                output["pages"] = 1
            else:
                params = {
                    "UserID": self.api_key,
                    "method": "GETDATA",
                    "DataSetName": "NIPA",
                    "TableName": ",".join(str(entry["dataset_id"]) for entry in entries),
                    "Format": "JSON",
                }
                fetched = request_payload(self.http_get, str(entries[0]["endpoint"]), params=params, timeout=self.timeout)
                output["requests"] = 1
                persist_records(self.store, self.parse(fetched, accessed_at=accessed), output)
                output["pages"] = 1
        except OfficialProviderError as exc:
            output["status"] = status_for_error(exc)
            output["error_class"] = exc.error_class
            output["retry_after"] = exc.retry_after
            output["errors"].append(str(exc))
        except (TypeError, ValueError) as exc:
            output["malformed"] = int(output["malformed"]) + 1
            output["errors"].append(str(exc))
        return finish(self.store, output)

    def _now_timestamp(self) -> str:
        return as_timestamp(self.now_fn(), "now")


BeaIngestor = BEAIngestor
BEAAdapter = BEAIngestor
