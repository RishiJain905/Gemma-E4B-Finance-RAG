"""src/ingestion/official/cftc.py
Weekly CFTC Commitments of Traders positioning ingestion.
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
    rows_from_payload,
    source_url,
    status_for_error,
    utc_now,
)

logger = logging.getLogger(__name__)


class CFTCIngestor:
    """Fetch weekly COT rows from the no-key CFTC public feed."""

    SOURCE_NAME = "cftc"

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
        """Parse COT CSV/API rows into open-interest and net-position observations."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        rows = rows_from_payload(payload)
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for row in rows:
            report_date = row_value(row, "Report_Date_as_YYYY-MM-DD", "report_date", "date")
            market_code = str(row_value(row, "Market_Code", "market_code") or "").strip()
            if not report_date or not market_code:
                continue
            vintage = row_value(row, "vintage_at", "vintage_date", "last_updated") or accessed
            long_position = parse_number(row_value(row, "Noncommercial_Positions_Long_All"))
            short_position = parse_number(row_value(row, "Noncommercial_Positions_Short_All"))
            for entry in entries:
                field = str(entry["field"])
                if field == "Noncommercial_Net":
                    value = None if long_position is None or short_position is None else long_position - short_position
                else:
                    value = parse_number(row_value(row, field))
                if value is None:
                    continue
                provider_id = f"{market_code}:{report_date}"
                records.append(
                    make_observation(
                        source_name=self.SOURCE_NAME,
                        source_category=str(entry["source_category"]),
                        metric_id=str(entry["name"]),
                        series_id=market_code,
                        value=value,
                        unit=str(entry["unit"]),
                        frequency=str(entry["frequency"]),
                        observation_period=report_date,
                        vintage=vintage,
                        provider_record_id=provider_id,
                        source_url_value=source_url(str(entry["endpoint"]), str(entry["endpoint"])),
                        accessed_at=accessed,
                        published_at=vintage,
                        metadata=metadata_values(
                            dataset_id=str(entry["dataset_id"]),
                            series_id=market_code,
                            tags=str(row_value(row, "Market_and_Exchange_Names", "market") or ""),
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
        """Ingest CFTC data independently of keyed agencies."""
        output = result(self.SOURCE_NAME, "official_positioning")
        entries = catalog_entries(self.SOURCE_NAME, entry_names)
        accessed = self._now_timestamp()
        try:
            if payload is not None:
                persist_records(self.store, self.parse(payload, accessed_at=accessed), output)
                output["pages"] = 1
            else:
                fetched = request_payload(self.http_get, str(entries[0]["endpoint"]), timeout=self.timeout)
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


CftcIngestor = CFTCIngestor
CFTCAdapter = CFTCIngestor
