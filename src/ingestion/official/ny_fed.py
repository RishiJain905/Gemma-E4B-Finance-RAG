"""src/ingestion/official/ny_fed.py
New York Fed SOFR, EFFR, and market-operation ingestion.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

import requests

from src.storage.store import Store

from . import (
    OfficialProviderError,
    as_timestamp,
    catalog_entries,
    finish,
    make_event,
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


class NYFedIngestor:
    """Fetch New York Fed rates and operation rows without an API key."""

    SOURCE_NAME = "ny_fed"

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
        """Parse rate and operation rows into observations and events."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        if isinstance(payload, Mapping) and isinstance(payload.get("response"), Mapping):
            payload = payload["response"]
        rows = rows_from_payload(payload)
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        operation_entries = [entry for entry in entries if entry.get("name") == "ny_fed_market_operations"]
        rate_entries = [entry for entry in entries if entry.get("name") != "ny_fed_market_operations"]
        for row in rows:
            observation_date = row_value(row, "observation_date", "effective_date", "date")
            if not observation_date:
                continue
            vintage = row_value(row, "vintage_at", "vintage_date", "last_updated") or accessed
            for entry in rate_entries:
                value = parse_number(row_value(row, str(entry["field"])))
                if value is None:
                    continue
                field = str(entry["field"])
                records.append(
                    make_observation(
                        source_name=self.SOURCE_NAME,
                        source_category=str(entry["source_category"]),
                        metric_id=str(entry["name"]),
                        series_id=str(entry["dataset_id"]),
                        value=value,
                        unit=str(entry["unit"]),
                        frequency=str(entry["frequency"]),
                        observation_period=observation_date,
                        vintage=vintage,
                        provider_record_id=f"{observation_date}:{field}",
                        source_url_value=source_url(str(entry["endpoint"]), str(entry["endpoint"])),
                        accessed_at=accessed,
                        published_at=vintage,
                        period_start_value=observation_date,
                        metadata=metadata_values(
                            dataset_id=str(entry["dataset_id"]),
                            series_id=str(entry["dataset_id"]),
                            frequency=str(entry["frequency"]),
                        ),
                    )
                )
            if operation_entries:
                operation_id = str(row_value(row, "operation_id", "id") or "").strip()
                amount = parse_number(row_value(row, "operation_amount", "amount"))
                if operation_id and amount is not None:
                    entry = operation_entries[0]
                    records.append(
                        make_event(
                            source_name=self.SOURCE_NAME,
                            source_category=str(entry["source_category"]),
                            provider_record_id=operation_id,
                            event_type="market_operation",
                            effective_at=observation_date,
                            announced_at=vintage,
                            published_at=vintage,
                            source_url_value=source_url(str(entry["endpoint"]), str(entry["endpoint"])),
                            accessed_at=accessed,
                            amount=amount,
                            currency="USD",
                            metadata=metadata_values(
                                dataset_id=str(entry["dataset_id"]),
                                category=str(row_value(row, "operation_type", "type") or "operation"),
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
        """Ingest New York Fed data independently of keyed agencies."""
        output = result(self.SOURCE_NAME, "official_rates_and_operations")
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

    def _now_timestamp(self) -> str:
        return as_timestamp(self.now_fn(), "now")


NyFedIngestor = NYFedIngestor
NYFedAdapter = NYFedIngestor
