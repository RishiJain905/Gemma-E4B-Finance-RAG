"""src/ingestion/official/eia.py
EIA energy-price, production, inventory, and generation observations.
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
    rows_from_payload,
    source_url,
    status_for_error,
    utc_now,
)

logger = logging.getLogger(__name__)


class EIAIngestor:
    """Fetch only the cataloged EIA series and keep source revisions."""

    SOURCE_NAME = "eia"
    API_KEY_ENV = "EIA_API_KEY"

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
        """Parse EIA response rows into one observation per cataloged series."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        response = payload.get("response", payload) if isinstance(payload, Mapping) else {}
        rows = rows_from_payload(response)
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for row in rows:
            series_id = str(row_value(row, "seriesId", "series_id") or "").strip()
            period = row_value(row, "period", "observation_date")
            value = parse_number(row_value(row, "value", "data_value"))
            if not series_id or not period or value is None:
                continue
            for entry in entries:
                if series_id != str(entry["dataset_id"]):
                    continue
                vintage = row_value(row, "vintage_date", "vintage_at", "last_updated") or accessed
                provider_id = str(row_value(row, "record_id", "id") or f"{series_id}:{period}")
                unit = str(row_value(row, "unit") or entry["unit"])
                records.append(
                    make_observation(
                        source_name=self.SOURCE_NAME,
                        source_category=str(entry["source_category"]),
                        metric_id=str(entry["name"]),
                        series_id=series_id,
                        value=value,
                        unit=unit,
                        frequency=str(entry["frequency"]),
                        observation_period=period,
                        vintage=vintage,
                        provider_record_id=provider_id,
                        source_url_value=source_url(str(entry["endpoint"]), str(entry["endpoint"])),
                        accessed_at=accessed,
                        published_at=vintage,
                        metadata=metadata_values(
                            dataset_id=series_id,
                            series_id=series_id,
                            frequency=str(entry["frequency"]),
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
        """Ingest EIA data; missing credentials disable only EIA."""
        output = result(self.SOURCE_NAME, "official_energy_statistics")
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
                    "api_key": self.api_key,
                    "data[0]": ",".join(str(entry["dataset_id"]) for entry in entries),
                    "frequency": "daily",
                    "sort[0][column]": "period",
                    "sort[0][direction]": "desc",
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


EiaIngestor = EIAIngestor
EIAAdapter = EIAIngestor
