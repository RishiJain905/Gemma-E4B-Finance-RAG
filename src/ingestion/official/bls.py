"""src/ingestion/official/bls.py
BLS CPI, PPI, employment, wage observations, and release metadata ingestion.
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
    make_narrative,
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


class BLSIngestor:
    """Fetch only the BLS series and release metadata in the official catalog."""

    SOURCE_NAME = "bls"
    API_KEY_ENV = "BLS_API_KEY"

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        coverage_resolver: Optional[object] = None,
        api_key: Optional[str] = None,
        base_url: str = "",
        http_get: Optional[Callable[..., Any]] = None,
        http_post: Optional[Callable[..., Any]] = None,
        timeout: float = 30.0,
        now_fn: Callable[[], object] = utc_now,
    ) -> None:
        from . import api_key as resolve_api_key

        self.store = store or Store()
        self.coverage = coverage_resolver
        self.api_key = resolve_api_key(api_key, self.API_KEY_ENV)
        self.base_url = str(base_url).rstrip("/")
        self.http_get = http_get or requests.get
        self.http_post = http_post or requests.post
        self.timeout = max(float(timeout), 0.1)
        self.now_fn = now_fn

    def parse(
        self,
        payload: Mapping[str, object],
        *,
        entry_name: Optional[str] = None,
        accessed_at: Optional[str] = None,
    ) -> list[object]:
        """Parse a BLS series envelope and retain release/vintage metadata."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        results = payload.get("Results", payload.get("results", {})) if isinstance(payload, Mapping) else {}
        series_rows = results.get("series", []) if isinstance(results, Mapping) else []
        release = results.get("release", {}) if isinstance(results, Mapping) else {}
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for series in series_rows:
            if not isinstance(series, Mapping):
                continue
            series_id = str(series.get("seriesID") or series.get("series_id") or "").strip()
            entry_matches = [entry for entry in entries if str(entry.get("series_id")) == series_id]
            for entry in entry_matches:
                for row in series.get("data", []) or []:
                    if not isinstance(row, Mapping):
                        continue
                    value = parse_number(row_value(row, "value"))
                    if value is None:
                        continue
                    report_period = row_value(row, "period", "periodName") or ""
                    period = row_value(row, "period_end") or f"{row.get('year', '')}{report_period}"
                    vintage = row_value(row, "vintage_date", "vintage_at") or row_value(
                        release if isinstance(release, Mapping) else {}, "release_date"
                    ) or accessed
                    provider_id = f"{series_id}:{row.get('year', '')}:{row.get('period', '')}"
                    release_date = row_value(release, "release_date", "published_at") if isinstance(release, Mapping) else None
                    release_link = row_value(release, "link", "url") if isinstance(release, Mapping) else None
                    records.append(
                        make_observation(
                            source_name=self.SOURCE_NAME,
                            source_category=str(entry["source_category"]),
                            metric_id=str(entry["name"]),
                            series_id=series_id,
                            value=value,
                            unit=str(entry["unit"]),
                            frequency=str(entry["frequency"]),
                            observation_period=period,
                            vintage=vintage,
                            provider_record_id=provider_id,
                            source_url_value=source_url(release_link, str(entry["endpoint"])),
                            accessed_at=accessed,
                            published_at=release_date or vintage,
                            period_start_value=row_value(row, "period_start"),
                            metadata=metadata_values(
                                dataset_id=str(entry["dataset_id"]),
                                series_id=series_id,
                                release_id=row_value(release, "id", "release_id") if isinstance(release, Mapping) else None,
                                report_period=str(report_period),
                                frame=str(row_value(row, "periodName") or ""),
                            ),
                        )
                    )
        release_entries = [entry for entry in entries if entry.get("series_id") == "release"]
        if release_entries and isinstance(release, Mapping) and release:
            release_id = str(row_value(release, "id", "release_id") or "bls-release")
            release_date = row_value(release, "release_date", "published_at") or accessed
            title = str(row_value(release, "name", "title") or "BLS economic indicators release")
            link = source_url(row_value(release, "link", "url"), str(release_entries[0]["endpoint"]))
            records.append(
                make_narrative(
                    source_name=self.SOURCE_NAME,
                    source_category=str(release_entries[0]["source_category"]),
                    provider_record_id=release_id,
                    title=title,
                    body=f"{title} published by the Bureau of Labor Statistics on {release_date}.",
                    source_url_value=link,
                    published_at=release_date,
                    accessed_at=accessed,
                    metadata=metadata_values(
                        dataset_id=str(release_entries[0]["dataset_id"]),
                        release_id=release_id,
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
        """Ingest BLS rows; a missing API key disables only BLS."""
        output = result(self.SOURCE_NAME, "official_labor_statistics")
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
                body = {
                    "seriesid": [str(entry["series_id"]) for entry in entries if entry.get("series_id") != "release"],
                    "startyear": accessed[:4],
                    "endyear": accessed[:4],
                    "registrationkey": self.api_key,
                }
                fetched = request_payload(
                    self.http_post,
                    str(entries[0]["endpoint"]),
                    json_body=body,
                    timeout=self.timeout,
                )
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


BlsIngestor = BLSIngestor
BLSAdapter = BLSIngestor
