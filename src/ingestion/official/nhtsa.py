"""src/ingestion/official/nhtsa.py
NHTSA vehicle recalls and investigations with exact manufacturer mapping.
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
    exact_security_ids,
    finish,
    make_event,
    make_narrative,
    metadata_values,
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


class NHTSAIngestor:
    """Fetch NHTSA safety events without an API key."""

    SOURCE_NAME = "nhtsa"

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        coverage_resolver: Optional[object] = None,
        api_key: Optional[str] = None,
        base_url: str = "",
        http_get: Optional[Callable[..., Any]] = None,
        timeout: float = 30.0,
        max_pages: int = 20,
        now_fn: Callable[[], object] = utc_now,
    ) -> None:
        self.store = store or Store()
        self.coverage = coverage_resolver
        self.base_url = str(base_url).rstrip("/")
        self.http_get = http_get or requests.get
        self.timeout = max(float(timeout), 0.1)
        self.max_pages = max(int(max_pages), 1)
        self.now_fn = now_fn

    def parse(
        self,
        payload: Mapping[str, object],
        *,
        entry_name: Optional[str] = None,
        accessed_at: Optional[str] = None,
    ) -> list[object]:
        """Parse NHTSA rows into linked narrative and event records."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        rows = rows_from_payload(payload)
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for row in rows:
            event_type = str(row_value(row, "event_type", "type") or "").lower()
            provider_id = str(
                row_value(
                    row,
                    "NHTSA_ID",
                    "nhtsa_id",
                    "nhtsa_action_number",
                    "id",
                )
                or ""
            ).strip()
            reported = row_value(
                row,
                "ReportDate",
                "report_date",
                "report_received_date",
                "open_date",
                "date",
            )
            manufacturer = str(row_value(row, "Manufacturer", "manufacturer") or "").strip()
            if not provider_id or not reported:
                continue
            for entry in entries:
                configured_type = str(entry.get("event_type") or "").lower()
                if configured_type and event_type and configured_type != event_type:
                    continue
                actual_type = event_type or configured_type or "safety_event"
                title = str(
                    row_value(
                        row,
                        "Summary",
                        "summary",
                        "defect_summary",
                        "subject",
                        "Component",
                    )
                    or f"NHTSA {actual_type} {provider_id}"
                )
                component = str(row_value(row, "Component", "component") or "")
                raw_url = row_value(
                    row,
                    "URL",
                    "url",
                    "link",
                    "recall_link",
                    "nhtsa_link",
                )
                if isinstance(raw_url, Mapping):
                    raw_url = raw_url.get("url")
                url = source_url(raw_url, str(entry["endpoint"]))
                narrative = make_narrative(
                    source_name=self.SOURCE_NAME,
                    source_category=str(entry["source_category"]),
                    provider_record_id=provider_id,
                    title=title,
                    body="\n\n".join(value for value in (title, component, manufacturer) if value),
                    source_url_value=url,
                    published_at=reported,
                    accessed_at=accessed,
                    document_family="official_regulatory",
                    metadata=metadata_values(
                        dataset_id=str(entry["dataset_id"]),
                        release_id=provider_id,
                        category=actual_type,
                        tags=str(row_value(row, "ModelYear", "model_year") or ""),
                    ),
                )
                records.append(narrative)
                records.append(
                    make_event(
                        source_name=self.SOURCE_NAME,
                        source_category=str(entry["source_category"]),
                        provider_record_id=provider_id,
                        event_type=actual_type,
                        effective_at=reported,
                        announced_at=reported,
                        published_at=reported,
                        source_url_value=url,
                        accessed_at=accessed,
                        security_ids=exact_security_ids(self.store, (manufacturer,)),
                        source_corpus_item_ids=(narrative.corpus_item_id,),
                        metadata=metadata_values(
                            dataset_id=str(entry["dataset_id"]),
                            release_id=provider_id,
                            category=actual_type,
                            tags=manufacturer,
                        ),
                    )
                )
                break
        return records

    def ingest(
        self,
        payload: Mapping[str, object] | None = None,
        *,
        entry_names: Optional[list[str]] = None,
    ) -> dict[str, object]:
        """Ingest NHTSA events independently of keyed agencies."""
        output = result(self.SOURCE_NAME, "vehicle_safety_events")
        entries = catalog_entries(self.SOURCE_NAME, entry_names)
        accessed = self._now_timestamp()
        try:
            if payload is not None:
                persist_records(self.store, self.parse(payload, accessed_at=accessed), output)
                output["pages"] = 1
            else:
                for entry in entries:
                    params = {
                        "$limit": int(entry.get("limit") or 100),
                        "$order": str(entry.get("order_by") or ""),
                    }
                    params = {key: value for key, value in params.items() if value != ""}
                    fetched = request_payload(
                        self.http_get,
                        str(entry["endpoint"]),
                        params=params,
                        timeout=self.timeout,
                    )
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


NhtsaIngestor = NHTSAIngestor
NHTSAAdapter = NHTSAIngestor
