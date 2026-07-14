"""src/ingestion/official/openfda.py
openFDA recalls, safety events, and shortages with conservative registry mapping.
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


class OpenFDAIngestor:
    """Fetch only the cataloged openFDA event families."""

    SOURCE_NAME = "openfda"
    API_KEY_ENV = "OPENFDA_API_KEY"

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
        self.api_key = resolve_api_key(api_key, self.API_KEY_ENV)
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
        """Parse openFDA rows into linked narrative and structured event records."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        rows = rows_from_payload(payload)
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for row in rows:
            event_type = str(row_value(row, "event_type", "event", "type") or "").lower()
            provider_id = str(
                row_value(row, "report_id", "event_id", "recall_number", "id") or ""
            ).strip()
            reported = row_value(row, "report_date", "event_date", "date")
            if not provider_id or not reported:
                continue
            manufacturer_values = _identifiers(
                row_value(row, "manufacturer", "recalling_firm", "manufacturer_name")
            )
            security_ids = exact_security_ids(self.store, manufacturer_values)
            for entry in entries:
                configured_type = str(entry.get("event_type") or "").lower()
                if configured_type and event_type and configured_type != event_type:
                    continue
                if configured_type and not event_type:
                    event_type = configured_type
                title = str(
                    row_value(row, "product_description", "product", "title")
                    or f"openFDA {event_type} {provider_id}"
                )
                reason = str(
                    row_value(row, "reason_for_recall", "reason", "description", "status") or ""
                )
                url = source_url(row_value(row, "url", "link"), str(entry["endpoint"]))
                body = "\n\n".join(value for value in (title, reason) if value)
                release_id = row_value(row, "recall_number", "report_id", "event_id")
                narrative = make_narrative(
                    source_name=self.SOURCE_NAME,
                    source_category=str(entry["source_category"]),
                    provider_record_id=provider_id,
                    title=title,
                    body=body,
                    source_url_value=url,
                    published_at=reported,
                    accessed_at=accessed,
                    document_family="official_regulatory",
                    metadata=metadata_values(
                        dataset_id=str(entry["dataset_id"]),
                        release_id=release_id,
                        category=event_type,
                        tags=str(row_value(row, "status", "classification") or ""),
                    ),
                )
                records.append(narrative)
                records.append(
                    make_event(
                        source_name=self.SOURCE_NAME,
                        source_category=str(entry["source_category"]),
                        provider_record_id=provider_id,
                        event_type=event_type or configured_type or "safety",
                        effective_at=reported,
                        announced_at=reported,
                        published_at=reported,
                        source_url_value=url,
                        accessed_at=accessed,
                        security_ids=security_ids,
                        source_corpus_item_ids=(narrative.corpus_item_id,),
                        status=str(row_value(row, "status", "classification") or "reported").lower(),
                        metadata=metadata_values(
                            dataset_id=str(entry["dataset_id"]),
                            release_id=release_id,
                            category=event_type,
                            tags=", ".join(manufacturer_values),
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
        """Ingest openFDA records; missing credentials disable only openFDA."""
        output = result(self.SOURCE_NAME, "healthcare_regulatory_events")
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
                for entry in entries:
                    page_url = str(entry["endpoint"])
                    seen: set[str] = set()
                    while page_url and len(seen) < self.max_pages:
                        if page_url in seen:
                            raise ValueError("openFDA repeated pagination URL")
                        seen.add(page_url)
                        fetched = request_payload(
                            self.http_get,
                            page_url,
                            params={"api_key": self.api_key},
                            timeout=self.timeout,
                        )
                        output["requests"] = int(output["requests"]) + 1
                        persist_records(
                            self.store,
                            self.parse(fetched, entry_name=str(entry["name"]), accessed_at=accessed),
                            output,
                        )
                        output["pages"] = int(output["pages"]) + 1
                        from . import next_page

                        page_url = next_page(fetched, self.base_url)
                    if page_url:
                        raise ValueError("openFDA pagination limit reached")
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


def _identifiers(value: object) -> tuple[str, ...]:
    """Normalize one or many exact manufacturer identifiers."""
    if isinstance(value, Mapping):
        value = value.get("manufacturer_name") or value.get("name")
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value or "").strip()
    return (text,) if text else ()


OpenFDAAdapter = OpenFDAIngestor
