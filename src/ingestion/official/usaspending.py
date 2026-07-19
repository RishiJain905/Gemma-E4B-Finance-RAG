"""src/ingestion/official/usaspending.py
USAspending federal-award events with exact recipient/UEI registry mapping.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
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


class USAspendingIngestor:
    """Fetch cataloged federal awards without requiring an API key."""

    SOURCE_NAME = "usaspending"

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
        max_pages: int = 20,
        now_fn: Callable[[], object] = utc_now,
    ) -> None:
        self.store = store or Store()
        self.coverage = coverage_resolver
        self.base_url = str(base_url).rstrip("/")
        self.http_get = http_get or requests.get
        self.http_post = http_post or requests.post
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
        """Parse USAspending award rows into linked narrative and event records."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        rows = rows_from_payload(payload)
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for row in rows:
            entry = entries[0] if entries else None
            if entry is None:
                continue
            award_id = str(row_value(row, "Award ID", "award_id", "AwardID") or "").strip()
            recipient = str(row_value(row, "Recipient Name", "recipient_name") or "").strip()
            uei = str(row_value(row, "Recipient UEI", "recipient_uei", "uei") or "").strip()
            start_date = row_value(
                row,
                "Period of Performance Start Date",
                "Start Date",
                "start_date",
            )
            modified = row_value(
                row,
                "Last Modified Date",
                "last_modified_date",
                "date",
            ) or start_date or accessed
            if not award_id or not start_date or not recipient:
                continue
            url = source_url(row_value(row, "Link", "link", "url"), str(entry["endpoint"]))
            description = str(row_value(row, "Description", "description", "Award Description") or "")
            agency = str(row_value(row, "Awarding Agency", "awarding_agency") or "")
            title = f"Federal award {award_id} to {recipient}"
            narrative = make_narrative(
                source_name=self.SOURCE_NAME,
                source_category=str(entry["source_category"]),
                provider_record_id=award_id,
                title=title,
                body="\n\n".join(value for value in (title, description, agency) if value),
                source_url_value=url,
                published_at=modified,
                accessed_at=accessed,
                document_family="official_award",
                metadata=metadata_values(
                    dataset_id=str(entry["dataset_id"]),
                    release_id=award_id,
                    tags=recipient,
                ),
            )
            records.append(narrative)
            records.append(
                make_event(
                    source_name=self.SOURCE_NAME,
                    source_category=str(entry["source_category"]),
                    provider_record_id=award_id,
                    event_type=str(entry.get("event_type") or "award"),
                    effective_at=start_date,
                    announced_at=modified,
                    published_at=modified,
                    source_url_value=url,
                    accessed_at=accessed,
                    security_ids=exact_security_ids(self.store, (recipient, uei)),
                    source_corpus_item_ids=(narrative.corpus_item_id,),
                    status="awarded",
                    amount=parse_number(row_value(row, "Award Amount", "award_amount", "amount")),
                    currency="USD",
                    metadata=metadata_values(
                        dataset_id=str(entry["dataset_id"]),
                        release_id=award_id,
                        tags=f"{recipient}; {uei}; {agency}".strip("; "),
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
        """Ingest USAspending awards independently of keyed agencies."""
        output = result(self.SOURCE_NAME, "government_awards")
        entries = catalog_entries(self.SOURCE_NAME, entry_names)
        accessed = self._now_timestamp()
        try:
            if payload is not None:
                persist_records(self.store, self.parse(payload, accessed_at=accessed), output)
                output["pages"] = 1
            else:
                now = datetime.fromisoformat(accessed.replace("Z", "+00:00"))
                if now.tzinfo is None:
                    now = now.replace(tzinfo=timezone.utc)
                body = {
                    "filters": {
                        "time_period": [{
                            "start_date": (now - timedelta(days=30)).date().isoformat(),
                            "end_date": now.date().isoformat(),
                        }],
                        "award_type_codes": ["A", "B", "C", "D"],
                    },
                    "fields": [
                        "Award ID",
                        "Recipient Name",
                        "Start Date",
                        "Award Amount",
                        "Awarding Agency",
                        "Description",
                    ],
                    "page": 1,
                    "limit": 100,
                    "sort": "Start Date",
                    "order": "desc",
                }
                fetched = request_payload(
                    self.http_post,
                    str(entries[0]["endpoint"]),
                    json_body=body,
                    timeout=self.timeout,
                )
                output["requests"] = 1
                persist_records(
                    self.store,
                    self.parse(fetched, accessed_at=accessed),
                    output,
                )
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


UsaspendingIngestor = USAspendingIngestor
USAspendingAdapter = USAspendingIngestor
