"""src/ingestion/official/federal_reserve.py
Federal Reserve RSS ingestion for policy, minutes, speeches, regulation, and releases.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, Callable, Optional

import requests

from src.storage.store import Store

from . import (
    catalog_entries,
    finish,
    make_narrative,
    metadata_values,
    persist_records,
    request_text,
    result,
    source_url,
    status_for_error,
    utc_now,
    OfficialProviderError,
)

logger = logging.getLogger(__name__)


class FederalReserveIngestor:
    """Fetch and normalize cataloged Federal Reserve RSS items."""

    SOURCE_NAME = "federal_reserve"

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
        payload: str | bytes,
        *,
        entry_name: Optional[str] = None,
        accessed_at: Optional[str] = None,
        ignore_max_items: bool = False,
    ) -> list[object]:
        """Parse RSS items into narrative records without contacting the provider."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        root = ET.fromstring(payload)
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for entry in entries:
            category_key = str(entry.get("category") or "").lower()
            max_items = (
                0
                if ignore_max_items
                else max(int(entry.get("max_items") or 0), 0)
            )
            matched_items = 0
            for item in root.iter():
                if item.tag.rsplit("}", 1)[-1].lower() != "item":
                    continue
                values = {
                    child.tag.rsplit("}", 1)[-1].lower(): (child.text or "").strip()
                    for child in list(item)
                }
                category = values.get("category", "").lower()
                title = values.get("title", "").strip()
                if category_key and category_key not in f"{category} {title.lower()}":
                    continue
                link = values.get("link", "").strip()
                description = values.get("description", "").strip()
                rdf_about = next(
                    (
                        str(value).strip()
                        for key, value in item.attrib.items()
                        if key.rsplit("}", 1)[-1].lower() == "about"
                    ),
                    "",
                )
                provider_id = values.get("guid") or rdf_about or link
                published_at = values.get("pubdate") or values.get("date")
                if not title or not provider_id or not link or not published_at:
                    raise ValueError(
                        "Federal Reserve RSS item requires an identity, title, link, "
                        "and publication date"
                    )
                records.append(
                    make_narrative(
                        source_name=self.SOURCE_NAME,
                        source_category=str(entry["source_category"]),
                        provider_record_id=provider_id,
                        title=title,
                        body="\n\n".join(value for value in (title, description) if value),
                        source_url_value=source_url(link, str(entry["endpoint"])),
                        published_at=published_at,
                        accessed_at=accessed,
                        metadata=metadata_values(
                            category=category or category_key,
                            dataset_id=str(entry["dataset_id"]),
                            release_id=provider_id,
                        ),
                    )
                )
                matched_items += 1
                if max_items and matched_items >= max_items:
                    break
        return records

    def ingest(
        self,
        payload: str | bytes | None = None,
        *,
        entry_names: Optional[list[str]] = None,
    ) -> dict[str, object]:
        """Ingest cataloged RSS feeds, or parse one supplied offline payload."""
        output = result(self.SOURCE_NAME, "official_releases")
        entries = catalog_entries(self.SOURCE_NAME, entry_names)
        accessed = self._now_timestamp()
        try:
            if payload is not None:
                persist_records(self.store, self.parse(payload, accessed_at=accessed), output)
                output["pages"] = 1
            else:
                for entry in entries:
                    text = request_text(
                        self.http_get,
                        str(entry["endpoint"]),
                        timeout=self.timeout,
                    )
                    output["requests"] = int(output["requests"]) + 1
                    persist_records(
                        self.store,
                        self.parse(text, entry_name=str(entry["name"]), accessed_at=accessed),
                        output,
                    )
                    output["pages"] = int(output["pages"]) + 1
        except OfficialProviderError as exc:
            output["status"] = status_for_error(exc)
            output["error_class"] = exc.error_class
            output["retry_after"] = exc.retry_after
            output["errors"].append(str(exc))
        except (ET.ParseError, TypeError, ValueError) as exc:
            output["malformed"] = int(output["malformed"]) + 1
            output["errors"].append(str(exc))
        return finish(self.store, output)

    def ingest_history(
        self,
        *,
        batch_size: int = 50,
        max_batches: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> dict[str, object]:
        """Backfill older statistical releases with durable batch checkpoints."""
        if not 1 <= int(batch_size) <= 200:
            raise ValueError("batch_size must be between 1 and 200")
        if max_batches is not None and int(max_batches) < 1:
            raise ValueError("max_batches must be positive")

        partition = "historical_statistical_releases"
        entry = catalog_entries(
            self.SOURCE_NAME,
            ["federal_reserve_statistical_releases"],
        )[0]
        output = result(self.SOURCE_NAME, "official_release_history")
        state = self.store.get_source_cursor_state(self.SOURCE_NAME, partition)
        if state and state.get("status") == "complete":
            output.update(
                {
                    "status": "complete",
                    "cursor_after": state.get("cursor_value"),
                    "remaining": 0,
                }
            )
            return output

        try:
            text = request_text(
                self.http_get,
                str(entry["endpoint"]),
                timeout=self.timeout,
            )
            output["requests"] = 1
            output["pages"] = 1
            records = self.parse(
                text,
                entry_name=str(entry["name"]),
                accessed_at=self._now_timestamp(),
                ignore_max_items=True,
            )

            cursor = str(state.get("cursor_value") or "") if state else ""
            if cursor:
                cursor_index = next(
                    (
                        index
                        for index, record in enumerate(records)
                        if str(record.provider_record_id) == cursor
                    ),
                    None,
                )
                if cursor_index is None:
                    raise ValueError("Federal Reserve history cursor is no longer in the feed")
                start = cursor_index + 1
            else:
                start = max(int(entry.get("max_items") or 0), 0)

            batches = 0
            while start < len(records):
                if max_batches is not None and batches >= int(max_batches):
                    break
                batch = records[start : start + int(batch_size)]
                malformed_before = int(output["malformed"])
                persist_records(self.store, batch, output)
                if int(output["malformed"]) != malformed_before:
                    output["status"] = "partial"
                    break

                start += len(batch)
                batches += 1
                cursor = str(batch[-1].provider_record_id)
                complete = start >= len(records)
                self.store.set_source_cursor(
                    self.SOURCE_NAME,
                    partition,
                    cursor,
                    cursor_type="page_token",
                    last_successful_run_id=run_id,
                    status="complete" if complete else "partial",
                )

            output["cursor_after"] = cursor or None
            output["remaining"] = max(len(records) - start, 0)
            if output["status"] == "ok":
                output["status"] = (
                    "complete" if int(output["remaining"]) == 0 else "partial"
                )
        except OfficialProviderError as exc:
            output["status"] = status_for_error(exc)
            output["error_class"] = exc.error_class
            output["retry_after"] = exc.retry_after
            output["errors"].append(str(exc))
        except (ET.ParseError, TypeError, ValueError) as exc:
            output["status"] = "partial"
            output["malformed"] = int(output["malformed"]) + 1
            output["errors"].append(str(exc))
        return finish(self.store, output, partition=partition)

    def _now_timestamp(self) -> str:
        value = self.now_fn()
        if isinstance(value, str):
            from . import as_timestamp

            return as_timestamp(value, "now")
        from . import as_timestamp

        return as_timestamp(value, "now")


FederalReserveAdapter = FederalReserveIngestor
