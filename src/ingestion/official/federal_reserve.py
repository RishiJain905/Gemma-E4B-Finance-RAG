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
    ) -> list[object]:
        """Parse RSS items into narrative records without contacting the provider."""
        entries = catalog_entries(self.SOURCE_NAME, [entry_name] if entry_name else None)
        root = ET.fromstring(payload)
        accessed = accessed_at or self._now_timestamp()
        records: list[object] = []
        for entry in entries:
            category_key = str(entry.get("category") or "").lower()
            for item in root.iter():
                if item.tag.rsplit("}", 1)[-1].lower() != "item":
                    continue
                values = {
                    child.tag.rsplit("}", 1)[-1].lower(): (child.text or "").strip()
                    for child in list(item)
                }
                category = values.get("category", "").lower()
                if category_key not in category and not category_key.startswith(category):
                    continue
                title = values.get("title", "").strip()
                link = values.get("link", "").strip()
                description = values.get("description", "").strip()
                provider_id = values.get("guid") or link
                if not title or not provider_id or not link or not values.get("pubdate"):
                    raise ValueError("Federal Reserve RSS item requires guid, title, link, and pubDate")
                records.append(
                    make_narrative(
                        source_name=self.SOURCE_NAME,
                        source_category=str(entry["source_category"]),
                        provider_record_id=provider_id,
                        title=title,
                        body="\n\n".join(value for value in (title, description) if value),
                        source_url_value=source_url(link, str(entry["endpoint"])),
                        published_at=values["pubdate"],
                        accessed_at=accessed,
                        metadata=metadata_values(
                            category=category or category_key,
                            dataset_id=str(entry["dataset_id"]),
                            release_id=provider_id,
                        ),
                    )
                )
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

    def _now_timestamp(self) -> str:
        value = self.now_fn()
        if isinstance(value, str):
            from . import as_timestamp

            return as_timestamp(value, "now")
        from . import as_timestamp

        return as_timestamp(value, "now")


FederalReserveAdapter = FederalReserveIngestor
