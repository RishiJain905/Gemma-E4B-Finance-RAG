"""src/ingestion/official/__init__.py
Shared bounded helpers for first-party macro, regulatory, and sector feeds.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
from calendar import monthrange
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import yaml

from src.ingestion.errors import (
    ProviderError,
    error_class_for_http,
    parse_retry_after,
    safe_message,
)
from src.ingestion.normalization import (
    NORMALIZATION_VERSION,
    content_hash,
    normalize_canonical_url,
    parse_timestamp,
)
from src.ingestion.records import EventRecord, NarrativeRecord, ObservationRecord

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).resolve().parents[3] / "configs" / "official_sources.yaml"
PUBLIC_LICENSE = "public_record"


class OfficialProviderError(ProviderError):
    """Provider response failure classified for source-status reporting."""

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
        reset_at: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            error_class=error_class,
            status_code=status_code,
            retry_after=retry_after,
            reset_at=reset_at,
        )


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def as_timestamp(value: object, field_name: str = "timestamp") -> str:
    """Normalize date, ISO timestamp, or RFC-822 provider values to UTC."""
    if isinstance(value, datetime):
        parsed = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    try:
        parsed = parse_timestamp(text, field_name)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text).astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 or RFC-822 value") from exc
    return parsed.isoformat().replace("+00:00", "Z")


def period_end(value: object, field_name: str = "period_end") -> str:
    """Normalize daily, monthly, quarterly, annual, and ISO period labels."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = (int(part) for part in text.split("-"))
        return date(year, month, monthrange(year, month)[1]).isoformat()
    quarter = re.fullmatch(r"(\d{4})Q([1-4])", text, flags=re.IGNORECASE)
    if quarter:
        year, quarter_number = int(quarter.group(1)), int(quarter.group(2))
        month = quarter_number * 3
        return date(year, month, monthrange(year, month)[1]).isoformat()
    month_period = re.fullmatch(r"(\d{4})M(\d{2})", text, flags=re.IGNORECASE)
    if month_period:
        year, month = int(month_period.group(1)), int(month_period.group(2))
        return date(year, month, monthrange(year, month)[1]).isoformat()
    annual = re.fullmatch(r"(\d{4})A\d{2}", text, flags=re.IGNORECASE)
    if annual:
        return date(int(annual.group(1)), 12, 31).isoformat()
    if re.fullmatch(r"\d{4}", text):
        return date(int(text), 12, 31).isoformat()
    return as_timestamp(text, field_name)[:10]


def as_date(value: object, field_name: str = "date") -> str:
    """Return a normalized ISO date for provider period or event fields."""
    return period_end(value, field_name)


def parse_number(value: object) -> Optional[float]:
    """Parse a provider number while treating disclosure sentinels as missing."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"na", "n/a", "null", "(d)", "(z)", "-"}:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def format_number(value: float) -> str:
    """Format a numeric value without inventing provider precision."""
    return str(int(value)) if value.is_integer() else format(value, ".15g")


def normalize_key(value: object) -> str:
    """Normalize a response key for tolerant provider column lookup."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def row_value(row: Mapping[str, object], *names: str) -> object:
    """Return the first case/punctuation-insensitive value from one provider row."""
    values = {normalize_key(key): value for key, value in row.items()}
    for name in names:
        key = normalize_key(name)
        if key in values:
            return values[key]
    return None


def rows_from_payload(payload: object) -> list[dict[str, object]]:
    """Extract bounded row lists from common JSON and CSV response envelopes."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        return [dict(row) for row in csv.DictReader(io.StringIO(payload))]
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("data", "results", "items", "observations", "records", "spending_by_award"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, Mapping)]
        if isinstance(value, Mapping):
            nested = rows_from_payload(value)
            if nested:
                return nested
    return []


def load_catalog() -> dict[str, object]:
    """Load the non-secret curated official-source catalog."""
    with CATALOG_PATH.open(encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict) or not isinstance(loaded.get("entries"), list):
        raise ValueError("official source catalog must contain an entries list")
    return loaded


def catalog_entries(agency: str, entry_names: Optional[Iterable[str]] = None) -> list[dict[str, object]]:
    """Return only cataloged entries for one agency and optional names."""
    entries = [
        entry for entry in load_catalog()["entries"]
        if isinstance(entry, dict) and entry.get("agency") == agency
    ]
    if entry_names is None:
        return entries
    wanted = {str(name) for name in entry_names}
    known = {str(entry.get("name")) for entry in entries}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"uncataloged {agency} entries: {', '.join(unknown)}")
    return [entry for entry in entries if str(entry.get("name")) in wanted]


def request_payload(
    http_get: Callable[..., Any],
    url: str,
    *,
    params: Optional[Mapping[str, object]] = None,
    timeout: float = 30.0,
) -> object:
    """Perform one provider request and classify its response without retrying."""
    try:
        response = http_get(url, params=dict(params or {}), timeout=timeout)
    except TypeError:
        response = http_get(url, params=dict(params or {}))
    status_code = int(getattr(response, "status_code", 200))
    if status_code >= 400:
        message = _response_error_text(response)
        raise OfficialProviderError(
            message or f"official provider HTTP {status_code}",
            error_class=provider_error_class(status_code, message),
            status_code=status_code,
            retry_after=retry_after(response),
        )
    json_reader = getattr(response, "json", None)
    if callable(json_reader):
        try:
            return json_reader()
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return getattr(response, "text", getattr(response, "content", ""))


def request_text(
    http_get: Callable[..., Any],
    url: str,
    *,
    params: Optional[Mapping[str, object]] = None,
    timeout: float = 30.0,
) -> str:
    """Perform one request and return text for RSS/XML/CSV providers."""
    try:
        response = http_get(url, params=dict(params or {}), timeout=timeout)
    except TypeError:
        response = http_get(url, params=dict(params or {}))
    status_code = int(getattr(response, "status_code", 200))
    if status_code >= 400:
        message = _response_error_text(response)
        raise OfficialProviderError(
            message or f"official provider HTTP {status_code}",
            error_class=provider_error_class(status_code, message),
            status_code=status_code,
            retry_after=retry_after(response),
        )
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    content = getattr(response, "content", b"")
    return content.decode("utf-8") if isinstance(content, bytes) else str(content)


def provider_error_class(status_code: int, message: str = "") -> str:
    """Map HTTP/provider messages to the established entitlement state model."""
    return error_class_for_http(status_code, message).value


def retry_after(response: object) -> Optional[float]:
    """Read numeric/date Retry-After seconds for scheduler handoff."""
    headers = getattr(response, "headers", {})
    value = headers.get("Retry-After") or headers.get("retry-after") if hasattr(headers, "get") else None
    window = parse_retry_after(value, now=datetime.now(timezone.utc))
    return window.delay_seconds if window is not None else None


def _response_error_text(response: object) -> str:
    json_reader = getattr(response, "json", None)
    if callable(json_reader):
        try:
            payload = json_reader()
            if isinstance(payload, Mapping):
                for key in ("error", "message", "detail", "status"):
                    if payload.get(key):
                        return safe_message(payload[key])
            return safe_message(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return safe_message(getattr(response, "text", ""))


def status_for_error(error: OfficialProviderError) -> str:
    """Convert a classified provider error into the common source status."""
    return {
        "authentication": "disabled_authentication",
        "entitlement": "disabled_entitlement",
        "rate_limited": "rate_limited",
    }.get(error.error_class, "error")


def source_url(value: object, fallback: str) -> str:
    """Return an absolute canonical source URL or fail closed."""
    candidate = str(value or fallback).strip()
    normalized = normalize_canonical_url(candidate)
    if not normalized:
        raise ValueError("official source URL is required")
    return normalized


def safe_identity(value: object, fallback: str) -> str:
    """Create a bounded stable identifier component."""
    text = str(value or "").strip()
    if not text:
        text = fallback
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:160]
    return normalized or hashlib.sha256(fallback.encode("utf-8")).hexdigest()


def metadata_values(**values: object) -> dict[str, object]:
    """Keep only scalar metadata values accepted by the normalized record contract."""
    return {key: value for key, value in values.items() if value is not None and value != ""}


def make_observation(
    *,
    source_name: str,
    source_category: str,
    metric_id: str,
    series_id: Optional[str],
    value: float,
    unit: str,
    frequency: str,
    observation_period: object,
    vintage: object,
    provider_record_id: str,
    source_url_value: str,
    accessed_at: str,
    published_at: Optional[object] = None,
    period_start_value: Optional[object] = None,
    metadata: Optional[Mapping[str, object]] = None,
    sector: Optional[str] = None,
    security_ids: tuple[str, ...] = (),
    tickers: tuple[str, ...] = (),
) -> ObservationRecord:
    """Build a structured observation with complete official provenance."""
    period = period_end(observation_period, "observation_period")
    vintage_at = as_timestamp(vintage or accessed_at, "vintage_at")
    provider_id = str(provider_record_id).strip()
    identity = f"{source_name}/{metric_id}/{provider_id}/{vintage_at}"
    return ObservationRecord(
        observation_id=f"observation/{safe_identity(identity, provider_id)}",
        metric_id=metric_id,
        series_id=series_id,
        value_text=format_number(value),
        value_numeric=value,
        unit=unit,
        frequency=frequency,
        period_start=period_end(period_start_value, "period_start") if period_start_value else None,
        period_end=period,
        vintage_at=vintage_at,
        as_of_at=vintage_at,
        scope="sector" if sector else "global",
        security_ids=security_ids,
        tickers=tickers,
        sector=sector,
        source_name=source_name,
        source_category=source_category,
        provider_record_id=provider_id,
        original_publisher=None,
        source_url=source_url_value,
        canonical_url=source_url_value,
        published_at=as_timestamp(published_at, "published_at") if published_at else None,
        observed_at=as_timestamp(accessed_at, "observed_at"),
        accessed_at=accessed_at,
        ingested_at=accessed_at,
        license_label=PUBLIC_LICENSE,
        normalization_version=NORMALIZATION_VERSION,
        metadata=dict(metadata or {}),
        evidence_authority="official",
    )


def make_narrative(
    *,
    source_name: str,
    source_category: str,
    provider_record_id: str,
    title: str,
    body: str,
    source_url_value: str,
    published_at: object,
    accessed_at: str,
    metadata: Optional[Mapping[str, object]] = None,
    document_family: str = "official_release",
) -> NarrativeRecord:
    """Build a release narrative for the SQLite metadata plus Chroma path."""
    provider_id = str(provider_record_id).strip()
    corpus_item_id = f"official/{source_name}/{safe_identity(provider_id, title)}"
    published = as_timestamp(published_at, "published_at")
    return NarrativeRecord(
        corpus_item_id=corpus_item_id,
        source_name=source_name,
        source_category=source_category,
        provider_record_id=provider_id,
        original_publisher=source_name,
        item_type="official_release",
        title=title.strip(),
        body=body.strip(),
        summary=body.strip()[:4_000],
        published_at=published,
        observed_at=accessed_at,
        accessed_at=accessed_at,
        ingested_at=accessed_at,
        source_url=source_url_value,
        canonical_url=source_url_value,
        license_label=PUBLIC_LICENSE,
        normalization_version=NORMALIZATION_VERSION,
        content_hash=content_hash(body.strip()),
        document_family=document_family,
        metadata=dict(metadata or {}),
        evidence_authority="official",
    )


def make_event(
    *,
    source_name: str,
    source_category: str,
    provider_record_id: str,
    event_type: str,
    effective_at: object,
    source_url_value: str,
    accessed_at: str,
    security_ids: tuple[str, ...] = (),
    source_corpus_item_ids: tuple[str, ...] = (),
    published_at: Optional[object] = None,
    announced_at: Optional[object] = None,
    status: str = "reported",
    amount: Optional[float] = None,
    currency: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> EventRecord:
    """Build a structured official event while retaining the agency identity."""
    provider_id = str(provider_record_id).strip()
    return EventRecord(
        event_id=f"event/{source_name}/{safe_identity(provider_id, event_type)}",
        event_type=event_type,
        effective_at=as_timestamp(effective_at, "effective_at") if effective_at else None,
        announced_at=as_timestamp(announced_at, "announced_at") if announced_at else None,
        status=status,
        security_ids=security_ids,
        source_corpus_item_ids=source_corpus_item_ids,
        source_name=source_name,
        source_category=source_category,
        provider_record_id=provider_id,
        original_publisher=source_name,
        source_url=source_url_value,
        canonical_url=source_url_value,
        published_at=as_timestamp(published_at, "published_at") if published_at else None,
        observed_at=accessed_at,
        accessed_at=accessed_at,
        ingested_at=accessed_at,
        license_label=PUBLIC_LICENSE,
        normalization_version=NORMALIZATION_VERSION,
        amount=amount,
        currency=currency,
        action_date=as_date(effective_at, "action_date") if effective_at else None,
        metadata=dict(metadata or {}),
        evidence_authority="official",
    )


def exact_security_ids(store: object, identifiers: Iterable[object]) -> tuple[str, ...]:
    """Resolve only exact registry identifiers; ambiguous candidates remain unattached."""
    security_ids: set[str] = set()
    for identifier in identifiers:
        value = str(identifier or "").strip()
        if not value:
            continue
        resolver = getattr(store, "resolve_exact_security", None)
        resolved = resolver(value) if callable(resolver) else getattr(store, "resolve_security")(value)
        if isinstance(resolved, Mapping) and resolved.get("security_id"):
            security_ids.add(str(resolved["security_id"]))
    return tuple(sorted(security_ids))


def result(source: str, capability: str) -> dict[str, object]:
    """Create a consistent adapter result envelope."""
    return {
        "status": "ok",
        "source": source,
        "capability": capability,
        "stored": 0,
        "updated": 0,
        "duplicates": 0,
        "malformed": 0,
        "requests": 0,
        "pages": 0,
        "errors": [],
    }


def persist_records(store: object, records: Iterable[object], output: dict[str, object]) -> None:
    """Persist normalized records through Store and classify replay/change outcomes."""
    for record in records:
        try:
            if isinstance(record, NarrativeRecord):
                stored = store.upsert_narrative(record)
                if stored.get("indexing_status") == "error":
                    output["malformed"] = int(output["malformed"]) + 1
                    output["errors"].append(str(stored.get("index_error") or "narrative indexing failed"))
                elif stored.get("created"):
                    output["stored"] = int(output["stored"]) + 1
                else:
                    output["duplicates"] = int(output["duplicates"]) + 1
            elif isinstance(record, ObservationRecord):
                stored = store.upsert_observation(record)
                if stored.get("created"):
                    output["stored"] = int(output["stored"]) + 1
                elif stored.get("changed"):
                    output["updated"] = int(output["updated"]) + 1
                else:
                    output["duplicates"] = int(output["duplicates"]) + 1
            elif isinstance(record, EventRecord):
                stored = store.upsert_event(record)
                if stored.get("created"):
                    output["stored"] = int(output["stored"]) + 1
                elif stored.get("changed"):
                    output["updated"] = int(output["updated"]) + 1
                else:
                    output["duplicates"] = int(output["duplicates"]) + 1
            else:
                raise TypeError(f"unsupported official record: {type(record).__name__}")
        except Exception as exc:  # noqa: BLE001 - one provider row must not abort its agency
            output["malformed"] = int(output["malformed"]) + 1
            output["errors"].append(safe_message(exc))


def finish(store: object, output: dict[str, object], *, partition: str = "catalog") -> dict[str, object]:
    """Persist source status and return the final bounded result."""
    if output["status"] == "ok" and output["malformed"]:
        output["status"] = "partial"
    try:
        store.set_source_status(
            str(output["source"]),
            partition,
            str(output["status"]),
            error_class=str(output.get("error_class") or ("contract" if output["malformed"] else "")) or None,
            error_message=(str(output["errors"][0]) if output["errors"] else None),
            retry_after=output.get("retry_after"),
        )
    except Exception:  # noqa: BLE001 - status bookkeeping cannot couple agencies
        logger.warning("Could not persist official source status for %s", output["source"], exc_info=True)
    return output


def disabled(store: object, output: dict[str, object], env_name: str) -> dict[str, object]:
    """Return the explicit missing-key capability state without making a request."""
    output["status"] = "disabled_missing_key"
    output["error_class"] = "authentication"
    try:
        store.set_source_status(
            str(output["source"]),
            "catalog",
            "disabled_missing_key",
            error_class="authentication",
            error_message=f"{env_name} is not configured",
        )
    except Exception:  # noqa: BLE001 - status bookkeeping cannot couple agencies
        logger.warning("Could not persist missing-key status for %s", output["source"], exc_info=True)
    return output


def api_key(value: Optional[str], env_name: str) -> str:
    """Resolve an explicit constructor key or one environment variable without loading .env."""
    return str(value if value is not None else os.environ.get(env_name, "")).strip()


def next_page(payload: object, base_url: str = "") -> Optional[str]:
    """Read a provider continuation URL for bounded scheduler-owned pagination."""
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("next_url") or payload.get("next") or payload.get("next_page") or payload.get("links")
    if isinstance(value, Mapping):
        value = value.get("url") or value.get("href") or value.get("next")
    if not value:
        return None
    text = str(value).strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}/{text.lstrip('/')}"


__all__ = [
    "OfficialProviderError",
    "api_key",
    "as_date",
    "as_timestamp",
    "catalog_entries",
    "disabled",
    "exact_security_ids",
    "finish",
    "format_number",
    "load_catalog",
    "make_event",
    "make_narrative",
    "make_observation",
    "metadata_values",
    "next_page",
    "parse_number",
    "period_end",
    "persist_records",
    "provider_error_class",
    "request_payload",
    "request_text",
    "result",
    "rows_from_payload",
    "safe_identity",
    "source_url",
    "status_for_error",
    "utc_now",
]
