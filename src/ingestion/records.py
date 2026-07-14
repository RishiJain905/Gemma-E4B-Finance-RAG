"""src/ingestion/records.py
Frozen normalized record contracts shared by Phase 2.3 provider adapters.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from src.ingestion.normalization import NORMALIZATION_VERSION, parse_timestamp


INDEXING_STATUSES = frozenset({"pending", "indexed", "error", "not_applicable"})
OBSERVATION_SCOPES = frozenset({"security", "sector", "global"})
MAX_TITLE_CHARS = 1_000
MAX_SUMMARY_CHARS = 4_000
MAX_BODY_CHARS = 1_000_000
MAX_EXPLANATION_CHARS = 4_000
MAX_METADATA_VALUE_CHARS = 2_000

METADATA_ALLOWLIST = frozenset({
    "accession",
    "adjusted",
    "authors",
    "category",
    "dataset_id",
    "exchange",
    "exhibit",
    "filing_item",
    "form",
    "frame",
    "fiscal_period",
    "fiscal_year",
    "market_date",
    "provider_revision",
    "release_id",
    "report_period",
    "series_id",
    "tags",
    "taxonomy",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METADATA_SCALARS = (str, int, float, bool, type(None))


def _required_text(name: str, value: object, *, max_chars: int = 2_000) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    if len(value) > max_chars:
        raise ValueError(f"{name} exceeds {max_chars} characters")


def _optional_text(name: str, value: object, *, max_chars: int = 2_000) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")
    if len(value) > max_chars:
        raise ValueError(f"{name} exceeds {max_chars} characters")


def _timestamp(name: str, value: str | None, *, required: bool = False) -> None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return
    parse_timestamp(value, name)


def _tuple_of_text(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list of strings")
    normalized = tuple(values)
    if len(normalized) > 1_000:
        raise ValueError(f"{name} exceeds 1000 entries")
    for value in normalized:
        _required_text(name, value, max_chars=200)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate values")
    return normalized


def validate_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    """Validate and freeze bounded provider metadata."""
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    unknown = sorted(set(metadata) - METADATA_ALLOWLIST)
    if unknown:
        raise ValueError(f"metadata contains unknown keys: {', '.join(unknown)}")
    validated: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(value, _METADATA_SCALARS):
            raise ValueError(f"metadata value for {key} must be a scalar")
        if isinstance(value, str) and len(value) > MAX_METADATA_VALUE_CHARS:
            raise ValueError(f"metadata value for {key} exceeds {MAX_METADATA_VALUE_CHARS} characters")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metadata value for {key} must be finite")
        validated[key] = value
    return MappingProxyType(validated)


def _validate_provenance(record: object) -> None:
    for name in ("source_name", "source_category", "source_url", "license_label", "normalization_version"):
        _required_text(name, getattr(record, name))
    for name in ("provider_record_id", "original_publisher", "canonical_url"):
        _optional_text(name, getattr(record, name))
    for name in ("published_at", "observed_at"):
        _timestamp(name, getattr(record, name))
    for name in ("accessed_at", "ingested_at"):
        _timestamp(name, getattr(record, name), required=True)
    authority = getattr(record, "evidence_authority")
    _required_text("evidence_authority", authority)
    if authority == "direct_sec" and getattr(record, "source_name").lower() != "sec":
        raise ValueError("direct_sec evidence authority is reserved for the SEC source")
    object.__setattr__(record, "metadata", validate_metadata(getattr(record, "metadata")))


@dataclass(frozen=True)
class NarrativeRecord:
    """Narrative evidence stored as SQLite metadata plus Chroma content."""

    corpus_item_id: str
    source_name: str
    source_category: str
    provider_record_id: str | None
    original_publisher: str | None
    item_type: str
    title: str
    body: str
    published_at: str | None
    observed_at: str | None
    accessed_at: str
    ingested_at: str
    source_url: str
    canonical_url: str | None
    license_label: str
    normalization_version: str
    content_hash: str
    document_family: str
    event_type: str | None = None
    summary: str | None = None
    language: str = "en"
    effective_at: str | None = None
    as_of_at: str | None = None
    security_ids: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()
    index_codes: tuple[str, ...] = ()
    sectors: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    indexing_status: str = "pending"
    evidence_authority: str = "provider"

    def __post_init__(self) -> None:
        _validate_provenance(self)
        for name in ("corpus_item_id", "item_type", "title", "body", "language", "document_family"):
            max_chars = MAX_BODY_CHARS if name == "body" else MAX_TITLE_CHARS if name == "title" else 2_000
            _required_text(name, getattr(self, name), max_chars=max_chars)
        _optional_text("event_type", self.event_type)
        _optional_text("summary", self.summary, max_chars=MAX_SUMMARY_CHARS)
        _timestamp("effective_at", self.effective_at)
        _timestamp("as_of_at", self.as_of_at)
        if not _SHA256.fullmatch(self.content_hash):
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
        if self.indexing_status not in INDEXING_STATUSES:
            raise ValueError(f"indexing_status must be one of {sorted(INDEXING_STATUSES)}")
        for name in ("security_ids", "tickers", "index_codes", "sectors"):
            object.__setattr__(self, name, _tuple_of_text(name, getattr(self, name)))


@dataclass(frozen=True)
class ObservationRecord:
    """Structured numeric or textual observation that is never embedded."""

    observation_id: str
    metric_id: str
    series_id: str | None
    value_text: str
    value_numeric: float | None
    unit: str
    frequency: str
    period_start: str | None
    period_end: str
    vintage_at: str | None
    as_of_at: str | None
    scope: str
    security_ids: tuple[str, ...]
    tickers: tuple[str, ...]
    sector: str | None
    source_name: str
    source_category: str
    provider_record_id: str | None
    original_publisher: str | None
    source_url: str
    canonical_url: str | None
    published_at: str | None
    observed_at: str | None
    accessed_at: str
    ingested_at: str
    license_label: str
    normalization_version: str = NORMALIZATION_VERSION
    metadata: Mapping[str, object] = field(default_factory=dict)
    evidence_authority: str = "provider"

    def __post_init__(self) -> None:
        _validate_provenance(self)
        for name in ("observation_id", "metric_id", "value_text", "unit", "frequency", "period_end"):
            _required_text(name, getattr(self, name))
        _optional_text("series_id", self.series_id)
        _optional_text("sector", self.sector)
        for name in ("period_start", "period_end", "vintage_at", "as_of_at"):
            _timestamp(name, getattr(self, name), required=name == "period_end")
        if self.value_numeric is not None and not math.isfinite(self.value_numeric):
            raise ValueError("value_numeric must be finite")
        if self.scope not in OBSERVATION_SCOPES:
            raise ValueError(f"scope must be one of {sorted(OBSERVATION_SCOPES)}")
        object.__setattr__(self, "security_ids", _tuple_of_text("security_ids", self.security_ids))
        object.__setattr__(self, "tickers", _tuple_of_text("tickers", self.tickers))
        if self.scope == "security" and not self.security_ids:
            raise ValueError("security scope requires at least one security_id")


@dataclass(frozen=True)
class EventRecord:
    """Structured company, regulatory, or corporate-action event."""

    event_id: str
    event_type: str
    effective_at: str | None
    announced_at: str | None
    status: str
    security_ids: tuple[str, ...]
    source_corpus_item_ids: tuple[str, ...]
    source_name: str
    source_category: str
    provider_record_id: str | None
    original_publisher: str | None
    source_url: str
    canonical_url: str | None
    published_at: str | None
    observed_at: str | None
    accessed_at: str
    ingested_at: str
    license_label: str
    normalization_version: str = NORMALIZATION_VERSION
    amount: float | None = None
    currency: str | None = None
    rate: float | None = None
    ratio: float | None = None
    action_date: str | None = None
    classifier_version: str | None = None
    explanation: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    evidence_authority: str = "provider"

    def __post_init__(self) -> None:
        _validate_provenance(self)
        for name in ("event_id", "event_type", "status"):
            _required_text(name, getattr(self, name))
        for name in ("currency", "classifier_version"):
            _optional_text(name, getattr(self, name))
        _optional_text("explanation", self.explanation, max_chars=MAX_EXPLANATION_CHARS)
        for name in ("effective_at", "announced_at", "action_date"):
            _timestamp(name, getattr(self, name))
        for name in ("amount", "rate", "ratio"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "security_ids", _tuple_of_text("security_ids", self.security_ids))
        object.__setattr__(
            self,
            "source_corpus_item_ids",
            _tuple_of_text("source_corpus_item_ids", self.source_corpus_item_ids),
        )
