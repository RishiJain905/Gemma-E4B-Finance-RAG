"""
src/middleware/evidence_taxonomy.py
Versioned finance evidence vocabulary, normalization, filtering, and bounded ranking.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from datetime import datetime
from typing import Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

TAXONOMY_VERSION = "finance-evidence-taxonomy-v1"

SOURCE_CATEGORIES = (
    "sec", "issuer", "market_data", "company_news", "central_bank",
    "treasury", "economic_agency", "regulator", "sector_agency",
    "transcript", "estimates", "global_news",
)

ITEM_TYPES = (
    "filing", "filing_exhibit", "press_release", "news", "speech",
    "policy_release", "economic_release", "market_observation",
    "corporate_action", "regulatory_event", "contract_award", "recall",
    "transcript", "estimate",
)

SEC_EVENT_TYPES = (
    "debt_raise", "equity_raise", "convertible_offering",
    "shelf_registration", "prospectus_update", "acquisition", "divestiture",
    "earnings_release", "guidance_change", "leadership_change", "auditor_change",
    "buyback", "dividend_change", "insider_transaction",
    "beneficial_ownership_change",
)

MACRO_SECTOR_EVENT_TYPES = (
    "monetary_policy_decision", "economic_release", "treasury_auction",
    "market_operation", "enforcement_action", "investigation", "recall",
    "safety_alert", "shortage", "contract_award", "split", "dividend",
    "corporate_action",
)

EVENT_TYPES = SEC_EVENT_TYPES + MACRO_SECTOR_EVENT_TYPES

AUTHORITY_TIERS = (
    "primary", "structured", "licensed", "analysis", "discovery",
)

AUTHORITY_MAX_BOOST = 0.025
ENTITY_MATCH_MAX_BOOST = 0.04
TYPE_MATCH_MAX_BOOST = 0.03
DATE_MATCH_MAX_BOOST = 0.02
RECENCY_STRONG_MAX_BOOST = 0.05
RECENCY_WEAK_MAX_BOOST = 0.005
DEFAULT_RRF_MAX_SCORE = 2.0 / 61.0

_CENTRAL_BANK_SOURCES = {
    "federal_reserve", "federal reserve", "fed", "new_york_fed", "ny_fed",
}
_TREASURY_SOURCES = {"treasury", "us_treasury", "u.s. treasury"}
_ECONOMIC_SOURCES = {"bea", "bls"}
_REGULATOR_SOURCES = {"sec", "cftc", "finra", "fdic", "occ", "federal_register"}
_SECTOR_SOURCES = {"eia", "nhtsa", "openfda", "fda", "usaspending"}

_SOURCE_CATEGORY_MAP = {
    "regulatory_filing": "sec",
    "sec_filing": "sec",
    "issuer_release": "issuer",
    "ir": "issuer",
    "news_vendor": "company_news",
    "vendor_news": "company_news",
    "corporate_action": "market_data",
    "vendor_filing_metadata": "market_data",
    "official_government": "treasury",
    "official_market_operation": "central_bank",
    "official_release": "economic_agency",
    "official_macro": "economic_agency",
    "official_regulatory": "regulator",
    "transcripts": "transcript",
    "estimate": "estimates",
    "gdelt": "global_news",
}

_ITEM_TYPE_MAP = {
    "sec_filing": "filing",
    "filing": "filing",
    "sec_exhibit": "filing_exhibit",
    "exhibit": "filing_exhibit",
    "issuer_release": "press_release",
    "official_release": "economic_release",
    "market_bar": "market_observation",
    "observation": "market_observation",
    "corporate_action": "corporate_action",
    "event": "regulatory_event",
    "award": "contract_award",
    "transcripts": "transcript",
    "estimates": "estimate",
}

_EVENT_TYPE_MAP = {
    "award": "contract_award",
    "safety": "safety_alert",
    "cash_dividend": "dividend",
    "stock_dividend": "dividend",
    "reverse_split": "split",
    "stock_split": "split",
}

_LIST_FIELDS = {
    "security": ("canonical_security", "ticker", "tickers", "security_id", "security_ids"),
    "ticker": ("canonical_security", "ticker", "tickers"),
    "alias": ("ticker", "tickers", "aliases"),
    "index_membership": ("index_memberships", "index_codes"),
    "sector": ("sector", "sectors"),
    "industry": ("industry", "industries"),
    "source_category": ("source_category",),
    "source": ("source", "source_name"),
    "item_type": ("item_type",),
    "event_type": ("event_type",),
    "form": ("form", "filing_type"),
    "item": ("item", "filing_item", "items"),
    "exhibit": ("exhibit", "exhibits", "document_type"),
    "freshness_status": ("freshness_status", "freshness"),
    "indexing_status": ("indexing_status", "indexing_state"),
    "authority_tier": ("authority_tier", "evidence_authority"),
    "coverage_tier": ("coverage_tier", "discovery_scope"),
}

_DATE_FILTERS = {
    "published": "published_at",
    "effective": "effective_at",
    "as_of": "as_of_at",
}

_FILTER_PLURALS = {
    "security": "securities",
    "alias": "aliases",
    "index_membership": "index_memberships",
    "industry": "industries",
    "source_category": "source_categories",
    "freshness_status": "freshness_statuses",
    "indexing_status": "indexing_statuses",
    "authority_tier": "authority_tiers",
    "coverage_tier": "coverage_tiers",
}


def _metadata(row: Mapping) -> dict:
    metadata = row.get("metadata") if isinstance(row, Mapping) else None
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _first(row: Mapping, metadata: Mapping, *names: str) -> object:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
        value = metadata.get(name)
        if value not in (None, ""):
            return value
    return None


def _values(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = value
    else:
        parts = (value,)
    return tuple(str(part).strip() for part in parts if str(part).strip())


def _source_category(raw: object, source: str) -> str:
    category = str(raw or "").strip().lower()
    source_key = source.strip().lower()
    if category in SOURCE_CATEGORIES:
        return category
    # Content semantics win over provider identity: one provider may expose
    # market observations, filing metadata, and news through separate feeds.
    if category in {
        "regulatory_filing", "sec_filing", "issuer_release", "ir",
        "news_vendor", "vendor_news", "corporate_action",
        "vendor_filing_metadata", "transcripts", "estimate", "gdelt",
    }:
        return _SOURCE_CATEGORY_MAP[category]
    if source_key == "sec":
        return "sec"
    if source_key.startswith("sec_"):
        return "sec"
    if source_key in {"yfinance", "yahoo_finance", "massive", "polygon"}:
        return "market_data"
    if source_key in {"finnhub"}:
        return "company_news"
    if source_key in {"fred"}:
        return "central_bank"
    if source_key in {"earnings_transcripts", "transcript"}:
        return "transcript"
    if source_key in {"estimate", "estimates"}:
        return "estimates"
    if source_key in _CENTRAL_BANK_SOURCES:
        return "central_bank"
    if source_key in _TREASURY_SOURCES:
        return "treasury"
    if source_key in _ECONOMIC_SOURCES:
        return "economic_agency"
    if source_key in _REGULATOR_SOURCES:
        return "regulator"
    if source_key in _SECTOR_SOURCES:
        return "sector_agency"
    if source_key == "gdelt":
        return "global_news"
    return _SOURCE_CATEGORY_MAP.get(category, "company_news" if category else "global_news")


def _item_type(raw: object, source_category: str, event_type: Optional[str]) -> str:
    value = str(raw or "").strip().lower()
    if value in ITEM_TYPES:
        return value
    if event_type == "recall":
        return "recall"
    if event_type == "contract_award":
        return "contract_award"
    if source_category in {"regulator", "sector_agency"} and event_type:
        return "regulatory_event"
    if source_category == "central_bank":
        return "policy_release"
    if source_category in {"treasury", "economic_agency"}:
        return "economic_release"
    return _ITEM_TYPE_MAP.get(value, "news" if source_category in {"company_news", "global_news"} else "press_release")


def _authority_rank(raw: object, source_category: str, item_type: str) -> int:
    value = str(raw or "").strip().lower()
    if source_category in {"sec", "issuer", "central_bank", "treasury", "economic_agency", "regulator", "sector_agency"}:
        return 1
    if source_category == "global_news":
        return 5
    if source_category in {"transcript", "estimates"}:
        return 4
    if source_category == "market_data" and item_type in {
        "market_observation", "corporate_action",
    }:
        return 2
    if (
        source_category in {"company_news", "market_data"}
        or value in {"provider", "licensed"}
    ):
        return 3
    return 5


def _date_semantics(row: Mapping, metadata: Mapping, item_type: str) -> dict:
    published = _first(row, metadata, "published_at", "filing_date", "date")
    effective = _first(row, metadata, "effective_at", "action_date")
    observation = _first(row, metadata, "observation_period", "period_end", "period")
    vintage = _first(row, metadata, "as_of_at", "vintage_at", "as_of")
    ingested = _first(row, metadata, "ingested_at", "accessed_at")
    if item_type in {"market_observation", "estimate"}:
        domain_kind, domain = ("as_of_at", vintage or observation or published)
    elif item_type in {"corporate_action", "regulatory_event", "contract_award", "recall"}:
        domain_kind, domain = ("effective_at", effective or published or vintage)
    else:
        domain_kind, domain = ("published_at", published or effective or vintage)
    return {
        "published_at": published,
        "effective_at": effective,
        "observation_period": observation,
        "source_vintage": vintage,
        "ingested_at": ingested,
        "domain_timestamp": domain,
        "domain_timestamp_kind": domain_kind,
    }


def normalize_evidence(row: Mapping) -> dict:
    """Return one evidence row with additive stable taxonomy metadata."""
    if not isinstance(row, Mapping):
        return {}
    result = dict(row)
    metadata = _metadata(row)
    source = str(_first(row, metadata, "source", "source_name", "source_type") or "unknown")
    raw_category = _first(row, metadata, "source_category")
    category = _source_category(raw_category, source)
    raw_event = _first(row, metadata, "event_type")
    raw_event_value = str(raw_event or "").strip().lower()
    event_type = _EVENT_TYPE_MAP.get(raw_event_value, raw_event_value) or None
    if event_type not in EVENT_TYPES:
        event_type = None
    raw_item = _first(row, metadata, "item_type")
    if not raw_item and _first(row, metadata, "metric") is not None:
        if category == "estimates":
            raw_item = "estimate"
        elif category == "sec":
            raw_item = "filing"
        else:
            raw_item = "market_observation"
    item_type = _item_type(raw_item, category, event_type)
    raw_authority = _first(row, metadata, "authority_tier", "evidence_authority")
    authority_rank = _authority_rank(raw_authority, category, item_type)
    semantics = _date_semantics(row, metadata, item_type)
    canonical = _first(row, metadata, "canonical_security", "ticker")
    coverage = _first(row, metadata, "coverage_tier", "discovery_scope") or (
        "global" if category in {"central_bank", "treasury", "economic_agency", "global_news"} else "unknown"
    )

    if raw_category and str(raw_category).lower() != category:
        metadata.setdefault("provider_source_category", raw_category)
    if raw_item and str(raw_item).lower() != item_type:
        metadata.setdefault("provider_item_type", raw_item)
    if raw_event and str(raw_event).lower() != event_type:
        metadata.setdefault("provider_event_type", raw_event)
    metadata.update({
        "taxonomy_version": TAXONOMY_VERSION,
        "source": source,
        "source_category": category,
        "item_type": item_type,
        "authority_tier": str(raw_authority or AUTHORITY_TIERS[authority_rank - 1]),
        "authority_rank": authority_rank,
        "canonical_security": str(canonical).upper() if canonical else None,
        "coverage_tier": str(coverage).lower(),
        "date_semantics": semantics,
        "domain_timestamp": semantics["domain_timestamp"],
        "domain_timestamp_kind": semantics["domain_timestamp_kind"],
        "publication_time": semantics["published_at"],
        "effective_time": semantics["effective_at"],
        "observation_period": semantics["observation_period"],
        "source_vintage": semantics["source_vintage"],
        "ingestion_time": semantics["ingested_at"],
    })
    if event_type:
        metadata["event_type"] = event_type
    result["metadata"] = metadata
    return result


def _requested_values(filters: Mapping, singular: str) -> tuple[str, ...]:
    value = filters.get(singular)
    if value in (None, ""):
        value = filters.get(_FILTER_PLURALS.get(singular, f"{singular}s"))
    return _values(value)


def _field_values(row: Mapping, metadata: Mapping, fields: Iterable[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        values.extend(_values(_first(row, metadata, field)))
    return tuple(dict.fromkeys(values))


def evidence_matches_filters(row: Mapping, filters: Optional[Mapping] = None) -> bool:
    """Return whether one row satisfies every supplied stable evidence facet."""
    if not filters:
        return True
    normalized = normalize_evidence(row)
    metadata = normalized.get("metadata") or {}
    for name, fields in _LIST_FIELDS.items():
        requested = _requested_values(filters, name)
        if not requested:
            continue
        actual = _field_values(normalized, metadata, fields)
        requested_keys = {value.casefold() for value in requested}
        actual_keys = {value.casefold() for value in actual}
        if not requested_keys.intersection(actual_keys):
            if name == "authority_tier":
                rank = str(metadata.get("authority_rank", ""))
                stable = AUTHORITY_TIERS[int(rank) - 1] if rank.isdigit() and 1 <= int(rank) <= 5 else ""
                if not requested_keys.intersection({rank.casefold(), stable.casefold()}):
                    return False
            else:
                return False
    for prefix, field in _DATE_FILTERS.items():
        value = str(metadata.get(field) or "")
        lower = filters.get(f"{prefix}_from")
        upper = filters.get(f"{prefix}_to")
        if lower and (not value or value < str(lower)):
            return False
        if upper and (not value or value > str(upper)):
            return False
    return True


def filter_evidence(rows: Iterable[Mapping], filters: Optional[Mapping] = None) -> list[dict]:
    """Normalize and composably filter a bounded iterable of evidence rows."""
    return [normalize_evidence(row) for row in rows if evidence_matches_filters(row, filters)]


def _relevance(row: Mapping) -> float:
    for key in ("rerank_score", "similarity"):
        value = row.get(key)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass
    fusion_score = row.get("fusion_score")
    if fusion_score is not None:
        try:
            # Default two-channel RRF scores occupy roughly [0, 2/(k+1)].
            # Normalize that range before applying bounded policy boosts so a
            # 0.025 authority term cannot overwhelm lexical/semantic relevance.
            return max(0.0, min(1.0, float(fusion_score) / DEFAULT_RRF_MAX_SCORE))
        except (TypeError, ValueError):
            pass
    distance = row.get("distance")
    if distance is not None:
        try:
            return max(0.0, min(1.0, 1.0 - float(distance)))
        except (TypeError, ValueError):
            pass
    score = row.get("score")
    try:
        return float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _query_years(query: str) -> set[str]:
    return set(re.findall(r"\b(?:19|20)\d{2}\b", query))


def _recency_mode(query: str, rows: list[dict]) -> str:
    lowered = query.lower()
    if "historical" in lowered or "history" in lowered or _query_years(query):
        return "weak"
    if re.search(r"\b(latest|newest|current|recent|today|news|what happened)\b", lowered):
        return "strong"
    item_types = {str((row.get("metadata") or {}).get("item_type") or "") for row in rows}
    if item_types.intersection({"news", "corporate_action", "regulatory_event", "recall"}):
        return "strong"
    return "weak" if item_types and item_types <= {"filing", "filing_exhibit"} else "neutral"


def rank_evidence(
    rows: Iterable[Mapping],
    *,
    query: str,
    filters: Optional[Mapping] = None,
    authority_max_boost: float = AUTHORITY_MAX_BOOST,
    recency_max_boost: Optional[float] = None,
) -> list[dict]:
    """Rank after relevance using bounded exact, recency, and authority boosts.

    The additive formula is ``relevance + entity(<=.04) + type(<=.03) +
    date(<=.02) + recency(<=.05 latest/.005 historical) + authority(<=.025)``.
    Authority is deliberately the smallest policy term and cannot overturn a
    relevance difference greater than its configured bound when other terms tie.
    """
    normalized = [normalize_evidence(row) for row in rows]
    mode = _recency_mode(query, normalized)
    if recency_max_boost is None:
        recency_max_boost = (
            RECENCY_STRONG_MAX_BOOST if mode == "strong"
            else RECENCY_WEAK_MAX_BOOST if mode == "weak" else 0.015
        )
    timestamps = sorted({
        str((row.get("metadata") or {}).get("domain_timestamp") or "")
        for row in normalized
        if (row.get("metadata") or {}).get("domain_timestamp")
    })
    parsed_timestamps: dict[str, datetime] = {}
    for value in timestamps:
        try:
            parsed_timestamps[value] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
    newest_timestamp = max(parsed_timestamps.values(), default=None)
    years = _query_years(query)
    requested_security = {
        value.casefold() for value in _requested_values(filters or {}, "security")
    } | {value.casefold() for value in _requested_values(filters or {}, "ticker")}
    requested_types = {
        value.casefold() for field in ("item_type", "event_type")
        for value in _requested_values(filters or {}, field)
    }

    ranked: list[dict] = []
    for row in normalized:
        metadata = row.get("metadata") or {}
        relevance = _relevance(row)
        actual_security = {
            value.casefold() for value in _field_values(
                row, metadata, _LIST_FIELDS["security"]
            )
        }
        entity_boost = ENTITY_MATCH_MAX_BOOST if requested_security.intersection(actual_security) else 0.0
        actual_types = {
            str(metadata.get("item_type") or "").casefold(),
            str(metadata.get("event_type") or "").casefold(),
        }
        type_boost = TYPE_MATCH_MAX_BOOST if requested_types.intersection(actual_types) else 0.0
        domain_timestamp = str(metadata.get("domain_timestamp") or "")
        date_boost = DATE_MATCH_MAX_BOOST if years and any(domain_timestamp.startswith(year) for year in years) else 0.0
        recency_boost = 0.0
        if domain_timestamp and newest_timestamp and recency_max_boost:
            parsed = parsed_timestamps.get(domain_timestamp)
            if parsed is not None:
                age_days = max(0.0, (newest_timestamp - parsed).total_seconds() / 86400)
                recency_boost = float(recency_max_boost) * math.exp(-age_days / 7.0)
        authority_rank = int(metadata.get("authority_rank") or 5)
        authority_boost = max(0.0, float(authority_max_boost)) * (5 - authority_rank) / 4
        ranking_score = relevance + entity_boost + type_boost + date_boost + recency_boost + authority_boost
        ranked_row = dict(row)
        ranked_row.update({
            "relevance_score": relevance,
            "entity_match_boost": round(entity_boost, 8),
            "type_match_boost": round(type_boost, 8),
            "date_match_boost": round(date_boost, 8),
            "recency_boost": round(recency_boost, 8),
            "authority_boost": round(authority_boost, 8),
            "ranking_score": round(ranking_score, 8),
        })
        ranked.append(ranked_row)
    return sorted(ranked, key=lambda row: (-row["ranking_score"], str(row.get("id") or "")))


def _coverage_key(row: Mapping) -> str:
    metadata = _metadata(row)
    explicit = _first(row, metadata, "event_id", "coverage_key")
    if explicit:
        return f"event:{explicit}"
    event_type = metadata.get("event_type")
    security = metadata.get("canonical_security") or metadata.get("ticker")
    timestamp = str(metadata.get("domain_timestamp") or "")[:10]
    if event_type and (security or timestamp):
        return f"derived:{security}:{event_type}:{timestamp}"
    return f"item:{row.get('id') or hashlib.sha256(str(row).encode()).hexdigest()}"


def _duplicate_key(row: Mapping) -> str:
    metadata = _metadata(row)
    syndicated = _first(row, metadata, "syndicated_key")
    if syndicated:
        return f"syndicated:{syndicated}"
    body = str(row.get("document") or row.get("text") or row.get("content") or "")
    normalized = re.sub(r"\W+", " ", body.lower()).strip()
    return f"body:{hashlib.sha256(normalized.encode()).hexdigest()}" if normalized else f"id:{row.get('id')}"


def pack_event_coverage(
    rows: Iterable[Mapping], *, limit: int, max_secondary_per_event: int = 2,
) -> list[dict]:
    """Pack primary coverage plus bounded material secondaries without duplicates."""
    if limit <= 0:
        return []
    normalized_rows = [normalize_evidence(row) for row in rows]
    primary_by_event: dict[str, tuple[tuple, dict]] = {}
    for row in normalized_rows:
        event = _coverage_key(row)
        current = primary_by_event.get(event)
        metadata = _metadata(row)
        raw_authority = str(metadata.get("authority_tier") or "").lower()
        priority = (
            int(metadata.get("authority_rank") or 5),
            0 if raw_authority == "direct_sec" else 1,
            -float(row.get("ranking_score") or _relevance(row)),
            str(row.get("id") or ""),
        )
        if current is None or priority < current[0]:
            primary_by_event[event] = (priority, row)

    candidates_by_event: dict[str, list[dict]] = {}
    seen_event_duplicates: set[tuple[str, str]] = set()
    for row in normalized_rows:
        event = _coverage_key(row)
        identity = (event, _duplicate_key(row))
        if identity in seen_event_duplicates:
            continue
        seen_event_duplicates.add(identity)
        candidates_by_event.setdefault(event, []).append(row)

    allowed_by_event: dict[str, set[int]] = {}
    for event, candidates in candidates_by_event.items():
        primary = primary_by_event[event][1]
        secondaries = [candidate for candidate in candidates if candidate is not primary]
        secondaries.sort(key=lambda candidate: (
            1 if _first(candidate, _metadata(candidate), "syndicated_key") else 0,
            int(_metadata(candidate).get("authority_rank") or 5),
            -float(candidate.get("ranking_score") or _relevance(candidate)),
            str(candidate.get("id") or ""),
        ))
        selected = [primary, *secondaries[:max(0, int(max_secondary_per_event))]]
        allowed_by_event[event] = {id(candidate) for candidate in selected}

    packed: list[dict] = []
    seen_duplicates: set[str] = set()
    event_counts: dict[str, int] = {}
    max_per_event = 1 + max(0, int(max_secondary_per_event))
    for row in normalized_rows:
        event = _coverage_key(row)
        primary = primary_by_event[event][1]
        if event_counts.get(event, 0) == 0 and primary is not row:
            primary_duplicate = _duplicate_key(primary)
            if primary_duplicate not in seen_duplicates:
                seen_duplicates.add(primary_duplicate)
                event_counts[event] = 1
                packed.append(primary)
                if len(packed) >= limit:
                    break
        if id(row) not in allowed_by_event[event]:
            continue
        duplicate = _duplicate_key(row)
        if duplicate in seen_duplicates:
            continue
        if event_counts.get(event, 0) >= max_per_event:
            continue
        seen_duplicates.add(duplicate)
        event_counts[event] = event_counts.get(event, 0) + 1
        packed.append(row)
        if len(packed) >= limit:
            break
    return packed
