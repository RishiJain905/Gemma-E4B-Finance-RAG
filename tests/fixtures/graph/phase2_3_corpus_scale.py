"""tests/fixtures/graph/phase2_3_corpus_scale.py
Deterministic builder that seeds a Phase 2.3-scale corpus for explorer tests.

Bulk-inserts securities, overlapping index memberships, corpus items,
observations, and events straight into SQLite so the read-only corpus explorer
can be measured against a realistic aggregation surface without embedding any
document. Chroma is only stubbed with a handful of records for detail paths.

The builder is fully deterministic given ``seed`` and returns the canonical
per-dimension tallies so tests can prove aggregate counts against seeded rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

# 11 GICS-style sectors, each with a couple of industries.
SECTORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Technology", ("Semiconductors", "Software")),
    ("Health Care", ("Biotechnology", "Medical Devices")),
    ("Financials", ("Banks", "Insurance")),
    ("Consumer Discretionary", ("Retail", "Automobiles")),
    ("Communication Services", ("Interactive Media", "Telecom")),
    ("Industrials", ("Aerospace", "Machinery")),
    ("Consumer Staples", ("Beverages", "Household Products")),
    ("Energy", ("Oil & Gas", "Renewables")),
    ("Utilities", ("Electric Utilities", "Water Utilities")),
    ("Real Estate", ("REITs", "Real Estate Services")),
    ("Materials", ("Chemicals", "Metals & Mining")),
)

# Narrative source category -> (sources, item types) that land in corpus_items.
NARRATIVE_CATEGORIES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("sec", ("sec_edgar",), ("filing", "exhibit")),
    ("issuer", ("ir",), ("issuer_release",)),
    ("market_data", ("massive", "yahoo"), ("market_summary", "corporate_action")),
    ("company_news", ("finnhub", "massive_news"), ("news",)),
    ("sector_agency", ("openfda", "nhtsa", "usaspending"), ("sector_event",)),
    ("transcript", ("transcripts",), ("transcript",)),
    ("estimates", ("estimates",), ("estimate",)),
    ("global_news", ("gdelt",), ("news",)),
)

# Structured observation feeds (never embedded; item_type derives to observation).
OBSERVATION_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("central_bank", "fred"),
    ("treasury", "treasury"),
    ("economic_agency", "bls"),
    ("economic_agency", "bea"),
    ("economic_agency", "eia"),
)

EVENT_TYPES: tuple[str, ...] = (
    "dividend", "split", "financing", "guidance", "recall",
)
NARRATIVE_EVENT_TYPES: tuple[Optional[str], ...] = (
    None, None, None, "earnings", "guidance",
)
INDEXING_STATES: tuple[str, ...] = (
    "indexed", "indexed", "indexed", "indexed", "indexed", "indexed",
    "pending", "error", "not_applicable", "indexed",
)


@dataclass
class ScaleSummary:
    """Canonical seeded tallies used to assert aggregate consistency."""

    securities: int = 0
    corpus_items: int = 0
    observations: int = 0
    events: int = 0
    linked_items: int = 0
    filing_accession: str = ""
    section_parent_ids: tuple[str, ...] = ()
    by_source_category: Counter = field(default_factory=Counter)
    by_item_type: Counter = field(default_factory=Counter)
    by_indexing_state: Counter = field(default_factory=Counter)
    by_index: Counter = field(default_factory=Counter)
    by_sector: Counter = field(default_factory=Counter)
    by_year: Counter = field(default_factory=Counter)
    by_event_type: Counter = field(default_factory=Counter)


_CORPUS_COLUMNS = (
    "corpus_item_id", "source", "source_category", "item_type", "event_type",
    "title", "normalized_headline", "language", "published_at", "accessed_at",
    "ingested_at", "source_url", "content_hash", "document_family",
    "document_family_id", "indexing_status", "license_label",
    "normalization_version", "evidence_authority", "narrative_bytes",
    "metadata_bytes",
)


def build_corpus_scale(
    store: Any,
    *,
    securities: int = 600,
    items: int = 100_000,
    observations: int = 2_000,
    events: int = 2_000,
    seed: int = 20_260_714,
) -> ScaleSummary:
    """Seed a deterministic, bounded-but-large corpus into ``store``.

    Returns a :class:`ScaleSummary` with the canonical per-dimension tallies.
    The layout is deterministic by construction; ``seed`` is reserved for future
    variation and does not perturb the seeded tallies today.
    """
    sqlite = store.sqlite
    summary = ScaleSummary()

    # -- Securities + overlapping index memberships ------------------------
    security_rows: list[tuple] = []
    membership_rows: list[tuple] = []
    security_index: list[frozenset[str]] = []
    now = "2026-06-01T00:00:00Z"
    for i in range(securities):
        sector, industries = SECTORS[i % len(SECTORS)]
        industry = industries[i % len(industries)]
        sid = f"SEC-{i:05d}"
        ticker = f"T{i:04d}"
        security_rows.append((
            sid, ticker, ticker.lower(), f"Company {i}", "NASDAQ",
            f"{1000000 + i:010d}", "common_stock", None, sector, industry,
            1, now, now, now,
        ))
        indexes: set[str] = set()
        # ~5/6 in sp500, ~1/6 in nasdaq100, deterministic overlap band.
        if i % 6 != 5:
            indexes.add("sp500")
        if i % 6 in (0, 5):
            indexes.add("nasdaq100")
        for index_code in sorted(indexes):
            membership_rows.append((
                sid, index_code, "2026-01-01", 1, "ivv", now,
            ))
        security_index.append(frozenset(indexes))

    with sqlite._connect() as conn:
        conn.executemany(
            "INSERT INTO securities (security_id, ticker, normalized_ticker, "
            "company_name, exchange, cik, security_type, share_class, sector, "
            "industry, active, first_seen_at, last_seen_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            security_rows,
        )
        conn.executemany(
            "INSERT INTO security_memberships (security_id, index_code, "
            "effective_from, active, source, observed_at) VALUES (?,?,?,?,?,?)",
            membership_rows,
        )
    summary.securities = securities

    # -- Corpus items (narrative metadata ledger) --------------------------
    corpus_rows: list[tuple] = []
    link_rows: list[tuple] = []
    for i in range(items):
        category, sources, item_types = NARRATIVE_CATEGORIES[
            i % len(NARRATIVE_CATEGORIES)]
        source = sources[i % len(sources)]
        item_type = item_types[i % len(item_types)]
        event_type = NARRATIVE_EVENT_TYPES[i % len(NARRATIVE_EVENT_TYPES)]
        state = INDEXING_STATES[i % len(INDEXING_STATES)]
        year = 2024 + (i % 3)
        month = 1 + (i % 12)
        published_at = f"{year:04d}-{month:02d}-15T00:00:00Z"
        cid = f"CI-{i:07d}"
        narrative_bytes = 200 + (i % 800)
        metadata_bytes = 50 + (i % 200)
        corpus_rows.append((
            cid, source, category, item_type, event_type,
            f"Item {i}", f"item {i}", "en", published_at, now, now,
            f"https://example.test/{source}/{i}", f"{i:064x}"[:64],
            f"fam-{i}", cid, state, "public", "v1", "provider",
            narrative_bytes, metadata_bytes,
        ))
        summary.by_source_category[category] += 1
        summary.by_item_type[item_type] += 1
        summary.by_indexing_state[state] += 1
        summary.by_year[str(year)] += 1
        summary.by_event_type[event_type or "none"] += 1
        # Link most items to a security (every 50th stays unlinked).
        if i % 50 != 49:
            sec_idx = i % securities
            sid = f"SEC-{sec_idx:05d}"
            ticker = f"T{sec_idx:04d}"
            link_rows.append((cid, sid, ticker))
            summary.linked_items += 1
            sector = SECTORS[sec_idx % len(SECTORS)][0]
            summary.by_sector[sector] += 1
            for index_code in security_index[sec_idx]:
                summary.by_index[index_code] += 1
        else:
            summary.by_sector["unclassified"] += 1
            summary.by_index["unlinked"] += 1

    with sqlite._connect() as conn:
        conn.executemany(
            f"INSERT INTO corpus_items ({', '.join(_CORPUS_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _CORPUS_COLUMNS)})",
            corpus_rows,
        )
        conn.executemany(
            "INSERT INTO corpus_item_securities (corpus_item_id, security_id, "
            "ticker) VALUES (?,?,?)",
            link_rows,
        )
    summary.corpus_items = items

    # -- Structured observations (never embedded) --------------------------
    observation_rows: list[tuple] = []
    obs_link_rows: list[tuple] = []
    for i in range(observations):
        category, source = OBSERVATION_CATEGORIES[i % len(OBSERVATION_CATEGORIES)]
        oid = f"OBS-{i:06d}"
        year = 2024 + (i % 3)
        observation_rows.append((
            oid, f"metric-{i % 40}", None, str(i), float(i), "index",
            "monthly", None, f"{year:04d}-{1 + i % 12:02d}-01", None, None,
            "global", "[]", None, source, category, f"prov-{i}", None,
            f"https://example.test/{source}/obs/{i}", None,
            f"{year:04d}-{1 + i % 12:02d}-05T00:00:00Z", None, now, now,
            "public", "v1", "provider",
        ))
        # Observations are global (unlinked): they land in the not_applicable
        # indexing state and the 'unlinked' index/sector buckets.
        summary.by_source_category[category] += 1
        summary.by_indexing_state["not_applicable"] += 1
        summary.by_year[str(year)] += 1
        summary.by_index["unlinked"] += 1
        summary.by_sector["unclassified"] += 1
        summary.by_event_type["none"] += 1
    with sqlite._connect() as conn:
        conn.executemany(
            "INSERT INTO corpus_observations (observation_id, metric_id, "
            "series_id, value_text, value_numeric, unit, frequency, "
            "period_start, period_end, vintage_at, as_of_at, scope, "
            "tickers_json, sector, source_name, source_category, "
            "provider_record_id, original_publisher, source_url, canonical_url, "
            "published_at, observed_at, accessed_at, ingested_at, license_label, "
            "normalization_version, evidence_authority) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            observation_rows,
        )
        if obs_link_rows:
            conn.executemany(
                "INSERT INTO observation_securities (observation_id, "
                "security_id) VALUES (?,?)",
                obs_link_rows,
            )
    summary.observations = observations
    summary.by_item_type["observation"] += observations

    # -- Structured events -------------------------------------------------
    event_rows: list[tuple] = []
    event_link_rows: list[tuple] = []
    for i in range(events):
        event_type = EVENT_TYPES[i % len(EVENT_TYPES)]
        eid = f"EVT-{i:06d}"
        sec_idx = i % securities
        year = 2024 + (i % 3)
        event_rows.append((
            eid, event_type, f"{year:04d}-{1 + i % 12:02d}-10", None,
            "confirmed", None, None, None, None, None, "v1", None,
            "sec_edgar", "sec", f"evt-prov-{i}", None,
            f"https://example.test/events/{i}", None,
            f"{year:04d}-{1 + i % 12:02d}-10T00:00:00Z", None, now, now,
            "public", "v1", "provider",
        ))
        event_link_rows.append((eid, f"SEC-{sec_idx:05d}"))
        summary.by_event_type[event_type] += 1
        summary.by_source_category["sec"] += 1
        summary.by_indexing_state["not_applicable"] += 1
        summary.by_year[str(year)] += 1
        summary.by_sector[SECTORS[sec_idx % len(SECTORS)][0]] += 1
        for index_code in security_index[sec_idx]:
            summary.by_index[index_code] += 1
    with sqlite._connect() as conn:
        conn.executemany(
            "INSERT INTO corpus_events (event_id, event_type, effective_at, "
            "announced_at, status, amount, currency, rate, ratio, action_date, "
            "classifier_version, explanation, source_name, source_category, "
            "provider_record_id, original_publisher, source_url, canonical_url, "
            "published_at, observed_at, accessed_at, ingested_at, license_label, "
            "normalization_version, evidence_authority) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            event_rows,
        )
        conn.executemany(
            "INSERT INTO event_securities (event_id, security_id) VALUES (?,?)",
            event_link_rows,
        )
    summary.events = events
    summary.by_item_type["event"] += events

    # -- Freshness rows (mixed states) for the overview/refresh surface ----
    fresh_rows: list[tuple] = []
    fresh_states = ("fresh", "stale", "error", "fresh")
    for i in range(min(securities, 200)):
        ticker = f"T{i:04d}"
        state = fresh_states[i % len(fresh_states)]
        error = "boom" if state == "error" else None
        fresh_rows.append((ticker, "finnhub", "all", now, now, state, error))
    with sqlite._connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO cache_meta (ticker, source, metric_scope, "
            "last_updated, next_scheduled_update, status, error_message) "
            "VALUES (?,?,?,?,?,?,?)",
            fresh_rows,
        )

    # Refresh planner statistics so filtered point lookups pick their indexes
    # instead of scanning the seeded ledger.
    with sqlite._connect() as conn:
        conn.execute("ANALYZE")

    # -- A single parsed filing + Chroma section stubs for detail paths ----
    accession = "ACC-SCALE-1"
    sqlite.register_filing(
        "T0000", "10-K", "2026-02-01", "2025-FY", accession,
        "https://example.test/filing/ACC-SCALE-1",
    )
    sqlite.mark_filing_parsed(
        accession, embedding_id=f"sec:{accession}:item_1",
        file_path=None, section_count=2, chunk_count=2,
    )
    _seed_chroma_stubs(store, accession)
    summary.filing_accession = accession
    summary.section_parent_ids = (
        f"sec:{accession}:item_1", f"sec:{accession}:item_2",
    )
    return summary


def _seed_chroma_stubs(store: Any, accession: str) -> None:
    """Attach a few metadata-only Chroma records for detail validation."""
    records = [
        {
            "id": f"sec:{accession}:item_1#0",
            "document": "Item 1. Business. " + "body " * 60,
            "metadata": {
                "source": "sec_filing", "ticker": "T0000",
                "accession": accession, "form": "10-K",
                "filing_date": "2026-02-01", "section_key": "item_1",
                "section_heading": "Item 1. Business", "section_index": 0,
                "parent_id": f"sec:{accession}:item_1", "chunk_index": 0,
                "chunk_count": 1, "source_url": "https://example.test/filing",
            },
        },
        {
            "id": f"sec:{accession}:item_2#0",
            "document": "Item 2. Risk factors",
            "metadata": {
                "source": "sec_filing", "ticker": "T0000",
                "accession": accession, "form": "10-K",
                "filing_date": "2026-02-01", "section_key": "item_2",
                "section_heading": "Item 2. Risk Factors", "section_index": 1,
                "parent_id": f"sec:{accession}:item_2", "chunk_index": 0,
                "chunk_count": 1, "source_url": "https://example.test/filing",
            },
        },
    ]
    chroma = getattr(store, "chroma", None)
    if hasattr(chroma, "records") and isinstance(chroma.records, list):
        chroma.records.extend(records)
