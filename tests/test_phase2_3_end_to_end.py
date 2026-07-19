"""tests/test_phase2_3_end_to_end.py
Phase 2.3.6.2 offline end-to-end integration harness for the broad-universe corpus.

Drives a seeded/mocked pilot slice (ORCL financing, NVDA high-volume news + GDELT,
AAPL multi-source + corporate action, JNJ openFDA, F NHTSA, LMT USAspending, and a
small official macro catalog) through the real Store, scheduler, retrieval,
Live-Trace projection, and Corpus-Explorer projection with no network or model.

Gates proven here (2.3.6.2 Step 1 / Step 4):
  - bootstrap then two identical refreshes create zero duplicate canonical
    records / corpus items / Chroma families (idempotence);
  - a simulated provider-wide 429 stops that provider after bounded attempts
    while every later source still runs;
  - a missing API key disables only that source with a truthful status (never
    "fresh"), leaving other sources running;
  - an indexing failure is recorded, repairable, and isolated to its item;
  - each enabled source reaches a truthful independent terminal status;
  - normalized rows, retrieval, the Live-Trace projection, and the Corpus-Explorer
    projection all see the pilot evidence;
  - the Phase 2.3 golden set's answerable classes deliver their labeled evidence
    ledger to generation, and its 2.3.7-dependent classes are explicitly deferred.

The 2.3.7-dependent acceptance gates (all-ticker inventory precision/recall,
deterministic-answer bypass, indirect-plan accuracy, seeded-scale lexical/hybrid,
storage benchmark, feature disposition) are NOT implemented here; they are tracked
outside this harness and asserted only as DEFERRED entries in the golden set.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

if "chromadb" not in sys.modules:  # keep Store import offline
    _chromadb = type(sys)("chromadb")
    _chromadb.EmbeddingFunction = object
    _chromadb.Documents = list
    _chromadb.Embeddings = list
    _chromadb.PersistentClient = MagicMock
    sys.modules["chromadb"] = _chromadb
    sys.modules["chromadb.api"] = MagicMock()

from eval import run_eval as R
from src.ingestion.errors import ErrorClass, ProviderError
from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash
from src.ingestion.records import EventRecord, NarrativeRecord
from src.scheduler import UnifiedScheduler
from src.scheduler.source_registry import SourceRegistry
from src.storage.sqlite_store import SQLiteStore
from src.storage.store import Store

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

# Pilot corpus store ids — kept in sync with tests/fixtures/evaluation/phase2_3_golden.json.
ORCL_FILING = "sec-orcl-notes-2026"
ORCL_ISSUER = "issuer-orcl-notes-2026"
ORCL_NEWS = "finnhub-orcl-notes-2026"
ORCL_SYND_A = "syndicated-orcl-notes-a"
ORCL_SYND_B = "syndicated-orcl-notes-b"
NVDA_NEWS = "finnhub-nvda-datacenter-2026"
NVDA_GDELT = "gdelt-nvda-datacenter-2026"
NVDA_STALE = "finnhub-nvda-stale-2025"
AAPL_FILING = "sec-aapl-8k-2026"
AAPL_NEWS = "finnhub-aapl-buyback-2026"
JNJ_OPENFDA = "openfda-jnj-recall-2026"
F_NHTSA = "nhtsa-f-recall-2026"
LMT_AWARD = "usaspending-lmt-award-2026"
BLS_CPI = "bls-cpi-2026-06"
FED_DECISION = "fed-decision-2026-06"
META_HISTORICAL = "meta-historical-2021"

ORCL_EVENT = "orcl-notes-2026"
AAPL_EVENT = "aapl-buyback-2026"


# ── Recording Chroma double ─────────────────────────────────────────────
#
# A faithful in-memory Chroma stand-in: it stores what ``add_document`` writes
# (honoring ``replace_family``), answers filtered lexical ``search`` so retrieval
# is real, and counts add calls / families so the idempotence gate can prove no
# duplicate family was written on a replay. ``fail_ids`` makes one document's
# indexing raise, to simulate an isolated indexing failure.


class RecordingChroma:
    """In-memory Chroma double that records writes and answers filtered search."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.add_calls: list[str] = []
        self.fail_ids: set[str] = set()

    def heartbeat(self) -> bool:
        return True

    def count(self) -> int:
        return len(self.records)

    @staticmethod
    def _matches(metadata: dict, where: dict | None) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(RecordingChroma._matches(metadata, item) for item in where["$and"])
        for key, expected in where.items():
            actual = metadata.get(key)
            if isinstance(expected, dict):
                if "$gte" in expected and (actual is None or actual < expected["$gte"]):
                    return False
                if "$lte" in expected and (actual is None or actual > expected["$lte"]):
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _family(metadata: dict, document_id: str) -> str:
        return str(metadata.get("document_family_id") or document_id)

    def add_document(self, *, document_id, text, ticker=None, source=None,
                     date=None, metadata=None, replace_family=False):
        self.add_calls.append(document_id)
        if document_id in self.fail_ids:
            raise RuntimeError("embedding service unavailable")
        md = dict(metadata or {})
        md.setdefault("ticker", ticker)
        md.setdefault("source", source)
        md.setdefault("date", date)
        if replace_family:
            family = self._family(md, document_id)
            self.records = [
                row for row in self.records
                if self._family(row["metadata"], row["id"]) != family
            ]
        self.records.append({"id": document_id, "document": text, "metadata": md})

    def delete_document_family(self, family_id):
        self.records = [
            row for row in self.records
            if self._family(row["metadata"], row["id"]) != family_id
        ]

    def count_filing_section_chunks(self, parent_id):
        return sum(
            1 for row in self.records
            if row["metadata"].get("parent_id") == parent_id or row["id"] == parent_id
        )

    def search(self, query=None, n_results=5, filter_dict=None):
        matches = [row for row in self.records if self._matches(row["metadata"], filter_dict)]
        terms = set(re.findall(r"[a-z0-9]+", (query or "").lower()))

        def score(row: dict) -> int:
            tokens = set(re.findall(r"[a-z0-9]+", (row.get("document") or "").lower()))
            return len(terms & tokens)

        matches.sort(key=score, reverse=True)
        return [dict(row) for row in matches[:n_results]]

    def get_source_counts(self, *, limit=200, offset=0):
        counts: dict[str, int] = {}
        for row in self.records:
            source = row["metadata"].get("source") or row["metadata"].get("source_name")
            if source:
                counts[source] = counts.get(source, 0) + 1
        rows = [{"source": s, "count": c} for s, c in sorted(counts.items())]
        return rows[offset:offset + limit]

    source_counts = get_source_counts

    def get_ticker_counts(self, *, limit=200, offset=0):
        grouped: dict[str, dict] = {}
        for row in self.records:
            ticker = row["metadata"].get("ticker")
            if not ticker:
                continue
            item = grouped.setdefault(
                ticker, {"ticker": ticker, "record_count": 0, "sources": set()})
            item["record_count"] += 1
            source = row["metadata"].get("source") or row["metadata"].get("source_name")
            if source:
                item["sources"].add(source)
        rows = [{**item, "sources": sorted(item["sources"])}
                for item in sorted(grouped.values(), key=lambda value: value["ticker"])]
        return rows[offset:offset + limit]

    ticker_counts = get_ticker_counts

    def get_metadata(self, *, where=None, limit=None, offset=0):
        rows = [row for row in self.records if self._matches(row["metadata"], where)][offset:]
        return rows[:limit] if limit is not None else rows


# ── Store + seed helpers ─────────────────────────────────────────────────


def _make_store(tmp_path: Path) -> tuple[Store, RecordingChroma]:
    """Build a Store with real SQLite and a recording Chroma double."""
    store = object.__new__(Store)
    store.sqlite = SQLiteStore(tmp_path / "phase2_3_e2e.db")
    store.chroma = RecordingChroma()
    return store, store.chroma


PILOT_SECURITIES = [
    {"symbol": "ORCL", "company_name": "Oracle Corporation", "index_code": "sp500",
     "sector": "Information Technology", "industry": "Software—Infrastructure",
     "cik": "0001341439"},
    {"symbol": "NVDA", "company_name": "NVIDIA Corporation", "index_code": "nasdaq100",
     "sector": "Information Technology", "industry": "Semiconductors",
     "cik": "0001045810"},
    {"symbol": "AAPL", "company_name": "Apple Inc.", "index_code": "nasdaq100",
     "sector": "Information Technology", "industry": "Consumer Electronics",
     "cik": "0000320193"},
    {"symbol": "JNJ", "company_name": "Johnson & Johnson", "index_code": "sp500",
     "sector": "Health Care", "industry": "Pharmaceuticals", "cik": "0000200406"},
    {"symbol": "F", "company_name": "Ford Motor Company", "index_code": "sp500",
     "sector": "Consumer Discretionary", "industry": "Automobile Manufacturers",
     "cik": "0000037996"},
    {"symbol": "LMT", "company_name": "Lockheed Martin Corporation", "index_code": "sp500",
     "sector": "Industrials", "industry": "Aerospace & Defense", "cik": "0000936468"},
    {"symbol": "META", "company_name": "Meta Platforms Inc.", "index_code": "nasdaq100",
     "sector": "Communication Services", "industry": "Interactive Media",
     "cik": "0001326801"},
    # Deep-research opt-in members named by configs/coverage.yaml (deep.tickers);
    # present in the registry so CoverageResolver's explicit-ticker validation
    # sees them inside the active index union.
    {"symbol": "MSFT", "company_name": "Microsoft Corporation", "index_code": "nasdaq100",
     "sector": "Information Technology", "industry": "Software—Infrastructure",
     "cik": "0000789019"},
    {"symbol": "AMD", "company_name": "Advanced Micro Devices Inc.", "index_code": "nasdaq100",
     "sector": "Information Technology", "industry": "Semiconductors", "cik": "0000002488"},
    {"symbol": "CRWD", "company_name": "CrowdStrike Holdings Inc.", "index_code": "nasdaq100",
     "sector": "Information Technology", "industry": "Software—Infrastructure",
     "cik": "0001535527"},
]


def seed_securities(store: Store) -> dict[str, str]:
    """Reconcile the pilot universe and return {ticker: security_id}."""
    store.upsert_universe_snapshot("ivv", "2026-07-01T00:00:00Z", PILOT_SECURITIES)
    ids = {
        row["symbol"]: store.resolve_security(row["symbol"])["security_id"]
        for row in PILOT_SECURITIES
    }
    # A former ticker so the historical-as-of case resolves FB -> META. The
    # former_ticker alias type is a registry-reconciliation identity, so it is
    # seeded directly (register_security_alias only accepts issuer/vendor types).
    normalized = store.sqlite._normalize_universe_symbol("FB")
    with store.sqlite._connect() as conn:
        conn.execute(
            "INSERT INTO security_aliases "
            "(security_id, alias, normalized_alias, alias_type, valid_from, valid_to, source) "
            "VALUES (?, ?, ?, 'former_ticker', ?, ?, 'registry')",
            (ids["META"], "FB", normalized, "2012-05-18", "2022-06-09"))
    return ids


def _narrative(item_id, *, source_name, source_category, item_type, title, body,
               published_at, evidence_authority, tickers=(), security_ids=(),
               index_codes=(), sectors=(), event_type=None, metadata=None,
               provider_record_id=None, canonical_url=None,
               original_publisher="Example Publisher") -> NarrativeRecord:
    return NarrativeRecord(
        corpus_item_id=item_id,
        source_name=source_name,
        source_category=source_category,
        provider_record_id=provider_record_id or item_id,
        original_publisher=original_publisher,
        item_type=item_type,
        title=title,
        body=body,
        summary=body[:200],
        published_at=published_at,
        observed_at=published_at,
        accessed_at="2026-07-14T18:00:00Z",
        ingested_at="2026-07-14T18:00:00Z",
        source_url=f"https://example.test/{item_id}",
        canonical_url=canonical_url,
        license_label="provider_summary",
        normalization_version=NORMALIZATION_VERSION,
        content_hash=content_hash(body),
        document_family=item_type,
        event_type=event_type,
        tickers=tuple(tickers),
        security_ids=tuple(security_ids),
        index_codes=tuple(index_codes),
        sectors=tuple(sectors),
        metadata=metadata or {},
        evidence_authority=evidence_authority,
    )


def pilot_narratives(ids: dict[str, str]) -> list[NarrativeRecord]:
    """The full set of pilot narrative payloads (one provider refresh)."""
    return [
        _narrative(
            ORCL_FILING, source_name="sec", source_category="sec", item_type="filing",
            title="Oracle 424B5 senior notes prospectus supplement",
            body="Oracle Corporation filed a 424B5 prospectus supplement for a senior "
                 "notes financing with the SEC, describing the notes and use of proceeds.",
            published_at="2026-07-10T13:00:00Z", evidence_authority="direct_sec",
            tickers=("ORCL",), security_ids=(ids["ORCL"],), index_codes=("sp500",),
            sectors=("Information Technology",), event_type="debt_raise",
            metadata={"form": "424B5", "filing_item": "2.03", "exhibit": "EX-10.1"}),
        _narrative(
            ORCL_ISSUER, source_name="oracle_ir", source_category="issuer",
            item_type="press_release",
            title="Oracle announces senior notes offering",
            body="Oracle described the notes offering, intended use of proceeds, and "
                 "maturity schedule in an investor relations press release.",
            published_at="2026-07-10T13:30:00Z", evidence_authority="issuer",
            tickers=("ORCL",), security_ids=(ids["ORCL"],), event_type="debt_raise"),
        _narrative(
            ORCL_NEWS, source_name="finnhub", source_category="company_news",
            item_type="news",
            title="Oracle notes financing draws investor interest",
            body="A licensed market-news summary added investor reaction to Oracle's "
                 "senior notes financing.",
            published_at="2026-07-10T14:00:00Z", evidence_authority="provider",
            tickers=("ORCL",), security_ids=(ids["ORCL"],), event_type="debt_raise",
            canonical_url="https://example.test/orcl-notes-analysis"),
        _narrative(
            ORCL_SYND_A, source_name="massive", source_category="company_news",
            item_type="news", title="Oracle launches senior notes offering",
            body="Oracle launches senior notes offering.",
            published_at="2026-07-10T14:05:00Z", evidence_authority="provider",
            tickers=("ORCL",), security_ids=(ids["ORCL"],), event_type="debt_raise"),
        _narrative(
            ORCL_SYND_B, source_name="gdelt", source_category="global_news",
            item_type="news", title="Oracle launches senior notes offering",
            body="Oracle launches senior notes offering.",
            published_at="2026-07-10T14:06:00Z", evidence_authority="discovery",
            tickers=("ORCL",), security_ids=(ids["ORCL"],), event_type="debt_raise"),
        _narrative(
            NVDA_NEWS, source_name="finnhub", source_category="company_news",
            item_type="news", title="NVIDIA data-center demand accelerates",
            body="NVIDIA reported accelerating data-center demand across cloud customers.",
            published_at="2026-07-12T12:00:00Z", evidence_authority="provider",
            tickers=("NVDA",), security_ids=(ids["NVDA"],)),
        _narrative(
            NVDA_GDELT, source_name="gdelt", source_category="global_news",
            item_type="news", title="Global coverage of NVIDIA data-center growth",
            body="Global news coverage corroborated NVIDIA data-center growth this week.",
            published_at="2026-07-12T12:30:00Z", evidence_authority="discovery",
            tickers=("NVDA",), security_ids=(ids["NVDA"],)),
        _narrative(
            NVDA_STALE, source_name="finnhub", source_category="company_news",
            item_type="news", title="NVIDIA data-center segment a year ago",
            body="An older summary describes NVIDIA's data-center segment last year.",
            published_at="2025-07-12T12:00:00Z", evidence_authority="provider",
            tickers=("NVDA",), security_ids=(ids["NVDA"],)),
        _narrative(
            AAPL_FILING, source_name="sec", source_category="sec", item_type="filing",
            title="Apple 8-K announces expanded buyback",
            body="Apple filed an 8-K announcing an expanded share buyback authorization "
                 "with an effective market date.",
            published_at="2026-07-08T20:00:00Z", evidence_authority="direct_sec",
            tickers=("AAPL",), security_ids=(ids["AAPL"],), index_codes=("nasdaq100",),
            sectors=("Information Technology",), event_type="buyback",
            metadata={"form": "8-K", "filing_item": "8.01", "market_date": "2026-07-08"}),
        _narrative(
            AAPL_NEWS, source_name="finnhub", source_category="company_news",
            item_type="news", title="Apple expands buyback program",
            body="A licensed news summary covered Apple's expanded buyback program.",
            published_at="2026-07-08T21:00:00Z", evidence_authority="provider",
            tickers=("AAPL",), security_ids=(ids["AAPL"],), event_type="buyback"),
        _narrative(
            JNJ_OPENFDA, source_name="openfda", source_category="sector_agency",
            item_type="regulatory_event",
            title="openFDA enforcement report affecting Johnson & Johnson",
            body="An openFDA enforcement report referenced a Johnson & Johnson product.",
            published_at="2026-07-11T09:00:00Z", evidence_authority="official",
            tickers=("JNJ",), security_ids=(ids["JNJ"],), sectors=("Health Care",),
            event_type="regulatory_event"),
        _narrative(
            F_NHTSA, source_name="nhtsa", source_category="sector_agency",
            item_type="recall", title="NHTSA recall affecting Ford vehicles",
            body="NHTSA announced a vehicle safety recall affecting Ford Motor Company.",
            published_at="2026-07-11T15:00:00Z", evidence_authority="official",
            tickers=("F",), security_ids=(ids["F"],),
            sectors=("Consumer Discretionary",), event_type="recall"),
        _narrative(
            LMT_AWARD, source_name="usaspending", source_category="sector_agency",
            item_type="contract_award",
            title="USAspending contract award to Lockheed Martin",
            body="USAspending recorded a federal contract award to Lockheed Martin.",
            published_at="2026-07-09T16:00:00Z", evidence_authority="official",
            tickers=("LMT",), security_ids=(ids["LMT"],), sectors=("Industrials",),
            event_type="contract_award"),
        _narrative(
            BLS_CPI, source_name="bls", source_category="official_macro",
            item_type="economic_release",
            title="BLS consumer price index release for June 2026",
            body="The Bureau of Labor Statistics published the June consumer price index "
                 "release.",
            published_at="2026-07-12T12:30:00Z", evidence_authority="official",
            event_type="economic_release", original_publisher="Bureau of Labor Statistics"),
        _narrative(
            FED_DECISION, source_name="fed", source_category="central_bank",
            item_type="policy_release",
            title="Federal Reserve monetary policy decision",
            body="The Federal Reserve announced its latest monetary policy decision, "
                 "holding the target rate steady.",
            published_at="2026-07-12T18:00:00Z", evidence_authority="official",
            event_type="monetary_policy_decision", original_publisher="Federal Reserve"),
        _narrative(
            META_HISTORICAL, source_name="sec", source_category="sec", item_type="filing",
            title="Meta Q3 2021 quarterly report",
            body="The company reported quarterly results while its public ticker was FB.",
            published_at="2021-10-25T20:00:00Z", evidence_authority="direct_sec",
            tickers=("META",), security_ids=(ids["META"],), event_type="earnings_release",
            metadata={"form": "10-Q"}),
    ]


def pilot_events(ids: dict[str, str]) -> list[EventRecord]:
    """Structured events linking financing and corporate-action evidence."""
    common = dict(
        source_url="https://example.test/event",
        canonical_url=None, observed_at="2026-07-10T13:05:00Z",
        accessed_at="2026-07-14T18:00:00Z", ingested_at="2026-07-14T18:00:00Z",
        license_label="provider_summary", original_publisher="SEC",
    )
    return [
        EventRecord(
            event_id=ORCL_EVENT, event_type="debt_raise", effective_at="2026-07-10T00:00:00Z",
            announced_at="2026-07-10T13:00:00Z", status="confirmed",
            security_ids=(ids["ORCL"],),
            source_corpus_item_ids=(ORCL_FILING, ORCL_ISSUER),
            source_name="sec", source_category="sec", provider_record_id=ORCL_EVENT,
            published_at="2026-07-10T13:00:00Z", amount=5_000_000_000.0, currency="USD",
            evidence_authority="direct_sec", **common),
        EventRecord(
            event_id=AAPL_EVENT, event_type="buyback", effective_at="2026-07-08T00:00:00Z",
            announced_at="2026-07-08T20:00:00Z", status="confirmed",
            security_ids=(ids["AAPL"],), source_corpus_item_ids=(AAPL_FILING,),
            source_name="sec", source_category="sec", provider_record_id=AAPL_EVENT,
            published_at="2026-07-08T20:00:00Z", action_date="2026-07-08",
            metadata={"market_date": "2026-07-08"}, evidence_authority="direct_sec",
            **common),
    ]


def seed_pilot_corpus(store: Store) -> dict[str, str]:
    """Seed the full pilot slice (securities, narratives, events, fundamentals)."""
    ids = seed_securities(store)
    for record in pilot_narratives(ids):
        store.upsert_narrative(record)
    for event in pilot_events(ids):
        store.upsert_event(event)
    store.save_fundamental("NVDA", "total_revenue", 26.0, "usd", "2026-Q1")
    store.save_fundamental("ORCL", "total_debt", 84.0, "usd", "2026-Q1")
    return ids


@pytest.fixture
def pilot_store(tmp_path):
    store, chroma = _make_store(tmp_path)
    ids = seed_pilot_corpus(store)
    return store, chroma, ids


# ── Idempotence gate (bootstrap + two identical refreshes) ───────────────


class TestPilotIdempotence:
    def test_repeat_refresh_creates_no_duplicate_records_items_or_families(self, tmp_path):
        store, chroma = _make_store(tmp_path)
        ids = seed_securities(store)
        narratives = pilot_narratives(ids)
        events = pilot_events(ids)

        def refresh() -> None:
            for record in narratives:
                store.upsert_narrative(record)
            for event in events:
                store.upsert_event(event)

        refresh()  # bootstrap
        items_after_bootstrap = store.sqlite.count_corpus_items()
        events_after_bootstrap = store.sqlite.count_events()
        families_after_bootstrap = len(chroma.records)
        adds_after_bootstrap = len(chroma.add_calls)

        refresh()  # identical refresh 1
        refresh()  # identical refresh 2

        # Zero duplicate canonical records / corpus items / events / families.
        assert store.sqlite.count_corpus_items() == items_after_bootstrap
        assert store.sqlite.count_events() == events_after_bootstrap
        assert len(chroma.records) == families_after_bootstrap
        # A deduplicated replay is content-unchanged, so it never re-indexes.
        assert len(chroma.add_calls) == adds_after_bootstrap

    def test_syndicated_duplicates_collapse_to_one_family(self, pilot_store):
        store, chroma, _ = pilot_store
        # The two syndicated ORCL headlines share a headline/content, so only one
        # canonical corpus item and Chroma family survives.
        family_ids = {row["id"] for row in chroma.records}
        assert ORCL_SYND_A in family_ids
        assert ORCL_SYND_B not in family_ids


# ── Failure simulations (isolation + bounded rate-limit handling) ────────


class TestFailureSimulations:
    def _scheduler(self, store, *, environ):
        registry = SourceRegistry.load(environ=environ)
        return UnifiedScheduler(store=store, registry=registry, inter_source_delay=0)

    def test_provider_wide_429_stops_provider_but_later_sources_run(self, tmp_path):
        store, _ = _make_store(tmp_path)
        seed_securities(store)
        env = {name: "test-key" for name in (
            "FINNHUB_API_KEY", "MASSIVE_API_KEY", "FRED_API_KEY",
            "BLS_API_KEY", "BEA_API_KEY", "EIA_API_KEY", "OPENFDA_API_KEY")}
        scheduler = self._scheduler(store, environ=env)

        ran: list[str] = []

        def run_source(name, deep=False, force=False):
            ran.append(name)
            if name == "sec_filings":
                raise ProviderError(
                    "rate limited token=hidden",
                    error_class=ErrorClass.RATE_LIMITED, attempts=3, retry_after=300,
                    provider_wide=True, circuit_open=True)
            return {"status": "ok", "requests": 1, "items": 1}

        scheduler._run_source = run_source
        result = scheduler.run_daily(force=True)

        # The rate-limited provider stops (skipped, classified), later sources run.
        assert result["sec_filings"]["status"] == "skipped"
        assert result["sec_filings"]["error_class"] == "rate_limited"
        assert result["yfinance"]["status"] == "success"
        assert "yfinance" in ran
        report = scheduler.status_report()["runs"][0]
        source_status = {row["source"]: row for row in report["sources"]}
        assert source_status["sec_filings"]["error_class"] == "rate_limited"
        assert source_status["yfinance"]["status"] == "success"

    def test_missing_api_key_disables_only_that_source_truthfully(self, tmp_path):
        store, _ = _make_store(tmp_path)
        seed_securities(store)
        scheduler = self._scheduler(store, environ={})  # no keys at all
        scheduler._run_source = MagicMock(return_value={"status": "ok", "requests": 1})

        result = scheduler.run_daily(force=True)

        # A key-requiring source is disabled with a truthful reason; a keyless
        # source still runs to success.
        assert result["finnhub"]["reason"] == "disabled_missing_key"
        assert result["yfinance"]["status"] == "success"
        report = scheduler.status_report()["runs"][0]
        source_status = {row["source"]: row for row in report["sources"]}
        assert source_status["finnhub"]["error_class"] == "authentication"
        # A disabled/missing-key source is never reported "fresh".
        cache = store.get_cache_status("SCHEDULER", "unified:finnhub")
        assert cache is None or cache["status"] != "fresh"

    def test_indexing_failure_is_recorded_repairable_and_isolated(self, tmp_path):
        store, chroma = _make_store(tmp_path)
        ids = seed_securities(store)
        chroma.fail_ids = {NVDA_NEWS}  # one item's indexing raises

        healthy = next(r for r in pilot_narratives(ids) if r.corpus_item_id == ORCL_FILING)
        broken = next(r for r in pilot_narratives(ids) if r.corpus_item_id == NVDA_NEWS)
        store.upsert_narrative(healthy)
        broken_result = store.upsert_narrative(broken)

        # The failure is recorded and isolated — the healthy item still indexed.
        assert broken_result["indexing_status"] == "error"
        assert store.get_corpus_item(ORCL_FILING)["indexing_status"] == "indexed"
        assert store.get_corpus_item(NVDA_NEWS)["indexing_status"] == "error"

        # Repair reindexes stored content without contacting a provider.
        chroma.fail_ids = set()
        scheduler = UnifiedScheduler(store=store, inter_source_delay=0)
        repair = scheduler.run_repair(source="finnhub", limit=10)
        assert repair["completed"] == 1
        assert repair["failed"] == 0
        assert store.get_corpus_item(NVDA_NEWS)["indexing_status"] == "indexed"


# ── Truthful independent terminal statuses ───────────────────────────────


class TestTruthfulStatus:
    def test_each_source_reaches_a_truthful_independent_terminal_status(self, tmp_path):
        store, _ = _make_store(tmp_path)
        seed_securities(store)
        env = {name: "test-key" for name in (
            "MASSIVE_API_KEY", "BLS_API_KEY", "BEA_API_KEY", "EIA_API_KEY",
            "OPENFDA_API_KEY")}  # deliberately omit FINNHUB + FRED keys
        registry = SourceRegistry.load(environ=env)
        scheduler = UnifiedScheduler(store=store, registry=registry, inter_source_delay=0)

        def run_source(name, deep=False, force=False):
            if name == "sec_filings":
                raise ProviderError(
                    "429", error_class=ErrorClass.RATE_LIMITED, attempts=3, retry_after=60)
            return {"status": "ok", "requests": 1, "items": 1}

        scheduler._run_source = run_source
        result = scheduler.run_daily(force=True)

        # Three independent terminal outcomes coexist truthfully in one run.
        assert result["sec_filings"]["status"] == "skipped"
        assert result["sec_filings"]["error_class"] == "rate_limited"
        assert result["finnhub"]["reason"] == "disabled_missing_key"
        assert result["yfinance"]["status"] == "success"


# ── Normalized rows + retrieval see the pilot evidence ───────────────────


class TestNormalizedRowsAndRetrieval:
    def test_normalized_rows_carry_pilot_evidence(self, pilot_store):
        store, _, ids = pilot_store
        orcl = store.get_corpus_item(ORCL_FILING)
        assert orcl is not None
        assert orcl["item_type"] == "filing"
        securities = {row["security_id"] for row in store.list_corpus_item_securities(ORCL_FILING)}
        assert ids["ORCL"] in securities

        event = store.get_corpus_event(ORCL_EVENT)
        assert event["event_type"] == "debt_raise"
        assert ids["ORCL"] in set(event["security_ids"])
        # The event's source corpus-item links are read via the structured getter.
        linked = store.sqlite.get_event(ORCL_EVENT)
        assert ORCL_FILING in set(linked["source_corpus_item_ids"])

    def test_retrieval_surfaces_primary_and_secondary_pilot_evidence(self, pilot_store):
        store, _, _ = pilot_store
        result = store.search("Oracle senior notes financing", ticker="ORCL", n_results=10)
        ledger_ids = R.ledger_store_ids(R.capture_evidence_ledger(result))
        assert ORCL_FILING in ledger_ids
        assert ORCL_NEWS in ledger_ids

    def test_macro_retrieval_needs_no_ticker(self, pilot_store):
        store, _, _ = pilot_store
        result = store.search("consumer price index release", filters={"source": "bls"},
                              n_results=10)
        ledger_ids = R.ledger_store_ids(R.capture_evidence_ledger(result))
        assert BLS_CPI in ledger_ids


# ── Live-Trace projection sees the pilot evidence (redacted, complete) ────


class TestLiveTraceProjection:
    def test_mixed_source_live_trace_is_complete_and_redacted(self, pilot_store):
        from src.middleware.evidence import EvidenceItem, assign_evidence_ids
        from src.middleware.graph_observer import TraceHub, make_event_observer
        from src.middleware.stream_events import QueryEventEmitter

        store, _, _ = pilot_store
        result = store.search("Oracle senior notes financing", ticker="ORCL", n_results=10)
        by_id = {doc["id"]: doc for doc in result["documents"]}
        items = assign_evidence_ids([
            EvidenceItem.from_row(by_id[ORCL_FILING], kind="document"),
            EvidenceItem.from_row(by_id[ORCL_NEWS], kind="document"),
        ])

        hub = TraceHub()
        emitter = QueryEventEmitter(
            query_id="q-pilot", observers=[make_event_observer(hub)], include_counts=True)
        emitter.query_started(question="What was Oracle's most recent debt raise?")
        emitter.stage("retrieve", "completed", elapsed_ms=1.0)
        for rank, item in enumerate(items, 1):
            emitter.graph_evidence(
                evidence_id=item.evidence_id, kind=item.kind, excerpt=item.document,
                metadata=item.taxonomy_metadata(), source_type=item.source_type, rank=rank)
        emitter.error("done", terminal=True)

        snap = hub.snapshot("q-pilot")
        assert snap is not None
        assert snap["complete"] is True
        evidence_nodes = [n for n in snap["nodes"] if n["kind"] == "evidence"]
        assert len(evidence_nodes) == 2
        # Mixed sources (direct SEC + provider news) both appear, source-aware.
        authorities = {n["metadata"].get("authority_tier") for n in evidence_nodes}
        assert {"direct_sec", "provider"} <= authorities

        # Redacted by construction: bounded preview, no local paths or secrets.
        blob = json.dumps(snap)
        assert "F:\\" not in blob and "C:\\" not in blob
        assert "API_KEY" not in blob
        assert len(snap["question_preview"]) <= 200


# ── Corpus-Explorer projection sees the pilot evidence (revision-consistent) ──


class TestCorpusExplorerProjection:
    def test_aggregates_and_overview_reflect_pilot_corpus(self, pilot_store):
        from src.middleware.corpus_graph import CorpusGraph

        store, _, _ = pilot_store
        graph = CorpusGraph(store, overview_ttl_s=60)

        overview = graph.overview()
        assert overview["corpus_revision"] == store.retrieval_revision()

        aggregates = graph.aggregates("source_category")
        buckets = aggregates["aggregates"]["buckets"]
        categories = {str(bucket.get("key")) for bucket in buckets}
        assert "sec" in categories
        assert aggregates["corpus_revision"] == store.retrieval_revision()

    def test_corpus_accounting_counts_pilot_sources(self, pilot_store):
        store, _, _ = pilot_store
        rows = store.get_corpus_accounting("source", limit=50)
        by_source = {str(row.get("key")): row for row in rows}
        assert "sec" in by_source
        assert by_source["sec"]["count"] >= 1


# ── Phase 2.3 golden evaluation set (answerable ledgers + deferred gates) ──


class TestPhase2_3GoldenSet:
    def test_golden_set_schema_is_well_formed(self):
        golden = R.load_phase2_3_golden()
        assert golden["version"].startswith("phase2_3")
        answerable = R.phase2_3_answerable_cases(golden)
        assert answerable
        seen_ids: set[str] = set()
        for case in answerable:
            assert case["id"] not in seen_ids, f"duplicate golden id {case['id']}"
            seen_ids.add(case["id"])
            for key in ("question", "question_class", "answerability", "retrieval"):
                assert key in case, f"{case['id']} missing {key}"
            # Every answerable case carries a labeled ledger contract.
            assert "expected_evidence_ids" in case or "expected_fact" in case

    def test_deferred_classes_name_their_2_3_7_gate(self):
        deferred = R.phase2_3_deferred_cases()
        assert deferred
        for entry in deferred:
            assert entry["requires"].startswith("2.3.7"), entry
            assert entry.get("gate")
            assert entry.get("question_class")

    def test_answerable_cases_deliver_labeled_ledger_to_generation(self, pilot_store):
        store, _, _ = pilot_store
        for case in R.phase2_3_answerable_cases():
            hint = dict(case.get("retrieval") or {})
            ticker = hint.pop("ticker", None)
            result = store.search(case["question"], n_results=10, ticker=ticker,
                                  filters=hint or None)
            ledger = R.capture_evidence_ledger(result)
            ledger_ids = R.ledger_store_ids(ledger)

            for expected in case.get("expected_evidence_ids") or []:
                assert expected in ledger_ids, (
                    f"{case['id']}: {expected} not delivered to generation")

            if case["answerability"] == "unavailable":
                assert ledger_ids == [], f"{case['id']}: should abstain (empty ledger)"

            # Every delivered entry is citable: request-local id + durable store id.
            for entry in ledger:
                assert entry["evidence_id"].startswith("E")
                assert entry["store_id"]

            expected_fact = case.get("expected_fact")
            if expected_fact:
                facts = [e for e in ledger if e["kind"] == "fact"]
                assert any(
                    expected_fact["metric"] in str(f.get("metric") or "")
                    and f.get("source_type") == expected_fact["source_type"]
                    and f.get("value") is not None
                    for f in facts), f"{case['id']}: structured fact not delivered"

    def test_primary_evidence_outranks_secondary(self, pilot_store):
        from src.middleware.evidence_taxonomy import rank_evidence

        store, _, _ = pilot_store
        for case in R.phase2_3_answerable_cases():
            primary = case.get("primary_evidence_id")
            secondaries = case.get("secondary_evidence_ids") or []
            if not primary or not secondaries:
                continue
            hint = dict(case.get("retrieval") or {})
            ticker = hint.pop("ticker", None)
            result = store.search(case["question"], n_results=10, ticker=ticker,
                                  filters=hint or None)
            by_id = {doc["id"]: doc for doc in result["documents"]}
            for secondary in secondaries:
                docs = [dict(by_id[primary], fusion_score=0.03),
                        dict(by_id[secondary], fusion_score=0.03)]
                ranked = rank_evidence(docs, query=case["question"],
                                       authority_max_boost=0.025, recency_max_boost=0.0)
                order = [row["id"] for row in ranked]
                assert order.index(primary) < order.index(secondary), (
                    f"{case['id']}: primary {primary} must outrank {secondary}")
