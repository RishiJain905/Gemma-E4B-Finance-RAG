"""
src/sec/backfill.py
Shared SEC substantive-filings backfill (deep-watchlist 10-K/10-Q/8-K history).

The scheduler discovers filings from EDGAR daily indexes, whose bootstrap only
reaches ``daily_index_bootstrap_days`` (=7) back, so a deep ticker's historical
periodic filings (10-K/10-Q) predate the cursor and are never ingested. This
module discovers a bounded, most-recent-first slice per ticker via the
per-company EDGAR submissions endpoint (``SECEdgarFilingFetcher.discover_filings``,
which works regardless of date) and processes each filing through the SAME
``FilingProcessor`` path the scheduler uses — so full 10-K/10-Q section text is
indexed when ``sec.index_filing_text`` is on, and 8-Ks flow through the event
path.

Contracts:
  - Idempotent: a filing already stored AND parsed is skipped; a re-run over an
    already-backfilled ticker is a cheap no-op (only the bounded discovery
    calls run, and every discovered filing is skipped).
  - Per-filing failure isolation: one filing's discover/download/parse/index
    failure is logged and counted, never propagated; the next filing still runs.
  - Deep tickers default from ``configs/coverage.yaml`` (``sec_filing_text``
    scope), so the CLI and the scheduler onboarding hook share one source.

Both ``scripts/backfill_filings.py`` and the scheduler onboarding hook call in
here; the scripts stay thin wrappers.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from .daily_index import _DEEP_PERIODIC_FORMS
from .edgar_fetcher import SECEdgarFilingFetcher
from .filing_processor import FilingProcessor
from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)

# Manual-backfill defaults: most-recent-first counts per periodic/event form.
DEFAULT_FILING_TYPES: tuple[str, ...] = ("10-K", "10-Q", "8-K")
DEFAULT_COUNTS_PER_TYPE: dict[str, int] = {"10-K": 2, "10-Q": 4, "8-K": 8}

# Onboarding / reconcile-repair are intentionally smaller than a full manual
# backfill: just enough substantive periodic text to make a deep ticker
# answerable, keeping the self-heal path bounded.
ONBOARDING_FILING_TYPES: tuple[str, ...] = ("10-K", "10-Q")
ONBOARDING_COUNTS: dict[str, int] = {"10-K": 1, "10-Q": 2}

# "Substantive" periodic forms for onboarding-trigger + reconcile gap checks.
SUBSTANTIVE_PERIODIC_FORMS: frozenset[str] = frozenset({"10-K", "10-Q"})

# Fallback per-type count for a form not named in counts_per_type.
_FALLBACK_COUNT = 2

# Reconcile default: a deep ticker should have a periodic filing this recent.
RECENCY_DAYS_DEFAULT = 120

# Discovery cache source + TTL (mirrors FilingScheduler.DISCOVERY_SOURCE), so a
# backfill records freshness the way normal per-ticker discovery does.
_DISCOVERY_CACHE_SOURCE = "sec_filings_discovery"
_DISCOVERY_TTL_HOURS = 12

# Result buckets a single discovered filing can fall into.
_BUCKETS = ("ingested", "skipped", "failed", "would_ingest")


class FilingBackfiller:
    """Discover + process a bounded slice of a ticker's substantive filings.

    Usage:
        backfiller = FilingBackfiller(store)
        report = backfiller.run(["NVDA", "PLTR"])           # manual backfill
        backfiller.onboard_ticker("SNDK")                    # bounded self-heal
        backfiller.reconcile(repair=True)                    # periodic gap fix
    """

    def __init__(
        self,
        store: Store,
        *,
        processor: Optional[FilingProcessor] = None,
        fetcher: Optional[SECEdgarFilingFetcher] = None,
        coverage: Optional[CoverageResolver] = None,
    ) -> None:
        self.store = store
        self.processor = processor or FilingProcessor(store=store)
        # Discovery and download share one fetcher (one user agent + throttle).
        self.fetcher = (
            fetcher
            or getattr(self.processor, "fetcher", None)
            or SECEdgarFilingFetcher(store=store)
        )
        self.coverage = coverage or CoverageResolver(store)

    # ── Ticker sourcing ────────────────────────────────

    def deep_tickers(self) -> list[str]:
        """Deep watchlist tickers (configs/coverage.yaml sec_filing_text scope)."""
        return list(self.coverage.tickers_for("sec_filing_text"))

    def has_substantive_filing(self, ticker: str) -> bool:
        """True if a 10-K or 10-Q is already stored for ``ticker``."""
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            return False
        counts = self.store.count_filing_types(symbol)
        present = {
            str(row.get("filing_type") or "").upper()
            for row in counts
            if int(row.get("count") or 0) > 0
        }
        return bool(present & SUBSTANTIVE_PERIODIC_FORMS)

    # ── Backfill ───────────────────────────────────────

    def backfill_ticker(
        self,
        ticker: str,
        *,
        filing_types: tuple[str, ...] = DEFAULT_FILING_TYPES,
        counts_per_type: Optional[dict[str, int]] = None,
        dry_run: bool = False,
    ) -> dict:
        """Discover + process one ticker's filings, most recent first.

        Returns a per-ticker result dict with bucket counters and one row per
        discovered filing. Failures are isolated per filing (and per form for a
        failed discovery call).
        """
        symbol = str(ticker or "").strip().upper()
        counts = {**DEFAULT_COUNTS_PER_TYPE, **(counts_per_type or {})}
        result: dict = {"ticker": symbol, "discovered": 0, "filings": []}
        for bucket in _BUCKETS:
            result[bucket] = 0

        for form in filing_types:
            form_u = str(form).strip().upper()
            count = int(counts.get(form_u, counts.get(form, _FALLBACK_COUNT)))
            if count <= 0:
                continue
            try:
                discovered = self.fetcher.discover_filings(
                    symbol, filing_types=[form_u], count=count,
                )
            except Exception as exc:  # noqa: BLE001 - per-form discovery isolation
                logger.error(
                    "Backfill discovery failed for %s %s: %s", symbol, form_u, exc,
                )
                result["failed"] += 1
                result["filings"].append({
                    "ticker": symbol, "filing_type": form_u, "accession": None,
                    "filing_date": None, "status": "discover_failed",
                    "detail": str(exc),
                })
                continue
            result["discovered"] += len(discovered)
            for filing in discovered:
                bucket, row = self._handle_filing(filing, symbol, dry_run=dry_run)
                result[bucket] += 1
                result["filings"].append(row)

        if not dry_run and (result["ingested"] or result["skipped"]):
            self._mark_discovery_fresh(symbol)
        return result

    def _handle_filing(
        self, filing: dict, symbol: str, *, dry_run: bool,
    ) -> tuple[str, dict]:
        """Register + process one discovered filing, isolating its failures."""
        form = str(filing.get("filing_type") or "").upper()
        accession = str(filing.get("accession") or "").strip()
        filing_date = filing.get("filing_date")
        base = {
            "ticker": symbol, "filing_type": form,
            "accession": accession or None, "filing_date": filing_date,
        }
        if not accession:
            return "failed", {**base, "status": "no_accession"}

        existing = self.store.get_filing(accession)
        if existing and str(existing.get("status")) == "parsed":
            return "skipped", {**base, "status": "already_parsed"}

        if dry_run:
            status = "would_reprocess" if existing else "would_ingest"
            return "would_ingest", {**base, "status": status}

        # Deep tickers' periodic forms get the full-text section path; every
        # other form (8-K, amendments outside the periodic set) uses the event
        # path — the same scope split the daily-index discovery applies.
        scope = "deep" if form in _DEEP_PERIODIC_FORMS else "broad"
        try:
            self.store.register_filing(
                symbol,
                form,
                str(filing_date or ""),
                str(filing.get("period") or ""),
                accession,
                str(filing.get("source_url") or ""),
                cik=filing.get("cik"),
                primary_document=filing.get("primary_document"),
                discovery_scope=scope,
                items=filing.get("items"),
            )
            record = {
                **filing,
                "ticker": symbol,
                "filing_type": form,
                "accession": accession,
                "discovery_scope": scope,
            }
            success = self.processor._process_single_filing(record)
        except Exception as exc:  # noqa: BLE001 - per-filing processing isolation
            logger.error(
                "Backfill processing failed for %s %s (%s): %s",
                symbol, form, accession, exc,
            )
            return "failed", {**base, "status": "error", "detail": str(exc)}

        if success:
            return "ingested", {**base, "status": "ingested", "scope": scope}
        return "failed", {**base, "status": "process_incomplete", "scope": scope}

    def run(
        self,
        tickers: Optional[list[str]] = None,
        *,
        filing_types: tuple[str, ...] = DEFAULT_FILING_TYPES,
        counts_per_type: Optional[dict[str, int]] = None,
        dry_run: bool = False,
    ) -> dict:
        """Backfill each ticker (deep watchlist by default) with ticker isolation."""
        symbols = [
            str(t).strip().upper()
            for t in (tickers if tickers is not None else self.deep_tickers())
            if str(t).strip()
        ]
        per_ticker: dict[str, dict] = {}
        for symbol in symbols:
            try:
                per_ticker[symbol] = self.backfill_ticker(
                    symbol,
                    filing_types=filing_types,
                    counts_per_type=counts_per_type,
                    dry_run=dry_run,
                )
            except Exception as exc:  # noqa: BLE001 - per-ticker isolation
                logger.error("Backfill failed for ticker %s: %s", symbol, exc)
                failed_row = {
                    "ticker": symbol, "discovered": 0, "filings": [],
                    "error": str(exc),
                }
                for bucket in _BUCKETS:
                    failed_row[bucket] = 0
                failed_row["failed"] = 1
                per_ticker[symbol] = failed_row
        totals = {
            key: sum(int(row.get(key, 0)) for row in per_ticker.values())
            for key in ("discovered", *_BUCKETS)
        }
        return {"tickers": per_ticker, "totals": totals, "dry_run": dry_run}

    def onboard_ticker(self, ticker: str) -> dict:
        """Run one bounded onboarding backfill (10-K + 10-Q) for a deep ticker."""
        return self.backfill_ticker(
            ticker,
            filing_types=ONBOARDING_FILING_TYPES,
            counts_per_type=ONBOARDING_COUNTS,
        )

    # ── Reconcile / self-heal ──────────────────────────

    def find_gaps(
        self,
        tickers: Optional[list[str]] = None,
        *,
        recency_days: int = RECENCY_DAYS_DEFAULT,
        now: Optional[datetime] = None,
    ) -> dict[str, dict]:
        """Report, per deep ticker, whether recent periodic + indexed text exist."""
        symbols = [
            str(t).strip().upper()
            for t in (tickers if tickers is not None else self.deep_tickers())
            if str(t).strip()
        ]
        reference = (now or datetime.now(timezone.utc)).date()
        return {
            symbol: self._ticker_gap(symbol, recency_days=recency_days, today=reference)
            for symbol in symbols
        }

    def _ticker_gap(
        self, symbol: str, *, recency_days: int, today: date,
    ) -> dict:
        """Gap-detect one ticker: a recent 10-K/10-Q and any indexed filing text."""
        filings = self.store.list_filings(
            ticker=symbol, limit=Store.MAX_CORPUS_PAGE_LIMIT,
        )
        recent_periodic = False
        indexed_text = False
        latest_periodic: Optional[str] = None
        latest_date: Optional[date] = None
        for filing in filings:
            form = str(filing.get("filing_type") or "").upper()
            filing_date = self._parse_date(filing.get("filing_date"))
            if form in SUBSTANTIVE_PERIODIC_FORMS:
                if filing_date is not None and (
                    latest_date is None or filing_date > latest_date
                ):
                    latest_date = filing_date
                    latest_periodic = str(filing.get("filing_date"))
                if filing_date is not None and (today - filing_date).days <= recency_days:
                    recent_periodic = True
            if int(filing.get("index_section_count") or 0) > 0:
                indexed_text = True

        reasons: list[str] = []
        if not recent_periodic:
            reasons.append("no_recent_periodic")
        if not indexed_text:
            reasons.append("no_indexed_text")
        return {
            "ticker": symbol,
            "gap": bool(reasons),
            "reasons": reasons,
            "latest_periodic": latest_periodic,
            "filing_count": len(filings),
        }

    def reconcile(
        self,
        tickers: Optional[list[str]] = None,
        *,
        recency_days: int = RECENCY_DAYS_DEFAULT,
        repair: bool = False,
        now: Optional[datetime] = None,
    ) -> dict:
        """Report deep-ticker filing gaps and, when ``repair``, backfill them."""
        gaps = self.find_gaps(tickers, recency_days=recency_days, now=now)
        repairs: dict[str, dict] = {}
        if repair:
            for symbol, info in gaps.items():
                if info["gap"]:
                    repairs[symbol] = self.backfill_ticker(
                        symbol,
                        filing_types=ONBOARDING_FILING_TYPES,
                        counts_per_type=ONBOARDING_COUNTS,
                    )
        return {
            "recency_days": recency_days,
            "gaps": gaps,
            "gapped_tickers": sorted(s for s, info in gaps.items() if info["gap"]),
            "repairs": repairs,
        }

    # ── Internals ──────────────────────────────────────

    @staticmethod
    def _parse_date(value: object) -> Optional[date]:
        """Parse an ISO ``YYYY-MM-DD`` filing date, tolerating junk/None."""
        if not value:
            return None
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None

    def _mark_discovery_fresh(self, symbol: str) -> None:
        """Record discovery freshness like normal ingestion (best-effort)."""
        try:
            self.store.mark_cache_fresh(
                symbol, _DISCOVERY_CACHE_SOURCE, _DISCOVERY_TTL_HOURS,
            )
        except Exception as exc:  # noqa: BLE001 - freshness cannot break a backfill
            logger.warning(
                "Could not mark discovery cache fresh for %s: %s", symbol, exc,
            )
