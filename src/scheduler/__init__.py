"""
src/scheduler/__init__.py
Unified scheduler orchestrator — coordinates all data source ingestion
with staggered execution, shared TTL tracking, and status reporting.

Usage:
    orchestrator = UnifiedScheduler()
    orchestrator.run_daily()       # Morning market-open run
    orchestrator.run_hourly()      # Intraday news check
    orchestrator.run_weekly()      # Weekend deep-dive batch
    orchestrator.run_all_stale()   # Run everything that's due

CLI (for cron):
    python -m src.scheduler daily
    python -m src.scheduler hourly
    python -m src.scheduler weekly
    python -m src.scheduler all --force
    python -m src.scheduler status
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)


class UnifiedScheduler:
    """
    Orchestrates all data source ingestion with:
      - Staggered execution (sources run sequentially with delays)
      - Shared TTL tracking via the cache_meta table
      - Partial-failure handling (one source failing doesn't kill others)
      - A dead-letter queue for persistently failing sources (best-effort)
      - Unified status reporting

    Per-source scheduler cadence is tracked in cache_meta under a synthetic
    ticker ("SCHEDULER") with source identifiers of the form
    ``"unified:<source>"`` (e.g. ``"unified:yfinance"``).
    """

    # Synthetic ticker used for the scheduler's own cadence tracking.
    SCHEDULER_TICKER = "SCHEDULER"

    SOURCES = {
        "yfinance": {
            "class": "YFinanceIngestor",
            "ttl_key": "fundamentals",
            "weight": 1,
        },
        "sec_filings": {
            "class": "FilingScheduler",
            "ttl_key": "sec_filings",
            "weight": 2,
        },
        "sec_companyfacts": {
            "class": "SECCompanyFactsIngestor",
            "ttl_key": "sec_companyfacts",
            "weight": 3,
        },
        "fred": {
            "class": "FREDIngestor",
            "ttl_key": "macro",
            "weight": 4,
        },
        "gdelt": {
            "class": "GDELTIngestor",
            "ttl_key": "gdelt_news",
            "weight": 5,
        },
        "earnings_transcripts": {
            "class": "EarningsTranscriptIngestor",
            "ttl_key": "transcripts",
            "weight": 6,
        },
        "ir_pages": {
            "class": "IRIngestor",
            "ttl_key": "ir_pages",
            "weight": 7,
        },
        "estimates": {
            "class": "EstimatesIngestor",
            "ttl_key": "estimates",
            "weight": 8,
        },
    }

    # Run-mode source selections.
    DAILY_SOURCES = [
        "yfinance", "fred", "sec_filings", "sec_companyfacts", "ir_pages", "estimates",
    ]
    HOURLY_SOURCES = ["gdelt"]
    WEEKLY_SOURCES = ["earnings_transcripts", "sec_filings"]

    DEFAULT_WATCHLIST_PATH = Path(__file__).parent.parent.parent / "configs/watchlist.yaml"

    def __init__(
        self,
        store: Optional[Store] = None,
        inter_source_delay: float = 2.0,
        watchlist_path: Optional[Path] = None,
        coverage_resolver: Optional[CoverageResolver] = None,
    ):
        self.store = store or Store()
        self.inter_source_delay = inter_source_delay
        self.watchlist_path = watchlist_path or self.DEFAULT_WATCHLIST_PATH
        self.coverage = coverage_resolver or CoverageResolver(self.store)
        self.ttls = self._load_ttls()
        self._dlq = None  # lazy

    # ── Config ─────────────────────────────────────────

    def _load_ttls(self) -> dict:
        """Load the schedule TTL map (hours) from watchlist.yaml."""
        defaults = {
            "fundamentals": 24, "news": 6, "macro": 24, "sec_filings": 12,
            "sec_companyfacts": 24,
            "gdelt_news": 6, "transcripts": 168, "ir_pages": 24,
            "estimates": 24,
        }
        try:
            import yaml
            if self.watchlist_path.exists():
                with open(self.watchlist_path) as f:
                    cfg = yaml.safe_load(f) or {}
                defaults.update(cfg.get("schedule", {}) or {})
        except Exception:
            pass
        return defaults

    @property
    def dlq(self):
        """Lazily-created dead-letter queue (best-effort; None if unavailable)."""
        if self._dlq is None:
            try:
                from src.utils.resilience import DeadLetterQueue
                self._dlq = DeadLetterQueue(self.store)
            except Exception as e:  # noqa: BLE001
                logger.debug("Dead-letter queue unavailable: %s", e)
                self._dlq = False
        return self._dlq or None

    # ── Ordering / staleness helpers ───────────────────

    def _ordered_sources(self) -> list[tuple[str, dict]]:
        """Return (name, config) pairs sorted by ascending weight."""
        return sorted(self.SOURCES.items(), key=lambda kv: kv[1]["weight"])

    @staticmethod
    def _scheduler_cache_source(name: str) -> str:
        return f"unified:{name}"

    def _ttl_for(self, name: str) -> int:
        ttl_key = self.SOURCES[name]["ttl_key"]
        return int(self.ttls.get(ttl_key, 24))

    def _is_stale(self, name: str) -> bool:
        """True if the source has never run, is marked stale, or is past TTL."""
        cache_source = self._scheduler_cache_source(name)
        status = self.store.get_cache_status(self.SCHEDULER_TICKER, cache_source)
        if not status:
            return True
        if status.get("status") == "stale":
            return True
        age = Store._age_hours(status.get("last_updated"))
        if age is None:
            return True
        return age >= self._ttl_for(name)

    # ── Source execution ───────────────────────────────

    def _run_source(self, name: str, deep: bool = False, force: bool = False) -> dict:
        """Dispatch ingestion for a single source. Returns a detail dict.

        Raises on failure so the caller can record an error / DLQ entry.
        """
        if name == "yfinance":
            from src.ingestion.yfinance_ingestor import YFinanceIngestor
            YFinanceIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
            ).ingest_all()
            return {"action": "ingest_all"}

        if name == "sec_filings":
            from src.sec import FilingScheduler
            sched = FilingScheduler(
                store=self.store,
                coverage_resolver=self.coverage,
            )
            if deep:
                return sched.run_full_pipeline(force=force)
            return sched.run_discovery(force=force)

        if name == "sec_companyfacts":
            return self._run_sec_companyfacts()

        if name == "fred":
            from src.macros.fred_ingestor import FREDIngestor
            results = FREDIngestor(store=self.store).fetch_all_indicators()
            return {"indicators_fetched": sum(1 for v in results.values() if v is not None),
                    "indicators_total": len(results)}

        if name == "gdelt":
            from src.macros.gdelt_ingestor import GDELTIngestor
            results = GDELTIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
            ).fetch_and_store_all()
            return {"articles_stored": sum(results.values()), "tickers": len(results)}

        if name == "earnings_transcripts":
            from src.macros.earnings_transcripts import EarningsTranscriptIngestor
            results = EarningsTranscriptIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
            ).fetch_all_core()
            return {"tickers_processed": len(results)}

        if name == "ir_pages":
            from src.macros.ir_ingestor import IRIngestor
            results = IRIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
            ).fetch_all_core()
            stored = sum(r.get("items_stored", 0) for r in results.values())
            return {"tickers_processed": len(results), "items_stored": stored}

        if name == "estimates":
            from src.macros.estimates_ingestor import EstimatesIngestor
            results = EstimatesIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
            ).fetch_all_core()
            stored = sum(r.get("facts_stored", 0) for r in results.values())
            return {"tickers_processed": len(results), "facts_stored": stored}

        raise ValueError(f"Unknown source: {name}")

    def _load_core_tickers(self) -> list[str]:
        """Compatibility alias for CompanyFacts deep-coverage tickers."""
        return self.coverage.tickers_for("sec_companyfacts")

    def _run_sec_companyfacts(self) -> dict:
        """Ingest core tickers independently and update per-ticker freshness."""
        from src.sec import SECCompanyFactsIngestor

        ingestor = SECCompanyFactsIngestor(store=self.store)
        if not ingestor.enabled:
            return {"enabled": False, "tickers_processed": 0}

        tickers = self._load_core_tickers()
        results: dict[str, dict] = {}
        failed = 0
        totals = {"facts_seen": 0, "facts_written": 0, "facts_skipped": 0}
        ttl_hours = self._ttl_for("sec_companyfacts")

        for ticker in tickers:
            try:
                summary = ingestor.fetch_for_ticker(ticker)
                results[ticker] = summary
                for key in totals:
                    totals[key] += int(summary.get(key, 0))
                errors = summary.get("errors", [])
                if errors:
                    failed += 1
                    self._mark_companyfacts_stale(
                        ticker,
                        "; ".join(str(error) for error in errors),
                    )
                else:
                    self.store.mark_source_fresh(
                        ticker,
                        "sec_companyfacts",
                        ttl_hours,
                    )
            except Exception as exc:  # noqa: BLE001 - isolate each ticker
                failed += 1
                error = str(exc)
                logger.error("CompanyFacts ticker %s failed: %s", ticker, error)
                results[ticker] = {"ticker": ticker, "errors": [error]}
                self._mark_companyfacts_stale(ticker, error)

        return {
            "enabled": True,
            "tickers_processed": len(tickers),
            "tickers_failed": failed,
            **totals,
            "tickers": results,
        }

    def _mark_companyfacts_stale(self, ticker: str, error: str) -> None:
        """Record a ticker failure without compromising ticker isolation."""
        try:
            self.store.mark_source_stale(ticker, "sec_companyfacts", error)
        except Exception as stale_error:  # noqa: BLE001
            logger.error(
                "Could not mark CompanyFacts stale for %s: %s",
                ticker,
                stale_error,
            )

    def _run_sources(
        self, names: list[str], force: bool = False, deep_sec: bool = False,
    ) -> dict:
        """Run the given sources in weight order with staggered execution."""
        ordered = [(n, c) for n, c in self._ordered_sources() if n in names]
        results: dict[str, dict] = {}

        for i, (name, cfg) in enumerate(ordered):
            if not self.coverage.is_enabled(name):
                results[name] = {"status": "skipped", "reason": "policy_disabled"}
                continue
            if not force and not self._is_stale(name):
                results[name] = {"status": "skipped", "reason": "cache_fresh"}
                continue

            start = time.monotonic()
            try:
                detail = self._run_source(
                    name, deep=(deep_sec and name == "sec_filings"), force=force,
                )
                results[name] = {
                    "status": "success",
                    "duration_s": round(time.monotonic() - start, 2),
                    "details": detail,
                }
                self.store.mark_cache_fresh(
                    self.SCHEDULER_TICKER,
                    self._scheduler_cache_source(name),
                    self._ttl_for(name),
                )
            except Exception as e:  # noqa: BLE001 - isolate per-source failures
                logger.error("Scheduler source %s failed: %s", name, e)
                results[name] = {
                    "status": "error",
                    "duration_s": round(time.monotonic() - start, 2),
                    "error": str(e),
                }
                self.store.upsert_cache_stale(
                    self.SCHEDULER_TICKER,
                    self._scheduler_cache_source(name),
                    str(e),
                )
                if self.dlq is not None:
                    try:
                        self.dlq.add(name, "scheduler", str(e))
                    except Exception:  # noqa: BLE001
                        pass

            # Stagger delay between sources (not after the last one).
            if i < len(ordered) - 1 and self.inter_source_delay > 0:
                time.sleep(self.inter_source_delay)

        return results

    # ── Run modes ──────────────────────────────────────

    def run_all_stale(self, force: bool = False) -> dict:
        """Run every source whose cache TTL has expired (or all if force)."""
        return self._run_sources(list(self.SOURCES.keys()), force=force)

    def run_daily(self, force: bool = False) -> dict:
        """Morning market-open run: Yahoo Finance + FRED + SEC discovery."""
        return self._run_sources(self.DAILY_SOURCES, force=force)

    def run_hourly(self, force: bool = False) -> dict:
        """Intraday news check: GDELT only."""
        return self._run_sources(self.HOURLY_SOURCES, force=force)

    def run_weekly(self, force: bool = False) -> dict:
        """Weekend deep-dive: earnings transcripts + full SEC pipeline."""
        return self._run_sources(self.WEEKLY_SOURCES, force=force, deep_sec=True)

    # ── Status ─────────────────────────────────────────

    def status_report(self) -> dict:
        """Return freshness/run status for every source."""
        sources: dict[str, dict] = {}
        for name, cfg in self._ordered_sources():
            cache_source = self._scheduler_cache_source(name)
            status = self.store.get_cache_status(self.SCHEDULER_TICKER, cache_source)
            ttl = self._ttl_for(name)
            if not status:
                sources[name] = {
                    "status": "never_fetched",
                    "last_run": None,
                    "age_hours": None,
                    "ttl_hours": ttl,
                    "error": None,
                    "coverage": self.coverage.explain(name),
                }
                continue
            age = Store._age_hours(status.get("last_updated"))
            raw_status = status.get("status")
            if raw_status == "stale":
                state = "stale"
            elif age is not None and age < ttl:
                state = "fresh"
            else:
                state = "stale"
            sources[name] = {
                "status": state,
                "last_run": str(status.get("last_updated")) if status.get("last_updated") else None,
                "age_hours": round(age, 2) if age is not None else None,
                "ttl_hours": ttl,
                "error": status.get("error_message"),
                "coverage": self.coverage.explain(name),
            }

        # Persisted counters only: status must never invoke an embedding call.
        with self.store.sqlite._connect() as conn:
            filing_index = dict(
                conn.execute(
                    "SELECT "
                    "SUM(CASE WHEN status='index_pending' THEN 1 ELSE 0 END) "
                    "AS pending, "
                    "COALESCE(SUM(index_section_count), 0) AS sections, "
                    "COALESCE(SUM(index_chunk_count), 0) AS chunks "
                    "FROM filings"
                ).fetchone()
            )
        sources.setdefault("sec_filings", {})["filing_text_index"] = {
            "pending": int(filing_index.get("pending") or 0),
            "sections": int(filing_index.get("sections") or 0),
            "chunks": int(filing_index.get("chunks") or 0),
        }

        return {
            "sources": sources,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Reset ──────────────────────────────────────────

    def reset_schedule(self):
        """Mark every scheduler source stale so the next run re-runs all."""
        for name in self.SOURCES:
            self.store.upsert_cache_stale(
                self.SCHEDULER_TICKER, self._scheduler_cache_source(name),
            )


def main():
    """CLI entry point for cron jobs."""
    import argparse

    parser = argparse.ArgumentParser(description="Unified ingestion scheduler")
    parser.add_argument("mode", choices=["daily", "hourly", "weekly", "all", "status"])
    parser.add_argument("--force", action="store_true", help="Skip freshness checks")
    args = parser.parse_args()

    scheduler = UnifiedScheduler()
    if args.mode == "status":
        print(json.dumps(scheduler.status_report(), indent=2))
    elif args.mode == "all":
        print(json.dumps(scheduler.run_all_stale(force=args.force), indent=2))
    elif args.mode == "daily":
        print(json.dumps(scheduler.run_daily(force=args.force), indent=2))
    elif args.mode == "hourly":
        print(json.dumps(scheduler.run_hourly(force=args.force), indent=2))
    elif args.mode == "weekly":
        print(json.dumps(scheduler.run_weekly(force=args.force), indent=2))


if __name__ == "__main__":
    main()
