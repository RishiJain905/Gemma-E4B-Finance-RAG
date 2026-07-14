"""
src/sec/scheduler.py
SEC filing scheduling — manages discovery cadence, last-checked tracking,
and summary reports for incremental filing updates.

Extends the cache_aware pattern from Phase 1.3 to SEC filings.

Usage:
    scheduler = FilingScheduler()
    scheduler.run_discovery()       # Check all core tickers
    scheduler.run_full_pipeline()   # Discover + process new filings
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from .filing_processor import FilingProcessor
from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)


class FilingScheduler:
    """Handles scheduling and incremental update logic for SEC filings.

    Filing discovery is tracked in the cache_meta table with:
        source = "sec_filings_discovery"

    Each core ticker has its own cache entry marking when its filings
    were last checked. The scheduler respects the TTL from watchlist.yaml
    and only checks tickers whose cache is stale.
    """

    # Source identifiers for cache_meta
    DISCOVERY_SOURCE = "sec_filings_discovery"

    def __init__(
        self,
        store: Optional[Store] = None,
        processor: Optional[FilingProcessor] = None,
        coverage_resolver: Optional[CoverageResolver] = None,
    ):
        self.store = store or Store()
        self.processor = processor or FilingProcessor(store=self.store)
        self._coverage_injected = coverage_resolver is not None
        self.coverage = coverage_resolver or CoverageResolver(self.store)

        # Load the SEC filing TTL from watchlist config
        self.ttl_hours = self._load_filing_ttl()

    def _tickers_for(self, source_name: str) -> list[str]:
        """Resolve policy tickers, retaining the fresh-install legacy seam."""
        if not self._coverage_injected and not self.store.list_securities(
            active=None, limit=1, offset=0
        ):
            from src.ingestion.yfinance_ingestor import YFinanceIngestor

            return list(YFinanceIngestor(store=self.store).core_tickers)
        return self.coverage.tickers_for(source_name)

    def _load_filing_ttl(self) -> int:
        """Load the SEC filing discovery TTL from watchlist config."""
        from pathlib import Path

        wl_path = Path(__file__).parent.parent.parent / "configs/watchlist.yaml"
        if wl_path.exists():
            import yaml
            with open(wl_path) as f:
                config = yaml.safe_load(f)
            return config.get("schedule", {}).get("sec_filings", 12)
        return 12

    # ── Discovery Check ───────────────────────────────

    def run_discovery(self, force: bool = False) -> dict:
        """Run filing discovery for tickers whose cache is stale.

        Checks cache_meta for each core ticker. Skips tickers that
        were checked within the TTL window unless force=True.

        Args:
            force: If True, skip freshness check and discover all

        Returns:
            {
                "checked": N,       # tickers checked
                "skipped": N,       # tickers skipped (fresh cache)
                "new_filings": N,   # total new filings registered
                "details": {ticker: new_count}
            }
        """
        tickers = self._tickers_for("sec_filings")

        result = {
            "checked": 0,
            "skipped": 0,
            "new_filings": 0,
            "details": {},
        }

        for ticker in tickers:
            # Check freshness
            if not force:
                status = self.store.get_cache_status(ticker, self.DISCOVERY_SOURCE)
                if status and status.get("status") == "fresh":
                    from datetime import datetime, timezone

                    last_updated = status.get("last_updated")
                    if last_updated:
                        try:
                            if isinstance(last_updated, str):
                                from dateutil import parser
                                updated_dt = parser.parse(last_updated)
                                if updated_dt.tzinfo is None:
                                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                            else:
                                updated_dt = last_updated
                            age = (datetime.now(timezone.utc) - updated_dt).total_seconds() / 3600
                            if age < self.ttl_hours:
                                result["skipped"] += 1
                                logger.debug(
                                    "Filing discovery for %s still fresh (%.1f hours old), skipping",
                                    ticker, age,
                                )
                                continue
                        except Exception:
                            pass

            # Run discovery
            try:
                new_count = self.processor.discover_new_filings(ticker)
                result["details"][ticker] = new_count
                result["new_filings"] += new_count
                result["checked"] += 1

                # Mark discovery cache as fresh
                self.store.mark_cache_fresh(ticker, self.DISCOVERY_SOURCE, self.ttl_hours)

            except Exception as e:
                logger.error("Filing discovery failed for %s: %s", ticker, e)
                self.store.mark_cache_stale(
                    ticker, self.DISCOVERY_SOURCE, error=str(e),
                )
                result["details"][ticker] = -1  # Error indicator

        logger.info(
            "Filing discovery: %d checked, %d skipped, %d new filings found",
            result["checked"], result["skipped"], result["new_filings"],
        )
        return result

    def run_full_pipeline(self, force: bool = False) -> dict:
        """Run discovery + processing for all core tickers.

        This is the main entry point for cron-based scheduling:
          1. Check for new filings (respects cache TTL unless force=True)
          2. Process all unprocessed filings through TraceAlchemy parser
          3. Return combined results

        Args:
            force: If True, skip freshness check on discovery

        Returns:
            {
                "discovery": {...},
                "processing": {processed: N, failed: N, errors: [...]},
                "timestamp": "2026-06-01T12:00:00Z",
            }
        """
        logger.info("Starting full filing pipeline run%s...", " (forced)" if force else "")

        discovery_result = self.run_discovery(force=force)

        broad_tickers = set(self._tickers_for("sec_filings"))
        deep_tickers = self._tickers_for("sec_filing_text")
        if broad_tickers == set(deep_tickers):
            processing_result = self.processor.process_pending_filings(limit=50)
        else:
            processing_result = {"processed": 0, "failed": 0, "errors": []}
            for ticker in deep_tickers:
                ticker_result = self.processor.process_ticker(ticker, limit=50)
                processing_result["processed"] += int(ticker_result.get("processed", 0))
                processing_result["failed"] += int(ticker_result.get("failed", 0))
                processing_result["errors"].extend(ticker_result.get("errors", []) or [])

        report = {
            "discovery": discovery_result,
            "processing": processing_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Filing pipeline complete: %d new filings, %d processed, %d failed, "
            "%d sections, %d chunks, %d replacements, %d pending",
            discovery_result["new_filings"],
            processing_result["processed"],
            processing_result["failed"],
            processing_result.get("sections_written", 0),
            processing_result.get("chunks_written", 0),
            processing_result.get("replacements", 0),
            processing_result.get("index_pending", 0),
        )
        return report

    # ── Status ─────────────────────────────────────────

    def status_report(self) -> dict:
        """Report on filing discovery freshness and pipeline health.

        Returns:
            {
                "discovery": [{ticker, status, last_checked, age_hours}, ...],
                "pipeline": {total_unprocessed, total_parsed, ...},
            }
        """
        discovery_status = []

        for ticker in self._tickers_for("sec_filings"):
            cache = self.store.get_cache_status(ticker, self.DISCOVERY_SOURCE)
            if cache:
                age = None
                last_updated = cache.get("last_updated")
                if last_updated:
                    try:
                        if isinstance(last_updated, str):
                            from dateutil import parser
                            dt = parser.parse(last_updated)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = last_updated
                        age = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                    except Exception:
                        pass

                discovery_status.append({
                    "ticker": ticker,
                    "status": cache.get("status", "unknown"),
                    "last_checked": str(cache.get("last_updated", "")),
                    "age_hours": round(age, 1) if age is not None else None,
                })
            else:
                discovery_status.append({
                    "ticker": ticker,
                    "status": "never_checked",
                    "last_checked": "",
                    "age_hours": None,
                })

        pipeline_status = self.processor.status_report()

        return {
            "discovery": discovery_status,
            "pipeline": pipeline_status,
        }

    # ── Reset ──────────────────────────────────────────

    def reset_discovery_cache(self):
        """Reset all filing discovery cache entries to stale.

        Next discovery run will re-check all tickers regardless of TTL.
        """
        for ticker in self._tickers_for("sec_filings"):
            self.store.upsert_cache_stale(ticker, self.DISCOVERY_SOURCE)
        logger.info("Filing discovery cache reset for all policy tickers")
