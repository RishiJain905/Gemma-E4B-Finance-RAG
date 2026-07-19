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
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from src.ingestion.errors import (
    ErrorClass,
    ProviderError,
    normalize_error_class,
    safe_message,
)
from src.scheduler.budget import BudgetExhaustedError, RunBudget
from src.scheduler.cursors import CursorManager
from src.scheduler.status import build_coverage_health
from src.scheduler.source_registry import VALID_SCOPES, SourceRegistry, SourceSpec
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
    BUDGETED_HTTP_SOURCES = frozenset(
        {
            "finnhub",
            "massive",
            "federal_reserve",
            "treasury",
            "bls",
            "bea",
            "eia",
            "ny_fed",
            "cftc",
            "openfda",
            "nhtsa",
            "usaspending",
        }
    )

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
    BOOTSTRAP_SOURCE_ORDER = (
        "universe_nasdaq100",
        "universe_ivv",
        "universe_sec",
        "sec_filings",
        "finnhub",
        "massive",
        "fred",
        "federal_reserve",
        "treasury",
        "bls",
        "bea",
        "eia",
        "ny_fed",
        "cftc",
        "openfda",
        "nhtsa",
        "usaspending",
    )
    BOUNDED_REFRESH_SOURCES = {
        "yfinance_fundamentals": "yfinance",
        "yfinance_news": "yfinance",
        "finnhub_news": "finnhub",
        "massive_market": "massive",
        "massive_actions": "massive",
        "sec_companyfacts": "sec_companyfacts",
        "gdelt_news": "gdelt",
        "estimates": "estimates",
    }

    DEFAULT_WATCHLIST_PATH = Path(__file__).parent.parent.parent / "configs/watchlist.yaml"
    DEFAULT_SOURCES_PATH = Path(__file__).parent.parent.parent / "configs/sources.yaml"

    def __init__(
        self,
        store: Optional[Store] = None,
        inter_source_delay: float = 2.0,
        watchlist_path: Optional[Path] = None,
        coverage_resolver: Optional[CoverageResolver] = None,
        registry: Optional[SourceRegistry] = None,
        sources_path: Optional[Path] = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.store = store or Store()
        self.inter_source_delay = inter_source_delay
        self.watchlist_path = watchlist_path or self.DEFAULT_WATCHLIST_PATH
        self.registry = registry or SourceRegistry.load(
            sources_path or self.DEFAULT_SOURCES_PATH
        )
        # Compatibility surface for callers that enumerate scheduler.SOURCES.
        self.SOURCES = dict(self.registry.sources)
        self.DAILY_SOURCES = [
            spec.name
            for spec in self.registry.select("daily", include_disabled=True)
        ]
        self.HOURLY_SOURCES = [
            spec.name
            for spec in self.registry.select("hourly", include_disabled=True)
        ]
        self.WEEKLY_SOURCES = [
            spec.name
            for spec in self.registry.select("weekly", include_disabled=True)
        ]
        self.coverage = coverage_resolver or CoverageResolver(self.store)
        self.ttls = self._load_ttls()
        self.cursors = CursorManager(self.store)
        self._selected_partitions: dict[str, list[str]] = {}
        self._active_budgets: dict[str, RunBudget] = {}
        self._provider_policies: dict[str, object] = {}
        self._budget_windows: dict[str, tuple[str, str]] = {}
        self._dlq = None  # lazy
        self._now_fn = now_fn
        self._bootstrap_context: Optional[dict[str, object]] = None
        self.bootstrap_config = self._load_bootstrap_config(sources_path or self.DEFAULT_SOURCES_PATH)

    # ── Config ─────────────────────────────────────────

    def _load_ttls(self) -> dict:
        """Return the registry-owned TTL map keyed for legacy consumers."""
        return {
            spec.ttl_key: spec.ttl_hours
            for spec in self.registry.sources.values()
            if spec.status != "invalid_configuration"
        }

    @staticmethod
    def _load_bootstrap_config(path: Path) -> dict[str, object]:
        """Load bounded bootstrap windows without reading credentials."""
        defaults: dict[str, object] = {
            "sec_lookback_days": 30,
            "company_news_days": 30,
            "market_history_days": 30,
            "macro_history_observations": 5,
            "max_sec_partitions": 90,
            "max_run_summaries": 100,
        }
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            block = loaded.get("bootstrap") if isinstance(loaded, dict) else None
            if isinstance(block, dict):
                defaults.update(block)
        except (OSError, ValueError, TypeError):
            logger.warning("Could not load bootstrap configuration from %s", path)
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

    def _ordered_sources(self) -> list[tuple[str, SourceSpec]]:
        """Return source specifications sorted by configured priority."""
        return sorted(
            self.SOURCES.items(),
            key=lambda item: (item[1].priority, item[0]),
        )

    @staticmethod
    def _scheduler_cache_source(name: str) -> str:
        return f"unified:{name}"

    def _ttl_for(self, name: str) -> int:
        return int(self.SOURCES[name].ttl_hours)

    def _new_budget(self, spec: SourceSpec) -> RunBudget:
        """Create a run budget from durable provider usage windows."""
        now = self._now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        day_start = now.date().isoformat()
        minute_start = now.strftime("%Y-%m-%dT%H:%MZ")
        usage = self.store.get_source_budget_usage(
            spec.name,
            day_start=day_start,
            minute_start=minute_start,
        )
        provider_remaining = usage.get("provider_remaining")
        provider_reset = usage.get("provider_reset")
        provider_state = self.store.get_source_cursor_state(
            spec.name, "__provider__"
        ) or {}
        if (
            provider_remaining is None
            and provider_state.get("status") == "circuit_open"
            and provider_state.get("cursor_value")
        ):
            provider_remaining = 0
            provider_reset = provider_state["cursor_value"]
        if provider_remaining is not None and provider_reset is not None:
            try:
                reset_at = float(provider_reset)
            except ValueError:
                try:
                    reset_at = datetime.fromisoformat(
                        str(provider_reset).replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    reset_at = None
            if reset_at is not None and reset_at <= now.timestamp():
                provider_remaining = None
                provider_reset = None
        self._budget_windows[spec.name] = (day_start, minute_start)
        return RunBudget(
            requests_per_minute=spec.requests_per_minute,
            requests_per_day=spec.requests_per_day,
            requests_per_run=spec.requests_per_run,
            max_work_items=spec.max_work_items_per_run,
            day_requests=int(usage.get("day_requests") or 0),
            minute_requests=int(usage.get("minute_requests") or 0),
            provider_remaining=provider_remaining,
            provider_reset=provider_reset,
            wall_now_fn=self._now_fn,
        )

    def _remember_budget(self, name: str, budget: RunBudget) -> None:
        """Persist request attempts and provider limits for later processes."""
        day_start, minute_start = self._budget_windows[name]
        self.store.record_source_budget_usage(
            name,
            day_start=day_start,
            minute_start=minute_start,
            attempted_requests=budget.attempted_requests,
            successful_requests=budget.successful_requests,
            provider_remaining=budget.provider_remaining,
            provider_reset=budget.provider_reset,
        )

    def _remember_budget_safely(self, name: str, budget: RunBudget) -> None:
        """Persist accounting without allowing bookkeeping to stop later sources."""
        try:
            self._remember_budget(name, budget)
        except Exception as exc:  # noqa: BLE001 - preserve source isolation
            logger.error("Could not persist budget usage for %s: %s", name, exc)

    def _mark_scheduler_stale(self, name: str, error: str) -> None:
        """Best-effort scheduler freshness update for one failed source."""
        try:
            self.store.upsert_cache_stale(
                self.SCHEDULER_TICKER,
                self._scheduler_cache_source(name),
                error,
            )
        except Exception as exc:  # noqa: BLE001 - preserve source isolation
            logger.error("Could not mark scheduler source %s stale: %s", name, exc)

    def _budgeted_http_request(
        self,
        name: str,
        request_fn,
        *,
        wait_for_minute: bool = False,
    ):
        """Return a budgeted, retrying HTTP callable with one source circuit."""
        from src.utils.resilience import provider_request_policy

        budgeted_request = self._active_budgets[name].wrap_http_get(
            request_fn,
            wait_for_minute=wait_for_minute,
            max_wait_seconds=60,
        )
        policy = self._provider_policies.get(name)
        if policy is None:
            policy = provider_request_policy(
                name,
                self.SOURCES[name].retry_policy,
                now_fn=self._now_fn,
            )
            self._provider_policies[name] = policy

        def request(*args: object, **kwargs: object) -> object:
            return policy.request(lambda: budgeted_request(*args, **kwargs))

        return request

    def _budgeted_http_get(self, name: str, *, wait_for_minute: bool = False):
        """Return a budgeted GET callable."""
        import requests

        return self._budgeted_http_request(
            name,
            requests.get,
            wait_for_minute=wait_for_minute,
        )

    def _budgeted_http_post(self, name: str):
        """Return a budgeted POST callable."""
        import requests

        return self._budgeted_http_request(name, requests.post)

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
        if name in {"universe_nasdaq100", "universe_ivv", "universe_sec"}:
            return self._run_universe_source(name)

        if name == "finnhub":
            from src.ingestion.finnhub_ingestor import FinnhubIngestor

            tickers = self._selected_partitions.get(name, [])
            lookback_days = int(self.bootstrap_config.get("company_news_days", 30))
            return FinnhubIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
                http_get=self._budgeted_http_get(name),
                overlap_hours=int((self.SOURCES[name].overlap or "0h")[:-1]),
                initial_lookback_days=lookback_days,
            ).ingest_news(tickers=tickers)

        if name == "massive":
            from src.ingestion.massive_ingestor import MassiveIngestor

            if self._bootstrap_context is not None:
                return MassiveIngestor(
                    store=self.store,
                    coverage_resolver=self.coverage,
                    http_get=self._budgeted_http_get(name, wait_for_minute=True),
                    overlap_days=int((self.SOURCES[name].overlap or "0d")[:-1]),
                    initial_lookback_days=int(
                        self.bootstrap_config.get("market_history_days", 30)
                    ),
                ).ingest_all(
                    start_date=str(
                        self._bootstrap_context.get("source_since")
                        or self._bootstrap_context.get("since")
                        or ""
                    ) or None,
                    end_date=str(self._bootstrap_context.get("through") or "") or None,
                    include_news=True,
                )
            return MassiveIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
                http_get=self._budgeted_http_get(name),
                overlap_days=int((self.SOURCES[name].overlap or "0d")[:-1]),
            ).ingest_all()

        official = self._official_ingestor(name)
        if official is not None:
            kwargs = {
                "store": self.store,
                "coverage_resolver": self.coverage,
                "http_get": self._budgeted_http_get(name),
            }
            if name in {"bls", "usaspending"}:
                kwargs["http_post"] = self._budgeted_http_post(name)
            return official(
                **kwargs,
            ).ingest()

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
            if sched.daily_index_discovery is not None and name in self._active_budgets:
                sched.daily_index_discovery.http_get = self._budgeted_http_get(name)
            if deep:
                return sched.run_full_pipeline(force=force, allow_bootstrap=False)
            return sched.run_discovery(force=force, allow_bootstrap=False)

        if name == "sec_companyfacts":
            return self._run_sec_companyfacts()

        if name == "fred":
            from src.macros.fred_ingestor import FREDIngestor
            ingestor = FREDIngestor(store=self.store)
            if self._bootstrap_context is not None:
                results = ingestor.fetch_all_indicators(
                    limit=int(self.bootstrap_config.get("macro_history_observations", 5)),
                    store_history=True,
                )
            else:
                results = ingestor.fetch_all_indicators()
            fetched = sum(1 for value in results.values() if value is not None)
            total = len(results)
            failed = max(total - fetched, 0)
            return {
                "status": (
                    "success" if failed == 0
                    else "partial" if fetched else "error"
                ),
                "indicators_fetched": fetched,
                "indicators_total": total,
                "failed": failed,
            }

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

    def run_bounded_security_refresh(self, ticker: str, logical_source: str) -> dict:
        """Refresh one security/source through registry and durable run budgets."""
        symbol = str(ticker or "").strip().upper()
        logical = str(logical_source or "").strip().lower()
        if not symbol or len(symbol) > 16:
            raise ValueError("ticker must be a bounded security symbol")
        registry_name = self.BOUNDED_REFRESH_SOURCES.get(logical)
        if registry_name is None:
            raise ValueError(f"source is not bounded-refresh capable: {logical}")
        spec = self.registry.get(registry_name)
        if not spec.is_available:
            raise ProviderError(
                spec.disabled_reason or spec.status,
                error_class=(
                    ErrorClass.AUTHENTICATION
                    if spec.status == "disabled_missing_key"
                    else ErrorClass.ENTITLEMENT
                ),
                provider_wide=True,
            )

        budget = self._new_budget(spec)
        self._active_budgets[registry_name] = budget
        self._provider_policies.pop(registry_name, None)
        try:
            wrapped_http = logical in {
                "finnhub_news", "massive_market", "massive_actions",
            }
            reserved = (
                budget.reserve_work(work_items=1)
                if wrapped_http
                else budget.reserve(requests=1, work_items=1)
            )
            if not reserved:
                raise BudgetExhaustedError(
                    f"bounded refresh rejected: {budget.exhausted_reason}"
                )

            if logical in {"yfinance_fundamentals", "yfinance_news"}:
                from src.ingestion.yfinance_ingestor import YFinanceIngestor

                ingestor = YFinanceIngestor(
                    store=self.store, coverage_resolver=self.coverage,
                )
                provider_ticker = ingestor._fetch_ticker(symbol)
                if provider_ticker is None:
                    raise RuntimeError(f"yfinance returned no ticker for {symbol}")
                if logical == "yfinance_fundamentals":
                    ingestor._ingest_ticker_fundamentals(symbol, provider_ticker)
                else:
                    ingestor._ingest_ticker_news(symbol, provider_ticker)
                detail = {"status": "success", "items": 1}
            elif logical == "finnhub_news":
                from src.ingestion.finnhub_ingestor import FinnhubIngestor

                detail = FinnhubIngestor(
                    store=self.store,
                    coverage_resolver=self.coverage,
                    http_get=self._budgeted_http_get(registry_name),
                ).ingest_ticker_news(symbol)
            elif logical in {"massive_market", "massive_actions"}:
                from src.ingestion.massive_ingestor import MassiveIngestor

                ingestor = MassiveIngestor(
                    store=self.store,
                    coverage_resolver=self.coverage,
                    http_get=self._budgeted_http_get(registry_name),
                )
                detail = (
                    ingestor.ingest_market_data(tickers=[symbol])
                    if logical == "massive_market"
                    else ingestor.ingest_corporate_actions(tickers=[symbol])
                )
            elif logical == "sec_companyfacts":
                from src.sec import SECCompanyFactsIngestor

                detail = SECCompanyFactsIngestor(
                    store=self.store,
                ).fetch_for_ticker(symbol)
            elif logical == "gdelt_news":
                from src.macros.gdelt_ingestor import GDELTIngestor

                stored = GDELTIngestor(
                    store=self.store, coverage_resolver=self.coverage,
                ).fetch_and_store_for_ticker(symbol)
                detail = {"status": "success", "items": int(stored or 0)}
            else:
                from src.macros.estimates_ingestor import EstimatesIngestor

                detail = EstimatesIngestor(
                    store=self.store, coverage_resolver=self.coverage,
                ).fetch_for_ticker(symbol)

            status, reason = self._classify_detail(detail or {})
            if status != "success":
                raise RuntimeError(reason or f"{logical} bounded refresh failed")
            cfg = Store.FRESHNESS_SOURCES.get(logical)
            if cfg:
                self.store.mark_source_fresh(
                    symbol,
                    str(cfg["cache_source"]),
                    self._ttl_for(registry_name),
                )
            return {
                "status": "success",
                "source": logical,
                "ticker": symbol,
                "details": detail,
                "budget": budget.snapshot(),
            }
        finally:
            self._active_budgets.pop(registry_name, None)
            self._remember_budget_safely(registry_name, budget)

    def _run_universe_source(self, name: str) -> dict:
        """Fetch and atomically reconcile one configured universe snapshot."""
        from src.universe.providers import provider_from_config
        from src.universe.registry import UniverseRegistry

        provider = provider_from_config(
            name,
            sec_user_agent=os.environ.get("SEC_EDGAR_USER_AGENT"),
        )
        rows = provider.fetch()
        result = UniverseRegistry(self.store).refresh(
            provider.source,
            datetime.now(timezone.utc).isoformat(),
            rows,
        )
        return asdict(result)

    @staticmethod
    def _official_ingestor(name: str):
        """Return the configured official adapter class, if ``name`` is official."""
        if name == "federal_reserve":
            from src.ingestion.official.federal_reserve import FederalReserveIngestor

            return FederalReserveIngestor
        if name == "treasury":
            from src.ingestion.official.treasury import TreasuryIngestor

            return TreasuryIngestor
        if name == "bls":
            from src.ingestion.official.bls import BLSIngestor

            return BLSIngestor
        if name == "bea":
            from src.ingestion.official.bea import BEAIngestor

            return BEAIngestor
        if name == "eia":
            from src.ingestion.official.eia import EIAIngestor

            return EIAIngestor
        if name == "ny_fed":
            from src.ingestion.official.ny_fed import NYFedIngestor

            return NYFedIngestor
        if name == "cftc":
            from src.ingestion.official.cftc import CFTCIngestor

            return CFTCIngestor
        if name == "openfda":
            from src.ingestion.official.openfda import OpenFDAIngestor

            return OpenFDAIngestor
        if name == "nhtsa":
            from src.ingestion.official.nhtsa import NHTSAIngestor

            return NHTSAIngestor
        if name == "usaspending":
            from src.ingestion.official.usaspending import USAspendingIngestor

            return USAspendingIngestor
        return None

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

    @staticmethod
    def _classify_detail(detail: dict) -> tuple[str, Optional[str]]:
        """Map fail-soft adapter envelopes to scheduler run/freshness states."""
        skipped = {
            "disabled_missing_key",
            "disabled_authentication",
            "disabled_entitlement",
            "rate_limited",
            "skipped_provider_circuit",
            "skipped_no_key",
        }
        top_status = str(detail.get("status") or "").lower()
        if top_status in skipped:
            return "skipped", top_status
        if top_status == "error":
            return "error", "provider_error"
        if top_status == "partial":
            return "partial", "provider_partial"

        nested_statuses = {
            str(value.get("status") or "").lower()
            for value in detail.values()
            if isinstance(value, dict) and value.get("status")
        }
        adverse = nested_statuses & (skipped | {"error", "partial"})
        if adverse:
            healthy = nested_statuses & {"ok", "success", "completed"}
            if healthy or "partial" in adverse:
                return "partial", "provider_partial"
            if "error" in adverse:
                return "error", "provider_error"
            if len(adverse) == 1:
                return "skipped", next(iter(adverse))
            return "error", "provider_error"

        failure_count = 0
        healthy_count = 0
        failure_keys = {"failed", "tickers_failed", "items_failed"}
        healthy_keys = {
            "downloaded", "registered", "stored", "items", "accepted_items",
            "new_filings", "indicators_fetched", "tickers_processed",
            "facts_written",
        }
        pending = [detail]
        while pending:
            value = pending.pop()
            for key, nested in value.items():
                if isinstance(nested, dict):
                    pending.append(nested)
                    continue
                if key == "errors" and isinstance(nested, (list, tuple)):
                    failure_count += len(nested)
                    continue
                try:
                    count = max(int(nested or 0), 0)
                except (TypeError, ValueError):
                    continue
                if key in failure_keys:
                    failure_count += count
                elif key in healthy_keys:
                    healthy_count += count
        if failure_count:
            return (
                ("partial", "provider_partial")
                if healthy_count else ("error", "provider_error")
            )
        return "success", None

    @staticmethod
    def _detail_value(detail: object, key: str) -> object:
        """Find one top-level-or-nested adapter result value."""
        if not isinstance(detail, dict):
            return None
        if detail.get(key) not in (None, ""):
            return detail[key]
        for value in detail.values():
            found = UnifiedScheduler._detail_value(value, key)
            if found not in (None, ""):
                return found
        return None

    @staticmethod
    def _detail_metric(detail: object, keys: tuple[str, ...]) -> int:
        """Read aggregate counters without double-counting nested envelopes."""
        if not isinstance(detail, dict):
            return 0
        present = [key for key in keys if key in detail]
        if present:
            return sum(int(detail.get(key) or 0) for key in present)
        return sum(
            UnifiedScheduler._detail_metric(value, keys)
            for value in detail.values()
            if isinstance(value, dict)
        )

    def _add_observability(
        self,
        source: str,
        result: dict,
        budget: Optional[RunBudget],
    ) -> None:
        """Attach the bounded common source-result contract in place."""
        detail = result.get("details")
        detail_requests = self._detail_metric(detail, ("requests",))
        budget_attempts = budget.attempted_requests if budget is not None else 0
        result.setdefault("attempts", max(detail_requests, budget_attempts))
        result.setdefault("requests", max(detail_requests, budget_attempts))
        result.setdefault(
            "accepted_items",
            self._detail_metric(detail, ("stored", "updated", "duplicates")),
        )
        result.setdefault(
            "rejected_items",
            self._detail_metric(detail, ("malformed", "rejected")),
        )
        result.setdefault("terminal_status", result.get("status", "error"))
        detail_error = self._detail_value(detail, "error_class")
        result["error_class"] = normalize_error_class(
            result.get("error_class") or detail_error
        )
        policy = self._provider_policies.get(source)
        retry_timestamps = self._detail_value(detail, "retry_timestamps")
        result.setdefault(
            "retry_timestamps",
            (
                list(retry_timestamps)
                if isinstance(retry_timestamps, list)
                else list(getattr(policy, "retry_timestamps", []))
            ),
        )
        detail_reset = self._detail_value(detail, "reset_at")
        policy_reset = getattr(getattr(policy, "last_error", None), "reset_at", None)
        result.setdefault(
            "reset_at",
            detail_reset
            or policy_reset
            or (budget.provider_reset if budget is not None else None),
        )
        result.setdefault(
            "circuit_opened_at",
            getattr(policy, "circuit_opened_at", None),
        )
        result.setdefault(
            "last_committed_cursor",
            self._detail_value(detail, "cursor_after"),
        )
        result.setdefault(
            "remaining_work_skipped",
            bool(self._detail_value(detail, "remaining_work_skipped")),
        )

    def _dead_letter_source_failure(
        self,
        source: str,
        error: ProviderError,
    ) -> None:
        """Best-effort one-row provider-wide failure recording."""
        if self.dlq is None:
            return
        try:
            self.dlq.add_failure(
                source=source,
                partition="__provider__" if error.provider_wide else "scheduler",
                provider_record_id=None,
                cursor=None,
                error_class=error.error_class.value,
                message=error.safe_message,
                attempts=error.attempts,
            )
        except Exception:  # noqa: BLE001 - DLQ cannot couple sources
            logger.debug("Could not add %s provider failure to DLQ", source, exc_info=True)

    def _persist_provider_status(
        self,
        source: str,
        status: str,
        *,
        error_class: Optional[str] = None,
        message: Optional[str] = None,
        retry_after: Optional[float] = None,
        reset_at: Optional[str] = None,
    ) -> None:
        """Persist one source-wide circuit state without touching data cursors."""
        try:
            self.store.set_source_cursor(
                source,
                "__provider__",
                reset_at,
                cursor_type="timestamp" if reset_at else "none",
                status=status,
                error_class=normalize_error_class(error_class),
                error_message=safe_message(message) if message else None,
                retry_after=retry_after,
            )
        except Exception:  # noqa: BLE001 - observability cannot couple sources
            logger.debug("Could not persist provider status for %s", source, exc_info=True)

    def _run_sources(
        self, names: list[str], force: bool = False, deep_sec: bool = False,
    ) -> dict:
        """Run configured sources sequentially with isolated failures and budgets."""
        ordered = [(n, c) for n, c in self._ordered_sources() if n in names]
        results: dict[str, dict] = {}

        for i, (name, spec) in enumerate(ordered):
            if not spec.is_available:
                results[name] = {
                    "status": "skipped",
                    "reason": spec.status,
                    "detail": spec.disabled_reason,
                    "error_class": (
                        ErrorClass.AUTHENTICATION.value
                        if spec.status == "disabled_missing_key"
                        else None
                    ),
                    "remaining_work_skipped": True,
                }
                if spec.status == "disabled_missing_key":
                    self._mark_scheduler_stale(name, spec.disabled_reason or "missing provider key")
                continue
            try:
                policy_enabled = self.coverage.is_enabled(name)
            except ValueError as exc:
                results[name] = {
                    "status": "skipped",
                    "reason": "policy_unavailable",
                    "detail": safe_message(exc),
                }
                continue
            except Exception as exc:  # noqa: BLE001 - isolate preflight failures
                message = safe_message(exc)
                logger.error("Scheduler preflight for %s failed: %s", name, message)
                results[name] = {
                    "status": "error",
                    "reason": "preflight_error",
                    "error": message,
                }
                self._mark_scheduler_stale(name, message)
                continue
            if not policy_enabled:
                results[name] = {"status": "skipped", "reason": "policy_disabled"}
                continue
            if not force:
                try:
                    if not self._is_stale(name):
                        results[name] = {
                            "status": "skipped",
                            "reason": "cache_fresh",
                        }
                        continue
                except Exception as exc:  # noqa: BLE001 - isolate preflight failures
                    message = safe_message(exc)
                    logger.error("Freshness preflight for %s failed: %s", name, message)
                    results[name] = {
                        "status": "error",
                        "reason": "preflight_error",
                        "error": message,
                    }
                    self._mark_scheduler_stale(name, message)
                    continue

            try:
                unavailable_dependencies = [
                    dependency
                    for dependency in spec.dependencies
                    if not self.registry.get(dependency).is_available
                ]
            except Exception as exc:  # noqa: BLE001 - isolate preflight failures
                message = safe_message(exc)
                logger.error("Dependency preflight for %s failed: %s", name, message)
                results[name] = {
                    "status": "error",
                    "reason": "preflight_error",
                    "error": message,
                }
                self._mark_scheduler_stale(name, message)
                continue
            if unavailable_dependencies:
                results[name] = {
                    "status": "skipped",
                    "reason": "dependency_unavailable",
                    "detail": unavailable_dependencies,
                }
                continue

            try:
                budget = self._new_budget(spec)
                self._active_budgets[name] = budget
                self._provider_policies.pop(name, None)
                if name == "finnhub":
                    partitions = self.cursors.order_partitions(
                        "finnhub_news",
                        self.coverage.tickers_for(name),
                    )
                    selected: list[str] = []
                    for partition in partitions:
                        if not budget.reserve_work(work_items=1):
                            break
                        selected.append(partition)
                    self._selected_partitions[name] = selected
                    has_capacity = bool(selected)
                elif name in self.BUDGETED_HTTP_SOURCES:
                    has_capacity = budget.reserve_work(work_items=1)
                else:
                    has_capacity = budget.reserve(
                        requests=spec.batch_size,
                        work_items=1,
                        force=force,
                    )
            except Exception as exc:  # noqa: BLE001 - isolate preflight failures
                message = safe_message(exc)
                logger.error("Budget preflight for %s failed: %s", name, message)
                results[name] = {
                    "status": "error",
                    "reason": "preflight_error",
                    "error": message,
                }
                self._mark_scheduler_stale(name, message)
                continue
            if not has_capacity:
                reason = budget.exhausted_reason or "no_work_items"
                is_cooldown = reason == "provider_cooldown"
                results[name] = {
                    "status": "skipped",
                    "reason": "provider_cooldown" if is_cooldown else "budget_exhausted",
                    "detail": reason,
                    "error_class": ErrorClass.QUOTA_EXHAUSTED.value,
                    "remaining_work_skipped": True,
                    "budget": budget.snapshot(),
                }
                self._remember_budget_safely(name, budget)
                continue

            start = time.monotonic()
            try:
                detail = self._run_source(
                    name, deep=(deep_sec and name == "sec_filings"), force=force,
                )
                request_count = int(detail.get("requests", 0) or 0)
                run_status, result_reason = self._classify_detail(detail)
                if (
                    name not in self.BUDGETED_HTTP_SOURCES
                    and run_status in {"success", "partial"}
                ):
                    budget.record_success(request_count or spec.batch_size)
                results[name] = {
                    "status": run_status,
                    "duration_s": round(time.monotonic() - start, 2),
                    "details": detail,
                    "budget": budget.snapshot(),
                }
                if result_reason is not None:
                    results[name]["reason"] = result_reason
                detail_error_class = normalize_error_class(
                    self._detail_value(detail, "error_class")
                )
                if detail_error_class in {
                    ErrorClass.AUTHENTICATION.value,
                    ErrorClass.ENTITLEMENT.value,
                    ErrorClass.RATE_LIMITED.value,
                    ErrorClass.QUOTA_EXHAUSTED.value,
                }:
                    retry_after = self._detail_value(detail, "retry_after")
                    reset_at = self._detail_value(detail, "reset_at")
                    if reset_at is None and retry_after is not None:
                        reset_at = (
                            self._now_fn()
                            + timedelta(
                                seconds=max(float(retry_after), 0.0)
                            )
                        ).isoformat().replace("+00:00", "Z")
                    budget.open_provider_circuit(
                        reset_at=str(reset_at) if reset_at else None,
                        retry_after=float(retry_after) if retry_after is not None else None,
                    )
                    results[name]["error_class"] = detail_error_class
                    results[name]["remaining_work_skipped"] = True
                    results[name]["reset_at"] = reset_at
                    provider_error = ProviderError(
                        self._detail_value(detail, "error")
                        or self._detail_value(detail, "errors")
                        or result_reason
                        or detail_error_class,
                        error_class=detail_error_class,
                        retry_after=(
                            float(retry_after) if retry_after is not None else None
                        ),
                        reset_at=str(reset_at) if reset_at else None,
                        attempts=max(request_count, 1),
                        circuit_open=True,
                    )
                    self._dead_letter_source_failure(name, provider_error)
                    self._persist_provider_status(
                        name,
                        "circuit_open",
                        error_class=detail_error_class,
                        message=provider_error.safe_message,
                        retry_after=provider_error.retry_after,
                        reset_at=provider_error.reset_at,
                    )
                if run_status == "success":
                    self.store.mark_cache_fresh(
                        self.SCHEDULER_TICKER,
                        self._scheduler_cache_source(name),
                        self._ttl_for(name),
                    )
                    if self.store.get_source_cursor_state(
                        name, "__provider__"
                    ) is not None:
                        self._persist_provider_status(name, "success")
                else:
                    self._mark_scheduler_stale(name, result_reason or run_status)
            except ProviderError as e:
                if e.provider_wide:
                    budget.open_provider_circuit(
                        reset_at=e.reset_at,
                        retry_after=e.retry_after,
                    )
                status = "skipped" if e.provider_wide else "error"
                logger.error(
                    "Scheduler source %s failed [%s]: %s",
                    name,
                    e.error_class.value,
                    e.safe_message,
                )
                results[name] = {
                    "status": status,
                    "reason": e.error_class.value,
                    "duration_s": round(time.monotonic() - start, 2),
                    "error": e.safe_message,
                    "error_class": e.error_class.value,
                    "attempts": e.attempts,
                    "requests": max(e.attempts, budget.attempted_requests),
                    "retry_timestamps": list(e.retry_timestamps),
                    "reset_at": e.reset_at,
                    "circuit_opened_at": (
                        self._now_fn().isoformat().replace("+00:00", "Z")
                        if e.circuit_open
                        else None
                    ),
                    "remaining_work_skipped": e.provider_wide,
                    "budget": budget.snapshot(),
                }
                self._mark_scheduler_stale(name, e.safe_message)
                self._dead_letter_source_failure(name, e)
                self._persist_provider_status(
                    name,
                    "circuit_open" if e.provider_wide else "error",
                    error_class=e.error_class.value,
                    message=e.safe_message,
                    retry_after=e.retry_after,
                    reset_at=e.reset_at,
                )
            except BudgetExhaustedError as e:
                results[name] = {
                    "status": "skipped",
                    "reason": "budget_exhausted",
                    "duration_s": round(time.monotonic() - start, 2),
                    "error": safe_message(e),
                    "error_class": ErrorClass.QUOTA_EXHAUSTED.value,
                    "remaining_work_skipped": True,
                    "budget": budget.snapshot(),
                }
                self._mark_scheduler_stale(name, safe_message(e))
            except Exception as e:  # noqa: BLE001 - isolate per-source failures
                message = safe_message(e)
                logger.error("Scheduler source %s failed: %s", name, message)
                results[name] = {
                    "status": "error",
                    "duration_s": round(time.monotonic() - start, 2),
                    "error": message,
                    "error_class": ErrorClass.PERMANENT.value,
                    "budget": budget.snapshot(),
                }
                self._mark_scheduler_stale(name, message)
                if self.dlq is not None:
                    try:
                        self.dlq.add_failure(
                            source=name,
                            partition="scheduler",
                            provider_record_id=None,
                            cursor=None,
                            error_class=ErrorClass.PERMANENT.value,
                            message=message,
                        )
                    except Exception:  # noqa: BLE001
                        pass

            self._remember_budget_safely(name, budget)

            # Stagger delay between sources (not after the last one).
            if i < len(ordered) - 1 and self.inter_source_delay > 0:
                time.sleep(self.inter_source_delay)

        for source_name, source_result in results.items():
            self._add_observability(
                source_name,
                source_result,
                self._active_budgets.get(source_name),
            )
        return results

    # -- Durable run summaries ----------------------------------------------

    def _status_now(self) -> str:
        """Return the injected scheduler clock as a UTC ISO timestamp."""
        value = self._now_fn()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _result_metric(result: dict, keys: tuple[str, ...]) -> int:
        """Read a direct metric or sum independent nested capability metrics."""
        values = [result]
        details = result.get("details")
        if isinstance(details, dict):
            values.append(details)
            for value in details.values():
                if isinstance(value, dict):
                    values.append(value)
        for value in values:
            for key in keys:
                try:
                    if value.get(key) is not None:
                        return max(int(value.get(key) or 0), 0)
                except (TypeError, ValueError):
                    continue
        nested_results = [
            value
            for value in result.values()
            if isinstance(value, dict)
            and any(
                marker in value
                for marker in ("status", "terminal_status", "capability")
            )
        ]
        if nested_results:
            return sum(
                UnifiedScheduler._result_metric(value, keys)
                for value in nested_results
            )
        return 0

    def _source_run_summary(
        self,
        run_id: str,
        mode: str,
        source: str,
        result: dict,
        *,
        started_at: str,
        ended_at: str,
    ) -> dict:
        """Convert one source result into the stable status-table contract."""
        status = str(result.get("terminal_status") or result.get("status") or "error")
        completed = int(status == "success")
        skipped = int(status == "skipped")
        failed = int(status in {"error", "partial"})
        budget = result.get("budget") if isinstance(result.get("budget"), dict) else {}
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        cache = self.store.get_cache_status(
            self.SCHEDULER_TICKER, self._scheduler_cache_source(source)
        )
        provider = result.get("provider") if isinstance(result.get("provider"), dict) else {}
        cooldown_reset = (
            result.get("reset_at")
            or provider.get("reset_at")
            or budget.get("provider_reset")
        )
        error_message = result.get("error") or result.get("detail")
        if isinstance(error_message, list):
            error_message = "; ".join(str(value) for value in error_message[:3])
        if error_message:
            error_message = safe_message(error_message)
        duration = result.get("duration_s")
        try:
            duration = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None
        cursor_before = details.get("cursor_before")
        cursor_after = details.get("cursor_after") or result.get("last_committed_cursor")
        quota_remaining = budget.get("provider_remaining")
        try:
            quota_remaining = int(quota_remaining) if quota_remaining is not None else None
        except (TypeError, ValueError):
            quota_remaining = None
        return {
            "run_id": run_id,
            "source": source,
            "mode": mode,
            "policy_revision": self.coverage.revision,
            "config_revision": self.registry.version,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration,
            "status": status,
            "requested": 1,
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "partitions": self._result_metric(
                result, ("partitions", "tickers", "checked", "pages")
            ),
            "items": self._result_metric(
                result, ("items", "accepted_items", "stored", "new_filings", "facts_written")
            ),
            "requests": self._result_metric(result, ("requests", "attempts")),
            "new_items": self._result_metric(result, ("new", "items_new", "stored", "new_filings")),
            "updated_items": self._result_metric(result, ("updated", "items_updated")),
            "duplicates": self._result_metric(result, ("duplicates",)),
            "cursor_before": cursor_before,
            "cursor_after": cursor_after,
            "quota_remaining": quota_remaining,
            "freshness": cache.get("status") if cache else "never_fetched",
            "last_success": cache.get("last_updated") if cache else None,
            "next_due": cache.get("next_scheduled_update") if cache else None,
            "cooldown_reset": cooldown_reset,
            "error_class": normalize_error_class(
                result.get("error_class") or details.get("error_class")
            ),
            "error_message": error_message,
            "details": details,
        }

    def _persist_run_status(
        self,
        run_id: str,
        mode: str,
        requested_sources: list[str],
        results: dict[str, dict],
        *,
        started_at: str,
        started_monotonic: float,
    ) -> None:
        """Persist one run and its source rows without coupling source execution."""
        ended_at = self._status_now()
        completed = [
            name for name, result in results.items()
            if str(result.get("terminal_status") or result.get("status")) == "success"
        ]
        skipped = [
            name for name, result in results.items()
            if str(result.get("terminal_status") or result.get("status")) == "skipped"
        ]
        failed = [
            name for name, result in results.items()
            if str(result.get("terminal_status") or result.get("status")) in {"error", "partial"}
        ]
        has_partial = any(
            str(result.get("terminal_status") or result.get("status")) == "partial"
            for result in results.values()
        )
        if has_partial:
            run_status = "partial"
        elif failed:
            run_status = "partial" if completed else "error"
        elif skipped:
            run_status = "partial" if completed else "skipped"
        else:
            run_status = "success"
        terminal = next((results[name] for name in failed + skipped if name in results), {})
        try:
            for name in requested_sources:
                result = results.get(name, {
                    "status": "skipped", "reason": "not_selected", "remaining_work_skipped": True,
                })
                summary = self._source_run_summary(
                    run_id,
                    mode,
                    name,
                    result,
                    started_at=started_at,
                    ended_at=ended_at,
                )
                self.store.record_scheduler_source_summary(summary)
            self.store.complete_scheduler_run(
                run_id,
                status=run_status,
                ended_at=ended_at,
                duration_seconds=round(time.monotonic() - started_monotonic, 3),
                completed_sources=completed,
                skipped_sources=skipped,
                failed_sources=failed,
                terminal_error_class=normalize_error_class(terminal.get("error_class")),
                terminal_error_message=(
                    safe_message(terminal.get("error") or terminal.get("detail"))
                    if terminal.get("error") or terminal.get("detail")
                    else None
                ),
            )
            self.store.prune_scheduler_history(
                int(self.bootstrap_config.get("max_run_summaries", 100))
            )
        except Exception as exc:  # noqa: BLE001 - status cannot stop ingestion
            logger.warning("Could not persist scheduler run %s: %s", run_id, safe_message(exc))

    # ── Run modes ──────────────────────────────────────

    def _run_mode(
        self,
        mode: str,
        *,
        force: bool,
        source: Optional[str],
        scope: Optional[str],
    ) -> dict:
        specs = self.registry.select(
            mode,
            source=source,
            scope=scope,
            include_disabled=True,
        )
        names = [spec.name for spec in specs]
        started_at = self._status_now()
        started_monotonic = time.monotonic()
        run_id = str(uuid.uuid4())
        try:
            self.store.start_scheduler_run(
                mode,
                policy_revision=self.coverage.revision,
                config_revision=self.registry.version,
                requested_sources=names,
                run_id=run_id,
                started_at=started_at,
            )
        except Exception as exc:  # noqa: BLE001 - operational state is best effort
            logger.warning("Could not start scheduler run: %s", safe_message(exc))
        results: dict[str, dict] = {}
        try:
            results = self._run_sources(
                names,
                force=force,
                deep_sec=mode == "weekly",
            )
            return results
        finally:
            self._persist_run_status(
                run_id,
                mode,
                names,
                results,
                started_at=started_at,
                started_monotonic=started_monotonic,
            )

    def run_all_stale(
        self,
        force: bool = False,
        source: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> dict:
        """Run every source whose cache TTL has expired (or all if force)."""
        return self._run_mode(
            "all", force=force, source=source, scope=scope
        )

    def run_daily(
        self,
        force: bool = False,
        source: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> dict:
        """Run registry sources whose allowed cadence includes daily."""
        return self._run_mode(
            "daily", force=force, source=source, scope=scope
        )

    def run_hourly(
        self,
        force: bool = False,
        source: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> dict:
        """Run registry sources whose allowed cadence includes hourly."""
        return self._run_mode(
            "hourly", force=force, source=source, scope=scope
        )

    def run_weekly(
        self,
        force: bool = False,
        source: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> dict:
        """Run weekly sources, including the full SEC pipeline."""
        return self._run_mode(
            "weekly", force=force, source=source, scope=scope
        )

    # -- Explicit bootstrap -------------------------------------------------

    @staticmethod
    def _validate_date(value: Optional[str], field: str = "date") -> Optional[str]:
        """Validate a CLI date without accepting ambiguous provider formats."""
        if value is None:
            return None
        try:
            parsed = datetime.strptime(str(value), "%Y-%m-%d").date().isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be YYYY-MM-DD") from exc
        if parsed != str(value):
            raise ValueError(f"{field} must be YYYY-MM-DD")
        return parsed

    def _bootstrap_specs(self, source: Optional[str]) -> list[SourceSpec]:
        """Return explicitly bootstrap-capable registry entries."""
        if source is not None:
            if source not in self.BOOTSTRAP_SOURCE_ORDER:
                raise ValueError(f"source is not bootstrap-capable: {source}")
            return [self.registry.get(source)]
        return [
            self.registry.get(name)
            for name in self.BOOTSTRAP_SOURCE_ORDER
            if name in self.registry.sources
        ]

    def _bootstrap_partitions(self, source: str, since: Optional[str]) -> list[str]:
        """Build deterministic bounded bootstrap partitions for one source."""
        if source.startswith("universe_"):
            return ["snapshot"]
        if source == "sec_filings":
            end = self._now_fn().date()
            start = (
                datetime.strptime(since, "%Y-%m-%d").date()
                if since
                else end - timedelta(days=int(self.bootstrap_config.get("sec_lookback_days", 30)))
            )
            if start > end:
                raise ValueError("since cannot be after the bootstrap end date")
            dates: list[str] = []
            current = start
            while current <= end:
                if current.weekday() < 5:
                    dates.append(current.isoformat())
                current += timedelta(days=1)
            return dates or [end.isoformat()]
        if source == "finnhub":
            return [str(ticker).upper() for ticker in self.coverage.tickers_for(source)]
        if source == "yfinance":
            return [str(ticker).upper() for ticker in self.coverage.tickers_for(source)]
        return ["__global__"]

    def _run_bootstrap_partition(
        self,
        source: str,
        partition: str,
        *,
        since: Optional[str],
        run_id: str,
    ) -> dict:
        """Run one bounded bootstrap partition through normal source adapters."""
        if source == "sec_filings":
            from src.sec.scheduler import FilingScheduler

            sched = FilingScheduler(store=self.store, coverage_resolver=self.coverage)
            discovery = getattr(sched, "daily_index_discovery", None)
            if discovery is not None:
                if source in self._active_budgets:
                    discovery.http_get = self._budgeted_http_get(source)
                return discovery.discover_dates([partition])
            return sched.run_discovery(force=True, allow_bootstrap=False)

        if source == "federal_reserve":
            from src.ingestion.official.federal_reserve import FederalReserveIngestor

            return FederalReserveIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
                http_get=self._budgeted_http_get(source),
            ).ingest_history(run_id=run_id)

        self._bootstrap_context = {
            "source": source,
            "since": since,
            "source_since": since,
            "through": self._now_fn().date().isoformat(),
            "run_id": run_id,
        }
        if source == "massive":
            through = self._now_fn().date()
            lookback = max(int(self.bootstrap_config.get("market_history_days", 30)), 1)
            self._bootstrap_context["source_since"] = (
                through - timedelta(days=lookback)
            ).isoformat()
        self._selected_partitions[source] = (
            [] if partition == "__global__" or partition == "snapshot" else [partition]
        )
        try:
            return self._run_source(source, force=True)
        finally:
            self._bootstrap_context = None
            self._selected_partitions.pop(source, None)

    def run_bootstrap(
        self,
        *,
        source: Optional[str] = None,
        since: Optional[str] = None,
        resume: bool = False,
    ) -> dict:
        """Populate bounded initial coverage through an explicit resumable run."""
        since = self._validate_date(since, "since")
        if since is None:
            configured = self.bootstrap_config.get("sec_start_date")
            since = self._validate_date(str(configured), "sec_start_date") if configured else None
        existing = self.store.get_resumable_bootstrap_run(source) if resume else None
        if resume and existing is None:
            raise ValueError("no resumable bootstrap run matches the request")
        if existing is not None:
            try:
                requested_names = json.loads(
                    str(existing.get("requested_sources_json") or "[]")
                )
            except (TypeError, ValueError):
                requested_names = []
            names = [str(name) for name in requested_names]
            run_source_names = list(names)
            if source is not None:
                if source not in names:
                    raise ValueError(f"source was not part of resumable run: {source}")
                names = [source]
            specs = [self.registry.get(name) for name in names]
        else:
            specs = self._bootstrap_specs(source)
            names = [spec.name for spec in specs]
            run_source_names = list(names)
        run_id = str(existing["run_id"]) if existing else str(uuid.uuid4())
        started_at = str(existing.get("started_at")) if existing else self._status_now()
        started_monotonic = time.monotonic()
        if existing is None:
            manifest = {
                spec.name: self._bootstrap_partitions(spec.name, since)
                for spec in specs
            }
            self.store.start_scheduler_run(
                "bootstrap",
                policy_revision=self.coverage.revision,
                config_revision=self.registry.version,
                requested_sources=names,
                run_id=run_id,
                started_at=started_at,
                bootstrap_manifest=manifest,
            )

        results: dict[str, dict] = {}
        try:
            for spec in specs:
                source_result: dict[str, object] = {
                    "status": "success",
                    "partitions": 0,
                    "items": 0,
                    "new": 0,
                    "updated": 0,
                    "duplicates": 0,
                    "details": {},
                }
                if not spec.is_available:
                    source_result.update({
                        "status": "skipped",
                        "reason": spec.status,
                        "detail": spec.disabled_reason,
                        "error_class": (
                            ErrorClass.AUTHENTICATION.value
                            if spec.status == "disabled_missing_key" else None
                        ),
                    })
                    results[spec.name] = source_result
                    continue

                partition_rows = self.store.list_bootstrap_partitions(run_id, spec.name)
                partitions = [str(row["partition_key"]) for row in partition_rows]
                prior = {
                    str(row["partition_key"]): row
                    for row in partition_rows
                    if row.get("status") == "completed"
                }
                source_result["partitions"] = len(prior)
                source_result["items"] = sum(int(row.get("items") or 0) for row in prior.values())
                source_result["new"] = sum(int(row.get("new_items") or 0) for row in prior.values())
                source_result["updated"] = sum(
                    int(row.get("updated_items") or 0) for row in prior.values()
                )
                source_result["duplicates"] = sum(
                    int(row.get("duplicates") or 0) for row in prior.values()
                )
                budget = self._new_budget(spec)
                self._active_budgets[spec.name] = budget
                self._provider_policies.pop(spec.name, None)
                details: list[dict] = []
                processed_this_invocation = 0
                invocation_cap = (
                    max(int(self.bootstrap_config.get("max_sec_partitions", 90)), 1)
                    if spec.name == "sec_filings"
                    else spec.max_work_items_per_run
                )
                try:
                    for partition in partitions:
                        if partition in prior:
                            continue
                        if processed_this_invocation >= invocation_cap:
                            source_result.update({
                                "status": "partial",
                                "reason": "bounded_work_remaining",
                            })
                            break
                        if not budget.reserve_work(work_items=1):
                            source_result.update({
                                "status": "partial",
                                "reason": "budget_exhausted",
                                "error_class": ErrorClass.QUOTA_EXHAUSTED.value,
                            })
                            break
                        previous = next(
                            (row for row in partition_rows if str(row["partition_key"]) == partition),
                            {},
                        )
                        attempts = int(previous.get("attempts") or 0) + 1
                        processed_this_invocation += 1
                        part_started = self._status_now()
                        self.store.record_bootstrap_partition(
                            run_id,
                            spec.name,
                            partition,
                            status="running",
                            started_at=part_started,
                            attempts=attempts,
                        )
                        try:
                            detail = self._run_bootstrap_partition(
                                spec.name,
                                partition,
                                since=since,
                                run_id=run_id,
                            ) or {}
                            part_status, reason = self._classify_detail(detail)
                            if part_status not in {"success", "partial"}:
                                raise ProviderError(
                                    reason or "bootstrap partition failed",
                                    error_class=normalize_error_class(
                                        detail.get("error_class")
                                    ) or ErrorClass.PERMANENT.value,
                                    provider_wide=bool(
                                        detail.get(
                                            "provider_wide",
                                            part_status == "skipped",
                                        )
                                    ),
                                    retry_after=detail.get("retry_after"),
                                    reset_at=detail.get("reset_at"),
                                    circuit_open=bool(detail.get("circuit_open")),
                                )
                            accepted = self._result_metric(detail, ("items", "accepted_items", "stored", "registered"))
                            new_items = self._result_metric(
                                detail, ("new", "stored", "registered")
                            )
                            updated_items = self._result_metric(detail, ("updated",))
                            duplicate_items = self._result_metric(detail, ("duplicates",))
                            checkpoint_status = (
                                "partial"
                                if spec.name == "federal_reserve"
                                and part_status == "partial"
                                else "completed"
                            )
                            self.store.record_bootstrap_partition(
                                run_id,
                                spec.name,
                                partition,
                                status=checkpoint_status,
                                started_at=part_started,
                                ended_at=self._status_now(),
                                attempts=attempts,
                                items=accepted,
                                new_items=new_items,
                                updated_items=updated_items,
                                duplicates=duplicate_items,
                            )
                            if checkpoint_status == "completed":
                                source_result["partitions"] = int(source_result["partitions"]) + 1
                            source_result["items"] = int(source_result["items"]) + accepted
                            source_result["new"] = int(source_result["new"]) + new_items
                            source_result["updated"] = int(source_result["updated"]) + updated_items
                            source_result["duplicates"] = int(source_result["duplicates"]) + duplicate_items
                            details.append({"partition": partition, **detail})
                            if part_status == "partial":
                                errors = detail.get("errors")
                                first_error = (
                                    str(errors[0])
                                    if isinstance(errors, list) and errors
                                    else None
                                )
                                source_result.update({
                                    "status": "partial",
                                    "reason": reason or "provider_partial",
                                    "error": first_error,
                                    "error_class": normalize_error_class(
                                        detail.get("error_class")
                                    ) or ErrorClass.ITEM.value,
                                })
                        except ProviderError as exc:
                            if exc.provider_wide:
                                budget.open_provider_circuit(
                                    reset_at=exc.reset_at,
                                    retry_after=exc.retry_after,
                                )
                                self._persist_provider_status(
                                    spec.name,
                                    "circuit_open",
                                    error_class=exc.error_class.value,
                                    message=exc.safe_message,
                                    retry_after=exc.retry_after,
                                    reset_at=exc.reset_at,
                                )
                            source_result.update({
                                "status": (
                                    "partial"
                                    if int(source_result["items"]) > 0
                                    else "error" if not exc.provider_wide else "skipped"
                                ),
                                "reason": exc.error_class.value,
                                "error": exc.safe_message,
                                "error_class": exc.error_class.value,
                                "reset_at": exc.reset_at,
                            })
                            self.store.record_bootstrap_partition(
                                run_id,
                                spec.name,
                                partition,
                                status="error",
                                started_at=part_started,
                                ended_at=self._status_now(),
                                attempts=attempts,
                                error_class=exc.error_class.value,
                                error_message=exc.safe_message,
                            )
                            break
                        except Exception as exc:  # noqa: BLE001 - one partition cannot abort bootstrap
                            message = safe_message(exc)
                            source_result.update({
                                "status": "error",
                                "reason": "partition_error",
                                "error": message,
                                "error_class": ErrorClass.ITEM.value,
                            })
                            self.store.record_bootstrap_partition(
                                run_id,
                                spec.name,
                                partition,
                                status="error",
                                started_at=part_started,
                                ended_at=self._status_now(),
                                attempts=attempts,
                                error_class=ErrorClass.ITEM.value,
                                error_message=message,
                            )
                            break
                finally:
                    self._active_budgets.pop(spec.name, None)
                source_result["budget"] = budget.snapshot()
                source_result["details"] = {"partitions": details}
                remaining = len(partitions) - int(source_result["partitions"])
                if remaining > 0 and source_result.get("status") == "success":
                    source_result.update({
                        "status": "partial",
                        "reason": "bounded_work_remaining",
                    })
                results[spec.name] = source_result
                self._remember_budget_safely(spec.name, budget)
        finally:
            self._persist_run_status(
                run_id,
                "bootstrap",
                names,
                results,
                started_at=started_at,
                started_monotonic=started_monotonic,
            )

        statuses = [str(result.get("status")) for result in results.values()]
        overall = (
            "error" if any(status == "error" for status in statuses)
            else "partial" if any(status in {"skipped", "partial"} for status in statuses)
            else "success"
        )
        all_checkpoints = self.store.list_bootstrap_partitions(run_id)
        checkpoint_sources: dict[str, list[dict]] = {
            name: [
                row for row in all_checkpoints
                if str(row.get("source")) == name
            ]
            for name in run_source_names
        }
        complete_sources = [
            name for name, rows in checkpoint_sources.items()
            if rows and all(str(row.get("status")) == "completed" for row in rows)
        ]
        failed_sources = [
            name for name, rows in checkpoint_sources.items()
            if any(str(row.get("status")) == "error" for row in rows)
        ]
        pending_sources = [
            name for name in run_source_names
            if name not in complete_sources and name not in failed_sources
        ]
        if len(complete_sources) < len(run_source_names):
            overall = "partial" if complete_sources or pending_sources else "error"
            self.store.complete_scheduler_run(
                run_id,
                status=overall,
                completed_sources=complete_sources,
                skipped_sources=pending_sources,
                failed_sources=failed_sources,
                terminal_error_class=(
                    ErrorClass.ITEM.value if failed_sources else None
                ),
                terminal_error_message=(
                    "bootstrap partitions remain retryable"
                    if failed_sources else "bootstrap partitions remain pending"
                ),
            )
        return {"mode": "bootstrap", "run_id": run_id, "status": overall, "sources": results}

    # -- Repair/reindex -----------------------------------------------------

    def run_repair(
        self,
        *,
        source: Optional[str] = None,
        limit: int = 100,
        security: Optional[str] = None,
        item_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> dict:
        """Retry stored indexing records without invoking a provider adapter."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        query_source = {"sec_filings": "sec", "sec_filing_text": "sec"}.get(source, source)
        narrative_items = self.store.list_retryable_corpus_items(
            limit,
            source=query_source,
            security=security,
            item_id=item_id,
            date_from=date_from,
            date_to=date_to,
            run_id=run_id,
        )
        filing_items = []
        if source in {None, "sec", "sec_filings", "sec_filing_text"}:
            filing_items = self.store.list_retryable_filings(
                limit,
                security=security,
                item_id=item_id,
                date_from=date_from,
                date_to=date_to,
                run_id=run_id,
            )
        work_items = [("narrative", item) for item in narrative_items]
        work_items.extend(("filing", item) for item in filing_items)
        work_items = work_items[:limit]
        source_names = [source] if source else sorted({
            "sec_filings" if kind == "filing" else str(item.get("source") or "repair")
            for kind, item in work_items
        })
        source_names = source_names or ["repair"]
        started_at = self._status_now()
        started_monotonic = time.monotonic()
        run_id_value = str(uuid.uuid4())
        self.store.start_scheduler_run(
            "repair",
            policy_revision=self.coverage.revision,
            config_revision=self.registry.version,
            requested_sources=source_names,
            run_id=run_id_value,
            started_at=started_at,
        )
        completed = failed = skipped = 0
        by_source: dict[str, dict] = {
            name: {"status": "success", "items": 0, "new": 0, "updated": 0, "duplicates": 0}
            for name in source_names
        }
        for kind, item in work_items:
            item_source = (
                "sec_filings"
                if kind == "filing"
                else str(item.get("source") or source or "repair")
            )
            try:
                result = (
                    self.store.repair_filing_index(str(item["accession"]))
                    if kind == "filing"
                    else self.store.repair_corpus_item(str(item["corpus_item_id"]))
                )
            except Exception as exc:  # noqa: BLE001 - repair stays item-isolated
                result = {
                    "status": "failed",
                    "error": safe_message(exc),
                }
            target = by_source.setdefault(item_source, {
                "status": "success", "items": 0, "new": 0, "updated": 0, "duplicates": 0,
            })
            target["items"] += 1
            if result.get("status") == "completed":
                completed += 1
            elif result.get("status") == "skipped":
                skipped += 1
                target["status"] = "partial"
            else:
                failed += 1
                target["status"] = "error"
                target["error"] = result.get("error") or result.get("reason")
                target["error_class"] = ErrorClass.ITEM.value
        results = {name: value for name, value in by_source.items()}
        self._persist_run_status(
            run_id_value,
            "repair",
            list(results),
            results,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        overall = "error" if failed and not completed else "partial" if failed or skipped else "success"
        return {
            "mode": "repair",
            "run_id": run_id_value,
            "status": overall,
            "requested": len(work_items),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "sources": results,
        }

    # -- Explicit retention -------------------------------------------------

    def run_retention(
        self,
        *,
        as_of: Optional[str] = None,
        preview: bool = False,
        apply: bool = False,
        eligible_ids: Optional[list[str]] = None,
    ) -> dict:
        """Preview or explicitly apply the displayed retention target set."""
        if preview and apply:
            raise ValueError("retention accepts either preview or apply, not both")
        started_at = self._status_now()
        started_monotonic = time.monotonic()
        run_id = str(uuid.uuid4())
        self.store.start_scheduler_run(
            "retention",
            policy_revision=self.coverage.revision,
            config_revision=self.registry.version,
            requested_sources=["retention"],
            run_id=run_id,
            started_at=started_at,
        )
        try:
            target = self.store.run_retention(
                as_of=as_of, apply=False, eligible_ids=eligible_ids,
            )
            if apply:
                ids = (
                    list(eligible_ids)
                    if eligible_ids is not None
                    else list(target["document_family_ids"])
                )
                result = self.store.run_retention(
                    as_of=as_of, apply=True, eligible_ids=ids,
                )
                result["preview"] = target
            else:
                result = target
        except Exception as exc:  # noqa: BLE001 - always terminalize the run
            result = {
                "apply": apply,
                "eligible": 0,
                "expired": 0,
                "failed": 1,
                "document_family_ids": [],
                "error": safe_message(exc),
            }
        failed = int(result.get("failed") or 0)
        expired = int(result.get("expired") or 0)
        terminal_status = (
            "success" if failed == 0
            else "partial" if expired else "error"
        )
        source_result = {
            "status": terminal_status,
            "items": int(result.get("eligible") or 0),
            "new": expired,
            "details": result,
        }
        if failed:
            source_result.update({
                "error": result.get("error") or f"{failed} retention item(s) failed",
                "error_class": ErrorClass.ITEM.value,
            })
        self._persist_run_status(
            run_id,
            "retention",
            ["retention"],
            {"retention": source_result},
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        return {
            "mode": "retention",
            "run_id": run_id,
            "status": terminal_status,
            **result,
        }

    # ── Status ─────────────────────────────────────────

    def status_report(
        self,
        source: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> dict:
        """Return freshness/run status for every source."""
        sources: dict[str, dict] = {}
        if scope is not None and scope not in VALID_SCOPES:
            raise ValueError(f"invalid source scope: {scope}")
        selected = (
            [self.registry.get(source)]
            if source is not None
            else list(self.registry.sources.values())
        )
        if scope is not None:
            selected = [spec for spec in selected if spec.scope == scope]
        selected.sort(key=lambda spec: (spec.priority, spec.name))
        for spec in selected:
            name = spec.name
            cache_source = self._scheduler_cache_source(name)
            status = self.store.get_cache_status(self.SCHEDULER_TICKER, cache_source)
            ttl = self._ttl_for(name)
            try:
                coverage = self.coverage.explain(name)
            except ValueError as exc:
                coverage = {"source": name, "enabled": False, "error": str(exc)}
            registry_status = {
                "registry_status": spec.status,
                "registry_reason": spec.disabled_reason,
                "cadence": spec.cadence,
                "run_modes": list(spec.run_modes),
                "scope": spec.scope,
                "cursor_kind": spec.cursor_kind,
            }
            if not spec.is_available:
                sources[name] = {
                    "status": spec.status,
                    "last_run": None,
                    "age_hours": None,
                    "ttl_hours": ttl,
                    "error": spec.disabled_reason,
                    "coverage": coverage,
                    **registry_status,
                }
                continue
            if not status:
                sources[name] = {
                    "status": "never_fetched",
                    "last_run": None,
                    "age_hours": None,
                    "ttl_hours": ttl,
                    "error": None,
                    "coverage": coverage,
                    **registry_status,
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
                "coverage": coverage,
                **registry_status,
            }

        now = self._now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        day_start = now.date().isoformat()
        minute_start = now.strftime("%Y-%m-%dT%H:%MZ")
        for name, source_status in sources.items():
            provider_state = self.store.get_source_cursor_state(
                name, "__provider__"
            ) or {}
            usage = self.store.get_source_budget_usage(
                name,
                day_start=day_start,
                minute_start=minute_start,
            )
            reset_value = usage.get("provider_reset")
            if reset_value is None and provider_state.get("status") == "circuit_open":
                reset_value = provider_state.get("cursor_value")
            reset_at = None
            if reset_value is not None:
                try:
                    reset_at = datetime.fromtimestamp(
                        float(reset_value), tz=timezone.utc
                    )
                except (TypeError, ValueError, OSError, OverflowError):
                    try:
                        reset_at = datetime.fromisoformat(
                            str(reset_value).replace("Z", "+00:00")
                        )
                    except ValueError:
                        reset_at = None
            cooldown = bool(
                usage.get("provider_remaining") == 0
                and reset_at is not None
                and reset_at > now.astimezone(timezone.utc)
            )
            source_status["provider"] = {
                "status": provider_state.get("status"),
                "error_class": normalize_error_class(
                    provider_state.get("error_class")
                ),
                "message": (
                    safe_message(provider_state.get("error_message"))
                    if provider_state.get("error_message")
                    else None
                ),
                "retry_after": provider_state.get("retry_after"),
                "reset_at": str(reset_value) if reset_value is not None else None,
                "circuit_opened_at": (
                    provider_state.get("updated_at")
                    if provider_state.get("status") == "circuit_open"
                    else None
                ),
                "cooldown_active": cooldown,
            }

        # Persisted counters only: status must never invoke an embedding call.
        with self.store.sqlite._connect() as conn:
            filing_index = dict(
                conn.execute(
                    "SELECT "
                    "SUM(CASE WHEN status='index_pending' THEN 1 ELSE 0 END) "
                    "AS pending, "
                    "SUM(CASE WHEN status='index_pending' AND index_error IS NOT NULL "
                    "THEN 1 ELSE 0 END) AS errors, "
                    "COALESCE(SUM(index_section_count), 0) AS sections, "
                    "COALESCE(SUM(index_chunk_count), 0) AS chunks "
                    "FROM filings"
                ).fetchone()
            )
        if "sec_filings" in sources:
            sources["sec_filings"]["filing_text_index"] = {
                "pending": int(filing_index.get("pending") or 0),
                "errors": int(filing_index.get("errors") or 0),
                "sections": int(filing_index.get("sections") or 0),
                "chunks": int(filing_index.get("chunks") or 0),
            }

        try:
            runs = self.store.list_scheduler_runs(limit=20, source=source)
        except Exception as exc:  # noqa: BLE001 - status remains useful
            logger.warning("Could not read scheduler run history: %s", safe_message(exc))
            runs = []
        try:
            coverage_health = build_coverage_health(
                self.store,
                self.coverage,
                self.registry,
                as_of=now,
            )
        except Exception as exc:  # noqa: BLE001 - one health section cannot hide status
            logger.warning("Could not calculate coverage health: %s", safe_message(exc))
            coverage_health = {
                "policy_revision": str(getattr(self.coverage, "revision", "unknown")),
                "config_revision": str(self.registry.version),
                "error": safe_message(exc),
            }

        return {
            "sources": sources,
            "runs": runs,
            "coverage_health": coverage_health,
            "registry_version": self.registry.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Reset ──────────────────────────────────────────

    def reset_schedule(self) -> None:
        """Mark every scheduler source stale so the next run re-runs all."""
        for name in self.SOURCES:
            self.store.upsert_cache_stale(
                self.SCHEDULER_TICKER, self._scheduler_cache_source(name),
            )


def main() -> None:
    """CLI entry point for cron jobs."""
    import argparse

    parser = argparse.ArgumentParser(description="Unified ingestion scheduler")
    parser.add_argument(
        "mode",
        choices=["bootstrap", "daily", "hourly", "weekly", "all", "repair", "retention", "status"],
    )
    parser.add_argument("--force", action="store_true", help="Skip freshness checks")
    parser.add_argument("--source", help="Run or report one registered source")
    parser.add_argument(
        "--scope",
        choices=["universe", "broad", "deep", "sector", "global"],
        help="Run or report sources in one coverage scope",
    )
    parser.add_argument("--since", help="Bootstrap start date (YYYY-MM-DD)")
    parser.add_argument("--resume", action="store_true", help="Resume an unfinished bootstrap run")
    parser.add_argument("--limit", type=int, help="Maximum repair items")
    parser.add_argument("--preview", action="store_true", help="Preview retention candidates")
    parser.add_argument("--apply", action="store_true", help="Apply the displayed retention candidates")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive retention after its preview is displayed",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    scheduler = UnifiedScheduler()
    if args.mode == "bootstrap":
        if args.force or args.scope or args.limit or args.preview or args.apply:
            parser.error("bootstrap accepts only --source, --since, and --resume")
        result = scheduler.run_bootstrap(
            source=args.source,
            since=args.since,
            resume=args.resume,
        )
    elif args.mode == "repair":
        if args.force or args.scope or args.since or args.resume or args.preview or args.apply:
            parser.error("repair accepts only --source and --limit")
        result = scheduler.run_repair(
            source=args.source,
            limit=args.limit if args.limit is not None else 100,
        )
    elif args.mode == "retention":
        if args.source or args.scope or args.force or args.since or args.resume or args.limit:
            parser.error("retention accepts only --preview or --apply")
        if args.preview and args.apply:
            parser.error("retention accepts either --preview or --apply")
        if not args.preview and not args.apply:
            parser.error("retention requires --preview or --apply")
        if args.preview:
            result = scheduler.run_retention(preview=True)
        elif not args.confirm:
            result = {
                "mode": "retention",
                "status": "rejected",
                "error": "destructive retention requires --confirm",
            }
        else:
            preview = scheduler.run_retention(preview=True)
            print(json.dumps({"retention_preview": preview}, indent=2, default=str))
            result = scheduler.run_retention(
                apply=True,
                eligible_ids=preview.get("document_family_ids", []),
            )
    elif args.mode == "status":
        if args.force or args.since or args.resume or args.limit or args.preview or args.apply:
            parser.error("status accepts only --source, --scope, and --json")
        if args.source is None and args.scope is None:
            result = scheduler.status_report()
        else:
            result = scheduler.status_report(source=args.source, scope=args.scope)
    elif args.mode == "all":
        if args.source is None and args.scope is None:
            result = scheduler.run_all_stale(force=args.force)
        else:
            result = scheduler.run_all_stale(
                force=args.force, source=args.source, scope=args.scope
            )
    elif args.mode == "daily":
        if args.source is None and args.scope is None:
            result = scheduler.run_daily(force=args.force)
        else:
            result = scheduler.run_daily(
                force=args.force, source=args.source, scope=args.scope
            )
    elif args.mode == "hourly":
        if args.source is None and args.scope is None:
            result = scheduler.run_hourly(force=args.force)
        else:
            result = scheduler.run_hourly(
                force=args.force, source=args.source, scope=args.scope
            )
    elif args.mode == "weekly":
        if args.source is None and args.scope is None:
            result = scheduler.run_weekly(force=args.force)
        else:
            result = scheduler.run_weekly(
                force=args.force, source=args.source, scope=args.scope
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
