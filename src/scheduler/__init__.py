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

    # ── Config ─────────────────────────────────────────

    def _load_ttls(self) -> dict:
        """Return the registry-owned TTL map keyed for legacy consumers."""
        return {
            spec.ttl_key: spec.ttl_hours
            for spec in self.registry.sources.values()
            if spec.status != "invalid_configuration"
        }

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

    def _budgeted_http_get(self, name: str):
        """Return a budgeted, retrying HTTP callable with one source circuit."""
        import requests
        from src.utils.resilience import provider_request_policy

        budgeted_get = self._active_budgets[name].wrap_http_get(requests.get)
        policy = self._provider_policies.get(name)
        if policy is None:
            policy = provider_request_policy(
                name,
                self.SOURCES[name].retry_policy,
                now_fn=self._now_fn,
            )
            self._provider_policies[name] = policy

        def request(*args: object, **kwargs: object) -> object:
            return policy.request(lambda: budgeted_get(*args, **kwargs))

        return request

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
            return FinnhubIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
                http_get=self._budgeted_http_get(name),
                overlap_hours=int((self.SOURCES[name].overlap or "0h")[:-1]),
            ).ingest_news(tickers=tickers)

        if name == "massive":
            from src.ingestion.massive_ingestor import MassiveIngestor

            return MassiveIngestor(
                store=self.store,
                coverage_resolver=self.coverage,
                http_get=self._budgeted_http_get(name),
                overlap_days=int((self.SOURCES[name].overlap or "0d")[:-1]),
            ).ingest_all()

        official = self._official_ingestor(name)
        if official is not None:
            return official(
                store=self.store,
                coverage_resolver=self.coverage,
                http_get=self._budgeted_http_get(name),
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

    def _run_universe_source(self, name: str) -> dict:
        """Fetch and atomically reconcile one configured universe snapshot."""
        from src.universe.providers import (
            IVVHoldingsProvider,
            Nasdaq100Provider,
            SECCompanyTickersProvider,
        )
        from src.universe.registry import UniverseRegistry

        if name == "universe_nasdaq100":
            provider = Nasdaq100Provider()
        elif name == "universe_ivv":
            provider = IVVHoldingsProvider()
        else:
            provider = SECCompanyTickersProvider(
                user_agent=os.environ.get("SEC_EDGAR_USER_AGENT")
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
        return self._run_sources(
            [spec.name for spec in specs],
            force=force,
            deep_sec=mode == "weekly",
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
                    "COALESCE(SUM(index_section_count), 0) AS sections, "
                    "COALESCE(SUM(index_chunk_count), 0) AS chunks "
                    "FROM filings"
                ).fetchone()
            )
        if "sec_filings" in sources:
            sources["sec_filings"]["filing_text_index"] = {
                "pending": int(filing_index.get("pending") or 0),
                "sections": int(filing_index.get("sections") or 0),
                "chunks": int(filing_index.get("chunks") or 0),
            }

        return {
            "sources": sources,
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
    parser.add_argument("mode", choices=["daily", "hourly", "weekly", "all", "status"])
    parser.add_argument("--force", action="store_true", help="Skip freshness checks")
    parser.add_argument("--source", help="Run or report one registered source")
    parser.add_argument(
        "--scope",
        choices=["universe", "broad", "deep", "sector", "global"],
        help="Run or report sources in one coverage scope",
    )
    args = parser.parse_args()

    scheduler = UnifiedScheduler()
    if args.mode == "status":
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
