"""
src/middleware/tools/data_tools.py
Analytical tools over middleware data sources.
"""

import logging
import os
from datetime import datetime, timezone

from . import sanity
from .base import Tool, register

logger = logging.getLogger(__name__)

_TOOL_REGISTRY_SOURCES = {
    "yfinance_fundamentals": "yfinance",
    "yfinance_news": "yfinance",
    "finnhub_news": "finnhub",
    "massive_market": "massive",
    "gdelt_news": "gdelt",
    "estimates": "estimates",
    "sec_filings": "sec_filings",
    "earnings_transcripts": "earnings_transcripts",
    "ir_pages": "ir_pages",
}


def _provider_reset_at(value: object) -> datetime | None:
    """Parse persisted provider reset values used by scheduler budgets."""
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromtimestamp(float(str(value)), tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _refresh_tool_preflight(store, logical: str) -> tuple[bool, str | None, str | None]:
    """Check registry availability, durable budgets, and provider cooldowns."""
    try:
        from src.scheduler.source_registry import SourceRegistry

        registry = SourceRegistry.load(environ=os.environ)
        registry_name = _TOOL_REGISTRY_SOURCES.get(logical)
        spec = registry.sources.get(registry_name) if registry_name else None
        if spec is None:
            return False, "invalid_source", registry_name
        if not spec.is_available:
            return False, spec.status, registry_name

        now = datetime.now(timezone.utc)
        day_start = now.date().isoformat()
        minute_start = now.strftime("%Y-%m-%dT%H:%MZ")
        usage = store.get_source_budget_usage(
            registry_name,
            day_start=day_start,
            minute_start=minute_start,
        )
        provider_state = store.get_source_cursor_state(
            registry_name, "__provider__",
        ) or {}
        state_reset = _provider_reset_at(provider_state.get("cursor_value"))
        if (
            provider_state.get("status") == "circuit_open"
            and (state_reset is None or state_reset > now)
        ):
            return False, "provider_cooldown", registry_name
        if int(usage.get("day_requests") or 0) >= spec.requests_per_day:
            return False, "requests_per_day", registry_name
        if int(usage.get("minute_requests") or 0) >= spec.requests_per_minute:
            return False, "requests_per_minute", registry_name
        remaining = usage.get("provider_remaining")
        usage_reset = _provider_reset_at(usage.get("provider_reset"))
        if (
            remaining is not None
            and int(remaining) <= 0
            and (usage_reset is None or usage_reset > now)
        ):
            return False, "provider_cooldown", registry_name
        return True, None, registry_name
    except Exception as exc:  # noqa: BLE001 - a guard failure must fail closed
        logger.warning("Refresh preflight failed for %s: %s", logical, exc)
        return False, "preflight_error", None


QUARTER_ESTIMATE_METRICS = [
    "estimate_revenue_current_q",
    "estimate_revenue_next_q",
    "estimate_eps_current_q",
    "estimate_eps_next_q",
]

YEAR_ESTIMATE_METRICS = [
    "estimate_revenue_current_y",
    "estimate_revenue_next_y",
    "estimate_eps_current_y",
    "estimate_eps_next_y",
]

ESTIMATE_METRICS = QUARTER_ESTIMATE_METRICS + YEAR_ESTIMATE_METRICS

PRICE_TARGET_METRICS = [
    "price_target_mean",
    "price_target_high",
    "price_target_low",
    "num_analysts",
    "recommendation_mean",
]


def list_metrics_handler(store, ticker=None):
    """Return the metrics and tickers available in `fundamentals`, so the model never invents one."""
    return {
        "metrics": store.sqlite.list_metrics(ticker.upper() if ticker else None),
        "tickers": store.sqlite.list_tickers(),
    }


def query_facts_handler(store, metric, tickers=None, order="asc", limit=10,
                        op=None, value=None, latest_only=True):
    """Rank, filter, or threshold stocks by a fundamental metric, with data-sanity exclusions applied."""
    known = store.sqlite.list_metrics()
    if metric not in known:
        return {"error": f"unknown metric '{metric}'", "available_metrics": known}
    if (op is None) != (value is None):
        return {
            "error": "op and value must be provided together for a threshold filter",
            "metric": metric,
        }

    rows = store.sqlite.query_metric(
        metric,
        tickers=tickers,
        order=order,
        limit=limit,
        op=op,
        value=value,
        latest_only=latest_only,
        exclude=sanity.excluded_symbols(metric),
        sane_range=sanity.sane_range(metric),
    )
    return {"metric": metric, "results": rows}


def get_fundamentals_handler(store, ticker, metrics=None):
    """Return targeted fundamentals for one ticker."""
    ticker = ticker.upper()
    try:
        return {
            "ticker": ticker,
            "fundamentals": store.get_fundamentals_batch(ticker, metrics),
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("get_fundamentals failed for %s", ticker)
        return {"error": str(e), "ticker": ticker}


def search_documents_handler(store, query, ticker=None, n_results=5):
    """Return compact qualitative document chunks."""
    ticker_out = ticker.upper() if ticker else None
    try:
        n_results = max(1, min(int(n_results), 10))
        result = store.search(query=query, n_results=n_results, ticker=ticker)
        documents = []
        for doc in result.get("documents", []):
            documents.append({
                "id": doc.get("id"),
                "text": (doc.get("document") or "")[:1500],
                "metadata": doc.get("metadata", {}),
            })
        return {"documents": documents}
    except Exception as e:  # noqa: BLE001
        logger.exception("search_documents failed for %s", ticker_out)
        return {"error": str(e), "ticker": ticker_out}


def get_macro_snapshot_handler(store):
    """Return cached macro-economic indicators."""
    try:
        from src.middleware.app import _macro_snapshot_data

        return {"macro": _macro_snapshot_data()}
    except Exception as e:  # noqa: BLE001
        logger.exception("get_macro_snapshot failed")
        return {"error": str(e), "ticker": "MACRO"}


def get_sentiment_handler(store, ticker, days=7):
    """Return cached GDELT sentiment summary for one ticker."""
    ticker = ticker.upper()
    try:
        from src.middleware.app import _sentiment_data

        days = max(1, min(int(days), 90))
        return _sentiment_data(ticker, days=days)
    except Exception as e:  # noqa: BLE001
        logger.exception("get_sentiment failed for %s", ticker)
        return {"error": str(e), "ticker": ticker}


def get_guidance_handler(store, ticker):
    """Return latest extracted earnings guidance for one ticker."""
    ticker = ticker.upper()
    try:
        from src.middleware.app import _guidance_data

        result = _guidance_data(ticker)
        if not result:
            return {"ticker": ticker, "guidance": {}, "status": "not_found"}
        return {"ticker": ticker, "guidance": result, "status": "found"}
    except Exception as e:  # noqa: BLE001
        logger.exception("get_guidance failed for %s", ticker)
        return {"error": str(e), "ticker": ticker}


def _estimate_metrics_for_horizon(horizon):
    if horizon is None:
        return ESTIMATE_METRICS
    horizon = str(horizon).lower()
    if horizon == "quarter":
        return QUARTER_ESTIMATE_METRICS
    if horizon == "year":
        return YEAR_ESTIMATE_METRICS
    raise ValueError("horizon must be one of: quarter, year")


def _estimate_fact(row):
    if not row or row.get("source_type") != "estimates":
        return None
    if row.get("value") is None or not row.get("period"):
        return None
    return {"value": row.get("value"), "period": row.get("period")}


def _growth_vs_realized(estimate_value, realized_value):
    if estimate_value is None or realized_value in (None, 0):
        return None
    return round((float(estimate_value) - float(realized_value)) / float(realized_value), 10)


def get_estimates_handler(store, ticker, horizon=None):
    """Return forward-looking analyst estimate facts for one ticker."""
    ticker = ticker.upper()
    try:
        normalized_horizon = None if horizon is None else str(horizon).lower()
        metrics = _estimate_metrics_for_horizon(normalized_horizon)
        estimates = {}
        for metric in metrics:
            fact = _estimate_fact(store.get_fundamental(ticker, metric))
            if fact:
                estimates[metric] = fact

        growth = {}
        realized_revenue = store.get_fundamental(ticker, "total_revenue")
        realized_eps = store.get_fundamental(ticker, "eps_diluted")
        realized_revenue_value = realized_revenue.get("value") if realized_revenue else None
        realized_eps_value = realized_eps.get("value") if realized_eps else None

        for metric, fact in estimates.items():
            realized_value = (
                realized_revenue_value
                if metric.startswith("estimate_revenue_")
                else realized_eps_value
            )
            metric_growth = _growth_vs_realized(fact["value"], realized_value)
            if metric_growth is not None:
                growth[metric] = metric_growth

        return {
            "ticker": ticker,
            "horizon": normalized_horizon,
            "estimates": estimates,
            "growth_vs_realized": growth,
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("get_estimates failed for %s", ticker)
        return {"error": str(e), "ticker": ticker}


def get_price_targets_handler(store, ticker):
    """Return forward-looking analyst price target facts for one ticker."""
    ticker = ticker.upper()
    try:
        price_targets = {}
        for metric in PRICE_TARGET_METRICS:
            fact = _estimate_fact(store.get_fundamental(ticker, metric))
            if fact:
                price_targets[metric] = fact
        return {"ticker": ticker, "price_targets": price_targets}
    except Exception as e:  # noqa: BLE001
        logger.exception("get_price_targets failed for %s", ticker)
        return {"error": str(e), "ticker": ticker}


def check_freshness_handler(store, ticker):
    """Return freshness status for one ticker."""
    ticker = ticker.upper()
    try:
        return store.get_freshness_report(ticker)
    except Exception as e:  # noqa: BLE001
        logger.exception("check_freshness failed for %s", ticker)
        return {"error": str(e), "ticker": ticker}


def refresh_data_handler(store, ticker, sources=None):
    """Refresh stale data for one ticker through the middleware refresh path.

    Only truly per-ticker sources are refreshable from the model: sources in
    SCHEDULER_SOURCE_MAP (SEC filings, earnings transcripts, IR pages) run
    watchlist-wide via the UnifiedScheduler, so the tool skips them and
    reports them instead of triggering a scheduler-scale ingestion.
    """
    ticker = str(ticker or "").strip().upper()
    try:
        requested = [str(value).strip().lower() for value in (sources or [])]
        if any(value in {"all", "*", "broad", "universe"} for value in requested):
            return {
                "error": "refresh request is unbounded; name bounded per-security sources",
                "ticker": ticker,
                "valid_sources": sorted(_TOOL_REGISTRY_SOURCES),
            }
        from src.middleware import app as middleware_app
        from src.middleware.app import (
            SCHEDULER_SOURCE_MAP,
            _SOURCE_ALIASES,
            _normalize_sources,
            _refresh_ticker_sources,
            _stale_source_names,
        )
        import time

        if not ticker or len(ticker) > 16:
            return {"error": "ticker must be a bounded security symbol", "ticker": ticker}

        if requested:
            logical = _normalize_sources(requested)
            if not logical:
                return {
                    "error": "no valid sources requested",
                    "ticker": ticker,
                    "valid_sources": sorted(set(_SOURCE_ALIASES.values())),
                }
        else:
            report = middleware_app.store.get_freshness_report(ticker)
            logical = _stale_source_names(report)

        if len(logical) > 8:
            return {
                "error": "refresh request is unbounded; request at most 8 sources",
                "ticker": ticker,
                "valid_sources": sorted(set(_SOURCE_ALIASES.values())),
            }

        skipped = [s for s in logical if s in SCHEDULER_SOURCE_MAP]
        logical = [s for s in logical if s not in SCHEDULER_SOURCE_MAP]
        if not logical:
            note = (
                "requested sources are scheduler-managed; run the scheduler instead"
                if skipped else "all sources fresh"
            )
            return {
                "ticker": ticker,
                "refreshed": [],
                "errors": [],
                "skipped_scheduler_managed": skipped,
                "note": note,
            }
        allowed: list[str] = []
        guard_errors: list[str] = []
        for logical_name in logical:
            ok, reason, _registry_name = _refresh_tool_preflight(store, logical_name)
            if ok:
                allowed.append(logical_name)
            else:
                guard_errors.append(f"{logical_name}: {reason or 'refresh rejected'}")
        logical = allowed
        if not logical:
            return {
                "ticker": ticker,
                "refreshed": [],
                "errors": guard_errors,
                "skipped_scheduler_managed": skipped,
                "note": "all requested refreshes were rejected by registry or operational controls",
            }
        logger.warning("REFRESH tool invoked: %s sources=%s", ticker, logical)
        start = time.time()
        refreshed, errors = _refresh_ticker_sources(ticker, logical)
        return {
            "ticker": ticker,
            "refreshed": refreshed,
            "errors": guard_errors + errors,
            "skipped_scheduler_managed": skipped,
            "duration_s": round(time.time() - start, 2),
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("refresh_data failed for %s", ticker)
        return {"error": str(e), "ticker": ticker}


get_fundamentals = get_fundamentals_handler
search_documents = search_documents_handler
get_macro_snapshot = get_macro_snapshot_handler
get_sentiment = get_sentiment_handler
get_guidance = get_guidance_handler
get_estimates = get_estimates_handler
get_price_targets = get_price_targets_handler
check_freshness = check_freshness_handler
refresh_data = refresh_data_handler


register(
    Tool(
        name="list_metrics",
        description=(
            "List the fundamental metric names and tickers actually available in the database. "
            "Call this first if you are unsure whether a metric name (e.g. 'forward_pe', "
            "'revenue_ttm') or ticker exists before calling query_facts, so you never guess "
            "an invalid metric."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Optional ticker to scope the metric list to (e.g. 'NVDA').",
                },
            },
            "required": [],
        },
        handler=list_metrics_handler,
        write=False,
    )
)

register(
    Tool(
        name="get_fundamentals",
        description=(
            "Use for a targeted lookup of one ticker's fundamental metrics. "
            "For ranking, filtering, or comparison across tickers, use query_facts instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker to retrieve fundamentals for, e.g. 'NVDA'.",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional metric names to retrieve.",
                },
            },
            "required": ["ticker"],
        },
        handler=get_fundamentals_handler,
        write=False,
    )
)

register(
    Tool(
        name="search_documents",
        description=(
            "Use for qualitative context from filings, news, or transcripts. "
            "Do not use for structured numeric facts; use get_fundamentals or query_facts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language document search query.",
                },
                "ticker": {
                    "type": "string",
                    "description": "Optional ticker filter, e.g. 'NVDA'.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of document chunks to return, clamped to 1-10.",
                },
            },
            "required": ["query"],
        },
        handler=search_documents_handler,
        write=False,
    )
)

register(
    Tool(
        name="get_macro_snapshot",
        description=(
            "Use when a question needs current cached macro indicators such as GDP, CPI, "
            "Fed funds, unemployment, Treasury yield, or yield-curve spread."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=get_macro_snapshot_handler,
        write=False,
    )
)

register(
    Tool(
        name="get_sentiment",
        description=(
            "Use for recent GDELT news sentiment for one ticker when tone, article count, "
            "or positive/negative news balance matters."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker to summarize sentiment for, e.g. 'NVDA'.",
                },
                "days": {
                    "type": "integer",
                    "description": "Lookback window in days, clamped to 1-90.",
                },
            },
            "required": ["ticker"],
        },
        handler=get_sentiment_handler,
        write=False,
    )
)

register(
    Tool(
        name="get_guidance",
        description=(
            "Use when the answer needs latest extracted earnings-call guidance for one ticker."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker to retrieve latest guidance for, e.g. 'NVDA'.",
                },
            },
            "required": ["ticker"],
        },
        handler=get_guidance_handler,
        write=False,
    )
)

register(
    Tool(
        name="get_estimates",
        description=(
            "Use for forward-looking analyst consensus on next-quarter or next-year "
            "expectations for one ticker, including revenue and EPS estimates. "
            "Values are estimates, not realized results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker to retrieve estimates for, e.g. 'NVDA'.",
                },
                "horizon": {
                    "type": "string",
                    "enum": ["quarter", "year"],
                    "description": "Optional estimate horizon to retrieve.",
                },
            },
            "required": ["ticker"],
        },
        handler=get_estimates_handler,
        write=False,
    )
)

register(
    Tool(
        name="get_price_targets",
        description=(
            "Use for forward-looking analyst consensus price targets for one ticker, "
            "including mean, high, low, analyst count, and recommendation mean. "
            "Use for price targets; values are estimates, not realized results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker to retrieve price targets for, e.g. 'NVDA'.",
                },
            },
            "required": ["ticker"],
        },
        handler=get_price_targets_handler,
        write=False,
    )
)

register(
    Tool(
        name="check_freshness",
        description=(
            "Use before answering time-sensitive questions to verify data recency for one "
            "ticker and identify stale or never-fetched sources."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker to check freshness for, e.g. 'NVDA'.",
                },
            },
            "required": ["ticker"],
        },
        handler=check_freshness_handler,
        write=False,
    )
)

register(
    Tool(
        name="query_facts",
        description=(
            "Rank, filter, or threshold tracked stocks by a fundamental metric — e.g. 'which "
            "stock has the lowest forward P/E' or 'which stocks have revenue growth above 20%'. "
            "Use this instead of guessing from memory for any question that requires sorting, "
            "comparing, or aggregating across tickers. Call list_metrics first if unsure which "
            "metric names are valid. Non-equity symbols (ETFs/bonds) and implausible values are "
            "automatically excluded for equity-only metrics like P/E ratios."
        ),
        parameters={
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Metric name to query, e.g. 'forward_pe', 'revenue_growth'.",
                },
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of tickers to restrict the query to.",
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort direction — 'asc' for lowest first, 'desc' for highest first.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results to return (default 10, max 100).",
                },
                "op": {
                    "type": "string",
                    "enum": ["lt", "lte", "gt", "gte", "eq", "ne"],
                    "description": "Optional comparison operator to threshold-filter by `value`.",
                },
                "value": {
                    "type": "number",
                    "description": "Threshold value used together with `op`.",
                },
                "latest_only": {
                    "type": "boolean",
                    "description": "Use each ticker's most recent reporting period only (default true).",
                },
            },
            "required": ["metric"],
        },
        handler=query_facts_handler,
        write=False,
    )
)

register(
    Tool(
        name="refresh_data",
        description=(
            "Use ONLY when check_freshness shows stale or never_fetched sources and "
            "the answer needs current data. This is slow because it performs network "
            "fetches, is rate-limited per query, and refreshes ONE ticker only. "
            "Scheduler-managed sources (sec_filings, earnings_transcripts, ir_pages) "
            "cannot be refreshed from here and are reported as skipped."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker to refresh, e.g. 'NVDA'.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional logical source names, e.g. yfinance_fundamentals "
                        "or gdelt_news. Invalid names are dropped; omit to refresh "
                        "all stale sources."
                    ),
                },
            },
            "required": ["ticker"],
        },
        handler=refresh_data_handler,
        write=True,
    )
)
