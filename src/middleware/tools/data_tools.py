"""
src/middleware/tools/data_tools.py
Read-only analytical tools (`list_metrics`, `query_facts`) over the `fundamentals` table.
"""

import logging

from . import sanity
from .base import Tool, register

logger = logging.getLogger(__name__)


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


def check_freshness_handler(store, ticker):
    """Return freshness status for one ticker."""
    ticker = ticker.upper()
    try:
        return store.get_freshness_report(ticker)
    except Exception as e:  # noqa: BLE001
        logger.exception("check_freshness failed for %s", ticker)
        return {"error": str(e), "ticker": ticker}


get_fundamentals = get_fundamentals_handler
search_documents = search_documents_handler
get_macro_snapshot = get_macro_snapshot_handler
get_sentiment = get_sentiment_handler
get_guidance = get_guidance_handler
check_freshness = check_freshness_handler


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
