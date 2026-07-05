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
