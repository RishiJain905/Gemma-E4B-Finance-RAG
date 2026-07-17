"""src/middleware/tools/coverage_tools.py
Read-only capability inventory and corpus-coverage tool.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import Tool, register

logger = logging.getLogger(__name__)

OPERATIONS = (
    "summary",
    "list_securities",
    "contains_security",
    "security_sources",
    "list_sources",
    "list_item_types",
    "list_metrics",
)


def _unavailable_result() -> dict:
    """Return an honest deterministic result when inventory reads fail."""
    return {
        "status": "unavailable",
        "coverage_basis": "unavailable",
        "total_securities": 0,
        "active_securities": 0,
        "coverage_tiers": {"broad": 0, "deep": 0, "sector": 0},
        "securities": [],
        "source_categories": [],
        "sources": [],
        "item_types": [],
        "item_type_details": [],
        "metrics": [],
        "metric_details": [],
        "filters_applied": {},
        "result_count": 0,
        "total_matching": 0,
        "complete": False,
        "next_cursor": None,
        "data_revision": None,
        "universe_snapshot_at": None,
        "answer_origin": "deterministic_coverage",
        "message": "Coverage inventory is unavailable.",
    }


def describe_coverage_handler(
    store,
    operation: str = "summary",
    ticker: Optional[str] = None,
    filters: Optional[dict] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    ticker_only: bool = False,
    **filter_args,
) -> dict:
    """Dispatch one bounded Store coverage read without refresh or model access."""
    merged_filters = dict(filters or {})
    for key, value in filter_args.items():
        if value not in (None, "", [], ()):
            merged_filters[key] = value
    try:
        result = store.describe_coverage(
            operation=operation,
            ticker=ticker,
            filters=merged_filters,
            limit=limit,
            cursor=cursor,
            ticker_only=ticker_only,
        )
        if not isinstance(result, dict):
            return _unavailable_result()
        return result
    except Exception:  # noqa: BLE001 - coverage must fail closed, never guess
        logger.exception("describe_coverage failed")
        return _unavailable_result()


describe_coverage = describe_coverage_handler


register(
    Tool(
        name="describe_coverage",
        description=(
            "Read the authoritative security registry and corpus metadata to describe "
            "membership, stored evidence, source capability, item types, and metric "
            "families. This tool is read-only, bounded, network-free, and never "
            "triggers refresh or ingestion. Use it for questions about what the "
            "system knows or covers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": list(OPERATIONS),
                    "description": "Coverage inventory operation.",
                },
                "ticker": {
                    "type": "string",
                    "description": "Optional ticker or security identifier.",
                },
                "filters": {
                    "type": "object",
                    "description": (
                        "Optional filters: index, sector, industry, coverage_tier, "
                        "item_type, source, source_category, active, as_of, date_from, "
                        "or date_to."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Bounded page size; ticker-only inventory allows up to 1000.",
                },
                "cursor": {
                    "type": "string",
                    "description": "Opaque cursor returned by a previous bounded page.",
                },
                "ticker_only": {
                    "type": "boolean",
                    "description": "Request the complete sorted ticker view when safe.",
                },
            },
            "required": ["operation"],
        },
        handler=describe_coverage_handler,
        write=False,
    )
)
