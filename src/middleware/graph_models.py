"""src/middleware/graph_models.py
Focused wire schemas for the read-only live-trace and corpus-explorer graph API.

The concrete models live in :mod:`src.middleware.models`; this module re-exports
the graph-facing subset and adds the 2.3.5.3 aggregation-first request contracts
so the graph router imports a small, stable surface instead of the whole model
module.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .models import (
    CorpusGraphResponse,
    GraphHealthResponse,
    GraphTraceSnapshot,
    GraphTraceSummary,
)

__all__ = [
    "CorpusGraphResponse",
    "GraphHealthResponse",
    "GraphTraceSnapshot",
    "GraphTraceSummary",
    "CorpusFacetFilters",
    "CORPUS_FACET_QUERY_KEYS",
]

# The additive facet filter keys the corpus explorer accepts on search, groups,
# and facets. Kept in one place so the router and the projector agree.
CORPUS_FACET_QUERY_KEYS: tuple[str, ...] = (
    "source_category", "source", "item_type", "event_type", "security",
    "sector", "industry", "index", "coverage_tier", "year", "month",
    "indexing_state",
)


class CorpusFacetFilters(BaseModel):
    """Bounded, optional facet filter set shared by the aggregation endpoints."""

    source_category: Optional[str] = Field(None, max_length=64)
    source: Optional[str] = Field(None, max_length=64)
    item_type: Optional[str] = Field(None, max_length=64)
    event_type: Optional[str] = Field(None, max_length=64)
    security: Optional[str] = Field(None, max_length=64)
    sector: Optional[str] = Field(None, max_length=64)
    industry: Optional[str] = Field(None, max_length=64)
    index: Optional[str] = Field(None, max_length=32)
    coverage_tier: Optional[str] = Field(None, max_length=32)
    year: Optional[str] = Field(None, max_length=8)
    month: Optional[str] = Field(None, max_length=8)
    indexing_state: Optional[str] = Field(None, max_length=32)

    def as_filters(self) -> dict:
        """Return only the set (non-empty) filters as a plain dict."""
        return {
            key: value
            for key, value in self.model_dump().items()
            if value not in (None, "")
        }
