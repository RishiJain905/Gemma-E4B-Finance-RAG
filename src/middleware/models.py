"""
src/middleware/models.py
Pydantic models for request/response schemas.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming user query."""

    question: str = Field(..., min_length=1, max_length=2000,
                          description="Natural language financial question")
    ticker: Optional[str] = Field(None, description="Optional ticker override")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0,
                                         description="Model temperature override")
    max_tokens: Optional[int] = Field(None, ge=64, le=8192,
                                       description="Max response tokens")
    stream: bool = Field(False, description="Enable streaming response")
    refresh: bool = Field(True, description="Auto-refresh stale data before answering")
    include_sources: bool = Field(True, description="Include source citations")


class SourceCitation(BaseModel):
    """A single source citation for a fact used in the answer."""

    source_type: str = Field(..., description="e.g., 'sec_10k', 'yfinance', 'fred'")
    ticker: str
    metric: Optional[str] = None
    value: Optional[float] = None
    period: Optional[str] = None
    source_url: Optional[str] = None
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Structured response from the middleware."""

    answer: str = Field(..., description="Grounded answer from the model")
    citations: list[SourceCitation] = Field(default_factory=list,
                                            description="Sources used in the answer")
    detected_ticker: Optional[str] = None
    detected_intent: Optional[str] = None
    facts_used: int = 0
    documents_used: int = 0
    latency_ms: float = 0.0
    model_available: bool = True
    freshness: dict = Field(
        default_factory=lambda: {
            "overall": "unknown",
            "refreshed_during_query": [],
            "stale_sources_used": [],
            "warning": None,
        },
        description="Freshness metadata for the data used in the answer",
    )
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    storage: Optional[dict] = None
    model_available: bool = False
    scheduler: Optional[dict] = None
    freshness: Optional[dict] = None
    version: str = "1.0.0"


class SearchRequest(BaseModel):
    """Raw search request (bypasses model, returns retrieved data only)."""

    query: str = Field(..., min_length=1)
    ticker: Optional[str] = None
    n_results: int = Field(5, ge=1, le=20)


class SearchResponse(BaseModel):
    """Raw search results from both stores."""

    documents: list[dict] = Field(default_factory=list)
    facts: list[dict] = Field(default_factory=list)
    ticker: Optional[str] = None


class MacroSnapshotResponse(BaseModel):
    """Snapshot of key macro-economic indicators."""
    gdp: Optional[float] = None
    inflation_cpi: Optional[float] = None
    fed_rate: Optional[float] = None
    unemployment: Optional[float] = None
    ten_year_treasury: Optional[float] = None
    ten_two_spread: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class SentimentResponse(BaseModel):
    """Sentiment summary for a ticker."""
    ticker: str
    average_tone: Optional[float] = None
    article_count: int = 0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0


class FreshnessResponse(BaseModel):
    """Freshness report for a ticker across all data sources."""
    ticker: str
    overall: str = "unknown"
    sources: dict = Field(default_factory=dict)
    stale_sources: list[str] = Field(default_factory=list)


class RefreshRequest(BaseModel):
    """On-demand refresh request body."""
    sources: Optional[list[str]] = Field(
        None, description="Sources to refresh (default: all stale sources)")


class RefreshResponse(BaseModel):
    """Result of an on-demand refresh."""
    ticker: str
    refreshed: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
