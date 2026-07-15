"""
src/middleware/models.py
Pydantic models for request/response schemas.
"""

import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Absolute hard ceiling for the current raw question (2.2.2.1). The middleware
# never silently truncates it — an over-limit question is an explicit
# validation error (HTTP 422). configs/middleware.yaml mirrors this as
# conversation_max_question_chars; this constant is the enforced pydantic cap.
MAX_QUESTION_CHARS = 16000

# Safe opaque session identifier: bounded length, conservative character set.
# session_id is tracing metadata only — never a server-side lookup key — so
# this only guards against unbounded/pathological values, not against reuse.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ChatTurn(BaseModel):
    """One bounded, client-owned conversation turn (2.2.2.1).

    A turn is a single chat message. ``history`` on :class:`QueryRequest` is a
    flat, chronological list of these. The middleware validates and uses them
    but never persists them; the client (``scripts/chat.py``) owns the memory.
    """

    role: Literal["user", "assistant"] = Field(
        ..., description="Who produced this turn")
    content: str = Field(..., description="Turn text; must be non-blank")
    turn_id: Optional[str] = Field(
        None, description="Opaque per-turn id for tracing only; never a lookup key")
    context: Optional[dict] = Field(
        None,
        description="Structured context carried from a prior response "
                    "(resolved tickers/metrics/timeframe/grounding). Consumed by "
                    "follow-up rewriting (2.2.2.2); retained only on assistant turns.",
    )

    @field_validator("content")
    @classmethod
    def _content_nonblank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("chat turn content must be non-blank")
        return v


class QueryRequest(BaseModel):
    """Incoming user query."""

    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS,
                          description="Natural language financial question")
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="Bounded, client-owned conversation history (2.2.2.1). "
                    "Additive — an omitted/empty list preserves single-turn behavior.",
    )
    session_id: Optional[str] = Field(
        None,
        description="Opaque client-generated session id for tracing only. Not a "
                    "server-side lookup key; bounded/sanitized, never persisted.",
    )
    ticker: Optional[str] = Field(None, description="Optional ticker override")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0,
                                         description="Model temperature override")
    max_tokens: Optional[int] = Field(None, ge=64, le=8192,
                                       description="Max response tokens")
    stream: bool = Field(False, description="Enable streaming response")
    refresh: bool = Field(True, description="Auto-refresh stale data before answering")
    include_sources: bool = Field(True, description="Include source citations")
    answer_policy: Optional[str] = Field(
        None,
        description="Per-request override of the server's answer policy: strict|graded",
    )
    include_evidence_trace: bool = Field(
        False,
        description="Include the exact evidence trace (system/user prompts, usable "
                    "facts/documents, tool results) used to produce the answer. "
                    "Off by default; intended for evaluation requests (2.2.1.2).",
    )

    @field_validator("session_id")
    @classmethod
    def _session_id_safe(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if not _SESSION_ID_RE.match(v):
            raise ValueError(
                "session_id must be 1-128 chars of [A-Za-z0-9._:-]")
        return v


class SourceCitation(BaseModel):
    """A single source citation for a fact used in the answer."""

    source_type: str = Field(..., description="e.g., 'sec_10k', 'yfinance', 'fred'")
    ticker: str
    metric: Optional[str] = None
    value: Optional[float] = None
    period: Optional[str] = None
    source_url: Optional[str] = None
    relevance_score: Optional[float] = None


class EvidenceCitation(BaseModel):
    """A structured citation resolved against the model-visible evidence ledger.

    2.2.4.3: an ``[E#]`` citation resolves to a request-local evidence item
    (``support_status=supported``); an ``[E#]`` absent from the final ledger is
    ``missing``; a botched id (``[E]``, ``[E1x]``) is ``malformed`` — neither is
    ever converted into a real source citation. Legacy ``[Source: type/ticker]``
    labels are retained (``support_status=supported``) during the compatibility
    window with ``evidence_id`` unset.
    """

    evidence_id: Optional[str] = Field(
        None, description="Request-local [E#] id, or None for a legacy source label")
    source_type: Optional[str] = None
    ticker: Optional[str] = None
    metric: Optional[str] = None
    period: Optional[str] = None
    source_url: Optional[str] = None
    support_status: Literal["supported", "missing", "malformed"] = "supported"
    item_type: Optional[str] = None
    event_type: Optional[str] = None
    authority_tier: Optional[str] = None
    source: Optional[str] = None
    date_semantics: Optional[dict] = None
    canonical_security: Optional[str] = None
    coverage_tier: Optional[str] = None
    source_category: Optional[str] = None
    provider: Optional[str] = None
    publisher: Optional[str] = None


class QueryResponse(BaseModel):
    """Structured response from the middleware."""

    answer: str = Field(..., description="Grounded answer from the model")
    citations: list[SourceCitation] = Field(default_factory=list,
                                            description="Sources used in the answer")
    detected_ticker: Optional[str] = None
    detected_intent: Optional[str] = None
    facts_used: int = 0
    documents_used: int = 0
    grounding: Literal["grounded", "partial", "general", "refused"] = Field(
        "refused",
        description="Answer grounding mode used: grounded|partial|general|refused",
    )
    latency_ms: float = 0.0
    timings: Optional[dict] = Field(
        None,
        description="Optional per-stage latency breakdown in milliseconds",
    )
    model_available: bool = True
    retrieval_strategy: Optional[str] = Field(
        None, description="Document retrieval path used: vector|hybrid|hybrid+rerank")
    tools_used: Optional[list[str]] = Field(
        None, description="Names of middleware tools invoked while answering, if any")
    resolved_ticker: Optional[dict] = Field(
        None,
        description="Resolved ticker {'name','source'} when the resolver mapped a "
                    "non-exact company name or typo (omitted for known_ticker/override)",
    )
    evidence_trace: Optional[dict] = Field(
        None,
        description="Exact model-visible evidence trace (2.2.1.2): system/user prompts, "
                    "usable facts/documents, and tool results. Present only when the "
                    "request set include_evidence_trace=true and a model call succeeded "
                    "(never populated for a degraded/model-unavailable answer).",
    )
    conversation: Optional[dict] = Field(
        None,
        description="Conversation-history metadata (2.2.2.1): "
                    "{history_turns_received, history_turns_used, history_truncated, "
                    "topic_reset}. Present only when the request carried history; "
                    "omitted (null) for single-turn requests.",
    )
    retrieval_query: Optional[str] = Field(
        None,
        description="Compiled standalone retrieval query (2.2.2.2). Distinct from the "
                    "raw question — retrieval input only, never the user's wording. "
                    "Present only when follow-up rewriting ran (flag on + history).",
    )
    carried_context: Optional[dict] = Field(
        None,
        description="Follow-up carryover metadata (2.2.2.2): {entities, metrics, "
                    "timeframe, topic_reset, ambiguous_slots, resolution_sources}. "
                    "Only the slots actually carried from history. Present only when "
                    "rewriting ran.",
    )
    resolved_tickers: Optional[list[str]] = Field(
        None,
        description="Effective tickers used for retrieval after carryover (2.2.2.2). "
                    "Explicit signal for conversational eval; present only when "
                    "rewriting ran.",
    )
    resolved_metrics: Optional[list[str]] = Field(
        None,
        description="Effective metrics used for retrieval after carryover (2.2.2.2).",
    )
    resolved_timeframe: Optional[str] = Field(
        None,
        description="Effective timeframe used for retrieval after carryover (2.2.2.2).",
    )
    orchestration: Optional[dict] = Field(
        None,
        description="Bounded adaptive-RAG route metadata (2.2.3.4): {lane, "
                    "reason_codes, subqueries_executed, retrieval_rounds, "
                    "planning_calls, reranker_calls, deterministic_tools, "
                    "context_chars, evidence_dropped, fallback_reason}. These are "
                    "ACTUAL executed counters, not configured maxima. Present only "
                    "when enable_adaptive_rag is on; omitted (null) on the legacy "
                    "path so older clients are unaffected.",
    )
    evidence_sufficiency: Optional[dict] = Field(
        None,
        exclude_if=lambda value: value is None,
        description="Evidence sufficiency/answer-policy metadata (2.2.4.1): "
                    "{status, reason_codes, covered_subqueries, missing_subqueries, "
                    "corrective_action, retry_performed}. Omitted while the feature "
                    "flag is off.",
    )
    evidence_citations: Optional[list[EvidenceCitation]] = Field(
        None,
        exclude_if=lambda value: value is None,
        description="Structured citations resolved against the model-visible "
                    "evidence ledger (2.2.4.3). Present only when answer_validation "
                    "is report|enforce; omitted (null) when validation is off.",
    )
    answer_validation: Optional[dict] = Field(
        None,
        exclude_if=lambda value: value is None,
        description="Deterministic citation/numeric-validation metadata (2.2.4.3): "
                    "{validation_status, citation_support_rate, numeric_claims_supported, "
                    "numeric_claims_unsupported, numeric_claims_ambiguous, "
                    "mismatch_counts, enforcement, ...}. Omitted (null) when "
                    "answer_validation is off; validator errors report "
                    "validation_status=report_unavailable rather than failing the query.",
    )
    graph_trace_id: Optional[str] = Field(
        None,
        exclude_if=lambda value: value is None,
        description="Opaque id of this request's bounded local query-graph trace "
                    "(2.2.7.4). Present only when the local graph observer is enabled; "
                    "omitted (null) otherwise so the response stays byte-compatible for "
                    "clients and servers without the observer. Used by the chat client's "
                    "/graph trace command to deep-link the running trace.",
    )
    freshness: dict = Field(
        default_factory=lambda: {
            "overall": "unknown",
            "refreshed_during_query": [],
            "stale_sources_used": [],
            "fetched_on_miss": [],
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
    capabilities: Optional[dict] = Field(
        None,
        description="Active deployment capabilities and effective limits. Always "
                    "includes {'tools','streaming','answer_policy'}. 'streaming' is the "
                    "EFFECTIVE capability (2.2.6.1): whether /query/stream can be served "
                    "at all — false when tools are enabled but tool-final streaming is "
                    "off. 'streaming_tool_final' is true only when a tools-enabled request "
                    "streams its final synthesis after bounded non-streaming tool rounds. "
                    "On servers that support conversation memory (2.2.2) it also advertises "
                    "{'history','multiline','max_question_chars','conversation_max_turns',"
                    "'conversation_max_history_chars'} so clients can read the real "
                    "limits instead of guessing. Older servers omit the extra keys and "
                    "clients fall back to local defaults.",
    )
    version: str = "1.0.0"


class GraphTraceSummary(BaseModel):
    """Read-only summary of one bounded in-memory query graph trace."""

    schema_version: int = 1
    query_id: str
    created_at: float
    updated_at: float
    complete: bool = False
    node_count: int = 0
    edge_count: int = 0
    question_preview: str = ""
    question_digest: Optional[str] = None


class GraphTraceSnapshot(GraphTraceSummary):
    """Current graph nodes and edges reconstructed from stored deltas."""

    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    last_sequence: int = -1


class GraphHealthResponse(BaseModel):
    """Bounded TraceHub state exposed by the local read-only health endpoint."""

    enabled: bool
    trace_count: int = 0
    element_count: int = 0
    subscriber_count: int = 0
    limits: dict = Field(default_factory=dict)
    oldest_sequence: Optional[int] = None
    newest_sequence: Optional[int] = None
    dropped_events: int = 0
    reset_count: int = 0


class CorpusGraphResponse(BaseModel):
    """Common bounded wire shape for read-only corpus graph pages."""

    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    next_cursor: Optional[str] = None
    truncated: bool = False
    corpus_revision: int = 0
    refresh: Optional[dict] = None
    aggregates: Optional[dict] = None
    # 2.3.5.3 aggregation-first surfaces: applied filter set, paged group level,
    # facet counts, and visible/total counts for a bounded projection.
    applied_filters: Optional[dict] = None
    group_by: Optional[str] = None
    facets: Optional[dict] = None
    visible_count: Optional[int] = None
    total_count: Optional[int] = None
    total_items: Optional[int] = None


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
