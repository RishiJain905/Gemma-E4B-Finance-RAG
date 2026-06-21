"""
src/middleware/app.py
FastAPI application — the main entry point for the middleware layer.

Endpoints:
  GET  /health    — Health check (storage + model)
  POST /query     — Ask a financial question (full pipeline)
  POST /search    — Raw hybrid search (bypasses model)

Usage:
    uvicorn src.middleware.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

from src.storage.store import Store
from .config import MiddlewareConfig
from .models import (
    FreshnessResponse,
    HealthResponse,
    MacroSnapshotResponse,
    QueryRequest,
    QueryResponse,
    RefreshRequest,
    RefreshResponse,
    SearchRequest,
    SearchResponse,
    SentimentResponse,
    SourceCitation,
)

logger = logging.getLogger(__name__)

# ── Global state (set during lifespan) ─────────────────

config: Optional[MiddlewareConfig] = None
store: Optional[Store] = None
model_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global config, store, model_client

    logger.info("Starting middleware...")
    from src.utils.env import load_env
    load_env()  # load .env credentials before initializing components
    config = MiddlewareConfig()
    store = Store(
        embedding_endpoint=config.embedding_endpoint,
    )
    model_client = httpx.AsyncClient(timeout=60)

    yield  # App runs here

    # Shutdown
    if model_client:
        await model_client.aclose()
    logger.info("Middleware shut down.")


app = FastAPI(
    title="Gemma-E4B-Finance-RAG Middleware",
    description="Hybrid RAG query router — retrieves facts + documents, "
                "augments prompts, and returns grounded answers.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Enhanced health check with storage, model, scheduler, and freshness."""
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    storage_health = store.heartbeat()
    model_ok = await _check_model_health()

    # Scheduler status (best-effort).
    scheduler_status = None
    try:
        from src.scheduler import UnifiedScheduler
        sched = UnifiedScheduler(store=store)
        scheduler_status = sched.status_report()
    except Exception as e:  # noqa: BLE001
        logger.warning("Scheduler status unavailable: %s", e)

    # Per-ticker freshness summary (best-effort).
    freshness_summary: dict[str, str] = {}
    for ticker in ["NVDA", "AMD", "AAPL", "MSFT", "META", "CRWD"]:
        try:
            report = store.get_freshness_report(ticker)
            freshness_summary[ticker] = report["overall"]
        except Exception:  # noqa: BLE001
            freshness_summary[ticker] = "error"

    return HealthResponse(
        status="ok" if storage_health.get("sqlite") else "degraded",
        storage=storage_health,
        model_available=model_ok,
        scheduler=scheduler_status,
        freshness=freshness_summary,
    )


async def _check_model_health() -> bool:
    """Ping the llama-server to check if the model is available."""
    if not config or not model_client:
        return False
    try:
        resp = await model_client.get(
            config.llama_endpoint.replace("/v1/chat/completions", "/health"),
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ── Query (Full Pipeline) ──────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a financial question.

    Full pipeline:
      1. Parse intent (ticker, metrics, question type)
      2. Dual retrieval (SQLite facts + ChromaDB documents)
      3. Build augmented prompt
      4. Call TraceAlchemy model
      5. Return grounded answer with citations
    """
    start = time.time()

    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    # Step 1: Intent parsing (delegated to 1.5.2)
    from .intent_parser import IntentParser
    parser = IntentParser()
    intent = parser.parse(request.question, override_ticker=request.ticker)

    # Step 1b: Staleness-aware freshness check (Phase 1.7.4)
    freshness_meta = _evaluate_and_refresh(intent.get("ticker"), request.refresh)

    # Step 2: Dual retrieval (delegated to 1.5.3)
    from .retriever import Retriever
    retriever = Retriever(store=store, config=config)
    retrieval = retriever.retrieve(
        query=request.question,
        intent=intent,
        top_k_documents=config.top_k_documents,
        top_k_facts=config.top_k_facts,
    )

    # Step 3: Prompt augmentation (delegated to 1.5.4)
    from .prompt_augmenter import PromptAugmenter
    augmenter = PromptAugmenter(config=config)
    augmented_prompt = augmenter.build_prompt(
        question=request.question,
        intent=intent,
        retrieval=retrieval,
    )

    # Step 4: Call the model — or degrade gracefully when it is unavailable.
    model_available = await _check_model_health()
    if model_available:
        answer_text, citations = await _call_model(
            prompt=augmented_prompt,
            temperature=request.temperature or config.default_temperature,
            max_tokens=request.max_tokens or config.max_tokens,
        )
    else:
        logger.warning("Model unavailable — returning degraded answer")
        answer_text = _format_degraded_answer(retrieval, intent)
        citations = []

    elapsed_ms = round((time.time() - start) * 1000, 1)

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        detected_ticker=intent.get("ticker"),
        detected_intent=intent.get("question_type"),
        facts_used=len(retrieval.get("facts", [])),
        documents_used=len(retrieval.get("documents", [])),
        latency_ms=elapsed_ms,
        model_available=model_available,
        freshness=freshness_meta,
    )


def _format_degraded_answer(retrieval: dict, intent: dict) -> str:
    """Format retrieved data as a readable answer when the model is unavailable."""
    parts = ["⚠️ Model unavailable — showing raw retrieved data:\n"]
    facts = retrieval.get("facts", [])
    docs = retrieval.get("documents", [])

    if facts:
        parts.append("**Structured Facts:**")
        for f in facts[:5]:
            parts.append(
                f"- {f.get('metric')}: {f.get('value')} "
                f"({f.get('period', 'N/A')})"
            )
        parts.append("")

    if docs:
        parts.append("**Relevant Documents:**")
        for d in docs[:3]:
            meta = d.get("metadata", {}) or {}
            parts.append(
                f"- {d.get('id', 'unknown')} "
                f"({meta.get('source', d.get('source', 'unknown'))})"
            )
        parts.append("")

    if not facts and not docs:
        parts.append("No stored data found for this question.")
        parts.append("")

    parts.append("Start llama-server to get AI-grounded answers.")
    return "\n".join(parts)


# ── Freshness / Refresh (Phase 1.7.4) ──────────────────

# Maps user-facing short source aliases to the logical source names used by
# Store.FRESHNESS_SOURCES.
_SOURCE_ALIASES = {
    "fundamentals": "yfinance_fundamentals",
    "yfinance_fundamentals": "yfinance_fundamentals",
    "news": "yfinance_news",
    "yfinance_news": "yfinance_news",
    "sec": "sec_filings",
    "sec_filings": "sec_filings",
    "gdelt": "gdelt_news",
    "gdelt_news": "gdelt_news",
    "earnings": "earnings_transcripts",
    "earnings_transcripts": "earnings_transcripts",
    "ir": "ir_pages",
    "ir_pages": "ir_pages",
}


def _normalize_sources(sources: Optional[list[str]]) -> list[str]:
    """Map short aliases to logical source names, dropping unknown ones."""
    if not sources:
        return []
    out = []
    for s in sources:
        logical = _SOURCE_ALIASES.get(str(s).lower().strip())
        if logical and logical not in out:
            out.append(logical)
    return out


# Logical sources that have a scheduler-managed ingestion pipeline. Refreshing
# these routes through the UnifiedScheduler so TTL tracking, staggered
# execution, and dead-letter handling stay consistent with cron-driven runs.
SCHEDULER_SOURCE_MAP = {
    "sec_filings": "sec_filings",
    "earnings_transcripts": "earnings_transcripts",
    "ir_pages": "ir_pages",
}


def _refresh_via_scheduler(ticker: str, logical: str) -> None:
    """Route a per-ticker refresh through the UnifiedScheduler's source runner.

    For scheduler-managed sources (SEC filings, earnings transcripts, IR pages)
    this ensures the same TTL tracking and error handling as the cron path.
    The underlying ingestors mark per-ticker cache freshness; we additionally
    mark the requested ticker fresh so the freshness report reflects the run.
    Falls through to direct ingestion for non-scheduler sources.
    """
    source_name = SCHEDULER_SOURCE_MAP.get(logical)
    if not source_name:
        _refresh_one_source_direct(ticker, logical)
        return

    from src.scheduler import UnifiedScheduler
    from src.storage.store import Store

    sched = UnifiedScheduler(store=store)
    sched._run_source(source_name, force=True)

    # Reflect the run in this ticker's freshness even if the underlying
    # ingestor only marks the synthetic scheduler ticker.
    cfg = Store.FRESHNESS_SOURCES.get(logical)
    if cfg:
        ttl = store._schedule_ttls().get(cfg["ttl_key"], 24)
        store.mark_source_fresh(ticker, cfg["cache_source"], ttl)


def _refresh_one_source(ticker: str, logical: str) -> None:
    """Run the ingestion for a single logical source for one ticker.

    Scheduler-managed sources are routed through the UnifiedScheduler bridge;
    all others are ingested directly. Marks the source fresh in cache_meta on
    success. Raises on failure so the caller can record the error.
    """
    if logical in SCHEDULER_SOURCE_MAP:
        _refresh_via_scheduler(ticker, logical)
        return
    _refresh_one_source_direct(ticker, logical)


def _refresh_one_source_direct(ticker: str, logical: str) -> None:
    """Direct per-ticker ingestion for a single logical source.

    Marks the source fresh in cache_meta on success. Raises on failure so the
    caller can record the error.
    """
    from src.storage.store import Store
    cfg = Store.FRESHNESS_SOURCES.get(logical)
    cache_source = cfg["cache_source"] if cfg else logical
    ttl = store._schedule_ttls().get(cfg["ttl_key"], 24) if cfg else 24

    if logical == "yfinance_fundamentals":
        from src.ingestion.yfinance_ingestor import YFinanceIngestor
        ing = YFinanceIngestor(store=store)
        t = ing._fetch_ticker(ticker)
        if t is not None:
            ing._ingest_ticker_fundamentals(ticker, t)  # marks cache fresh
    elif logical == "yfinance_news":
        from src.ingestion.yfinance_ingestor import YFinanceIngestor
        ing = YFinanceIngestor(store=store)
        t = ing._fetch_ticker(ticker)
        if t is not None:
            ing._ingest_ticker_news(ticker, t)  # marks cache fresh
    elif logical == "sec_filings":
        from src.sec import FilingScheduler
        sched = FilingScheduler(store=store)
        sched.processor.discover_new_filings(ticker)
        store.mark_source_fresh(ticker, cache_source, ttl)
    elif logical == "gdelt_news":
        from src.macros.gdelt_ingestor import GDELTIngestor
        GDELTIngestor(store=store).fetch_and_store_for_ticker(ticker)
        store.mark_source_fresh(ticker, cache_source, ttl)
    elif logical == "earnings_transcripts":
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        EarningsTranscriptIngestor(store=store).fetch_and_process(ticker)
        store.mark_source_fresh(ticker, cache_source, ttl)
    elif logical == "ir_pages":
        from src.macros.ir_ingestor import IRIngestor
        IRIngestor(store=store).fetch_for_ticker(ticker)
        store.mark_source_fresh(ticker, cache_source, ttl)
    else:
        raise ValueError(f"Unknown source: {logical}")


def _refresh_ticker_sources(ticker: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """Refresh the given logical sources for a ticker.

    Returns (refreshed, errors).
    """
    refreshed: list[str] = []
    errors: list[str] = []
    for logical in sources:
        try:
            _refresh_one_source(ticker, logical)
            refreshed.append(logical)
        except Exception as e:  # noqa: BLE001 - never let a refresh crash the query
            logger.warning("Refresh failed for %s/%s: %s", ticker, logical, e)
            store.mark_source_stale(ticker, _logical_cache_source(logical), str(e))
            errors.append(f"{logical}: {e}")
    return refreshed, errors


def _logical_cache_source(logical: str) -> str:
    from src.storage.store import Store
    cfg = Store.FRESHNESS_SOURCES.get(logical)
    return cfg["cache_source"] if cfg else logical


def _evaluate_and_refresh(ticker: Optional[str], do_refresh: bool) -> dict:
    """Check freshness for a ticker and optionally refresh stale sources.

    Returns the freshness metadata block for the query response.
    """
    meta = {
        "overall": "unknown",
        "refreshed_during_query": [],
        "stale_sources_used": [],
        "warning": None,
    }
    if not ticker or not store:
        return meta

    try:
        report = store.get_freshness_report(ticker)
    except Exception as e:  # noqa: BLE001
        logger.warning("Freshness check failed for %s: %s", ticker, e)
        return meta

    meta["overall"] = report.get("overall", "unknown")
    # Only present-but-expired sources are auto-refreshed during a query;
    # never_fetched sources are left to the scheduler / explicit refresh.
    stale = [
        name for name, info in report.get("sources", {}).items()
        if info.get("status") == "stale"
    ]
    if not stale:
        return meta

    if do_refresh:
        refreshed, _errors = _refresh_ticker_sources(ticker, stale)
        meta["refreshed_during_query"] = refreshed
        meta["stale_sources_used"] = [s for s in stale if s not in refreshed]
    else:
        meta["stale_sources_used"] = stale
        meta["warning"] = (
            f"{ticker} has stale data for: {', '.join(stale)}. "
            "Answer may not reflect the latest information."
        )
    return meta


@app.get("/freshness/{ticker}", response_model=FreshnessResponse)
async def get_freshness(ticker: str):
    """Get a freshness report for a ticker across all data sources."""
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    report = store.get_freshness_report(ticker.upper())
    return FreshnessResponse(
        ticker=report["ticker"],
        overall=report["overall"],
        sources=report["sources"],
        stale_sources=report["stale_sources"],
    )


@app.post("/refresh/{ticker}", response_model=RefreshResponse)
async def refresh_ticker(ticker: str, body: Optional[RefreshRequest] = None):
    """Trigger on-demand refresh for a ticker.

    If ``sources`` is omitted, all currently-stale sources are refreshed.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    ticker = ticker.upper()
    start = time.time()

    # Distinguish "no sources field" (refresh all stale) from "sources field
    # provided but all unknown" (refresh nothing — silently skip unknowns).
    sources_provided = bool(body and body.sources)
    requested = _normalize_sources(body.sources if body else None)
    report = store.get_freshness_report(ticker)
    stale = [
        name for name, info in report.get("sources", {}).items()
        if info.get("status") in ("stale", "never_fetched")
    ]

    if sources_provided:
        to_refresh = requested
        skipped = [s for s in report.get("sources", {}) if s not in requested]
    else:
        to_refresh = stale
        skipped = [s for s in report.get("sources", {}) if s not in stale]

    refreshed, errors = _refresh_ticker_sources(ticker, to_refresh)

    return RefreshResponse(
        ticker=ticker,
        refreshed=refreshed,
        skipped=skipped,
        errors=errors,
        duration_s=round(time.time() - start, 2),
    )


async def _call_model(prompt: str, temperature: float,
                      max_tokens: int) -> tuple[str, list[SourceCitation]]:
    """Send the augmented prompt to TraceAlchemy and parse the response."""
    if not model_client or not config:
        return "Model unavailable. Please ensure llama-server is running.", []

    payload = {
        "model": config.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a financial research assistant. Answer the user's "
                    "question using ONLY the provided context. If the context "
                    "doesn't contain enough information, say so. "
                    "Cite sources inline using [Source: type/ticker] notation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = await model_client.post(config.llama_endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse citations from the response (simple heuristic)
        citations = _extract_citations(content)

        return content, citations
    except Exception as e:
        logger.error("Model call failed: %s", e)
        return f"Error calling model: {e}", []


def _extract_citations(text: str) -> list[SourceCitation]:
    """Extract [Source: ...] citations from model output."""
    import re
    citations = []
    pattern = r'\[Source:\s*([^\]]+)\]'
    for match in re.finditer(pattern, text):
        parts = match.group(1).split("/")
        citation = SourceCitation(
            source_type=parts[0] if len(parts) > 0 else "unknown",
            ticker=parts[1] if len(parts) > 1 else "",
        )
        citations.append(citation)
    return citations


# ── Raw Search (Bypass Model) ──────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Raw hybrid search — returns retrieved data without model inference."""
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    try:
        results = store.search(
            query=request.query,
            n_results=request.n_results,
            ticker=request.ticker,
        )
    except Exception as e:
        logger.warning("Search failed (embedding server may be down): %s", e)
        # Return empty results gracefully when embedding server is unavailable
        results = {"documents": [], "facts": [], "ticker": request.ticker}

    return SearchResponse(
        documents=results.get("documents", []),
        facts=results.get("facts", []),
        ticker=results.get("ticker"),
    )


# ── Macro / Sentiment / Guidance ───────────────────────

@app.get("/macro/snapshot", response_model=MacroSnapshotResponse)
async def macro_snapshot():
    """
    Get a quick snapshot of key macro-economic indicators.
    Returns cached data from SQLite (no live FRED API call).
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    macro = store.get_fundamentals_batch("MACRO", metrics=[
        "GDP", "CPIAUCSL", "FEDFUNDS", "UNRATE", "DGS10", "T10Y2Y",
    ])

    return MacroSnapshotResponse(
        gdp=macro.get("GDP"),
        inflation_cpi=macro.get("CPIAUCSL"),
        fed_rate=macro.get("FEDFUNDS"),
        unemployment=macro.get("UNRATE"),
        ten_year_treasury=macro.get("DGS10"),
        ten_two_spread=macro.get("T10Y2Y"),
    )


@app.get("/sentiment/{ticker}", response_model=SentimentResponse)
async def sentiment(ticker: str, days: int = 7):
    """
    Get GDELT sentiment summary for a ticker.

    Args:
        ticker: Stock ticker symbol
        days: Lookback period in days (default: 7)

    Returns average tone score, article count, and sentiment ratios.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    from src.macros.gdelt_ingestor import GDELTIngestor
    ingestor = GDELTIngestor(store=store)
    summary = ingestor.get_sentiment_summary(ticker.upper(), days=days)

    return SentimentResponse(
        ticker=summary.get("ticker", ticker.upper()),
        average_tone=summary.get("average_tone"),
        article_count=summary.get("article_count", 0),
        positive_ratio=summary.get("positive_ratio", 0.0),
        negative_ratio=summary.get("negative_ratio", 0.0),
    )


@app.get("/guidance/{ticker}", response_model=dict)
async def guidance(ticker: str):
    """
    Get the latest earnings guidance for a ticker.
    Returns revenue guidance range, EPS, and margin from the most
    recent earnings transcript.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    from src.macros.earnings_transcripts import EarningsTranscriptIngestor
    ingestor = EarningsTranscriptIngestor(store=store)
    guidance_data = ingestor.get_latest_guidance(ticker.upper())

    if not guidance_data:
        return {"ticker": ticker.upper(), "guidance": {}, "status": "not_found"}

    return {
        "ticker": ticker.upper(),
        "guidance": guidance_data,
        "status": "found",
    }


# ── Root ───────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Gemma-E4B-Finance-RAG Middleware",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
        "search": "POST /search",
    }
