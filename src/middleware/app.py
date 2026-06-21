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
    HealthResponse,
    MacroSnapshotResponse,
    QueryRequest,
    QueryResponse,
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
    """Check storage backends and model availability."""
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    storage_health = store.heartbeat()
    model_ok = await _check_model_health()

    return HealthResponse(
        status="ok" if storage_health.get("sqlite") else "degraded",
        storage=storage_health,
        model_available=model_ok,
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

    # Step 4: Call the model
    answer_text, citations = await _call_model(
        prompt=augmented_prompt,
        temperature=request.temperature or config.default_temperature,
        max_tokens=request.max_tokens or config.max_tokens,
    )

    elapsed_ms = round((time.time() - start) * 1000, 1)

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        detected_ticker=intent.get("ticker"),
        detected_intent=intent.get("question_type"),
        facts_used=len(retrieval.get("facts", [])),
        documents_used=len(retrieval.get("documents", [])),
        latency_ms=elapsed_ms,
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
