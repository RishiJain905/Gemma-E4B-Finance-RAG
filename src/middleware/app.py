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

import asyncio
import contextvars
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.storage.store import Store
from . import prompt_policy
from .config import MiddlewareConfig
from .evidence import evidence_counts, usable_documents, usable_facts
from .evidence_trace import EvidenceTraceCollector
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

# Backward-compatible aliases — prompt_policy.py is now the single owner of
# these strings (src/middleware/prompt_policy.py). Kept here so older
# imports/tests referencing middleware_app.SYSTEM_PROMPT still resolve.
SYSTEM_PROMPT = prompt_policy.STRICT_SYSTEM_PROMPT
TOOLS_SYSTEM_PROMPT = prompt_policy.STRICT_TOOLS_SYSTEM_PROMPT

GENERAL_FALLBACK_PREFIX = "Not from your data - general knowledge:"
GENERAL_FALLBACK_CAVEAT = "Please verify against a primary source before relying on it."
NO_GENERAL_FALLBACK_MESSAGE = (
    "I don't have enough data in my knowledge base to answer this. "
    "General-knowledge fallback is disabled for this deployment."
)

# ── Global state (set during lifespan) ─────────────────

config: Optional[MiddlewareConfig] = None
store: Optional[Store] = None
model_client: Optional[httpx.AsyncClient] = None
_tools_supported: bool = True
MACRO_SNAPSHOT_METRICS = ["GDP", "CPIAUCSL", "FEDFUNDS", "UNRATE", "DGS10", "T10Y2Y"]
_MODEL_TASKS_CACHE: Optional[dict] = None
FETCH_ON_MISS_MIN_CONFIDENCE = 0.9
retriever = None  # Shared Retriever (built on startup) — Phase 2.1.2
_MODEL_HEALTH_TTL_S = 10.0
_HEALTH_SUMMARY_TTL_S = 3.0
_model_health = {"ok": False, "ts": 0.0}
_health_cache = {"ts": 0.0, "value": None}
_scheduler = None

# Per-request state (Phase 2.1.8.3). Each incoming request runs in its own
# asyncio Task, which copies the context at creation time, so these never
# leak between concurrent requests as long as they're reset at the top of
# _build_query_context — the single entry point shared by /query and
# /query/stream.
_tools_used_var: contextvars.ContextVar[Optional[list[str]]] = contextvars.ContextVar(
    "tools_used", default=None
)
_answer_policy_override_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "answer_policy_override", default=None
)
_evidence_trace_var: contextvars.ContextVar[Optional[EvidenceTraceCollector]] = contextvars.ContextVar(
    "evidence_trace", default=None
)


def _grounding_level(retrieval: dict) -> str:
    """Return the graded grounding level from usable facts/documents.

    Blank document bodies and None-valued facts are not usable evidence
    (see src/middleware/evidence.py) and must not inflate grounding.
    """
    n_facts, n_docs = evidence_counts(retrieval)
    n = n_facts + n_docs
    return "grounded" if n >= 3 else "partial" if n >= 1 else "none"


def _answer_policy() -> str:
    """Return the effective answer policy: per-request override, else configured default."""
    override = _answer_policy_override_var.get()
    if override in ("strict", "graded"):
        return override
    policy = str(getattr(config, "answer_policy", "graded") or "graded").lower()
    return "strict" if policy == "strict" else "graded"


def _reset_request_scoped_state(answer_policy: Optional[str]) -> None:
    """Reset per-request contextvars: tool-call log, answer-policy override,
    and evidence-trace collector."""
    _tools_used_var.set([])
    _evidence_trace_var.set(None)
    normalized = str(answer_policy or "").strip().lower()
    _answer_policy_override_var.set(normalized if normalized in ("strict", "graded") else None)


def _record_tool_used(name: str) -> None:
    """Append a dispatched tool name to the current request's tool-call log."""
    used = _tools_used_var.get()
    if used is not None and name and name not in used:
        used.append(name)


def _get_tools_used() -> Optional[list[str]]:
    """Return the current request's dispatched tool names, or None if empty."""
    used = _tools_used_var.get()
    return list(used) if used else None


def _record_trace_prompt(system_prompt: str, user_prompt: str) -> None:
    """Record the exact system/user messages for the request's evidence trace
    (2.2.1.2). No-op when no trace was requested for this request."""
    collector = _evidence_trace_var.get()
    if collector is not None:
        collector.record_prompt(system_prompt=system_prompt, user_prompt=user_prompt)


def _record_trace_tool_result(name: str, arguments: dict, result: dict) -> None:
    """Append one dispatched tool call to the request's evidence trace, if any."""
    collector = _evidence_trace_var.get()
    if collector is not None:
        collector.record_tool_result(name, arguments, result)


def _discard_trace_tool_results() -> None:
    """Drop any tool results recorded so far on the request's evidence trace.

    Called before a plain-call fallback (tools unsupported / empty tool-mode
    response) records its messages, so the abandoned tool attempt's results
    never leak into the successful answer path's trace.
    """
    collector = _evidence_trace_var.get()
    if collector is not None:
        collector.discard_tool_results()


def _resolved_ticker_field(intent: dict) -> Optional[dict]:
    """Return {'name','source'} when the resolver mapped a non-exact ticker.

    Omitted for exact matches (known_ticker/override) so the field only
    fires for name lookups and typo-corrected fuzzy matches.
    """
    source = intent.get("ticker_source")
    name = intent.get("resolved_name")
    if not source or not name or source in ("known_ticker", "override"):
        return None
    return {"name": name, "source": source}


def _allow_general_fallback() -> bool:
    """Return whether no-context general fallback answers are allowed."""
    return bool(getattr(config, "allow_general_fallback", True))


def _system_prompt_for_request(
    intent: Optional[dict],
    grounding_level: str,
    tools_enabled: bool = False,
) -> str:
    """Return this request's system prompt via the shared prompt_policy builder.

    Shared by the plain, streaming, and tool-loop call sites so they cannot
    silently enforce different rules (2.2.1.1).
    """
    return prompt_policy.build_system_prompt(
        answer_policy=_answer_policy(),
        allow_general_fallback=_allow_general_fallback(),
        intent=intent,
        grounding_level=grounding_level,
        tools_enabled=tools_enabled,
    )


def _is_declined_answer(answer: str) -> bool:
    """Heuristically detect model refusals/unsafe declines."""
    text = (answer or "").strip().lower()
    if not text:
        return True
    decline_markers = (
        "i don't have enough data",
        "i do not have enough data",
        "i can't answer",
        "i cannot answer",
        "i'm unable to answer",
        "i am unable to answer",
        "cannot provide",
        "can't provide",
        "unsafe",
        "not enough information",
        "data is unavailable",
    )
    return any(marker in text for marker in decline_markers)


def _apply_answer_policy(answer: str, grounding_level: str) -> str:
    """Enforce deterministic labels/refusals that should not depend on sampling."""
    if _answer_policy() == "strict":
        return answer
    if not answer or answer.startswith(("Error calling model:", "Model unavailable.")):
        return answer
    if grounding_level != "none":
        return answer
    if not _allow_general_fallback():
        return NO_GENERAL_FALLBACK_MESSAGE
    if _is_declined_answer(answer):
        return answer

    labeled = answer.strip()
    if not labeled.lower().startswith(GENERAL_FALLBACK_PREFIX.lower()):
        labeled = f"{GENERAL_FALLBACK_PREFIX} {labeled}"
    if "verify against a primary source" not in labeled.lower():
        labeled = f"{labeled}\n\n{GENERAL_FALLBACK_CAVEAT}"
    return labeled


def _response_grounding(answer: str, grounding_level: str) -> str:
    """Map the actual answer path to response metadata."""
    if _is_declined_answer(answer):
        return "refused"
    if grounding_level == "grounded":
        return "grounded"
    if grounding_level == "partial":
        return "partial"
    if grounding_level == "none" and _allow_general_fallback():
        return "general"
    return "refused"


async def _invoke_model(
    prompt: str,
    temperature: float,
    max_tokens: int,
    intent: dict,
    grounding_level: str,
) -> tuple[str, list[SourceCitation]]:
    """Call _call_model while preserving old-signature test monkeypatches."""
    import inspect

    params = inspect.signature(_call_model).parameters
    if "intent" not in params:
        return await _call_model(prompt, temperature, max_tokens)
    return await _call_model(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        intent=intent,
        grounding_level=grounding_level,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global config, store, model_client, retriever

    logger.info("Starting middleware...")
    from src.utils.env import load_env
    load_env()  # load .env credentials before initializing components
    config = MiddlewareConfig()
    store = Store(
        embedding_endpoint=config.embedding_endpoint,
        embedding_cache_size=config.embedding_cache_size,
    )
    model_client = httpx.AsyncClient(timeout=60)

    # Shared retriever so the BM25 lexical index is built once and reused
    # across requests (Phase 2.1.2). Warm it eagerly on startup.
    from .retriever import Retriever
    retriever = Retriever(store=store, config=config)
    if config.enable_lexical:
        retriever.warm_lexical_index()

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


def _find_tasks_block(node) -> dict:
    if not isinstance(node, dict):
        return {}
    tasks = node.get("tasks")
    if isinstance(tasks, dict):
        return tasks
    for value in node.values():
        found = _find_tasks_block(value)
        if found:
            return found
    return {}


def _task_params(task_name) -> dict:
    """Return model task params from configs/model.yaml; fail soft."""
    global _MODEL_TASKS_CACHE
    try:
        if _MODEL_TASKS_CACHE is None:
            import yaml

            path = Path(__file__).resolve().parents[2] / "configs" / "model.yaml"
            with open(path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            _MODEL_TASKS_CACHE = _find_tasks_block(loaded)
        task = _MODEL_TASKS_CACHE.get(str(task_name), {})
        return task if isinstance(task, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load model task params: %s", exc)
        _MODEL_TASKS_CACHE = {}
        return {}


def _macro_snapshot_data() -> dict:
    return store.get_fundamentals_batch("MACRO", metrics=MACRO_SNAPSHOT_METRICS)


def _sentiment_data(ticker: str, days: int = 7) -> dict:
    from src.macros.gdelt_ingestor import GDELTIngestor

    return GDELTIngestor(store=store).get_sentiment_summary(ticker.upper(), days=days)


def _guidance_data(ticker: str) -> dict:
    from src.macros.earnings_transcripts import EarningsTranscriptIngestor

    return EarningsTranscriptIngestor(store=store).get_latest_guidance(ticker.upper())


# ── Health ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Enhanced health check with storage, model, scheduler, and freshness."""
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    storage_health = store.heartbeat()
    model_ok = await _check_model_health()
    summary = _cached_health_summary()

    capabilities = None
    if config:
        capabilities = {
            "tools": bool(config.enable_tools),
            "streaming": bool(getattr(config, "enable_streaming", True)),
            "answer_policy": str(getattr(config, "answer_policy", "graded") or "graded").lower(),
        }

    return HealthResponse(
        status="ok" if storage_health.get("sqlite") else "degraded",
        storage=storage_health,
        model_available=model_ok,
        scheduler=summary.get("scheduler"),
        freshness=summary.get("freshness"),
        capabilities=capabilities,
    )


@app.get("/tools")
async def tools():
    """List model-callable middleware tools and tool-gating state."""
    from .tools import REGISTRY

    return {
        "enabled": bool(config.enable_tools) if config else False,
        "allow_write_tools": bool(config.allow_write_tools) if config else False,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "write": tool.write,
            }
            for tool in REGISTRY.values()
        ],
    }


def _mark_model_health(ok: bool) -> None:
    """Refresh the in-process model-health cache."""
    _model_health["ok"] = bool(ok)
    _model_health["ts"] = time.monotonic()


def _get_scheduler():
    """Return the lazily-built scheduler used by /health."""
    global _scheduler
    if _scheduler is None:
        from src.scheduler import UnifiedScheduler
        _scheduler = UnifiedScheduler(store=store)
    return _scheduler


def _cached_health_summary(ttl: float = _HEALTH_SUMMARY_TTL_S) -> dict:
    """Return cached scheduler status and ticker freshness for /health."""
    now = time.monotonic()
    cached = _health_cache.get("value")
    if cached is not None and now - float(_health_cache.get("ts", 0.0)) < ttl:
        return cached

    scheduler_status = None
    try:
        scheduler_status = _get_scheduler().status_report()
    except Exception as e:  # noqa: BLE001
        logger.warning("Scheduler status unavailable: %s", e)

    freshness_summary: dict[str, str] = {}
    for ticker in ["NVDA", "AMD", "AAPL", "MSFT", "META", "CRWD"]:
        try:
            report = store.get_freshness_report(ticker)
            freshness_summary[ticker] = report["overall"]
        except Exception:  # noqa: BLE001
            freshness_summary[ticker] = "error"

    value = {"scheduler": scheduler_status, "freshness": freshness_summary}
    _health_cache["value"] = value
    _health_cache["ts"] = now
    return value


async def _check_model_health(ttl: float = _MODEL_HEALTH_TTL_S) -> bool:
    """Ping the llama-server to check if the model is available."""
    if not config or not model_client:
        return False
    now = time.monotonic()
    if now - float(_model_health.get("ts", 0.0)) < ttl:
        return bool(_model_health.get("ok", False))
    try:
        resp = await model_client.get(
            config.llama_endpoint.replace("/v1/chat/completions", "/health"),
            timeout=5,
        )
        ok = resp.status_code == 200
        _mark_model_health(ok)
        return ok
    except Exception:
        _mark_model_health(False)
        return False


# ── Query (Full Pipeline) ──────────────────────────────

async def _maybe_fetch_on_miss(ticker: str) -> dict:
    """Run fetch-on-miss ingestion off the event loop with a hard timeout."""
    from .on_demand import fetch_ticker_on_miss

    # Validate config BEFORE starting any work: once the to_thread task is
    # created the fetch runs (with network I/O) even if wait_for errors out.
    # Strict type check on purpose — mock/partial configs must not trigger
    # a live network fetch.
    timeout = getattr(config, "fetch_on_miss_timeout_s", None)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        logger.debug("Fetch-on-miss skipped for %s: invalid timeout config", ticker)
        return {"fetched": False, "ticker": ticker, "sources": [], "error": "invalid_config"}

    try:
        task = asyncio.create_task(asyncio.to_thread(fetch_ticker_on_miss, store, ticker))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Fetch-on-miss timed out for %s", ticker)
        return {"fetched": False, "ticker": ticker, "sources": [], "error": "timeout"}
    except Exception as e:  # noqa: BLE001 - never let fetch-on-miss crash a query
        logger.warning("Fetch-on-miss failed unexpectedly for %s: %s", ticker, e)
        return {"fetched": False, "ticker": ticker, "sources": [], "error": str(e)}


def _stage_timing(timings: dict[str, object], name: str, stage_start: float) -> float:
    """Record and return a stage duration in milliseconds."""
    elapsed = round((time.perf_counter() - stage_start) * 1000, 1)
    timings[name] = elapsed
    return elapsed


def _return_timings_enabled() -> bool:
    """Return whether response timing metadata should be included."""
    return_timings = getattr(config, "return_timings", True)
    return return_timings if isinstance(return_timings, bool) else True


async def _build_query_context(request: QueryRequest) -> dict:
    """Run the shared query pipeline up to the augmented prompt."""
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    start = time.time()
    timings: dict[str, object] = {}
    _reset_request_scoped_state(request.answer_policy)

    stage_start = time.perf_counter()
    from .intent_parser import IntentParser

    parser = IntentParser()
    intent = parser.parse(request.question, override_ticker=request.ticker)
    _stage_timing(timings, "intent_parse", stage_start)

    stage_start = time.perf_counter()
    freshness_meta = _evaluate_and_refresh(intent.get("ticker"), request.refresh)
    ticker = intent.get("ticker")
    if (
        getattr(config, "enable_fetch_on_miss", True)
        and ticker
        and freshness_meta.get("overall") == "never_fetched"
        and intent.get("ticker_confidence", 0.0) >= FETCH_ON_MISS_MIN_CONFIDENCE
    ):
        res = await _maybe_fetch_on_miss(ticker)
        if res.get("fetched"):
            freshness_meta.setdefault("fetched_on_miss", []).append(ticker)
            freshness_meta["overall"] = "fresh"
        else:
            warning = f"Couldn't fetch live data for {ticker} right now."
            if res.get("error"):
                warning = f"{warning} Reason: {res['error']}"
            freshness_meta["warning"] = warning
    _stage_timing(timings, "freshness_check", stage_start)

    stage_start = time.perf_counter()
    from .retriever import Retriever

    r = retriever or Retriever(store=store, config=config)
    retrieval = r.retrieve(
        query=request.question,
        intent=intent,
        top_k_documents=config.top_k_documents,
        top_k_facts=config.top_k_facts,
    )
    retrieval_ms = _stage_timing(timings, "retrieval", stage_start)
    retrieval_timings = retrieval.get("timings", {}) if isinstance(retrieval, dict) else {}
    timings["retrieval"] = {
        "total": retrieval_ms,
        "embedding": round(float(retrieval_timings.get("embedding", 0.0) or 0.0), 1),
        "chroma": round(float(retrieval_timings.get("chroma", 0.0) or 0.0), 1),
        "sqlite": round(float(retrieval_timings.get("sqlite", 0.0) or 0.0), 1),
    }
    grounding_level = _grounding_level(retrieval)

    # Request-scoped evidence-trace collector (2.2.1.2), created right after
    # evidence normalization so facts/documents are the exact usable rows
    # (see src/middleware/evidence.py) — untruncated, full provenance. Left
    # None (and thus omitted from the response) unless explicitly requested.
    trace_collector: Optional[EvidenceTraceCollector] = None
    if request.include_evidence_trace:
        trace_collector = EvidenceTraceCollector(
            answer_policy=_answer_policy(),
            grounding_level=grounding_level,
            raw_question=request.question,
            retrieval_query=request.question,
            facts=usable_facts(retrieval),
            documents=usable_documents(retrieval),
        )
    _evidence_trace_var.set(trace_collector)

    stage_start = time.perf_counter()
    from .prompt_augmenter import PromptAugmenter

    augmenter = PromptAugmenter(config=config)
    augmented_prompt = augmenter.build_prompt(
        question=request.question,
        intent=intent,
        retrieval=retrieval,
        grounding_level=grounding_level,
    )
    _stage_timing(timings, "prompt_build", stage_start)

    return {
        "start": start,
        "timings": timings,
        "intent": intent,
        "freshness": freshness_meta,
        "retrieval": retrieval,
        "grounding_level": grounding_level,
        "augmented_prompt": augmented_prompt,
        "include_evidence_trace": request.include_evidence_trace,
    }


def _task_settings(request: QueryRequest, intent: dict) -> tuple[float, int]:
    """Return temperature and max_tokens for this request."""
    task = _task_params("analysis") if intent.get("question_type") == "projection" else {}
    temperature = request.temperature or task.get("temperature") or config.default_temperature
    max_tokens = request.max_tokens or task.get("max_tokens") or config.max_tokens
    return temperature, max_tokens


def _build_query_response(
    *,
    context: dict,
    answer_text: str,
    citations: list[SourceCitation],
    model_available: bool,
) -> QueryResponse:
    """Build a QueryResponse from shared query context and model output."""
    intent = context["intent"]
    retrieval = context["retrieval"]
    elapsed_ms = round((time.time() - context["start"]) * 1000, 1)
    # Usable-evidence counts (2.2.1.1) — shared by /query and the
    # /query/stream terminal metadata event since both call this function.
    n_facts, n_docs = evidence_counts(retrieval)

    # Evidence trace (2.2.1.2) — opt-in, and only ever non-None when a model
    # call actually recorded a prompt (never for the degraded path).
    evidence_trace = None
    if context.get("include_evidence_trace"):
        collector = _evidence_trace_var.get()
        trace = collector.finalize() if collector is not None else None
        if trace is not None:
            evidence_trace = trace.to_dict()

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        detected_ticker=intent.get("ticker"),
        detected_intent=intent.get("question_type"),
        facts_used=n_facts,
        documents_used=n_docs,
        grounding=_response_grounding(answer_text, context["grounding_level"]),
        latency_ms=elapsed_ms,
        timings=context["timings"] if _return_timings_enabled() else None,
        model_available=model_available,
        evidence_trace=evidence_trace,
        freshness=context["freshness"],
        retrieval_strategy=retrieval.get("retrieval_strategy"),
        tools_used=_get_tools_used(),
        resolved_ticker=_resolved_ticker_field(intent),
    )


async def _answer_query_context(request: QueryRequest, context: dict) -> QueryResponse:
    """Complete a prepared query context through the non-streaming model path."""
    intent = context["intent"]
    retrieval = context["retrieval"]
    grounding_level = context["grounding_level"]
    stage_start = time.perf_counter()
    model_available = await _check_model_health()
    if model_available:
        temperature, max_tokens = _task_settings(request, intent)
        answer_text, citations = await _invoke_model(
            prompt=context["augmented_prompt"],
            temperature=temperature,
            max_tokens=max_tokens,
            intent=intent,
            grounding_level=grounding_level,
        )
        if not answer_text.startswith("Error calling model:"):
            _mark_model_health(True)
        if intent.get("question_type") == "projection":
            from .guardrails import apply_projection_guardrail

            answer_text, _flagged = apply_projection_guardrail(
                answer_text,
                context["augmented_prompt"],
            )
    else:
        logger.warning("Model unavailable - returning degraded answer")
        answer_text = _format_degraded_answer(retrieval, intent)
        citations = []
    _stage_timing(context["timings"], "model_call", stage_start)

    return _build_query_response(
        context=context,
        answer_text=answer_text,
        citations=citations,
        model_available=model_available,
    )


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

    Shares `_build_query_context` / `_answer_query_context` with
    `/query/stream` so the two paths cannot drift.
    """
    context = await _build_query_context(request)
    return await _answer_query_context(request, context)


def _response_to_dict(response: QueryResponse) -> dict:
    """Return a pydantic model as a JSON-serializable dict."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def _sse(event: str, data: dict) -> str:
    """Format one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_model_tokens(
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
    intent: dict,
    grounding_level: str,
):
    """Yield token deltas from llama-server's OpenAI-compatible stream."""
    if not model_client or not config:
        raise RuntimeError("model client unavailable")

    payload = {
        "model": config.model_name,
        "messages": [
            {
                "role": "system",
                "content": _system_prompt_for_request(
                    intent=intent,
                    grounding_level=grounding_level,
                    tools_enabled=False,
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    async with model_client.stream("POST", config.llama_endpoint, json=payload) as resp:
        if hasattr(resp, "raise_for_status"):
            resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = (line or "").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Ignoring malformed model stream line: %s", line[:120])
                continue
            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            token = delta.get("content")
            if token is None:
                token = (choice.get("message") or {}).get("content")
            if token:
                yield token
    _mark_model_health(True)


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream token deltas as SSE, followed by terminal query metadata."""
    if not bool(getattr(config, "enable_streaming", True)):
        raise HTTPException(status_code=404, detail="Streaming disabled")
    if bool(getattr(config, "enable_tools", False)):
        raise HTTPException(status_code=404, detail="Streaming disabled while tools are enabled")

    context = await _build_query_context(request)
    model_available = await _check_model_health()
    if not model_available:
        raise HTTPException(status_code=404, detail="Streaming unavailable when model is unavailable")

    async def events():
        stage_start = time.perf_counter()
        answer_parts: list[str] = []
        citations: list[SourceCitation] = []
        try:
            temperature, max_tokens = _task_settings(request, context["intent"])
            async for token in _stream_model_tokens(
                prompt=context["augmented_prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
                intent=context["intent"],
                grounding_level=context["grounding_level"],
            ):
                answer_parts.append(token)
                yield _sse("token", {"token": token})

            answer_text = _apply_answer_policy(
                "".join(answer_parts),
                context["grounding_level"],
            )
            citations = _extract_citations(answer_text)
            # Streaming is always tools_enabled=False (see the guard above),
            # so the exact system prompt is reproducible here for the
            # evidence trace (2.2.1.2) without a second model call.
            _record_trace_prompt(
                _system_prompt_for_request(
                    intent=context["intent"],
                    grounding_level=context["grounding_level"],
                    tools_enabled=False,
                ),
                context["augmented_prompt"],
            )
            if context["intent"].get("question_type") == "projection":
                from .guardrails import apply_projection_guardrail

                answer_text, _flagged = apply_projection_guardrail(
                    answer_text,
                    context["augmented_prompt"],
                )
        except Exception as exc:  # noqa: BLE001 - streaming must never fail a query
            logger.warning("Streaming model call failed; falling back server-side: %s", exc)
            fallback = await _answer_query_context(request, context)
            if fallback.answer:
                yield _sse("token", {"token": fallback.answer})
            metadata = _response_to_dict(fallback)
            metadata.pop("answer", None)
            yield _sse("metadata", metadata)
            return

        _stage_timing(context["timings"], "model_call", stage_start)
        response = _build_query_response(
            context=context,
            answer_text=answer_text,
            citations=citations,
            model_available=True,
        )
        metadata = _response_to_dict(response)
        metadata.pop("answer", None)
        yield _sse("metadata", metadata)

    return StreamingResponse(events(), media_type="text/event-stream")


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
    "estimates": "estimates",
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


def _stale_source_names(report: dict) -> list[str]:
    """Return logical source names that are stale or have never been fetched."""
    return [
        name for name, info in report.get("sources", {}).items()
        if info.get("status") in ("stale", "never_fetched")
    ]


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
    elif logical == "estimates":
        from src.macros.estimates_ingestor import EstimatesIngestor
        result = EstimatesIngestor(store=store).fetch_for_ticker(ticker)
        status = result.get("status")
        if status not in ("success", "no_data"):
            raise RuntimeError(
                f"estimates refresh failed for {ticker}: {status} "
                f"{result.get('errors') or []}"
            )
        # no_data still marks fresh: lack of analyst coverage (e.g. ETFs)
        # shouldn't trigger a refetch on every stale check.
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
        "fetched_on_miss": [],
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
    stale = _stale_source_names(report)

    if sources_provided:
        to_refresh = requested
        skipped = [s for s in report.get("sources", {}) if s not in requested]
    else:
        to_refresh = stale
        skipped = [s for s in report.get("sources", {}) if s not in stale]

    refreshed, errors = _refresh_ticker_sources(ticker, to_refresh)

    # Keep the BM25 lexical index fresh after ingestion (Phase 2.1.2.3).
    if retriever is not None:
        try:
            retriever.refresh_lexical_index()
        except Exception as e:  # noqa: BLE001 - never fail the refresh response
            logger.warning("Lexical index refresh failed: %s", e)

    return RefreshResponse(
        ticker=ticker,
        refreshed=refreshed,
        skipped=skipped,
        errors=errors,
        duration_s=round(time.time() - start, 2),
    )


async def _call_model(
    prompt: str,
    temperature: float,
    max_tokens: int,
    intent: Optional[dict] = None,
    grounding_level: str = "grounded",
) -> tuple[str, list[SourceCitation]]:
    """Send the augmented prompt to TraceAlchemy and parse the response."""
    global _tools_supported

    if not model_client or not config:
        return "Model unavailable. Please ensure llama-server is running.", []

    tools_on = bool(getattr(config, "enable_tools", False)) and _tools_supported
    messages = [
        {
            "role": "system",
            "content": _system_prompt_for_request(
                intent=intent,
                grounding_level=grounding_level,
                tools_enabled=False,
            ),
        },
        {"role": "user", "content": prompt},
    ]
    base_payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if not tools_on:
        answer, citations = await _post_and_parse(base_payload)
        _record_trace_prompt(messages[0]["content"], prompt)
        return _apply_answer_policy(answer, grounding_level), citations

    from .tools import ToolContext, dispatch_tool_traced, openai_schema

    # The tool loop keeps its own messages list so every fallback to
    # _post_and_parse(base_payload) still sends the exact pre-tools prompt.
    messages = [
        {
            "role": "system",
            "content": _system_prompt_for_request(
                intent=intent,
                grounding_level=grounding_level,
                tools_enabled=True,
            ),
        },
        {"role": "user", "content": prompt},
    ]
    schema = openai_schema()
    ctx = ToolContext(
        allow_write=config.allow_write_tools,
        max_refreshes=config.max_refreshes_per_query,
    )
    for iteration in range(config.max_tool_iterations):
        payload = {**base_payload, "messages": messages, "tools": schema}
        try:
            resp = await model_client.post(config.llama_endpoint, json=payload)
            resp.raise_for_status()
            _mark_model_health(True)
            data = resp.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text if e.response is not None else ""
            status = e.response.status_code if e.response is not None else None
            if status in (400, 404, 500) and "tool" in body.lower():
                logger.warning("Model tools unsupported; falling back to plain calls")
                _tools_supported = False
                answer, citations = await _post_and_parse(base_payload)
                _discard_trace_tool_results()
                _record_trace_prompt(base_payload["messages"][0]["content"], prompt)
                return _apply_answer_policy(answer, grounding_level), citations
            logger.error("Model call failed: %s", e)
            return f"Error calling model: {e}", []
        except Exception as e:  # noqa: BLE001
            logger.error("Model call failed: %s", e)
            return f"Error calling model: {e}", []

        try:
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
        except (AttributeError, IndexError, TypeError) as e:
            logger.error("Malformed model response in tool loop: %s", e)
            return f"Error calling model: malformed response ({e})", []
        if iteration == 0 and not tool_calls and not content.strip():
            logger.warning("Model returned empty content with tools; disabling tools")
            _tools_supported = False
            answer, citations = await _post_and_parse(base_payload)
            _discard_trace_tool_results()
            _record_trace_prompt(base_payload["messages"][0]["content"], prompt)
            return _apply_answer_policy(answer, grounding_level), citations
        if not tool_calls:
            answer = _apply_answer_policy(content, grounding_level)
            _record_trace_prompt(messages[0]["content"], prompt)
            return answer, _extract_citations(answer)

        messages.append(msg)
        for call in tool_calls:
            tool_name = (call.get("function") or {}).get("name")
            if tool_name:
                _record_tool_used(tool_name)
            result, dispatched_name, dispatched_args = await asyncio.to_thread(
                dispatch_tool_traced, call, store, ctx
            )
            # Recorded immediately after dispatch returns and before the
            # matching role=tool message is appended (2.2.1.2 Step 2).
            _record_trace_tool_result(dispatched_name or tool_name or "", dispatched_args, result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result),
                }
            )

    answer, citations = await _post_and_parse({**base_payload, "messages": messages})
    _record_trace_prompt(messages[0]["content"], prompt)
    return _apply_answer_policy(answer, grounding_level), citations


async def _post_and_parse(payload: dict) -> tuple[str, list[SourceCitation]]:
    """POST a chat payload and parse content plus inline citations."""
    try:
        resp = await model_client.post(config.llama_endpoint, json=payload)
        resp.raise_for_status()
        _mark_model_health(True)
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
    """Raw hybrid search — returns retrieved data without model inference.

    Documents come through the shared retriever's hybrid path (vector + BM25 +
    re-rank, per config) so ``fusion_score`` / ``rerank_score`` are exposed for
    inspecting ranking quality (Phase 2.1.2.3).
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    from .retriever import Retriever
    r = retriever or Retriever(store=store, config=config)

    documents: list[dict] = []
    facts: list[dict] = []
    ticker_out = request.ticker
    try:
        facts_results = store.search(
            query=request.query,
            n_results=request.n_results,
            ticker=request.ticker,
        )
        facts = facts_results.get("facts", [])
        ticker_out = facts_results.get("ticker") or request.ticker
    except Exception as e:  # noqa: BLE001
        logger.warning("Search facts failed (embedding server may be down): %s", e)

    try:
        documents = r.retrieve_documents(
            query=request.query,
            ticker=request.ticker,
            n_results=request.n_results,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Search documents failed: %s", e)

    return SearchResponse(
        documents=documents,
        facts=facts,
        ticker=ticker_out,
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

    macro = _macro_snapshot_data()

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

    summary = _sentiment_data(ticker, days=days)

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

    guidance_data = _guidance_data(ticker)

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
