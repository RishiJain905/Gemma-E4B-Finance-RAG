"""
eval/run_eval.py — Stage 1 of the evaluation harness: the runner.

Drives every golden case through the query pipeline and captures, per case:
  answer, detected_ticker, detected_intent, facts_used, documents_used,
  retrieved_sources, evidence_trace, trace_complete, answer_policy,
  grounding, dataset_digest, latency_ms, model_available.

Backend preference (per the 2.1.1.1 spec):
  1. Live middleware  — POST /query on :8000 (measures the real system).
  2. Direct pipeline  — import IntentParser/Retriever/PromptAugmenter in-process
                         (used when the middleware is down but the model/store are up).
  3. Error row         — if both fail, emit a well-formed row with model_available=False
                         so scoring never crashes on a half-dead environment.

2.2.1.2 replaced the old judge-context mechanism — a second, truncated
in-process retrieval (``_capture_context_and_sources``) run after the live
``/query`` call returned — with the exact evidence trace the endpoint itself
captured. Live requests set ``include_evidence_trace=true`` and the runner
persists the returned trace verbatim; nothing is re-retrieved, and nothing is
truncated. The direct/offline backend builds an equivalent trace in-process
using the same evidence helpers (``src/middleware/evidence.py``) and prompt
builder (``src/middleware/prompt_policy.py``) as the middleware, so both
backends produce a trace of the same shape.

A row whose model produced an answer but whose trace is missing or
incomplete is marked with ``error`` — it is an evaluation error, not
silently scorable context (see ``eval/metrics.py::trace_errors``).

Usage:
    python eval/run_eval.py                 # live :8000, fall back to direct
    python eval/run_eval.py --offline      # skip the live endpoint, direct only
    python eval/run_eval.py --limit 5      # just the first 5 cases
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Make ``src.*`` importable when this file is run as a script
# (sys.path[0] would otherwise be eval/, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden" / "finance_qa.jsonl"
CONVERSATIONS = EVAL_DIR / "golden" / "finance_conversations.jsonl"
RUNS_DIR = EVAL_DIR / "runs"

DEFAULT_QUERY_URL = os.environ.get("EVAL_QUERY_URL", "http://127.0.0.1:8000/query")
DEFAULT_TIMEOUT = float(os.environ.get("EVAL_QUERY_TIMEOUT", "240"))

# Canonical source names a retrieval can surface. The golden dataset's
# ``expected_sources`` uses these; retrieved sources are normalized to them.
CANONICAL_SOURCES = ("yfinance", "sec", "fred", "gdelt", "ir", "earnings")

# Map raw source strings produced by the stores / model citations to canonical
# names. ``sqlite`` (company fundamentals) and ``news``/``analysis`` (news-ish
# chroma docs) collapse to ``yfinance``; SEC filing variants collapse to ``sec``.
_SOURCE_ALIASES = {
    "yfinance": "yfinance",
    "yfinance_fundamentals": "yfinance",
    "yfinance_news": "yfinance",
    "estimates": "yfinance",
    "sqlite": "yfinance",
    "news": "yfinance",
    "analysis": "yfinance",
    "sec": "sec",
    "sec_10k": "sec", "sec_10q": "sec", "sec_filings": "sec",
    "fred": "fred",
    "gdelt": "gdelt",
    "ir": "ir", "ir_pages": "ir",
    "earnings": "earnings",
    "earnings_call": "earnings",
    "earnings_transcripts": "earnings",
    "transcript": "earnings",
}

# The keys every result row MUST have (used by tests + score.py).
#
# 2.2.1.3 adds the conversational/compound fields: conversation_id/turn_index
# (None for single-turn rows), raw_question/retrieval_query (equal today — no
# rewrite step exists until 2.2.2), history_sent (turns the runner sent with
# this request), the expected vs. resolved ticker/metric/timeframe triples, the
# per-turn expected_carryover contract, and the compound-question subquestion
# ids + coverage. Every row carries all of them so single-turn and conversation
# artifacts share one shape.
RESULT_KEYS = (
    "id", "question", "answer", "detected_ticker", "detected_intent",
    "facts_used", "documents_used", "retrieved_sources", "context",
    "evidence_trace", "trace_complete", "answer_policy", "grounding",
    "dataset_digest", "latency_ms", "model_available", "error", "case",
    "conversation_id", "turn_index", "raw_question", "retrieval_query",
    "history_sent", "expected_tickers", "expected_metrics", "expected_timeframe",
    "expected_carryover", "resolved_tickers", "resolved_metrics",
    "resolved_timeframe", "subquestion_ids", "subquestion_coverage",
)

# Cached in-process Store/config (built once, reused across cases).
_cached_config = None
_cached_store = None


# ── Source normalization ───────────────────────────────────────────────

def normalize_source(source: Optional[str]) -> Optional[str]:
    """Map a raw source string to a canonical name, or None if unknown."""
    if not source:
        return None
    return _SOURCE_ALIASES.get(str(source).strip().lower())


def _extract_citation_types(answer: str) -> list[str]:
    """Pull [Source: type/...] source types out of a model answer."""
    if not answer:
        return []
    return [m.group(1).split("/")[0].strip()
            for m in re.finditer(r"\[Source:\s*([^\]]+)\]", answer, re.IGNORECASE)]


# ── Dataset digest (2.2.1.2 Step 3) ────────────────────────────────────

def dataset_digest(cases: list[dict]) -> str:
    """SHA-256 over the committed golden-dataset cases used for a run.

    Standard library only. Deterministic — cases are serialized with sorted
    keys so field order in the source file never changes the digest.
    """
    blob = json.dumps(cases, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── Row construction ───────────────────────────────────────────────────

def _expected_tickers(case: dict) -> list[str]:
    """Ordered expected tickers for a case/turn.

    Prefers the multi-company ``expected_tickers`` list; falls back to the
    single-turn ``expected_ticker`` so one code path serves both fixture
    shapes. Upper-cased; empty for macro/no-ticker cases.
    """
    ex = case.get("expected_tickers")
    if ex:
        return [str(t).strip().upper() for t in ex if t]
    t = case.get("expected_ticker")
    return [str(t).strip().upper()] if t else []


def _subquestion_fields(case: dict, answer: str) -> tuple[list, Optional[float]]:
    """Return (subquestion ids, coverage) for a compound case.

    Coverage is the deterministic fraction of subquestions whose ``must_mention``
    terms all appear in the answer (see ``metrics.subquestions_addressed``).
    ``(None, None)`` for a case with no subquestions.
    """
    subs = case.get("subquestions") or []
    if not subs:
        return [], None
    # Lazy import: metrics imports run_eval at module load, so importing it here
    # (at call time) avoids a circular import while reusing the one definition.
    from eval.metrics import subquestions_addressed

    addressed = subquestions_addressed(subs, answer or "")
    ids = [s.get("id") for s in subs]
    return ids, round(len(addressed) / len(subs), 4)


def _row(case: dict, *, answer: str, detected_ticker, detected_intent,
         facts_used, documents_used, retrieved_sources, context: str = "",
         evidence_trace: Optional[dict] = None, trace_complete: bool = True,
         answer_policy: Optional[str] = None, grounding: Optional[str] = None,
         dataset_digest: Optional[str] = None,
         latency_ms, model_available: bool, error: Optional[str] = None,
         conversation_id: Optional[str] = None, turn_index: Optional[int] = None,
         history_sent: int = 0, retrieval_query: Optional[str] = None,
         resolved_tickers: Optional[list] = None,
         resolved_metrics: Optional[list] = None,
         resolved_timeframe: Optional[str] = None) -> dict:
    """Assemble a well-formed result row (always has every RESULT_KEY)."""
    ans = answer or ""
    sq_ids, sq_cov = _subquestion_fields(case, ans)
    question = case.get("question", "")
    return {
        "id": case["id"],
        "question": question,
        "answer": ans,
        "detected_ticker": detected_ticker,
        "detected_intent": detected_intent,
        "facts_used": int(facts_used or 0),
        "documents_used": int(documents_used or 0),
        "retrieved_sources": sorted({s for s in (retrieved_sources or []) if s}),
        "context": context or "",
        "evidence_trace": evidence_trace,
        "trace_complete": bool(trace_complete),
        "answer_policy": answer_policy,
        "grounding": grounding,
        "dataset_digest": dataset_digest,
        "latency_ms": round(float(latency_ms or 0), 1),
        "model_available": bool(model_available),
        "error": error,
        "case": case,
        # ── Conversational / compound fields (2.2.1.3) ──
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "raw_question": question,
        "retrieval_query": retrieval_query if retrieval_query is not None else question,
        "history_sent": int(history_sent or 0),
        "expected_tickers": _expected_tickers(case),
        "expected_metrics": [str(m) for m in (case.get("expected_metrics") or [])],
        "expected_timeframe": case.get("expected_timeframe"),
        "expected_carryover": case.get("expected_carryover"),
        "resolved_tickers": list(resolved_tickers or []),
        "resolved_metrics": list(resolved_metrics or []),
        "resolved_timeframe": resolved_timeframe,
        "subquestion_ids": sq_ids,
        "subquestion_coverage": sq_cov,
    }


# ── Resolved-field derivation (2.2.1.3) ────────────────────────────────

def _resolved_tickers(data: dict, trace: Optional[dict]) -> list[str]:
    """Tickers the system actually resolved for a turn.

    Prefers an explicit ``resolved_tickers`` field on the response (which the
    middleware will populate once 2.2.2 lands and which tests inject directly).
    Until then it derives a best-effort set from the detected ticker plus any
    ticker present in the evidence trace, so a live run still yields a value.
    """
    explicit = data.get("resolved_tickers")
    if explicit is not None:
        return sorted({str(t).strip().upper() for t in explicit if t})
    out: set[str] = set()
    dt = data.get("detected_ticker")
    if dt:
        out.add(str(dt).strip().upper())
    if isinstance(trace, dict):
        for f in trace.get("facts") or []:
            t = f.get("ticker")
            if t and str(t).strip().upper() not in ("MACRO", "", "?"):
                out.add(str(t).strip().upper())
        for d in trace.get("documents") or []:
            meta = d.get("metadata", {}) or {}
            t = meta.get("ticker")
            if t and str(t).strip().upper() not in ("MACRO", "", "?"):
                out.add(str(t).strip().upper())
    return sorted(out)


def _resolved_metrics(data: dict, trace: Optional[dict]) -> list[str]:
    """Metrics the system actually surfaced (explicit field, else trace facts)."""
    explicit = data.get("resolved_metrics")
    if explicit is not None:
        return sorted({str(m).strip().lower() for m in explicit if m})
    out: set[str] = set()
    if isinstance(trace, dict):
        for f in trace.get("facts") or []:
            m = f.get("metric")
            if m:
                out.add(str(m).strip().lower())
    return sorted(out)


def _resolved_timeframe(data: dict, trace: Optional[dict]) -> Optional[str]:
    """Timeframe the system actually resolved (explicit field, else a trace period)."""
    explicit = data.get("resolved_timeframe")
    if explicit:
        return str(explicit)
    if isinstance(trace, dict):
        for f in trace.get("facts") or []:
            period = f.get("period")
            if period:
                return str(period)
    return None


def _trace_is_complete(trace: Optional[dict]) -> bool:
    """Structural completeness check mirroring EvidenceTrace.is_complete()."""
    if not isinstance(trace, dict):
        return False
    return bool(trace.get("system_prompt")) and bool(trace.get("user_prompt"))


def _format_trace_context(trace: Optional[dict]) -> str:
    """Render an evidence trace's facts/documents/tool_results as a compact
    display string. Untruncated (2.2.1.2) — kept on the row for readability/
    tooling; the LLM-judge reads the trace fields directly, not this string.
    """
    if not isinstance(trace, dict):
        return ""
    parts: list[str] = []
    facts = trace.get("facts") or []
    if facts:
        parts.append("Facts:")
        for f in facts:
            parts.append(f"- {f.get('metric')}: {f.get('value')} "
                         f"({f.get('period') or 'N/A'}, ticker={f.get('ticker')})")
    documents = trace.get("documents") or []
    if documents:
        from src.middleware.evidence import document_body

        parts.append("\nDocuments:")
        for d in documents:
            meta = d.get("metadata", {}) or {}
            parts.append(f"- [{meta.get('source', d.get('source', '?'))}/"
                         f"{meta.get('ticker', '?')}] {document_body(d)}")
    tool_results = trace.get("tool_results") or []
    if tool_results:
        parts.append("\nTool results:")
        for t in tool_results:
            parts.append(f"- {t.get('name')}({t.get('arguments')}) -> "
                         f"{json.dumps(t.get('result'))}")
    if not parts:
        return "(no evidence)"
    return "\n".join(parts)


def _sources_from_evidence(facts: list[dict], documents: list[dict],
                           tool_results: Optional[list[dict]] = None) -> list[str]:
    """Canonical sources surfaced by retrieval-shaped facts/documents, plus
    any source hints inside tool results. Shared by the live path (reading
    the endpoint's evidence trace) and the direct path (reading its own
    retrieval)."""
    sources: list[str] = []
    for f in facts or []:
        st = f.get("source_type") or ("fred" if f.get("ticker") == "MACRO" else "sqlite")
        n = normalize_source(st)
        if n:
            sources.append(n)
    for d in documents or []:
        meta = d.get("metadata", {}) or {}
        n = normalize_source(meta.get("source") or d.get("source"))
        if n:
            sources.append(n)
    for t in tool_results or []:
        result = t.get("result") if isinstance(t, dict) else None
        if not isinstance(result, dict):
            continue
        n = normalize_source(result.get("source") or result.get("source_type"))
        if n:
            sources.append(n)
        for row in result.get("results") or []:
            if isinstance(row, dict):
                n2 = normalize_source(row.get("source_type"))
                if n2:
                    sources.append(n2)
    return sources


def _sources_from_trace(trace: Optional[dict]) -> list[str]:
    if not isinstance(trace, dict):
        return []
    return _sources_from_evidence(
        trace.get("facts") or [], trace.get("documents") or [], trace.get("tool_results") or [])


def _ctx_kwargs(ctx: Optional[dict], data: dict, trace: Optional[dict]) -> dict:
    """Conversational row kwargs (2.2.1.3): conversation position, history
    count, and the resolved ticker/metric/timeframe triple derived from the
    response + trace. Shared by the endpoint and direct row builders."""
    ctx = ctx or {}
    return {
        "conversation_id": ctx.get("conversation_id"),
        "turn_index": ctx.get("turn_index"),
        "history_sent": ctx.get("history_sent", 0),
        "resolved_tickers": _resolved_tickers(data, trace),
        "resolved_metrics": _resolved_metrics(data, trace),
        "resolved_timeframe": _resolved_timeframe(data, trace),
    }


def _row_from_endpoint(case: dict, data: dict, latency_s: float,
                       *, dataset_digest: Optional[str] = None,
                       ctx: Optional[dict] = None) -> dict:
    """Shape a live /query (or injected) JSON response into a result row.

    ``retrieved_sources`` is derived entirely from the response's citations
    and its evidence trace (2.2.1.2) — no second retrieval is run.
    """
    trace = data.get("evidence_trace")
    model_available = bool(data.get("model_available", False))
    trace_complete = _trace_is_complete(trace) if model_available else True

    sources = [normalize_source(c.get("source_type"))
               for c in (data.get("citations") or [])
               if isinstance(c, dict)]
    sources.extend(_sources_from_trace(trace))
    sources = sorted({s for s in sources if s})

    error = None
    if model_available and not trace_complete:
        error = "incomplete evidence trace for model-produced answer"

    answer_policy = (trace or {}).get("answer_policy") if isinstance(trace, dict) else None

    return _row(
        case,
        answer=data.get("answer", ""),
        detected_ticker=data.get("detected_ticker"),
        detected_intent=data.get("detected_intent"),
        facts_used=data.get("facts_used", 0),
        documents_used=data.get("documents_used", 0),
        retrieved_sources=sources,
        context=_format_trace_context(trace),
        evidence_trace=trace,
        trace_complete=trace_complete,
        answer_policy=answer_policy,
        grounding=data.get("grounding"),
        dataset_digest=dataset_digest,
        latency_ms=latency_s * 1000,
        model_available=model_available,
        error=error,
        **_ctx_kwargs(ctx, data, trace),
    )


def _error_row(case: dict, err: BaseException, latency_s: float,
               *, live_err: Optional[BaseException] = None,
               dataset_digest: Optional[str] = None,
               ctx: Optional[dict] = None) -> dict:
    """A well-formed row for when every backend failed."""
    parts = []
    if live_err:
        parts.append(str(live_err))
    if err:
        parts.append(str(err))
    msg = "; ".join(parts) or "all backends failed"
    ctx = ctx or {}
    return _row(
        case,
        answer="",
        detected_ticker=case.get("expected_ticker"),
        detected_intent=None,
        facts_used=0,
        documents_used=0,
        retrieved_sources=[],
        context="",
        evidence_trace=None,
        trace_complete=True,  # no model answer was produced — nothing to omit
        answer_policy=None,
        grounding=None,
        dataset_digest=dataset_digest,
        latency_ms=latency_s * 1000,
        model_available=False,
        error=msg,
        conversation_id=ctx.get("conversation_id"),
        turn_index=ctx.get("turn_index"),
        history_sent=ctx.get("history_sent", 0),
    )


# ── In-process retrieval (used by the direct/offline backend) ─────────

def _get_config_and_store():
    """Lazily build (and cache) a MiddlewareConfig + Store for in-process
    retrieval. Cached so the 42-case run doesn't reopen SQLite/Chroma per case."""
    global _cached_config, _cached_store
    if _cached_config is not None and _cached_store is not None:
        return _cached_config, _cached_store
    from src.middleware.config import MiddlewareConfig
    from src.storage.store import Store
    from src.utils.env import load_env
    load_env()
    _cached_config = MiddlewareConfig()
    _cached_store = Store(embedding_endpoint=_cached_config.embedding_endpoint)
    return _cached_config, _cached_store


def retrieve_for_question(question: str):
    """Run intent parsing + hybrid retrieval in-process.

    Returns (intent, retrieval, config). This is the direct/offline backend's
    only retrieval pass — it both builds the prompt and supplies the
    evidence trace, so results are never re-retrieved for judging.
    """
    from src.middleware.intent_parser import IntentParser
    from src.middleware.retriever import Retriever

    config, store = _get_config_and_store()
    intent = IntentParser().parse(question)
    retrieval = Retriever(store=store, config=config).retrieve(
        query=question, intent=intent,
        top_k_documents=config.top_k_documents,
        top_k_facts=config.top_k_facts,
    )
    return intent, retrieval, config


# ── Backend 1: live middleware ─────────────────────────────────────────

def call_live_endpoint(question: str, *, query_url: str = DEFAULT_QUERY_URL,
                        timeout: float = DEFAULT_TIMEOUT,
                        client=None, history: Optional[list] = None) -> dict:
    """POST /query to the live middleware. Raises on any failure.

    Always requests the evidence trace (2.2.1.2) so the runner never needs a
    second, truncated in-process retrieval to judge the answer.

    ``history`` (2.2.1.3) is the runner-owned list of completed
    ``{question, answer}`` turns for the current conversation. It is sent in a
    forward-compatible ``history`` field the middleware ignores today (pydantic
    ``extra="ignore"``) and will start honoring in 2.2.2.1 — the runner owns
    conversation state; the middleware stays stateless.
    """
    import httpx

    own = client is None
    c = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))
    body = {"question": question, "refresh": False, "include_evidence_trace": True}
    if history:
        body["history"] = [{"question": h.get("question", ""),
                            "answer": h.get("answer", "")} for h in history]
    try:
        resp = c.post(query_url, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    finally:
        if own:
            c.close()


# ── Backend 2: direct in-process pipeline ─────────────────────────────

def _model_reachable(llama_endpoint: str, timeout: float = 5.0) -> bool:
    import httpx
    try:
        r = httpx.get(llama_endpoint.replace("/v1/chat/completions", "/health"),
                      timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _direct_system_prompt(config, intent: Optional[dict], grounding_level: str) -> str:
    """Build the system prompt via the shared prompt_policy builder — the
    same function the live middleware's plain/streaming/tool-loop calls use
    (2.2.1.1) — so the direct-eval path's trace can never silently drift
    from live policy."""
    from src.middleware import prompt_policy

    return prompt_policy.build_system_prompt(
        answer_policy=str(getattr(config, "answer_policy", "graded") or "graded"),
        allow_general_fallback=bool(getattr(config, "allow_general_fallback", True)),
        intent=intent,
        grounding_level=grounding_level,
        tools_enabled=False,
    )


def _call_model_sync(config, prompt: str, *, intent: Optional[dict] = None,
                     grounding_level: str = "grounded") -> str:
    """Call the model synchronously via the OpenAI-compatible chat endpoint.

    Builds its system message through ``_direct_system_prompt`` — the same
    prompt_policy.build_system_prompt call the live middleware's plain/
    streaming/tool-loop calls use — so the direct-eval path can never
    silently drift from live policy (2.2.1.1).
    """
    import httpx

    system_prompt = _direct_system_prompt(config, intent, grounding_level)
    payload = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": config.default_temperature,
        "max_tokens": config.max_tokens,
    }
    r = httpx.post(config.llama_endpoint, json=payload,
                   timeout=httpx.Timeout(120.0, connect=10.0))
    r.raise_for_status()
    return r.json().get("choices", [{}])[0].get("message", {}).get("content", "")


def _degraded_answer(retrieval: dict, intent: dict) -> str:
    """Mirror the middleware's degraded answer format (model unavailable)."""
    parts = ["⚠️ Model unavailable — showing raw retrieved data:\n"]
    facts = retrieval.get("facts", [])
    docs = retrieval.get("documents", [])
    if facts:
        parts.append("**Structured Facts:**")
        for f in facts[:5]:
            parts.append(f"- {f.get('metric')}: {f.get('value')} "
                         f"({f.get('period', 'N/A')})")
        parts.append("")
    if docs:
        parts.append("**Relevant Documents:**")
        for d in docs[:3]:
            meta = d.get("metadata", {}) or {}
            parts.append(f"- {d.get('id', 'unknown')} "
                         f"({meta.get('source', d.get('source', 'unknown'))})")
        parts.append("")
    if not facts and not docs:
        parts.append("No stored data found for this question.")
        parts.append("")
    parts.append("Start llama-server to get AI-grounded answers.")
    return "\n".join(parts)


def _direct_grounding(answer: str, grounding_level: str, allow_general_fallback: bool) -> str:
    """Mirror app.py's ``_response_grounding`` for the direct/offline path.

    ``_is_declined_answer`` is a pure function (no module globals), so this
    is safe to import lazily without touching the live app's request state.
    """
    from src.middleware.app import _is_declined_answer

    if _is_declined_answer(answer):
        return "refused"
    if grounding_level in ("grounded", "partial"):
        return grounding_level
    if grounding_level == "none" and allow_general_fallback:
        return "general"
    return "refused"


def call_pipeline_direct(question: str) -> dict:
    """Run the query pipeline in-process. Raises on any failure.

    Builds an evidence trace (2.2.1.2) using the same evidence helpers and
    prompt-policy builder the live middleware uses, from the single
    retrieval pass that also builds the model prompt.
    """
    from src.middleware.evidence import evidence_counts, usable_documents, usable_facts
    from src.middleware.evidence_trace import EvidenceTrace
    from src.middleware.prompt_augmenter import PromptAugmenter

    intent, retrieval, config = retrieve_for_question(question)
    # Usable-evidence counts (2.2.1.1) — same contract app.py uses, so the
    # direct-eval path's facts_used/documents_used/grounding_level match
    # what a live /query response would report for the same retrieval.
    n_facts, n_docs = evidence_counts(retrieval)
    n_evidence = n_facts + n_docs
    grounding_level = "grounded" if n_evidence >= 3 else "partial" if n_evidence >= 1 else "none"
    prompt = PromptAugmenter(config=config).build_prompt(
        question=question, intent=intent, retrieval=retrieval,
        grounding_level=grounding_level,
    )
    answer_policy = str(getattr(config, "answer_policy", "graded") or "graded")
    allow_general_fallback = bool(getattr(config, "allow_general_fallback", True))

    model_available = _model_reachable(config.llama_endpoint)
    trace: Optional[EvidenceTrace] = None
    if model_available:
        try:
            answer = _call_model_sync(config, prompt, intent=intent,
                                      grounding_level=grounding_level)
            trace = EvidenceTrace(
                answer_policy=answer_policy,
                grounding_level=grounding_level,
                raw_question=question,
                retrieval_query=question,
                system_prompt=_direct_system_prompt(config, intent, grounding_level),
                user_prompt=prompt,
                facts=usable_facts(retrieval),
                documents=usable_documents(retrieval),
                tool_results=[],
            )
        except Exception:  # noqa: BLE001 - degrade on model error
            answer = _degraded_answer(retrieval, intent)
            model_available = False
    else:
        answer = _degraded_answer(retrieval, intent)

    grounding = (
        _direct_grounding(answer, grounding_level, allow_general_fallback)
        if model_available else None
    )

    sources = _sources_from_evidence(usable_facts(retrieval), usable_documents(retrieval))
    for ct in _extract_citation_types(answer):
        n = normalize_source(ct)
        if n:
            sources.append(n)

    return {
        "answer": answer,
        "detected_ticker": intent.get("ticker"),
        "detected_intent": intent.get("question_type"),
        "facts_used": n_facts,
        "documents_used": n_docs,
        "retrieved_sources": sources,
        "evidence_trace": trace.to_dict() if trace is not None else None,
        "answer_policy": answer_policy if model_available else None,
        "grounding": grounding,
        "model_available": model_available,
    }


def _row_from_direct(case: dict, data: dict, latency_s: float,
                     *, dataset_digest: Optional[str] = None,
                     ctx: Optional[dict] = None) -> dict:
    trace = data.get("evidence_trace")
    model_available = bool(data.get("model_available", False))
    trace_complete = _trace_is_complete(trace) if model_available else True
    error = "incomplete evidence trace for model-produced answer" \
        if model_available and not trace_complete else None
    return _row(
        case,
        answer=data.get("answer", ""),
        detected_ticker=data.get("detected_ticker"),
        detected_intent=data.get("detected_intent"),
        facts_used=data.get("facts_used", 0),
        documents_used=data.get("documents_used", 0),
        retrieved_sources=data.get("retrieved_sources", []),
        context=_format_trace_context(trace),
        evidence_trace=trace,
        trace_complete=trace_complete,
        answer_policy=data.get("answer_policy"),
        grounding=data.get("grounding"),
        dataset_digest=dataset_digest,
        latency_ms=latency_s * 1000,
        model_available=model_available,
        error=error,
        **_ctx_kwargs(ctx, data, trace),
    )


# ── Public runner API ───────────────────────────────────────────────────

def run_case(case: dict, *, query_fn: Optional[Callable] = None,
             query_url: str = DEFAULT_QUERY_URL,
             timeout: float = DEFAULT_TIMEOUT,
             use_live: bool = True, allow_direct: bool = True,
             dataset_digest: Optional[str] = None,
             history: Optional[list] = None,
             conversation_id: Optional[str] = None,
             turn_index: Optional[int] = None) -> dict:
    """Run one golden case (or conversation turn) through the pipeline.

    Args:
        case:           A golden-dataset case dict, or a synthesized turn case.
        query_fn:       Optional injected backend. For single-turn use it is called
                         ``query_fn(case) -> endpoint_json``; when ``history`` is
                         provided (conversation mode) it is called
                         ``query_fn(case, history)`` so tests can assert exactly
                         which prior turns each request saw.
        query_url:      Live middleware /query URL.
        timeout:        Per-request timeout (seconds).
        use_live:       Try the live middleware first.
        allow_direct:   Fall back to the in-process pipeline if the live path fails.
        dataset_digest: SHA-256 of the golden dataset used for this run (denormalized
                        onto every row so score.py/gate.py can read it without the file).
        history:        Runner-owned completed ``{question, answer}`` turns for the
                         current conversation (``None`` for a single-turn case, an
                         empty list at the first turn of a conversation).
        conversation_id/turn_index: Position of this turn (``None`` for single-turn).

    Returns a dict with every key in ``RESULT_KEYS`` (never raises).
    """
    t0 = time.time()
    ctx = {
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "history_sent": len(history) if history is not None else 0,
    }

    def _call_injected():
        return query_fn(case, history) if history is not None else query_fn(case)

    # 1. Injected backend (tests).
    if query_fn is not None:
        try:
            data = _call_injected()
            return _row_from_endpoint(case, data, time.time() - t0,
                                      dataset_digest=dataset_digest, ctx=ctx)
        except Exception as e:  # noqa: BLE001
            if not allow_direct:
                return _error_row(case, e, time.time() - t0,
                                  dataset_digest=dataset_digest, ctx=ctx)
            try:
                return _row_from_direct(case, call_pipeline_direct(case["question"]),
                                         time.time() - t0, dataset_digest=dataset_digest, ctx=ctx)
            except Exception as e2:  # noqa: BLE001
                return _error_row(case, e2, time.time() - t0, live_err=e,
                                  dataset_digest=dataset_digest, ctx=ctx)

    # 2. Live middleware.
    live_err: Optional[BaseException] = None
    if use_live:
        try:
            data = call_live_endpoint(case["question"], query_url=query_url,
                                       timeout=timeout, history=history)
            return _row_from_endpoint(case, data, time.time() - t0,
                                      dataset_digest=dataset_digest, ctx=ctx)
        except Exception as e:  # noqa: BLE001
            live_err = e

    # 3. Direct in-process pipeline.
    if allow_direct:
        try:
            return _row_from_direct(case, call_pipeline_direct(case["question"]),
                                     time.time() - t0, dataset_digest=dataset_digest, ctx=ctx)
        except Exception as e:  # noqa: BLE001
            return _error_row(case, e, time.time() - t0, live_err=live_err,
                              dataset_digest=dataset_digest, ctx=ctx)

    # 4. Nothing left.
    return _error_row(case, live_err or RuntimeError("no backend available"),
                      time.time() - t0, dataset_digest=dataset_digest, ctx=ctx)


# ── Conversations (2.2.1.3) ─────────────────────────────────────────────

def _turn_case(conv: dict, turn: dict, index: int, category: str) -> dict:
    """Synthesize a golden-case dict for one conversation turn.

    Carries the turn's own ``expected_*`` fields plus stable id / conversation
    position, so ``metrics._case(row)`` sees the same fields it would for a
    single-turn case.
    """
    case = dict(turn)
    case.setdefault("id", f"{conv['id']}#{index}")
    case["question"] = turn.get("question", "")
    case.setdefault("category", category)
    case["conversation_id"] = conv["id"]
    case["turn_index"] = index
    return case


def run_conversation(conv: dict, *, query_fn: Optional[Callable] = None,
                     query_url: str = DEFAULT_QUERY_URL,
                     timeout: float = DEFAULT_TIMEOUT,
                     use_live: bool = True, allow_direct: bool = True,
                     dataset_digest: Optional[str] = None) -> list[dict]:
    """Run one conversation's turns sequentially, runner-owned history.

    History starts empty and accumulates the completed ``{question, answer}``
    of each turn before the next request; it is never shared across
    conversations (each call to this function owns its own list), which is what
    keeps two interleaved conversations from ever seeing each other's context.
    Returns one result row per turn.
    """
    conv_id = conv["id"]
    category = conv.get("category", "conversation")
    history: list[dict] = []
    rows: list[dict] = []
    for i, turn in enumerate(conv.get("turns") or []):
        turn_case = _turn_case(conv, turn, i, category)
        row = run_case(turn_case, query_fn=query_fn, query_url=query_url,
                       timeout=timeout, use_live=use_live, allow_direct=allow_direct,
                       dataset_digest=dataset_digest, history=list(history),
                       conversation_id=conv_id, turn_index=i)
        rows.append(row)
        history.append({"question": turn.get("question", ""),
                        "answer": row.get("answer", "")})
    return rows


# ── Dataset + run artifact IO ───────────────────────────────────────────

def load_cases(path: Path = GOLDEN) -> list[dict]:
    """Load the golden dataset, validating each line parses."""
    cases: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        cases.append(json.loads(line))
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def load_conversations(path: Path = CONVERSATIONS) -> list[dict]:
    """Load the conversation golden set (one conversation per line).

    Returns ``[]`` when the file is absent so a single-turn-only run still
    works. Each conversation is ``{id, category, turns:[{...}]}``.
    """
    if not path.exists():
        return []
    convs: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            convs.append(json.loads(line))
    return convs


# ── Fixture validation report (2.2.1.3 Step 2) ─────────────────────────

# Minimum coverage the challenge set must hit, per dimension.
FIXTURE_MINIMUMS = {
    "conversations": 12,
    "conversation_turns": 30,
    "paraphrase_pairs": 6,
    "compound": 6,
    "multi_ticker": 6,
    "timeframe": 6,
    "stale": 4,
    "unanswerable": 4,
}


def fixture_counts(cases: list[dict], conversations: list[dict]) -> dict:
    """Count coverage per challenge dimension (single-turn + conversation).

    Counted deterministically so the validation report can prove no single
    large bucket hides a missing dimension. Single-turn dimensions are keyed by
    ``category``; conversation stats come from the conversations file.
    """
    def cat(name: str) -> int:
        return sum(1 for c in cases if c.get("category") == name)

    return {
        "single_turn_cases": len(cases),
        "conversations": len(conversations),
        "conversation_turns": sum(len(cv.get("turns") or []) for cv in conversations),
        "paraphrase_pairs": len({c.get("paraphrase_group")
                                 for c in cases if c.get("paraphrase_group")}),
        "compound": cat("compound"),
        "multi_ticker": cat("multi_ticker"),
        "timeframe": cat("timeframe"),
        "stale": cat("stale"),
        "unanswerable": cat("unanswerable"),
    }


def fixture_id_collisions(cases: list[dict], conversations: list[dict]) -> list[str]:
    """Return any ids that are not unique across cases, conversations, and turns."""
    seen: dict[str, int] = {}
    ids: list[str] = [c["id"] for c in cases]
    for cv in conversations:
        ids.append(cv["id"])
        ids.extend(t["id"] for t in (cv.get("turns") or []) if "id" in t)
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
    return sorted(i for i, n in seen.items() if n > 1)


def validate_fixtures(cases: list[dict],
                      conversations: list[dict]) -> tuple[dict, list[str]]:
    """Return (counts, problems). Problems list unmet minimums + id collisions."""
    counts = fixture_counts(cases, conversations)
    problems: list[str] = []
    for dim, minimum in FIXTURE_MINIMUMS.items():
        if counts.get(dim, 0) < minimum:
            problems.append(f"{dim}: {counts.get(dim, 0)} < required {minimum}")
    collisions = fixture_id_collisions(cases, conversations)
    if collisions:
        problems.append(f"duplicate fixture/turn ids: {collisions}")
    return counts, problems


def print_fixture_report(cases: list[dict], conversations: list[dict]) -> bool:
    """Print the per-dimension coverage report. Returns True when all clear."""
    counts, problems = validate_fixtures(cases, conversations)
    print("=== Golden fixture coverage (2.2.1.3) ===")
    for dim in ("single_turn_cases", "conversations", "conversation_turns",
                "paraphrase_pairs", "compound", "multi_ticker", "timeframe",
                "stale", "unanswerable"):
        minimum = FIXTURE_MINIMUMS.get(dim)
        bar = f"  (min {minimum})" if minimum else ""
        print(f"  {dim:<22} {counts[dim]}{bar}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return False
    print("\nAll dimensions meet their minimums.")
    return True


def latest_run_path(runs_dir: Path = RUNS_DIR) -> Optional[Path]:
    """Most recent ``<ts>.jsonl`` run artifact (excludes .summary/.report)."""
    if not runs_dir.exists():
        return None
    runs = sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def write_run(rows: list[dict], runs_dir: Path = RUNS_DIR,
              timestamp: Optional[int] = None) -> Path:
    """Persist the raw run artifact. Returns the written path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp if timestamp is not None else int(time.time())
    out = runs_dir / f"{ts}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
                   encoding="utf-8")
    return out


def summarize_rows(rows: list[dict]) -> dict:
    """Lightweight run meta (full metrics live in score.py)."""
    n = len(rows)
    return {
        "n_cases": n,
        "n_model_available": sum(1 for r in rows if r.get("model_available")),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "avg_latency_ms": round(
            sum(r.get("latency_ms", 0) for r in rows) / n, 1) if n else 0.0,
        "answer_rate": round(
            sum(1 for r in rows if r.get("answer")) / n, 3) if n else 0.0,
    }


# ── CLI ────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the golden dataset through the pipeline.")
    p.add_argument("--golden", type=Path, default=GOLDEN, help="Golden dataset path.")
    p.add_argument("--conversations", type=Path, default=CONVERSATIONS,
                   help="Conversation golden set path.")
    p.add_argument("--query-url", default=DEFAULT_QUERY_URL,
                   help="Live middleware /query URL.")
    p.add_argument("--offline", action="store_true",
                   help="Skip the live endpoint; use the direct pipeline only.")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N single-turn cases.")
    p.add_argument("--no-conversations", action="store_true",
                   help="Skip the conversation golden set (single-turn cases only).")
    p.add_argument("--validate-fixtures", action="store_true",
                   help="Print the per-dimension fixture coverage report and exit.")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="Per-case timeout (seconds).")
    args = p.parse_args(argv)

    all_cases = load_cases(args.golden)
    conversations = load_conversations(args.conversations)

    if args.validate_fixtures:
        return 0 if print_fixture_report(all_cases, conversations) else 1

    # The run-level digest covers both golden inputs (2.2.1.3) so the gate's
    # dataset-digest check fails if either file changes since the baseline.
    convs = [] if args.no_conversations else conversations
    digest = dataset_digest(all_cases + convs)
    cases = all_cases[: args.limit] if args.limit else all_cases

    backend = "direct" if args.offline else "live+direct"
    print(f"Running {len(cases)} cases + {len(convs)} conversations "
          f"via {backend} (url={args.query_url}) ...")

    rows = [
        run_case(c, query_url=args.query_url, timeout=args.timeout,
                 use_live=not args.offline, allow_direct=True,
                 dataset_digest=digest)
        for c in cases
    ]
    for conv in convs:
        rows.extend(run_conversation(
            conv, query_url=args.query_url, timeout=args.timeout,
            use_live=not args.offline, allow_direct=True, dataset_digest=digest))

    out = write_run(rows)
    meta = summarize_rows(rows)
    print(f"\nWrote {len(rows)} results -> {out}")
    print(f"  model_available: {meta['n_model_available']}/{meta['n_cases']}")
    print(f"  errors:         {meta['n_errors']}")
    print(f"  avg latency:    {meta['avg_latency_ms']} ms")
    print(f"  answer_rate:    {meta['answer_rate']}")
    print("Next: python eval/score.py  &&  python eval/gate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
