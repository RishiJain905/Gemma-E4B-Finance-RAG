"""
eval/run_eval.py — Stage 1 of the evaluation harness: the runner.

Drives every golden case through the query pipeline and captures, per case:
  answer, detected_ticker, detected_intent, facts_used, documents_used,
  retrieved_sources, context (for the LLM-judge), latency_ms, model_available.

Backend preference (per the 2.1.1.1 spec):
  1. Live middleware  — POST /query on :8000 (measures the real system).
  2. Direct pipeline  — import IntentParser/Retriever/PromptAugmenter in-process
                         (used when the middleware is down but the model/store are up).
  3. Error row         — if both fail, emit a well-formed row with model_available=False
                         so scoring never crashes on a half-dead environment.

The live /query endpoint returns counts + citations but not the retrieved
context, so for the LLM-judge (2.1.1.2) the runner captures a compact
``context`` string via one in-process retrieval — the same Store/Retriever/
IntentParser the middleware uses, so it is faithful to what the middleware
retrieved. The run artifact is thus self-contained: score.py / the judge need
no store at scoring time.

The runner is import-safe and testable: ``run_case`` accepts an injectable
``query_fn`` and flags to force a backend, so tests never need the network.

Usage:
    python eval/run_eval.py                 # live :8000, fall back to direct
    python eval/run_eval.py --offline      # skip the live endpoint, direct only
    python eval/run_eval.py --limit 5      # just the first 5 cases
"""

from __future__ import annotations

import argparse
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
RESULT_KEYS = (
    "id", "question", "answer", "detected_ticker", "detected_intent",
    "facts_used", "documents_used", "retrieved_sources", "context",
    "latency_ms", "model_available", "error", "case",
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


# ── Row construction ───────────────────────────────────────────────────

def _row(case: dict, *, answer: str, detected_ticker, detected_intent,
         facts_used, documents_used, retrieved_sources, context: str,
         latency_ms, model_available: bool, error: Optional[str] = None) -> dict:
    """Assemble a well-formed result row (always has every RESULT_KEY)."""
    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer or "",
        "detected_ticker": detected_ticker,
        "detected_intent": detected_intent,
        "facts_used": int(facts_used or 0),
        "documents_used": int(documents_used or 0),
        "retrieved_sources": sorted({s for s in (retrieved_sources or []) if s}),
        "context": context or "",
        "latency_ms": round(float(latency_ms or 0), 1),
        "model_available": bool(model_available),
        "error": error,
        "case": case,
    }


def _row_from_endpoint(case: dict, data: dict, latency_s: float,
                       context: str = "", extra_sources: Optional[list] = None) -> dict:
    """Shape a live /query (or injected) JSON response into a result row.

    ``extra_sources`` (from the in-process context-retrieval pass) are merged
    with citation sources so ``retrieved_sources`` reflects what was actually
    retrieved, not only what the model chose to cite.
    """
    sources = [normalize_source(c.get("source_type"))
               for c in (data.get("citations") or [])
               if isinstance(c, dict)]
    for s in extra_sources or []:
        if s:
            sources.append(s)
    sources = sorted({s for s in sources if s})
    return _row(
        case,
        answer=data.get("answer", ""),
        detected_ticker=data.get("detected_ticker"),
        detected_intent=data.get("detected_intent"),
        facts_used=data.get("facts_used", 0),
        documents_used=data.get("documents_used", 0),
        retrieved_sources=sources,
        context=context or data.get("context", ""),
        latency_ms=latency_s * 1000,
        model_available=data.get("model_available", False),
    )


def _error_row(case: dict, err: BaseException, latency_s: float,
               *, live_err: Optional[BaseException] = None) -> dict:
    """A well-formed row for when every backend failed."""
    parts = []
    if live_err:
        parts.append(str(live_err))
    if err:
        parts.append(str(err))
    msg = "; ".join(parts) or "all backends failed"
    return _row(
        case,
        answer="",
        detected_ticker=case.get("expected_ticker"),
        detected_intent=None,
        facts_used=0,
        documents_used=0,
        retrieved_sources=[],
        context="",
        latency_ms=latency_s * 1000,
        model_available=False,
        error=msg,
    )


# ── In-process retrieval (shared by the direct path + live context capture) ─

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

    Returns (intent, retrieval). Shared by the direct pipeline path and by the
    live path's context capture so both use identical retrieval logic.
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


def _format_context(retrieval: dict, intent: dict, max_facts: int = 8,
                    max_docs: int = 3, doc_chars: int = 600) -> str:
    """Render retrieved facts + document excerpts as a compact context string.

    This is what the LLM-judge reads to score faithfulness. Capped so the run
    artifact stays small.
    """
    parts: list[str] = []
    facts = retrieval.get("facts", [])
    if facts:
        parts.append("Facts:")
        for f in facts[:max_facts]:
            parts.append(f"- {f.get('metric')}: {f.get('value')} "
                         f"({f.get('period') or 'N/A'}, ticker={f.get('ticker')})")
    docs = retrieval.get("documents", [])
    if docs:
        parts.append("\nDocuments:")
        for d in docs[:max_docs]:
            meta = d.get("metadata", {}) or {}
            text = (d.get("text") or d.get("content") or "").strip()
            text = text[:doc_chars] + ("…" if len(text) > doc_chars else "")
            parts.append(f"- [{meta.get('source', d.get('source', '?'))}/"
                         f"{meta.get('ticker', '?')}] {text}")
    if not parts:
        return "(no retrieved context)"
    return "\n".join(parts)


def _sources_from_retrieval(retrieval: dict) -> list[str]:
    """Canonical sources actually surfaced by a retrieval dict."""
    sources: list[str] = []
    for f in retrieval.get("facts", []):
        st = f.get("source_type") or ("fred" if f.get("ticker") == "MACRO" else "sqlite")
        n = normalize_source(st)
        if n:
            sources.append(n)
    for d in retrieval.get("documents", []):
        meta = d.get("metadata", {}) or {}
        n = normalize_source(meta.get("source") or d.get("source"))
        if n:
            sources.append(n)
    return sources


# ── Backend 1: live middleware ─────────────────────────────────────────

def call_live_endpoint(question: str, *, query_url: str = DEFAULT_QUERY_URL,
                        timeout: float = DEFAULT_TIMEOUT,
                        client=None) -> dict:
    """POST /query to the live middleware. Raises on any failure."""
    import httpx

    own = client is None
    c = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))
    try:
        resp = c.post(query_url, json={"question": question, "refresh": False},
                      timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    finally:
        if own:
            c.close()


def _capture_context_and_sources(question: str) -> tuple[str, list[str]]:
    """One in-process retrieval for a live-path row, returning (context, sources).

    Faithful to the middleware: the same Store/Retriever/IntentParser, so the
    retrieved sources here are what the middleware retrieved (the model may or
    may not cite them). Best-effort: returns ("", []) on any failure.
    """
    try:
        intent, retrieval, _cfg = retrieve_for_question(question)
        return _format_context(retrieval, intent), _sources_from_retrieval(retrieval)
    except Exception:  # noqa: BLE001 - context is best-effort; judge still runs
        return "", []


# ── Backend 2: direct in-process pipeline ─────────────────────────────

def _model_reachable(llama_endpoint: str, timeout: float = 5.0) -> bool:
    import httpx
    try:
        r = httpx.get(llama_endpoint.replace("/v1/chat/completions", "/health"),
                      timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _call_model_sync(config, prompt: str) -> str:
    """Call the model synchronously via the OpenAI-compatible chat endpoint."""
    import httpx
    from src.middleware.app import SYSTEM_PROMPT

    payload = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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


def call_pipeline_direct(question: str) -> dict:
    """Run the query pipeline in-process. Raises on any failure.

    Uses one retrieval pass for both the model prompt and the captured context.
    """
    from src.middleware.prompt_augmenter import PromptAugmenter

    intent, retrieval, config = retrieve_for_question(question)
    prompt = PromptAugmenter(config=config).build_prompt(
        question=question, intent=intent, retrieval=retrieval,
    )

    model_available = _model_reachable(config.llama_endpoint)
    if model_available:
        try:
            answer = _call_model_sync(config, prompt)
        except Exception:  # noqa: BLE001 - degrade on model error
            answer = _degraded_answer(retrieval, intent)
            model_available = False
    else:
        answer = _degraded_answer(retrieval, intent)

    sources = _sources_from_retrieval(retrieval)
    for ct in _extract_citation_types(answer):
        n = normalize_source(ct)
        if n:
            sources.append(n)

    return {
        "answer": answer,
        "detected_ticker": intent.get("ticker"),
        "detected_intent": intent.get("question_type"),
        "facts_used": len(retrieval.get("facts", [])),
        "documents_used": len(retrieval.get("documents", [])),
        "retrieved_sources": sources,
        "context": _format_context(retrieval, intent),
        "model_available": model_available,
    }


def _row_from_direct(case: dict, data: dict, latency_s: float) -> dict:
    return _row(
        case,
        answer=data.get("answer", ""),
        detected_ticker=data.get("detected_ticker"),
        detected_intent=data.get("detected_intent"),
        facts_used=data.get("facts_used", 0),
        documents_used=data.get("documents_used", 0),
        retrieved_sources=data.get("retrieved_sources", []),
        context=data.get("context", ""),
        latency_ms=latency_s * 1000,
        model_available=data.get("model_available", False),
    )


# ── Public runner API ───────────────────────────────────────────────────

def run_case(case: dict, *, query_fn: Optional[Callable] = None,
             query_url: str = DEFAULT_QUERY_URL,
             timeout: float = DEFAULT_TIMEOUT,
             use_live: bool = True, allow_direct: bool = True,
             capture_context: bool = True) -> dict:
    """Run one golden case through the pipeline and return a result row.

    Args:
        case:           A golden-dataset case dict.
        query_fn:       Optional callable ``(case) -> endpoint_json``. When given
                         it replaces the live endpoint (tests / programmatic use).
        query_url:      Live middleware /query URL.
        timeout:        Per-request timeout (seconds).
        use_live:       Try the live middleware first.
        allow_direct:   Fall back to the in-process pipeline if the live path fails.
        capture_context: For the live path, capture retrieved context via one
                         in-process retrieval (for the LLM-judge).

    Returns a dict with every key in ``RESULT_KEYS`` (never raises).
    """
    t0 = time.time()

    # 1. Injected backend (tests).
    if query_fn is not None:
        try:
            data = query_fn(case)
            ctx, extra = ("", [])
            if capture_context and not data.get("context"):
                ctx, extra = _capture_context_and_sources(case["question"])
            return _row_from_endpoint(case, data, time.time() - t0,
                                       context=ctx, extra_sources=extra)
        except Exception as e:  # noqa: BLE001
            if not allow_direct:
                return _error_row(case, e, time.time() - t0)
            try:
                return _row_from_direct(case, call_pipeline_direct(case["question"]),
                                         time.time() - t0)
            except Exception as e2:  # noqa: BLE001
                return _error_row(case, e2, time.time() - t0, live_err=e)

    # 2. Live middleware.
    live_err: Optional[BaseException] = None
    if use_live:
        try:
            data = call_live_endpoint(case["question"], query_url=query_url,
                                       timeout=timeout)
            ctx, extra = ("", [])
            if capture_context and not data.get("context"):
                ctx, extra = _capture_context_and_sources(case["question"])
            return _row_from_endpoint(case, data, time.time() - t0,
                                       context=ctx, extra_sources=extra)
        except Exception as e:  # noqa: BLE001
            live_err = e

    # 3. Direct in-process pipeline.
    if allow_direct:
        try:
            return _row_from_direct(case, call_pipeline_direct(case["question"]),
                                     time.time() - t0)
        except Exception as e:  # noqa: BLE001
            return _error_row(case, e, time.time() - t0, live_err=live_err)

    # 4. Nothing left.
    return _error_row(case, live_err or RuntimeError("no backend available"),
                      time.time() - t0)


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
    p.add_argument("--query-url", default=DEFAULT_QUERY_URL,
                   help="Live middleware /query URL.")
    p.add_argument("--offline", action="store_true",
                   help="Skip the live endpoint; use the direct pipeline only.")
    p.add_argument("--no-context", action="store_true",
                   help="Skip captured context (faster; the LLM-judge will be skipped).")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N cases.")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="Per-case timeout (seconds).")
    args = p.parse_args(argv)

    cases = load_cases(args.golden)
    if args.limit:
        cases = cases[: args.limit]

    backend = "direct" if args.offline else "live+direct"
    print(f"Running {len(cases)} cases via {backend} (url={args.query_url}) ...")

    rows = [
        run_case(c, query_url=args.query_url, timeout=args.timeout,
                 use_live=not args.offline, allow_direct=True,
                 capture_context=not args.no_context)
        for c in cases
    ]

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
