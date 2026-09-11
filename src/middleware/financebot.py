"""
src/middleware/financebot.py
FinanceBot-facing RAG adapter — hybrid retrieval with an explicit hit/miss contract.

FinanceBot (external assistant) treats this RAG as source of truth. The adapter
runs hybrid retrieval only (no local chat-model generation) and returns a
structured status so the caller can fall back to open-web search on misses.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from .deterministic_router import is_trade_bias_question
from .intent_parser import IntentParser
from .tools import REGISTRY, ToolContext, dispatch_named_tool

logger = logging.getLogger(__name__)

REQUIRED_RAG_TOOLS = frozenset({
    "list_metrics",
    "query_facts",
    "get_fundamentals",
    "search_documents",
    "get_macro_snapshot",
    "get_sentiment",
    "get_guidance",
    "get_estimates",
    "get_price_targets",
    "classify_trade_bias",
    "check_freshness",
    "refresh_data",
    "describe_coverage",
})

RagStatus = Literal["hit", "miss"]

_SCORE_KEYS = (
    "rerank_score",
    "fusion_score",
    "fused_score",
    "score",
    "relevance_score",
    "distance",
)


def document_score(doc: dict) -> float | None:
    """Best available ranking score for a retrieved document.

    Higher is better. Chroma ``distance`` (lower-is-better) is converted to a
    descending score via ``1 / (1 + distance)``.
    """
    if not isinstance(doc, dict):
        return None
    for key in _SCORE_KEYS:
        if key not in doc or doc[key] is None:
            continue
        try:
            value = float(doc[key])
        except (TypeError, ValueError):
            continue
        if key == "distance":
            return 1.0 / (1.0 + max(value, 0.0))
        return value
    return None


def classify_rag_hit(
    facts: list[dict],
    documents: list[dict],
    *,
    min_facts: int = 1,
    min_documents: int = 1,
    min_document_score: float = 0.0,
) -> tuple[RagStatus, list[dict], float | None]:
    """Decide hit vs miss for FinanceBot's RAG-first contract.

    A hit requires at least ``min_facts`` structured facts **or** at least
    ``min_documents`` documents whose score meets ``min_document_score``.
    Documents without a score are treated as score ``0.0`` so a populated
    corpus still counts as a hit when the threshold is 0.

    Returns ``(status, qualifying_documents, top_score)``.
    """
    facts = [f for f in (facts or []) if isinstance(f, dict)]
    documents = [d for d in (documents or []) if isinstance(d, dict)]

    scored: list[tuple[float | None, dict]] = []
    for doc in documents:
        score = document_score(doc)
        effective = 0.0 if score is None else score
        if effective >= min_document_score:
            scored.append((score, doc))

    top_score: float | None = None
    if scored:
        numeric = [s for s, _ in scored if s is not None]
        if numeric:
            top_score = max(numeric)

    qualifying = [doc for _, doc in scored]
    fact_hit = len(facts) >= max(1, int(min_facts))
    doc_hit = len(qualifying) >= max(1, int(min_documents))

    if fact_hit or doc_hit:
        return "hit", qualifying, top_score
    return "miss", [], top_score


def build_financebot_payload(
    *,
    query: str,
    facts: list[dict],
    documents: list[dict],
    ticker: str | None,
    status: RagStatus,
    qualifying_documents: list[dict],
    top_score: float | None,
    retrieval_strategy: str | None = None,
    trade_bias: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the FinanceBot response body (shared by the HTTP adapter)."""
    is_hit = status == "hit"
    evidence_docs = qualifying_documents if is_hit else []
    evidence_facts = facts if is_hit else []
    bias_hit = (
        isinstance(trade_bias, dict)
        and trade_bias.get("evidence_status") == "hit"
        and not trade_bias.get("error")
    )
    bias_miss = isinstance(trade_bias, dict) and not bias_hit
    if bias_hit:
        is_hit = True
        evidence_facts = facts
        evidence_docs = qualifying_documents
    web_search_allowed = (not is_hit) or bias_miss
    if bias_hit:
        message = (
            "RAG trade bias hit — you MUST answer long or short from "
            "trade_bias.bias; do not guess and do not use open web."
        )
    elif bias_miss:
        message = (
            "classify_trade_bias miss — no indexed long/short evidence; "
            "open-web fallback is allowed. Do not invent a directional call."
        )
    elif is_hit:
        message = (
            "RAG hit — answer from returned facts/documents only; do not use open web."
        )
    else:
        message = (
            "RAG miss — no relevant indexed evidence; open-web fallback is allowed."
        )
    return {
        "status": "hit" if is_hit else "miss",
        "web_search_allowed": web_search_allowed,
        "source_of_truth": "rag" if is_hit else None,
        "query": query,
        "ticker": ticker,
        "facts": evidence_facts,
        "documents": evidence_docs,
        "fact_count": len(evidence_facts),
        "document_count": len(evidence_docs),
        "top_score": top_score,
        "retrieval_strategy": retrieval_strategy,
        "model_generation": False,
        "trade_bias": trade_bias,
        "message": message,
    }


def _maybe_trade_bias(
    store: Any, query: str, ticker: str | None,
) -> dict[str, Any] | None:
    """Run classify_trade_bias when the question requires a long/short answer."""
    if not is_trade_bias_question(query):
        return None
    ticker_use = str(ticker or "").strip().upper()
    if not ticker_use:
        try:
            parsed = IntentParser().parse(query)
            ticker_use = str((parsed or {}).get("ticker") or "").strip().upper()
        except Exception:  # noqa: BLE001 — ticker resolution is best-effort
            ticker_use = ""
    if not ticker_use:
        return {
            "ticker": None,
            "bias": None,
            "must_answer": True,
            "evidence_status": "miss",
            "web_search_allowed": True,
            "signals": [],
            "error": "ticker is required for classify_trade_bias",
        }
    result, _, _ = dispatch_named_tool(
        "classify_trade_bias",
        {"ticker": ticker_use},
        store,
        ToolContext(allow_write=False, max_refreshes=0),
    )
    if isinstance(result, dict):
        return result
    return {"error": "classify_trade_bias failed", "bias": None, "must_answer": True}


def run_financebot_retrieval(
    *,
    store: Any,
    retriever: Any,
    query: str,
    ticker: str | None = None,
    n_results: int = 5,
    min_facts: int = 1,
    min_documents: int = 1,
    min_document_score: float = 0.0,
) -> dict[str, Any]:
    """Execute hybrid retrieval and classify hit/miss for FinanceBot."""
    facts: list[dict] = []
    documents: list[dict] = []
    ticker_out = ticker
    retrieval_strategy = None

    try:
        facts_results = store.search(
            query=query,
            n_results=n_results,
            ticker=ticker,
        )
        facts = list(facts_results.get("facts") or [])
        ticker_out = facts_results.get("ticker") or ticker
    except Exception as exc:  # noqa: BLE001 — miss soft; never fail FinanceBot
        logger.warning("FinanceBot fact retrieval failed: %s", exc)

    try:
        documents = list(
            retriever.retrieve_documents(
                query=query,
                ticker=ticker,
                n_results=n_results,
            )
            or []
        )
        retrieval_strategy = getattr(retriever, "_doc_retrieval_strategy", None)
        if retrieval_strategy is None and hasattr(retriever, "last_strategy"):
            retrieval_strategy = getattr(retriever, "last_strategy", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FinanceBot document retrieval failed: %s", exc)

    status, qualifying, top_score = classify_rag_hit(
        facts,
        documents,
        min_facts=min_facts,
        min_documents=min_documents,
        min_document_score=min_document_score,
    )
    try:
        trade_bias = _maybe_trade_bias(store, query, ticker_out)
    except Exception as exc:  # noqa: BLE001 — never fail FinanceBot retrieval
        logger.warning("FinanceBot classify_trade_bias failed: %s", exc)
        trade_bias = None
    return build_financebot_payload(
        query=query,
        facts=facts,
        documents=documents,
        ticker=ticker_out,
        status=status,
        qualifying_documents=qualifying,
        top_score=top_score,
        retrieval_strategy=retrieval_strategy,
        trade_bias=trade_bias,
    )


def list_financebot_tools() -> list[dict]:
    """Return every registered RAG tool (read and write) for FinanceBot."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "write": bool(tool.write),
        }
        for tool in REGISTRY.values()
    ]


def invoke_financebot_tool(
    *,
    store: Any,
    name: str,
    arguments: dict | None = None,
    allow_write: bool = False,
    max_refreshes: int = 2,
) -> dict[str, Any]:
    """Dispatch a registered RAG tool for FinanceBot without a local chat model.

    All tools attached to this RAG — including ``classify_trade_bias`` for
    long/short questions — stay callable through this adapter.
    """
    tool_name = str(name or "").strip()
    if not tool_name:
        return {"error": "tool name is required", "tool": name}
    if tool_name not in REGISTRY:
        return {
            "error": f"unknown tool: {tool_name}",
            "tool": tool_name,
            "available_tools": sorted(REGISTRY),
        }
    ctx = ToolContext(
        allow_write=bool(allow_write),
        max_refreshes=int(max_refreshes or 0),
    )
    result, resolved_name, validated = dispatch_named_tool(
        tool_name, arguments or {}, store, ctx
    )
    write = (
        bool(REGISTRY[resolved_name].write)
        if resolved_name in REGISTRY else False
    )
    payload: dict[str, Any] = {
        "tool": resolved_name,
        "arguments": validated,
        "result": result,
        "write": write,
    }
    if isinstance(result, dict) and result.get("error"):
        payload["error"] = result["error"]
    return payload
