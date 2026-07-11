"""
src/middleware/query_rewriter.py
Optional, bounded ambiguity fallback for follow-up rewriting (2.2.2.2 Step 3).

When deterministic compilation (``conversation.compile_question``) leaves a slot
genuinely ambiguous, this module may make *one* structured model call to
recover a standalone retrieval query. It is disabled by default, consumes the
shared Phase 2.2 planning-call budget (one call, hard timeout), and never sees
retrieved documents. Every value the model returns is validated against the
symbol resolver and the known metric catalog, and rejected if it names any
entity/metric/period/number not present in the current turn, the selected
history, or the catalog. On timeout, invalid JSON, entity drift, or model
unavailability it retains the deterministic compiled query and leaves the
ambiguity marked in metadata — a rewrite failure can never error a query.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from .conversation import KNOWN_METRICS, CompiledQuestion
from .models import ChatTurn

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM = (
    "You rewrite a user's latest chat turn into a single standalone search "
    "query for a finance retrieval system, using the prior turns only to fill "
    "in what the latest turn references implicitly. Respond with ONLY minified "
    "JSON of the form "
    '{"standalone_query": str, "entities": [str], "metrics": [str], '
    '"timeframe": str|null, "topic_reset": bool}. Use only tickers, metrics, '
    "periods, and numbers that appear in the provided turns. Never invent a "
    "company, figure, or period. No prose, no code fences."
)

# Uppercase tokens that look like tickers but must not count as invented
# entities when they appear in the rewritten query.
_STOPWORD_UPPER = {
    "I", "A", "AN", "THE", "IT", "IS", "BE", "TO", "OF", "IN", "ON", "AT",
    "BY", "AS", "OR", "IF", "NO", "GO", "DO", "WE", "HE", "US", "AND", "NOT",
    "FY", "Q", "YOY", "TTM", "YTD", "EPS", "PE", "GDP", "CPI", "USD",
}
_TICKERISH_RE = re.compile(r"\b[A-Z]{2,5}\b")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def should_use_llm_fallback(compiled: CompiledQuestion, config) -> bool:
    """Whether the bounded LLM fallback should run for this compiled turn.

    Only when both flags are on *and* deterministic compilation left a slot
    ambiguous — an unambiguous turn is answered from the deterministic query
    and makes zero rewrite-model calls (2.2.2.2 Step 4 gate).
    """
    return bool(
        getattr(config, "enable_conversation_rewrite", False)
        and getattr(config, "enable_llm_rewrite_fallback", False)
        and compiled.ambiguous_slots
    )


def rewrite_query(
    raw_question: str,
    selected_history: Optional[list[ChatTurn]],
    deterministic: CompiledQuestion,
    *,
    config,
    resolver=None,
    call_model: Optional[Callable[[str], str]] = None,
) -> CompiledQuestion:
    """Attempt one validated model rewrite; fall back to ``deterministic``.

    ``call_model(prompt) -> str`` returns the model's raw reply; injected by
    tests so no network is touched. Any failure (timeout, invalid JSON, drift,
    model down) returns the deterministic compiled question unchanged, with its
    ambiguity still marked.
    """
    timeout = _timeout_s(config)
    caller = call_model or _default_call_model(config, timeout)
    try:
        prompt = _build_prompt(raw_question, selected_history, deterministic)
        reply = caller(prompt)
        data = _parse_json(reply)
        if data is None:
            return deterministic
        rewritten = _validate(
            data, raw_question, selected_history, deterministic, resolver=resolver
        )
        return rewritten if rewritten is not None else deterministic
    except Exception as exc:  # noqa: BLE001 - a rewrite failure must never error a query
        logger.warning("Conversation rewrite fallback failed, keeping deterministic query: %s", exc)
        return deterministic


# ── Prompt + model call ────────────────────────────────────────────────────

def _build_prompt(
    raw_question: str,
    selected_history: Optional[list[ChatTurn]],
    deterministic: CompiledQuestion,
) -> str:
    lines = ["Prior turns (oldest first):"]
    for turn in selected_history or []:
        role = getattr(turn, "role", "?")
        content = (getattr(turn, "content", "") or "").replace("\n", " ")
        lines.append(f"- {role}: {content}")
    lines.append("")
    lines.append(f"Latest turn: {raw_question}")
    if deterministic.ambiguous_slots:
        lines.append(f"Unresolved slots: {', '.join(deterministic.ambiguous_slots)}")
    lines.append("")
    lines.append("Return the JSON object described in the system message.")
    return "\n".join(lines)


def _timeout_s(config) -> float:
    try:
        value = float(getattr(config, "conversation_rewrite_timeout_s", 15.0) or 15.0)
    except (TypeError, ValueError):
        return 15.0
    return value if value > 0 else 15.0


def _default_call_model(config, timeout: float) -> Callable[[str], str]:
    """Build the real one-shot model caller (httpx). Kept lazy for offline use."""

    def _call(prompt: str) -> str:
        import httpx

        endpoint = getattr(config, "llama_endpoint", "")
        payload = {
            "model": getattr(config, "model_name", "tracealchemy"),
            "messages": [
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        resp = httpx.post(endpoint, json=payload, timeout=httpx.Timeout(timeout, connect=5.0))
        resp.raise_for_status()
        return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""

    return _call


def _parse_json(reply: Optional[str]) -> Optional[dict]:
    """Extract the first JSON object from a model reply, or None."""
    if not reply:
        return None
    text = reply.strip()
    # Tolerate a leading ```json fence or surrounding prose by isolating the
    # first {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# ── Validation ─────────────────────────────────────────────────────────────

def _combined_text(raw_question: str, selected_history: Optional[list[ChatTurn]]) -> str:
    parts = [raw_question or ""]
    for turn in selected_history or []:
        parts.append(getattr(turn, "content", "") or "")
    return " ".join(parts)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _validate(
    data: dict,
    raw_question: str,
    selected_history: Optional[list[ChatTurn]],
    deterministic: CompiledQuestion,
    *,
    resolver=None,
) -> Optional[CompiledQuestion]:
    """Return a validated CompiledQuestion, or None to reject (→ deterministic)."""
    standalone = data.get("standalone_query")
    if not isinstance(standalone, str) or not standalone.strip():
        return None
    standalone = standalone.strip()

    combined = _combined_text(raw_question, selected_history)
    combined_lower = combined.lower()
    combined_norm = _norm(combined)

    resolver = resolver or _get_resolver()

    # Entities: must resolve to a real ticker AND be present in the turns.
    validated_entities: list[str] = []
    for raw_entity in data.get("entities") or []:
        entity = _norm_ticker(raw_entity)
        if not entity:
            continue
        resolution = resolver.resolve(entity)
        present = entity.lower() in combined_lower or (
            resolution.resolved_name and resolution.resolved_name.lower() in combined_lower
        )
        if not present:
            return None  # invented / drifted entity
        if resolution.ticker is None and entity not in _known_tickers():
            return None  # not a resolvable symbol
        if entity not in validated_entities:
            validated_entities.append(entity)

    # Metrics: must be in the known catalog.
    validated_metrics: list[str] = []
    for raw_metric in data.get("metrics") or []:
        metric = str(raw_metric).strip().lower().replace(" ", "_")
        if metric not in KNOWN_METRICS:
            return None
        if metric not in validated_metrics:
            validated_metrics.append(metric)

    # Timeframe: must appear in the turns.
    timeframe = data.get("timeframe")
    validated_timeframe: Optional[str] = None
    if timeframe:
        tf = str(timeframe).strip()
        if tf and _norm(tf) and _norm(tf) not in combined_norm:
            return None
        validated_timeframe = tf or None

    # Standalone query must not smuggle in an entity or number absent from the
    # turns and not among the validated entities.
    if not _query_within_bounds(standalone, validated_entities, combined_lower, combined_norm):
        return None

    topic_reset = bool(data.get("topic_reset", deterministic.topic_reset))
    resolution_sources = list(dict.fromkeys([*deterministic.resolution_sources, "llm_rewrite"]))

    return CompiledQuestion(
        raw_question=deterministic.raw_question,
        retrieval_query=standalone,
        carried_entities=validated_entities or deterministic.carried_entities,
        carried_metrics=validated_metrics or deterministic.carried_metrics,
        carried_timeframe=validated_timeframe
        if validated_timeframe is not None else deterministic.carried_timeframe,
        topic_reset=topic_reset,
        ambiguous_slots=[],
        resolution_sources=resolution_sources,
        entity=(validated_entities[0] if validated_entities else deterministic.entity),
        metrics=validated_metrics or deterministic.metrics,
        timeframe=validated_timeframe if validated_timeframe is not None else deterministic.timeframe,
    )


def _query_within_bounds(
    standalone: str,
    validated_entities: list[str],
    combined_lower: str,
    combined_norm: str,
) -> bool:
    """Reject a rewritten query that introduces a new ticker or number."""
    allowed = {e.upper() for e in validated_entities}
    for token in _TICKERISH_RE.findall(standalone):
        upper = token.upper()
        if upper in allowed or upper in _STOPWORD_UPPER:
            continue
        if upper.lower() in combined_lower:
            continue
        if upper in _known_tickers() and upper.lower() not in combined_lower:
            return False  # a real ticker the user never mentioned
    for number in _NUMBER_RE.findall(standalone):
        if _norm(number) and _norm(number) not in combined_norm:
            return False
    return True


def _norm_ticker(value) -> Optional[str]:
    if value is None:
        return None
    norm = str(value).strip().upper()
    return norm or None


def _get_resolver():
    from .symbol_resolver import get_default_resolver

    return get_default_resolver()


def _known_tickers() -> frozenset[str]:
    from .intent_parser import IntentParser

    return frozenset(IntentParser.KNOWN_TICKERS)
