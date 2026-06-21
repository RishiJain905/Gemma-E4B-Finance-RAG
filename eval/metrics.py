"""
eval/metrics.py — Stage 2 scoring functions for the eval harness (2.1.1.2).

Turns raw run rows (from run_eval.py) into scores:

  - intent_accuracy      detected_intent == expected_intent (over declaring cases)
  - ticker_accuracy      detected_ticker == expected_ticker (all cases; None==None)
  - retrieval_hit_rate   retrieved sources hit expected_sources (count fallback)
  - keyword_coverage     fraction of must_mention terms present in the answer
  - refusal_rate         fraction of model answers matching a refusal pattern
  - faithfulness         LLM-as-judge groundedness (0–1, or None if model down)
  - answer_relevance     LLM-as-judge relevance  (0–1, or None if model down)

Every function takes a list of run-row dicts. A row carries its golden case
either under ``row["case"]`` (real runs) or inline (synthetic test rows); the
``_case`` helper handles both. The LLM-judge accepts an injectable ``judge``
callable ``(prompt) -> text`` so unit tests never touch the network.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from . import judge_prompts
from .run_eval import normalize_source

DEFAULT_MODEL_ENDPOINT = "http://127.0.0.1:8087/v1/chat/completions"

# Phrasings that count as the system refusing to answer. Lowercased substring match.
REFUSAL_MARKERS = (
    "don't have enough data",
    "do not have enough data",
    "i do not have enough",
    "i don't have enough",
    "not enough information",
    "i don't have that information",
    "i don't have that data",
    "no data available",
    "i am unable to answer",
    "i can't answer",
    "cannot answer",
    "i cannot answer",
)

# Metric keys that are simple floats (used by the gate + console table).
SCALAR_METRICS = (
    "intent_accuracy",
    "ticker_accuracy",
    "retrieval_hit_rate",
    "keyword_coverage",
    "refusal_rate",
)


# ── Helpers ────────────────────────────────────────────────────────────

def _case(row: dict) -> dict:
    """Return the golden case for a row: row['case'] if present, else the row."""
    c = row.get("case")
    return c if isinstance(c, dict) else row


def _norm_ticker(t) -> Optional[str]:
    """Normalize a ticker for comparison (None / '' → None, else upper)."""
    if t is None:
        return None
    s = str(t).strip().upper()
    return s or None


def _norm_intent(t) -> Optional[str]:
    if t is None:
        return None
    return str(t).strip().lower() or None


# ── Cheap metrics ───────────────────────────────────────────────────────

def intent_accuracy(rows: list[dict]) -> float:
    """Fraction of declaring cases where detected_intent == expected_intent."""
    decl = [r for r in rows if _norm_intent(_case(r).get("expected_intent")) is not None]
    if not decl:
        return 0.0
    hit = sum(1 for r in decl
              if _norm_intent(r.get("detected_intent"))
              == _norm_intent(_case(r).get("expected_intent")))
    return round(hit / len(decl), 4)


def ticker_accuracy(rows: list[dict]) -> float:
    """Fraction of all cases where detected_ticker == expected_ticker.

    ``None == None`` counts as a correct (no-ticker) detection, so this also
    catches false positives (e.g. macro questions misdetected as a company).
    """
    if not rows:
        return 0.0
    hit = sum(1 for r in rows
              if _norm_ticker(r.get("detected_ticker"))
              == _norm_ticker(_case(r).get("expected_ticker")))
    return round(hit / len(rows), 4)


def retrieval_hit_rate(rows: list[dict]) -> float:
    """Fraction of cases with a relevant retrieval.

    Per case:
      - if expected_sources is non-empty → hit when any expected source is in
        the retrieved sources;
      - if expected_sources is empty → hit when anything was retrieved
        (retrieved_sources non-empty OR facts_used + documents_used > 0).
    """
    if not rows:
        return 0.0
    hits = 0
    for r in rows:
        c = _case(r)
        expected = {normalize_source(s) for s in c.get("expected_sources") or []}
        expected.discard(None)
        retrieved = set(r.get("retrieved_sources") or [])
        if expected:
            if expected & retrieved:
                hits += 1
        else:
            counts = int(r.get("facts_used") or 0) + int(r.get("documents_used") or 0)
            if retrieved or counts > 0:
                hits += 1
    return round(hits / len(rows), 4)


def keyword_coverage(rows: list[dict]) -> float:
    """Mean fraction of ``must_mention`` terms present in the answer.

    Rows with empty ``must_mention`` don't contribute. A term is "present" if it
    appears as a case-insensitive substring of the answer.
    """
    per_case_cov: list[float] = []
    for r in rows:
        terms = _case(r).get("must_mention") or []
        if not terms:
            continue
        ans = (r.get("answer") or "").lower()
        present = sum(1 for t in terms if t and t.lower() in ans)
        per_case_cov.append(present / len(terms))
    if not per_case_cov:
        return 0.0
    return round(sum(per_case_cov) / len(per_case_cov), 4)


def refusal_rate(rows: list[dict], *, only_model_answers: bool = True) -> float:
    """Fraction of (model-generated) answers matching a refusal pattern.

    Tracks over-strictness. When ``only_model_answers`` is true, only rows where
    the model produced the answer are counted, so model-unavailability (the
    degraded "⚠️ Model unavailable" answer) is not misread as a refusal.
    """
    pool = [r for r in rows if r.get("answer")] if only_model_answers else list(rows)
    pool = [r for r in pool if not only_model_answers or r.get("model_available")]
    if not pool:
        return 0.0
    refused = sum(1 for r in pool
                  if any(m in (r.get("answer") or "").lower() for m in REFUSAL_MARKERS))
    return round(refused / len(pool), 4)


# ── LLM-judge ──────────────────────────────────────────────────────────

def _parse_score(text: Optional[str]) -> Optional[float]:
    """Extract a 0–1 float from a judge reply.

    Accepts an explicit ``Score: X`` (preferred) or falls back to the last
    number in the text. Values in [0, 1] are used directly; values in [2, 10]
    are treated as a 0–10 scale and divided by 10. Values in (1, 2) are
    ambiguous (a 0–1 violation vs. a low 0–10 score) and rejected — for a
    quality gate it is safer to skip than to mis-score.
    """
    if not text:
        return None
    m = re.search(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1)
    else:
        nums = re.findall(r"[0-9]*\.?[0-9]+", text)
        if not nums:
            return None
        raw = nums[-1]
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return None
    if 0.0 <= val <= 1.0:
        return round(val, 4)
    if 2.0 <= val <= 10.0:
        return round(val / 10.0, 4)  # 0–10 scale
    return None  # (1, 2) ambiguous, or > 10


def default_judge(prompt: str, *, endpoint: str = DEFAULT_MODEL_ENDPOINT,
                  timeout: float = 120.0, max_tokens: int = 1024) -> Optional[str]:
    """Call the local model and return its reply text, or None if unavailable.

    The TraceAlchemy model is a *thinking* model: it writes reasoning into
    ``reasoning_content`` and the final answer into ``content``. We give it
    ample ``max_tokens`` so reasoning finishes and the ``Score:`` line actually
    lands in ``content`` (with too-small a cap the model spends every token
    reasoning and ``content`` comes back empty). We return ``content`` only —
    falling back to ``reasoning_content`` would risk parsing context numbers as
    the score.
    """
    import httpx
    payload = {
        "model": "tracealchemy",
        "messages": [
            {"role": "system", "content": judge_prompts.JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    try:
        r = httpx.post(endpoint, json=payload,
                       timeout=httpx.Timeout(timeout, connect=10.0))
        r.raise_for_status()
        return r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or None
    except Exception:
        return None


def _judge_scores(rows: list[dict], *, build_prompt: Callable, judge: Optional[Callable],
                  endpoint: str, timeout: float, need_context: bool) -> dict:
    """Shared driver for faithfulness / answer_relevance judging."""
    j = judge or (lambda p: default_judge(p, endpoint=endpoint, timeout=timeout))
    scores: list[float] = []
    per_case: list[dict] = []
    model_down = False

    for r in rows:
        ans = r.get("answer") or ""
        q = r.get("question") or ""
        cid = r.get("id")
        if not ans.strip():
            per_case.append({"id": cid, "score": None, "reason": "empty answer"})
            continue
        ctx = r.get("context") or ""
        # NOTE: empty context is NOT skipped — the faithfulness rubric handles it
        # (score 0 for hallucination with no context, 1 for a faithful refusal).
        prompt = build_prompt(q, ctx, ans) if need_context else build_prompt(q, ans)
        text = j(prompt)
        if text is None:
            model_down = True
            per_case.append({"id": cid, "score": None, "reason": "judge unavailable"})
            continue
        s = _parse_score(text)
        per_case.append({"id": cid, "score": s,
                         "reason": None if s is not None else "unparseable"})
        if s is not None:
            scores.append(s)

    return {
        "score": round(sum(scores) / len(scores), 4) if scores else None,
        "n_scored": len(scores),
        "n_skipped": len(rows) - len(scores),
        "model_available": not model_down,
        "per_case": per_case,
    }


def faithfulness(rows: list[dict], *, judge: Optional[Callable] = None,
                 endpoint: str = DEFAULT_MODEL_ENDPOINT,
                 timeout: float = 120.0) -> dict:
    """LLM-as-judge groundedness (0–1). Returns None score when the model is down."""
    return _judge_scores(
        rows, build_prompt=judge_prompts.faithfulness_prompt, judge=judge,
        endpoint=endpoint, timeout=timeout, need_context=True,
    )


def answer_relevance(rows: list[dict], *, judge: Optional[Callable] = None,
                     endpoint: str = DEFAULT_MODEL_ENDPOINT,
                     timeout: float = 120.0) -> dict:
    """LLM-as-judge answer relevance (0–1). Returns None score when the model is down."""
    return _judge_scores(
        rows, build_prompt=judge_prompts.relevance_prompt, judge=judge,
        endpoint=endpoint, timeout=timeout, need_context=False,
    )


# ── Aggregator ─────────────────────────────────────────────────────────

def score_all(rows: list[dict], *, judge: Optional[Callable] = None,
              endpoint: str = DEFAULT_MODEL_ENDPOINT, timeout: float = 120.0,
              run_judge: bool = True) -> dict:
    """Compute every metric for a run. Returns the summary block.

    ``run_judge=False`` skips the (slow) LLM-judge calls; faithfulness /
    answer_relevance come back as None. Per-category breakdowns cover the cheap
    scalar metrics.
    """
    n = len(rows)
    summary: dict = {
        "n_cases": n,
        "n_model_available": sum(1 for r in rows if r.get("model_available")),
        "n_errors": sum(1 for r in rows if r.get("error")),
    }
    for m in SCALAR_METRICS:
        summary[m] = globals()[m](rows)

    if run_judge:
        summary["faithfulness"] = faithfulness(rows, judge=judge, endpoint=endpoint,
                                                timeout=timeout)
        summary["answer_relevance"] = answer_relevance(rows, judge=judge,
                                                       endpoint=endpoint, timeout=timeout)
    else:
        summary["faithfulness"] = {"score": None, "n_scored": 0, "n_skipped": n,
                                    "model_available": False, "per_case": []}
        summary["answer_relevance"] = {"score": None, "n_scored": 0, "n_skipped": n,
                                       "model_available": False, "per_case": []}

    summary["per_category"] = _per_category(rows)
    return summary


def _per_category(rows: list[dict]) -> dict:
    """Per-category breakdown of the cheap scalar metrics."""
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        cat = _case(r).get("category", "uncategorized")
        by_cat.setdefault(cat, []).append(r)
    out: dict[str, dict] = {}
    for cat, cat_rows in by_cat.items():
        out[cat] = {
            "n": len(cat_rows),
            "intent_accuracy": intent_accuracy(cat_rows),
            "ticker_accuracy": ticker_accuracy(cat_rows),
            "retrieval_hit_rate": retrieval_hit_rate(cat_rows),
            "keyword_coverage": keyword_coverage(cat_rows),
            "refusal_rate": refusal_rate(cat_rows),
        }
    return out