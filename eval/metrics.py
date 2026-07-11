"""
eval/metrics.py — Stage 2 scoring functions for the eval harness (2.1.1.2, 2.2.1.2).

Turns raw run rows (from run_eval.py) into scores:

  - intent_accuracy         detected_intent == expected_intent (over declaring cases)
  - ticker_accuracy         detected_ticker == expected_ticker (all cases; None==None)
  - retrieval_hit_rate      retrieved sources hit expected_sources (count fallback)
  - keyword_coverage        fraction of must_mention terms present in the answer
  - refusal_rate            fraction of model answers matching a refusal pattern
  - grounded_faithfulness   LLM-as-judge groundedness of grounded/partial answers
                            against the exact evidence trace, incl. tool results
  - policy_compliance       LLM-as-judge policy compliance across all four
                            grounding modes (grounded/partial/general/refused)
  - answer_relevance        LLM-as-judge relevance (0-1, or None if model down)

Every function takes a list of run-row dicts. A row carries its golden case
either under ``row["case"]`` (real runs) or inline (synthetic test rows); the
``_case`` helper handles both. The LLM-judge accepts an injectable ``judge``
callable ``(prompt) -> text`` so unit tests never touch the network.

2.2.1.2 replaced the single ``faithfulness`` metric (scored against a
truncated re-retrieved ``context`` string, and thus inflated by strict
refusals) with two metrics scored against the row's exact
``evidence_trace`` (2.2.1.2):

  - ``grounded_faithfulness`` only scores ``grounded``/``partial`` answers —
    general and refused answers are excluded from its denominator so a
    refusal can no longer inflate it.
  - ``policy_compliance`` scores every mode against the rules for that mode.

A row with a model-produced answer (``model_available``) but a missing or
incomplete evidence trace (``trace_complete`` false) is never silently
scored — it is excluded from both judge metrics' eligible pool and reported
via ``trace_errors``.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from . import judge_prompts
from .run_eval import normalize_source

DEFAULT_MODEL_ENDPOINT = "http://127.0.0.1:8087/v1/chat/completions"

# Bump when the shape of score_all()'s summary changes in a way that would
# invalidate a naive comparison against an older baseline (2.2.1.2 Step 5).
SCORE_SCHEMA_VERSION = 1

# Grounding modes eligible for grounded_faithfulness (they claim grounding).
GROUNDED_MODES = ("grounded", "partial")

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

# Metric keys whose value is a judge-result dict ({"score": ..., ...}).
JUDGE_METRICS = ("grounded_faithfulness", "policy_compliance", "answer_relevance")


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


def _is_missing_trace(row: dict) -> bool:
    """True when a model-produced answer lacks a complete evidence trace.

    A degraded (``model_available=False``) row makes no model-visible-
    evidence claim at all, so it is never "missing" in this sense — there is
    nothing to have omitted.
    """
    return bool(row.get("model_available")) and not row.get("trace_complete", False)


def trace_errors(rows: list[dict]) -> list[dict]:
    """Rows with a model-produced answer but a missing/incomplete evidence trace.

    Never silently scored — the 2.1.7 gate must fail before comparing metrics
    when any of these exist (2.2.1.2 Step 5).
    """
    return [
        {"id": r.get("id"), "reason": r.get("error") or "incomplete evidence trace"}
        for r in rows if _is_missing_trace(r)
    ]


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
                  endpoint: str, timeout: float,
                  skip_fn: Callable) -> dict:
    """Shared driver for the trace-grounded judges (grounded_faithfulness /
    policy_compliance).

    ``skip_fn(row) -> Optional[str]`` returns a skip reason (rows that are
    empty-answer, degraded, missing-trace, or out-of-mode never reach the
    judge) or ``None`` when the row is eligible. ``n_eligible`` counts rows
    that passed ``skip_fn`` and had a non-empty answer, regardless of whether
    the judge itself was reachable — that is what the 2.1.7 gate's
    "eligible denominator" check compares run over run.
    """
    j = judge or (lambda p: default_judge(p, endpoint=endpoint, timeout=timeout))
    scores: list[float] = []
    per_case: list[dict] = []
    model_down = False
    n_eligible = 0

    for r in rows:
        ans = r.get("answer") or ""
        cid = r.get("id")
        if not ans.strip():
            per_case.append({"id": cid, "score": None, "reason": "empty answer"})
            continue
        skip_reason = skip_fn(r)
        if skip_reason:
            per_case.append({"id": cid, "score": None, "reason": skip_reason})
            continue
        n_eligible += 1
        trace = r.get("evidence_trace") or {}
        prompt = build_prompt(r, trace, ans)
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
        "n_eligible": n_eligible,
        "n_skipped": len(rows) - len(scores),
        "model_available": not model_down,
        "per_case": per_case,
    }


def _skip_not_model_answer(row: dict) -> Optional[str]:
    """Skip reason for rows that never reached a model call at all."""
    if not row.get("model_available"):
        return "model unavailable (degraded answer, not a policy choice)"
    if _is_missing_trace(row):
        return "missing/incomplete evidence trace"
    return None


def _skip_ungrounded_mode(row: dict) -> Optional[str]:
    reason = _skip_not_model_answer(row)
    if reason:
        return reason
    grounding = row.get("grounding")
    if grounding not in GROUNDED_MODES:
        return f"excluded: {grounding or 'unknown'} mode not scored for grounded faithfulness"
    return None


def grounded_faithfulness(rows: list[dict], *, judge: Optional[Callable] = None,
                          endpoint: str = DEFAULT_MODEL_ENDPOINT,
                          timeout: float = 120.0) -> dict:
    """LLM-as-judge groundedness (0-1) of grounded/partial answers against the
    exact evidence trace. General/refused answers and rows with a missing
    trace never enter the denominator, so refusals cannot inflate this score.
    """
    return _judge_scores(
        rows,
        build_prompt=lambda r, trace, ans: judge_prompts.grounded_faithfulness_prompt(
            r.get("question") or "", trace, ans),
        judge=judge, endpoint=endpoint, timeout=timeout,
        skip_fn=_skip_ungrounded_mode,
    )


def policy_compliance(rows: list[dict], *, judge: Optional[Callable] = None,
                      endpoint: str = DEFAULT_MODEL_ENDPOINT,
                      timeout: float = 120.0) -> dict:
    """LLM-as-judge policy compliance (0-1) across all four grounding modes."""
    return _judge_scores(
        rows,
        build_prompt=lambda r, trace, ans: judge_prompts.policy_compliance_prompt(
            r.get("question") or "", trace, ans, r.get("grounding")),
        judge=judge, endpoint=endpoint, timeout=timeout,
        skip_fn=_skip_not_model_answer,
    )


def answer_relevance(rows: list[dict], *, judge: Optional[Callable] = None,
                     endpoint: str = DEFAULT_MODEL_ENDPOINT,
                     timeout: float = 120.0) -> dict:
    """LLM-as-judge answer relevance (0-1). Returns None score when the model is down.

    Unlike the trace-grounded judges, relevance needs only the question and
    answer, so every non-empty answer (including general/refused) is eligible.
    """
    j = judge or (lambda p: default_judge(p, endpoint=endpoint, timeout=timeout))
    scores: list[float] = []
    per_case: list[dict] = []
    model_down = False
    n_eligible = 0

    for r in rows:
        ans = r.get("answer") or ""
        cid = r.get("id")
        if not ans.strip():
            per_case.append({"id": cid, "score": None, "reason": "empty answer"})
            continue
        n_eligible += 1
        prompt = judge_prompts.relevance_prompt(r.get("question") or "", ans)
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
        "n_eligible": n_eligible,
        "n_skipped": len(rows) - len(scores),
        "model_available": not model_down,
        "per_case": per_case,
    }


# ── Conversational / compound metrics (2.2.1.3) ────────────────────────
#
# All deterministic: they read captured structured fields (resolved vs.
# expected tickers/metrics/timeframe, subquestion coverage, staleness/answer
# disclosures) — never the model. Each returns a block with a ``score`` plus
# its ``n_eligible`` denominator and ``n_pass`` so a dropping/zero denominator
# is visible (the gate fails a required category with zero eligible fixtures).
# Only compound coverage may optionally consult an injected judge; it defaults
# to deterministic keyword matching.

# The Phase 2.2 conversational metrics, in report/gate order.
PHASE22_METRICS = (
    "entity_carryover_accuracy",
    "metric_carryover_accuracy",
    "timeframe_carryover_accuracy",
    "verbose_paraphrase_parity",
    "compound_subquestion_coverage",
    "stale_disclosure_rate",
    "unanswerable_numeric_hallucination_rate",
    "cross_session_leakage_rate",
)

# Phrasings that count as a visible data-freshness / staleness disclosure.
STALE_DISCLOSURE_MARKERS = (
    "stale",
    "outdated",
    "out of date",
    "out-of-date",
    "last updated",
    "last refreshed",
    "as-of",
    "as of ",
    "freshness",
    "may be out of date",
    "may not be current",
    "may not reflect",
    "not be current",
    "data is from",
    "data from ",
    "⚠️",
)

# An "invented financial figure": currency amounts, magnitudes, percentages, or
# valuation multiples. Bare years (2026) and quarter labels (Q1) do NOT match,
# so an unanswerable answer can name a period without being flagged.
_FINANCIAL_FIGURE = re.compile(
    r"""
      (?:[$€£]\s?\d)                                               # currency amount
    | (?:\b\d[\d,]*(?:\.\d+)?\s*%)                                 # percentage
    | (?:\b\d[\d,]*(?:\.\d+)?\s*(?:billion|million|trillion|bn|mn)\b)  # magnitude
    | (?:\b\d+(?:\.\d+)?\s*x\b)                                    # valuation multiple
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _norm_metric(m) -> Optional[str]:
    """Normalize a metric token for comparison (lower, spaces→underscores)."""
    if m is None:
        return None
    s = str(m).strip().lower().replace(" ", "_")
    return s or None


def _norm_timeframe(t) -> Optional[str]:
    """Normalize a timeframe token (upper, strip spaces/hyphens removed)."""
    if t is None:
        return None
    s = str(t).strip().upper().replace(" ", "").replace("-", "")
    return s or None


def subquestions_addressed(subquestions: list[dict], answer: str) -> list[str]:
    """Ids of subquestions whose ``must_mention`` terms all appear in the answer.

    Deterministic, case-insensitive substring match — the same rule
    ``keyword_coverage`` uses, applied per subquestion. A subquestion with no
    terms cannot be judged addressed and is treated as unaddressed.
    """
    ans = (answer or "").lower()
    out: list[str] = []
    for sq in subquestions or []:
        terms = sq.get("must_mention") or []
        if terms and all(str(t).lower() in ans for t in terms):
            out.append(sq.get("id"))
    return out


def contains_financial_figure(text: str) -> bool:
    """True when the text contains an invented-figure-shaped number.

    Deterministic guard for unanswerable cases: a genuine no-data answer names
    no dollar amount, percentage, magnitude, or multiple. Years/quarters are
    intentionally excluded so a refusal may still reference a period.
    """
    return bool(_FINANCIAL_FIGURE.search(text or ""))


def discloses_staleness(text: str) -> bool:
    """True when the answer carries a visible freshness/staleness disclosure."""
    low = (text or "").lower()
    return any(m in low for m in STALE_DISCLOSURE_MARKERS)


def _block(score: Optional[float], n_eligible: int, n_pass: int,
           **extra) -> dict:
    return {"score": score, "n_eligible": n_eligible, "n_pass": n_pass, **extra}


def _carryover_accuracy(rows: list[dict], field: str) -> dict:
    """Accuracy of one carried field over turns that declare it should carry.

    Eligible = turns whose ``expected_carryover`` names ``field``. A turn passes
    when the resolved value matches the carried expectation: exact for
    ticker/timeframe, subset for the metric list.
    """
    n_elig = 0
    n_pass = 0
    for r in rows:
        carry = _case(r).get("expected_carryover") or {}
        want = carry.get(field)
        if want in (None, "", []):
            continue
        n_elig += 1
        if field == "ticker":
            resolved = {_norm_ticker(t) for t in (r.get("resolved_tickers") or [])}
            ok = _norm_ticker(want) in resolved
        elif field == "metrics":
            resolved = {_norm_metric(m) for m in (r.get("resolved_metrics") or [])}
            ok = {_norm_metric(m) for m in want}.issubset(resolved)
        else:  # timeframe
            ok = _norm_timeframe(want) == _norm_timeframe(r.get("resolved_timeframe"))
        if ok:
            n_pass += 1
    return _block(round(n_pass / n_elig, 4) if n_elig else None, n_elig, n_pass)


def entity_carryover_accuracy(rows: list[dict]) -> dict:
    """Fraction of ticker-carryover turns whose resolved ticker is the carried one."""
    return _carryover_accuracy(rows, "ticker")


def metric_carryover_accuracy(rows: list[dict]) -> dict:
    """Fraction of metric-carryover turns whose resolved metrics cover the carried set."""
    return _carryover_accuracy(rows, "metrics")


def timeframe_carryover_accuracy(rows: list[dict]) -> dict:
    """Fraction of timeframe-carryover turns whose resolved timeframe matches."""
    return _carryover_accuracy(rows, "timeframe")


def _normalized_plan(row: dict) -> tuple:
    """The comparable retrieval plan for a row (2.2.1.3 paraphrase parity)."""
    return (
        tuple(sorted(_norm_ticker(t) for t in (row.get("resolved_tickers") or []))),
        tuple(sorted(_norm_metric(m) for m in (row.get("resolved_metrics") or []))),
        _norm_timeframe(row.get("resolved_timeframe")),
        _norm_intent(row.get("detected_intent")),
    )


def verbose_paraphrase_parity(rows: list[dict]) -> dict:
    """Fraction of paraphrase groups whose members resolve to the same plan.

    Eligible = groups (``paraphrase_group``) with at least two members (a
    verbose/concise pair). A group passes when every member's normalized
    retrieval plan is identical.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        g = _case(r).get("paraphrase_group")
        if g:
            groups.setdefault(g, []).append(r)
    n_elig = 0
    n_pass = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        n_elig += 1
        if len({_normalized_plan(m) for m in members}) == 1:
            n_pass += 1
    return _block(round(n_pass / n_elig, 4) if n_elig else None, n_elig, n_pass)


def compound_subquestion_coverage(rows: list[dict], *,
                                  judge: Optional[Callable] = None) -> dict:
    """Mean fraction of subquestions addressed across compound cases.

    Eligible = rows whose case declares ``subquestions``. ``n_pass`` counts
    fully-covered rows. Deterministic keyword matching by default; ``judge`` is
    accepted for future answer-coverage grading but unused in the default path.
    """
    covs: list[float] = []
    n_full = 0
    for r in rows:
        subs = _case(r).get("subquestions") or []
        if not subs:
            continue
        addressed = subquestions_addressed(subs, r.get("answer") or "")
        cov = len(addressed) / len(subs)
        covs.append(cov)
        if cov >= 1.0:
            n_full += 1
    score = round(sum(covs) / len(covs), 4) if covs else None
    return _block(score, len(covs), n_full)


def stale_disclosure_rate(rows: list[dict]) -> dict:
    """Fraction of stale-required cases whose answer discloses staleness.

    Eligible = cases with ``requires_stale_disclosure`` truthy. Higher is
    better; the Phase 2.2 gate requires 1.00.
    """
    elig = [r for r in rows if _case(r).get("requires_stale_disclosure")]
    if not elig:
        return _block(None, 0, 0)
    n_pass = sum(1 for r in elig if discloses_staleness(r.get("answer") or ""))
    return _block(round(n_pass / len(elig), 4), len(elig), n_pass)


def unanswerable_numeric_hallucination_rate(rows: list[dict]) -> dict:
    """Fraction of unanswerable cases whose answer invents a financial figure.

    Eligible = cases with ``answerability == "unanswerable"``. Lower is better;
    the Phase 2.2 gate requires 0.00. ``n_pass`` counts clean (non-hallucinated)
    answers.
    """
    elig = [r for r in rows if _case(r).get("answerability") == "unanswerable"]
    if not elig:
        return _block(None, 0, 0)
    n_halluc = sum(1 for r in elig if contains_financial_figure(r.get("answer") or ""))
    return _block(round(n_halluc / len(elig), 4), len(elig), len(elig) - n_halluc)


def _conversation_legit_tickers(rows: list[dict]) -> dict:
    """Map conversation_id -> set of tickers legitimately reachable in it.

    A ticker is legitimate for a conversation if any of its turns declares it in
    ``expected_tickers`` or carries it via ``expected_carryover.ticker``.
    """
    legit: dict[str, set] = {}
    for r in rows:
        cid = r.get("conversation_id")
        if cid is None:
            continue
        bucket = legit.setdefault(cid, set())
        for t in r.get("expected_tickers") or []:
            n = _norm_ticker(t)
            if n:
                bucket.add(n)
        carry = _case(r).get("expected_carryover") or {}
        n = _norm_ticker(carry.get("ticker"))
        if n:
            bucket.add(n)
    return legit


def cross_session_leakage_rate(rows: list[dict]) -> dict:
    """Fraction of conversation turns that resolved a ticker from another session.

    Eligible = all conversation turns. A turn leaks when a resolved ticker is
    foreign to its own conversation's legitimate set AND belongs to a *different*
    conversation's legitimate set — i.e. context crossed sessions. Lower is
    better; the Phase 2.2 gate requires 0.00.
    """
    conv_rows = [r for r in rows if r.get("conversation_id") is not None]
    if not conv_rows:
        return _block(None, 0, 0)
    legit = _conversation_legit_tickers(rows)
    n_leak = 0
    for r in conv_rows:
        cid = r["conversation_id"]
        own = legit.get(cid, set())
        others: set = set()
        for k, v in legit.items():
            if k != cid:
                others |= v
        resolved = {_norm_ticker(t) for t in (r.get("resolved_tickers") or [])}
        if (resolved - own) & others:
            n_leak += 1
    return _block(round(n_leak / len(conv_rows), 4), len(conv_rows), len(conv_rows) - n_leak)


def has_phase22_fixtures(rows: list[dict]) -> bool:
    """True when a run includes any Phase 2.2 conversational/compound fixture.

    Gates whether ``score_all`` emits the conversational metric blocks, so a
    pre-2.2 single-turn-only run isn't forced to satisfy Phase 2.2 categories.
    """
    keys = ("expected_carryover", "subquestions", "paraphrase_group",
            "requires_stale_disclosure", "answerability", "expected_metrics",
            "expected_timeframe", "expected_tickers")
    for r in rows:
        if r.get("conversation_id") is not None:
            return True
        c = _case(r)
        if any(c.get(k) for k in keys):
            return True
    return False


def conversational_metrics(rows: list[dict]) -> dict:
    """Compute every Phase 2.2 conversational metric block for a run."""
    return {
        "entity_carryover_accuracy": entity_carryover_accuracy(rows),
        "metric_carryover_accuracy": metric_carryover_accuracy(rows),
        "timeframe_carryover_accuracy": timeframe_carryover_accuracy(rows),
        "verbose_paraphrase_parity": verbose_paraphrase_parity(rows),
        "compound_subquestion_coverage": compound_subquestion_coverage(rows),
        "stale_disclosure_rate": stale_disclosure_rate(rows),
        "unanswerable_numeric_hallucination_rate":
            unanswerable_numeric_hallucination_rate(rows),
        "cross_session_leakage_rate": cross_session_leakage_rate(rows),
    }


# ── Adaptive orchestration metrics (2.2.3.4) ───────────────────────────
#
# Offline, deterministic per-lane / budget metrics computed from the
# ``orchestration`` block the middleware attaches to each response (and the
# eval harness copies onto each row). The judge-scored quality deltas
# (Recall@10 / nDCG@10 / faithfulness by lane) are produced from a live
# three-config comparison; these scalar metrics are the offline scaffolding
# that attributes the run to a config and proves the request stayed bounded.

# The single write tool must never be reachable via the deterministic route.
_WRITE_TOOLS = frozenset({"refresh_data"})

# Adaptive hard caps (mirror config._ADAPTIVE_* / ExecutionBudget). Used to
# assert no request exceeded a budget within a single run.
ADAPTIVE_CAPS = {
    "subqueries_executed": 3,
    "retrieval_rounds": 2,
    "planning_calls": 1,
    "reranker_calls": 1,
}


def _orch(row: dict) -> Optional[dict]:
    o = row.get("orchestration")
    return o if isinstance(o, dict) else None


def _adaptive_rows(rows: list[dict]) -> list[dict]:
    """Rows that carried an orchestration block (i.e. the adaptive path ran)."""
    return [r for r in rows if _orch(r) is not None]


def _lane_of(row: dict) -> Optional[str]:
    o = _orch(row)
    if o and o.get("lane"):
        return o["lane"]
    return row.get("lane")


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile of ``values`` (q in [0,1]); 0.0 when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return round(ordered[idx], 1)


def lane_distribution(rows: list[dict]) -> dict:
    """Count of adaptive rows per selected lane."""
    dist: dict[str, int] = {}
    for r in _adaptive_rows(rows):
        lane = _lane_of(r) or "unknown"
        dist[lane] = dist.get(lane, 0) + 1
    return dist


def adaptive_fallback_rate(rows: list[dict]) -> dict:
    """Fraction of adaptive rows that demoted to the legacy retrieval path."""
    adaptive = _adaptive_rows(rows)
    n = len(adaptive)
    n_fb = sum(1 for r in adaptive if (_orch(r) or {}).get("fallback_reason"))
    return {"rate": round(n_fb / n, 3) if n else None, "n_eligible": n, "n_fallback": n_fb}


def budget_exhaustion_rate(rows: list[dict]) -> dict:
    """Fraction of adaptive rows that hit any *_budget_exhausted reason code."""
    adaptive = _adaptive_rows(rows)
    n = len(adaptive)
    n_ex = 0
    for r in adaptive:
        codes = (_orch(r) or {}).get("reason_codes") or []
        if any(str(c).endswith("_budget_exhausted") for c in codes):
            n_ex += 1
    return {"rate": round(n_ex / n, 3) if n else None, "n_eligible": n, "n_exhausted": n_ex}


def write_tool_route_count(rows: list[dict]) -> int:
    """Count deterministic-tool routes to a write tool. MUST be 0 (safety)."""
    total = 0
    for r in _adaptive_rows(rows):
        tools = (_orch(r) or {}).get("deterministic_tools") or []
        total += sum(1 for t in tools if t in _WRITE_TOOLS)
    return total


def budget_cap_violations(rows: list[dict]) -> dict:
    """Per-counter count of adaptive rows that EXCEEDED a hard cap (must be 0)."""
    violations = {k: 0 for k in ADAPTIVE_CAPS}
    for r in _adaptive_rows(rows):
        o = _orch(r) or {}
        for key, cap in ADAPTIVE_CAPS.items():
            try:
                if int(o.get(key, 0) or 0) > cap:
                    violations[key] += 1
            except (TypeError, ValueError):
                continue
    return violations


def per_lane_metrics(rows: list[dict]) -> dict:
    """Per-lane breakdown of the cheap scalar metrics + latency percentiles.

    Only lanes actually observed are emitted. The judge/ranking metrics for the
    live comparison are recorded separately in RESULTS.md; here each lane gets
    the deterministic scalars plus p50/p95 latency for the latency gate.
    """
    by_lane: dict[str, list[dict]] = {}
    for r in _adaptive_rows(rows):
        by_lane.setdefault(_lane_of(r) or "unknown", []).append(r)
    out: dict[str, dict] = {}
    for lane, lane_rows in by_lane.items():
        latencies = [float(r.get("latency_ms") or 0.0) for r in lane_rows]
        out[lane] = {
            "n": len(lane_rows),
            "intent_accuracy": intent_accuracy(lane_rows),
            "ticker_accuracy": ticker_accuracy(lane_rows),
            "retrieval_hit_rate": retrieval_hit_rate(lane_rows),
            "p50_latency_ms": _pct(latencies, 0.50),
            "p95_latency_ms": _pct(latencies, 0.95),
        }
    return out


def _mean_counter(rows: list[dict], key: str) -> Optional[float]:
    vals = []
    for r in rows:
        o = _orch(r) or {}
        try:
            vals.append(float(o.get(key, 0) or 0))
        except (TypeError, ValueError):
            continue
    return round(sum(vals) / len(vals), 3) if vals else None


def _max_counter(rows: list[dict], key: str) -> int:
    best = 0
    for r in rows:
        o = _orch(r) or {}
        try:
            best = max(best, int(o.get(key, 0) or 0))
        except (TypeError, ValueError):
            continue
    return best


def adaptive_metrics(rows: list[dict]) -> dict:
    """The full adaptive-orchestration summary block for a run (2.2.3.4).

    Empty-but-present shape when no row carried orchestration, so a config
    comparison can always read the same keys.
    """
    adaptive = _adaptive_rows(rows)
    counters = ("subqueries_executed", "retrieval_rounds",
                "planning_calls", "reranker_calls")
    return {
        "n_adaptive": len(adaptive),
        "lane_distribution": lane_distribution(rows),
        "fallback_rate": adaptive_fallback_rate(rows),
        "budget_exhaustion_rate": budget_exhaustion_rate(rows),
        "write_tool_routes": write_tool_route_count(rows),
        "budget_cap_violations": budget_cap_violations(rows),
        "avg_counters": {c: _mean_counter(adaptive, c) for c in counters},
        "max_counters": {c: _max_counter(adaptive, c) for c in counters},
        "per_lane": per_lane_metrics(rows),
    }


def has_adaptive_rows(rows: list[dict]) -> bool:
    """True when at least one row carried an orchestration block."""
    return any(_orch(r) is not None for r in rows)


# ── Aggregator ─────────────────────────────────────────────────────────

def _run_field(rows: list[dict], key: str) -> Optional[str]:
    """A stable per-run value denormalized onto every row (e.g. answer_policy,
    dataset_digest) — returns it if every row agrees, else None."""
    values = {r.get(key) for r in rows if r.get(key)}
    return next(iter(values)) if len(values) == 1 else None


def score_all(rows: list[dict], *, judge: Optional[Callable] = None,
              endpoint: str = DEFAULT_MODEL_ENDPOINT, timeout: float = 120.0,
              run_judge: bool = True) -> dict:
    """Compute every metric for a run. Returns the summary block.

    ``run_judge=False`` skips the (slow) LLM-judge calls; the judge metrics
    come back with ``score: None``. Per-category breakdowns cover the cheap
    scalar metrics.
    """
    n = len(rows)
    errs = trace_errors(rows)
    summary: dict = {
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "n_cases": n,
        "n_model_available": sum(1 for r in rows if r.get("model_available")),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "trace_errors": errs,
        "n_trace_errors": len(errs),
        "answer_policy": _run_field(rows, "answer_policy"),
        "dataset_digest": _run_field(rows, "dataset_digest"),
    }
    for m in SCALAR_METRICS:
        summary[m] = globals()[m](rows)

    # Denominators for the cheap scalar metrics (2.2.1.2 Step 4: "report
    # denominators ... for every metric"). The judge metrics report their own
    # n_eligible/n_scored/n_skipped inline.
    summary["denominators"] = {
        "intent_accuracy": sum(
            1 for r in rows if _norm_intent(_case(r).get("expected_intent")) is not None),
        "ticker_accuracy": n,
        "retrieval_hit_rate": n,
        "keyword_coverage": sum(1 for r in rows if _case(r).get("must_mention")),
        "refusal_rate": sum(1 for r in rows if r.get("answer") and r.get("model_available")),
    }

    if run_judge:
        summary["grounded_faithfulness"] = grounded_faithfulness(
            rows, judge=judge, endpoint=endpoint, timeout=timeout)
        summary["policy_compliance"] = policy_compliance(
            rows, judge=judge, endpoint=endpoint, timeout=timeout)
        summary["answer_relevance"] = answer_relevance(
            rows, judge=judge, endpoint=endpoint, timeout=timeout)
    else:
        empty = {"score": None, "n_scored": 0, "n_eligible": 0, "n_skipped": n,
                 "model_available": False, "per_case": []}
        summary["grounded_faithfulness"] = dict(empty)
        summary["policy_compliance"] = dict(empty)
        summary["answer_relevance"] = dict(empty)

    # Phase 2.2 conversational/compound metrics (2.2.1.3) — deterministic, and
    # only emitted when the run actually contains Phase 2.2 fixtures so a legacy
    # single-turn-only run is not forced to satisfy conversational categories.
    if has_phase22_fixtures(rows):
        conv = conversational_metrics(rows)
        summary.update(conv)
        summary["conversational_denominators"] = {
            m: {"n_eligible": conv[m]["n_eligible"], "n_pass": conv[m]["n_pass"]}
            for m in PHASE22_METRICS
        }
        summary["conversational_by_category"] = _conversational_per_category(rows)

    # Adaptive-orchestration metrics (2.2.3.4) — emitted only when the run
    # actually exercised the adaptive path or was explicitly labeled, so a
    # legacy run's summary is byte-compatible with the pre-2.2.3.4 shape.
    label = _run_field(rows, "config_label")
    if has_adaptive_rows(rows) or label:
        summary["config_label"] = label
        summary["adaptive"] = adaptive_metrics(rows)

    summary["per_category"] = _per_category(rows)
    return summary


def _conversational_per_category(rows: list[dict]) -> dict:
    """Per-category eligible/pass counts for each Phase 2.2 metric.

    Lets the report show that no category silently lost its denominator, per
    2.2.1.3 Step 4 ("report overall and per-category denominators")."""
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        cat = _case(r).get("category", "uncategorized")
        by_cat.setdefault(cat, []).append(r)
    out: dict[str, dict] = {}
    for cat, cat_rows in by_cat.items():
        blocks = conversational_metrics(cat_rows)
        cat_out = {}
        for m in PHASE22_METRICS:
            b = blocks[m]
            if b["n_eligible"]:
                cat_out[m] = {"n_eligible": b["n_eligible"], "n_pass": b["n_pass"]}
        if cat_out:
            out[cat] = {"n": len(cat_rows), **cat_out}
    return out


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
