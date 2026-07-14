"""
eval/judge_prompts.py — LLM-as-judge prompt templates for the eval harness.

The judge is the local model on :8087 (the same TraceAlchemy model the pipeline
uses). Each builder returns a prompt string; the scorer (metrics.py) posts it to
the model and parses a 0–1 score from the reply.

Scoring protocol (shared by every judge):
  The model is instructed to reason briefly and then emit a final line of the
  form ``Score: <float>`` in the inclusive range 0.0–1.0. ``metrics._parse_score``
  extracts the first such number and clamps it.

2.2.1.2 replaced the truncated ``context`` string the faithfulness judge used
to read with the exact evidence trace (``src/middleware/evidence_trace.py``):
usable facts/documents (untruncated, full provenance) plus every tool result
the model actually saw. ``grounded_faithfulness_prompt`` and
``policy_compliance_prompt`` render the trace directly — no second retrieval,
no truncation.
"""

from __future__ import annotations

import json as _json
from typing import Optional

# The system message frames the model as an evaluator. Keeping it short avoids
# the small local model drifting off-format.
JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of financial Q&A answers. "
    "Read the inputs carefully and judge by the stated rubric. "
    "Always end your reply with a line of exactly the form: "
    "'Score: <float between 0.0 and 1.0>'."
)

GROUNDED_FAITHFULNESS_RUBRIC = """\
You are judging whether the ANSWER is faithful to (fully supported by) the EXACT EVIDENCE the
model was given: retrieved facts, retrieved documents, and tool results. This is the complete
model-visible evidence for this answer — nothing else may justify a claim in it.

Question:
{question}

Evidence facts:
{facts}

Evidence documents:
{documents}

Tool results:
{tool_results}

Answer:
{answer}

Rubric (0.0 = fully unsupported/hallucinated, 1.0 = fully supported by the evidence):
- 1.0  Every claim in the answer is directly supported by the facts/documents/tool results above; numbers match.
- 0.7  Mostly supported; minor unstated framing or rounding.
- 0.4  Partially supported; mixes supported and unsupported claims.
- 0.1  Answer makes claims not present in the evidence (hallucination) or contradicts it.
- If the evidence above is empty and the answer states specific figures anyway, score 0.0.

Reason concisely (under 80 words), then end your reply with exactly: Score: <float 0.0-1.0>"""

POLICY_COMPLIANCE_RUBRIC = """\
You are judging whether the ANSWER complies with the financial-assistant answer policy for its
grounding mode.

Question:
{question}

Grounding mode: {grounding}

Evidence facts:
{facts}

Evidence documents:
{documents}

Tool results:
{tool_results}

Answer:
{answer}

Policy by mode:
- grounded / partial: every factual claim must be traceable to the evidence above; never invent
  numbers (prices, P/E, targets, revenue, margins, growth rates, dates, counts).
- general: must be clearly labeled as general knowledge (e.g. an explicit "not from your data"
  prefix) with a primary-source verification caveat, and must not state specific figures as if
  they were retrieved.
- refused: must clearly decline rather than guess, and the refusal must fit a genuinely
  unanswerable, unsafe, or missing-data case — not dodge a question the evidence could answer.

Rubric (0.0 = violates the policy for its mode, 1.0 = fully compliant):
- 1.0  Fully compliant with the rules for its mode.
- 0.5  Partially compliant (e.g. missing the caveat, borderline labeling, minor overreach).
- 0.0  Clearly violates the policy (invents figures, mislabels general knowledge as grounded, or
  refuses despite sufficient evidence above).

Reason concisely (under 80 words), then end your reply with exactly: Score: <float 0.0-1.0>"""

RELEVANCE_RUBRIC = """\
You are judging whether the ANSWER addresses the user's QUESTION.

Question:
{question}

Answer:
{answer}

Rubric (0.0 = does not address the question, 1.0 = directly and correctly addresses it):
- 1.0  Directly answers what was asked, with the right entity/metric/period.
- 0.7  Addresses the question but is vague or slightly off-scope.
- 0.4  Tangential — related but not an answer to this question.
- 0.1  Does not address the question at all.
- A correct, on-topic refusal ("I don't have that data") scores 0.6 (honest but not an answer).

Reason concisely (under 80 words), then end your reply with exactly: Score: <float 0.0-1.0>"""


# ── Evidence-trace rendering (shared by both trace-grounded judges) ────────

def _format_facts_for_judge(facts: list[dict]) -> str:
    if not facts:
        return "(none)"
    lines = []
    for f in facts:
        unit = f.get("unit") or ""
        lines.append(
            f"- {f.get('metric')}: {f.get('value')} {unit} "
            f"({f.get('period') or 'N/A'}) "
            f"[Source: {f.get('source_type')}/{f.get('ticker')}]"
        )
    return "\n".join(lines)


def _document_body(document: dict) -> str:
    """Untruncated body via the canonical/legacy field chain (evidence.py)."""
    for key in ("document", "text", "content"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _format_documents_for_judge(documents: list[dict]) -> str:
    if not documents:
        return "(none)"
    lines = []
    for d in documents:
        meta = d.get("metadata") or {}
        source = meta.get("source", d.get("source", "?"))
        ticker = meta.get("ticker", d.get("ticker", "?"))
        lines.append(f"- [{source}/{ticker}] {_document_body(d)}")
    return "\n\n".join(lines)


def _format_tool_results_for_judge(tool_results: list[dict]) -> str:
    if not tool_results:
        return "(none)"
    lines = []
    for t in tool_results:
        lines.append(
            f"- {t.get('name')}({t.get('arguments')}) -> {_json.dumps(t.get('result'))}"
        )
    return "\n".join(lines)


def grounded_faithfulness_prompt(question: str, trace: Optional[dict], answer: str) -> str:
    """Build the grounded-faithfulness judge prompt from an evidence trace."""
    trace = trace or {}
    return GROUNDED_FAITHFULNESS_RUBRIC.format(
        question=question or "",
        facts=_format_facts_for_judge(trace.get("facts") or []),
        documents=_format_documents_for_judge(trace.get("documents") or []),
        tool_results=_format_tool_results_for_judge(trace.get("tool_results") or []),
        answer=answer or "(empty answer)",
    )


def policy_compliance_prompt(question: str, trace: Optional[dict], answer: str,
                             grounding: Optional[str]) -> str:
    """Build the policy-compliance judge prompt from an evidence trace."""
    trace = trace or {}
    return POLICY_COMPLIANCE_RUBRIC.format(
        question=question or "",
        grounding=grounding or "unknown",
        facts=_format_facts_for_judge(trace.get("facts") or []),
        documents=_format_documents_for_judge(trace.get("documents") or []),
        tool_results=_format_tool_results_for_judge(trace.get("tool_results") or []),
        answer=answer or "(empty answer)",
    )


def relevance_prompt(question: str, answer: str) -> str:
    """Build the answer-relevance judge prompt (evidence not required)."""
    return RELEVANCE_RUBRIC.format(
        question=question or "",
        answer=answer or "(empty answer)",
    )
