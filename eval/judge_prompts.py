"""
eval/judge_prompts.py — LLM-as-judge prompt templates for the eval harness.

The judge is the local model on :8087 (the same TraceAlchemy model the pipeline
uses). Each builder returns a prompt string; the scorer (metrics.py) posts it to
the model and parses a 0–1 score from the reply.

Scoring protocol (shared by both judges):
  The model is instructed to reason briefly and then emit a final line of the
  form ``Score: <float>`` in the inclusive range 0.0–1.0. ``metrics._parse_score``
  extracts the first such number and clamps it.
"""

from __future__ import annotations

# The system message frames the model as an evaluator. Keeping it short avoids
# the small local model drifting off-format.
JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of financial Q&A answers. "
    "Read the inputs carefully and judge by the stated rubric. "
    "Always end your reply with a line of exactly the form: "
    "'Score: <float between 0.0 and 1.0>'."
)

# Answer-level refusal phrasing the faithfulness judge should not penalize.
FAITHFULNESS_RUBRIC = """\
You are judging whether the ANSWER is faithful to (supported by) the retrieved CONTEXT.

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}

Rubric (0.0 = fully unsupported/hallucinated, 1.0 = fully supported by the context):
- 1.0  Every claim in the answer is directly supported by the context; numbers match.
- 0.7  Mostly supported; minor unstated framing or rounding.
- 0.4  Partially supported; mixes supported and unsupported claims.
- 0.1  Answer makes claims not in the context (hallucination) or contradicts it.
- If the answer correctly states it cannot answer / lacks data, score 1.0 (faithful refusal).
- If the context is empty and the answer asserts facts anyway, score 0.0.

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


def faithfulness_prompt(question: str, context: str, answer: str) -> str:
    """Build the faithfulness (groundedness) judge prompt."""
    return FAITHFULNESS_RUBRIC.format(
        question=question or "",
        context=context or "(no retrieved context)",
        answer=answer or "(empty answer)",
    )


def relevance_prompt(question: str, answer: str) -> str:
    """Build the answer-relevance judge prompt (context not required)."""
    return RELEVANCE_RUBRIC.format(
        question=question or "",
        answer=answer or "(empty answer)",
    )