"""
src/middleware/prompt_policy.py
Single owner of the strict/graded system prompts and tool-aware policy
guidance sent to the model. Every model call site — plain, streaming,
tool-loop (src/middleware/app.py), and the direct-eval path
(eval/run_eval.py) — builds its system message through
``build_system_prompt`` so live and offline requests can never silently
enforce different rules.

Authoritative answer-policy rules (what may be answered, how grounding
modes work, citation format, the "no fabricated numbers" rule) belong only
here, in the system message. ``PromptAugmenter`` owns the augmented user
message: retrieved evidence, intent-specific task guidance, the raw
question, and output format — it must not duplicate these rules.
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fixed_prefix_digest(system_prompt: str) -> str:
    """Short stable digest of the fixed system-policy prefix (2.2.6.2 Step 3).

    The system message built here is the reusable prompt prefix. Exposing a
    digest lets latency telemetry confirm the prefix stayed byte-stable across
    turns (a prerequisite for llama-server prompt reuse) without logging the
    prompt text. This module is the single owner of that prefix, so it owns its
    digest too.
    """
    return hashlib.sha256((system_prompt or "").encode("utf-8")).hexdigest()[:16]

# Byte-for-byte identical to the pre-2.2.1.1 middleware SYSTEM_PROMPT
# constant. Pinned by tests/test_answer_policy.py::test_strict_policy_is_unchanged
# — do not edit without updating that test deliberately.
STRICT_SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer the user's "
    "question using ONLY the provided context. If the context "
    "doesn't contain enough information, say so. "
    "Cite sources inline using [Source: type/ticker] notation."
)

# Used instead of STRICT_SYSTEM_PROMPT when tool-calling is enabled. The base
# prompt's "ONLY the provided context ... say so" rule contradicts tool use
# and makes the model refuse instead of calling a tool, so the tools-mode
# prompt replaces (not appends to) it. Tools-off requests never see this.
STRICT_TOOLS_SYSTEM_PROMPT = (
    "You are a financial research assistant with callable tools. Answer using "
    "the provided context and your tools. When the context does not already "
    "contain the answer — especially for ranking, filtering, or aggregating "
    "across stocks (use query_facts), targeted lookups (get_fundamentals), or "
    "data freshness (check_freshness) — call the appropriate tool rather than "
    "refusing. Only say the data is unavailable if the context and your tools "
    "cannot provide it. Answer strictly from context and tool results; never "
    "invent numbers. Cite sources inline using [Source: type/ticker] notation."
)


def _graded_mode_guidance(grounding_level: str, allow_general_fallback: bool) -> str:
    """Return the mode-specific paragraph for the graded system prompt."""
    mode_guidance = {
        "grounded": (
            "Mode: grounded. Answer using ONLY the retrieved facts/documents — "
            "every factual claim must come from them. Cite sourced claims "
            "inline using [Source: type/ticker]. Do not add outside knowledge "
            "and do not use the 'Not from your data' prefix."
        ),
        "partial": (
            "Mode: partial. Answer only the parts supported by retrieved data, "
            "explicitly name what is missing, and do not fill the gaps with "
            "outside knowledge or the 'Not from your data' prefix."
        ),
        "none": (
            "Mode: general fallback. No relevant stored facts/documents were "
            "retrieved. If the request is answerable from stable background "
            "knowledge, prefix the answer exactly with 'Not from your data - "
            "general knowledge:' and include a caveat to verify against a "
            "primary source. Refuse if the request is unsafe or genuinely "
            "unknowable."
        ),
        "general": (
            "Mode: general fallback. No usable requested evidence was found and "
            "the request needs no specific figures. Use only stable background "
            "knowledge, prefix the answer exactly with 'Not from your data - "
            "general knowledge:', and include a primary-source verification caveat."
        ),
        "refused": (
            "Mode: refuse. Required evidence is missing, stale, or conflicting. "
            "Briefly state the supplied missing-evidence reason and do not answer "
            "from background knowledge."
        ),
    }
    if grounding_level == "none" and not allow_general_fallback:
        mode_guidance["none"] = (
            "Mode: refuse. No relevant stored facts/documents were retrieved "
            "and general fallback is disabled. Say you do not have enough data "
            "instead of using background knowledge."
        )
    return mode_guidance.get(grounding_level, mode_guidance["none"])


def _graded_system_prompt(
    intent: Optional[dict],
    grounding_level: str,
    allow_general_fallback: bool,
    tools_enabled: bool,
) -> str:
    """Build the intent-aware graded answer-policy system prompt."""
    question_type = (intent or {}).get("question_type", "general")

    tool_guidance = ""
    if tools_enabled:
        tool_guidance = (
            "\n- Tools are available. Prefer calling the appropriate tool for "
            "targeted facts, ranking/filtering, or freshness before refusing. "
            "For long vs short / buy vs sell, you MUST call classify_trade_bias "
            "and report its bias (long, short, or neutral); do not guess."
        )

    return (
        "You are a financial research assistant.\n\n"
        "## Answer policy\n"
        "Choose one response mode from the grounding level provided.\n"
        "- Grounded: sufficient facts/docs -> answer and cite [Source: ...].\n"
        "- Partial: some relevant data -> answer what is supported and state "
        "what is missing.\n"
        "- General fallback: no relevant data -> clearly label general "
        "knowledge and include a primary-source verification caveat.\n"
        "- Refuse: only for genuinely unknowable or unsafe asks.\n\n"
        "Hard rule in every mode: never invent specific numbers such as "
        "prices, P/E, targets, revenue, margins, growth rates, dates, or "
        "counts. Specific figures must come from context or tools.\n"
        "The 'Not from your data - general knowledge:' prefix is reserved for "
        "the general-fallback mode only — never use it when any relevant data "
        "was retrieved.\n\n"
        f"Intent: {question_type}\n"
        f"Grounding level: {grounding_level}\n"
        f"{_graded_mode_guidance(grounding_level, allow_general_fallback)}"
        f"{tool_guidance}"
    )


def build_system_prompt(
    *,
    answer_policy: str,
    allow_general_fallback: bool,
    intent: Optional[dict],
    grounding_level: str,
    tools_enabled: bool = False,
) -> str:
    """Return the system prompt for a model call under the given policy.

    This is the only function that should build a system message anywhere
    in the pipeline — plain, streaming, tool-loop, and direct-eval calls all
    route through it so they cannot drift apart.

    Args:
        answer_policy: "strict" (byte-for-byte legacy regression prompt) or
            "graded" (intent/grounding-aware policy).
        allow_general_fallback: whether no-context general-knowledge answers
            are permitted when grounding_level is "none" (graded mode only).
        intent: parsed request intent, used for the "Intent: {question_type}"
            line in graded mode.
        grounding_level: "grounded" | "partial" | "general" | "refused";
            ``none`` remains the legacy no-evidence alias.
        tools_enabled: whether tool-calling guidance should be included.
    """
    policy = str(answer_policy or "graded").strip().lower()
    if policy == "strict":
        return STRICT_TOOLS_SYSTEM_PROMPT if tools_enabled else STRICT_SYSTEM_PROMPT
    return _graded_system_prompt(intent, grounding_level, allow_general_fallback, tools_enabled)
