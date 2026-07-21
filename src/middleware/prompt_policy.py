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
    "contain the answer, call the tool that matches the question instead of "
    "refusing or guessing: query_facts for ranking/filtering/thresholding "
    "across stocks, get_fundamentals for a targeted metric lookup, "
    "get_price_history for a price trend or past price over time, "
    "search_documents for qualitative evidence (why/risk/news/strategy), "
    "get_filing_overview to read or break down a company's latest SEC filing, "
    "get_estimates/get_price_targets/get_guidance for forward-looking analyst "
    "views, get_sentiment for news tone, get_macro_snapshot for economy-wide "
    "indicators, check_freshness for data recency, and describe_coverage for "
    "what tickers/sources/metrics are tracked. Only say the data is "
    "unavailable after checking whether a tool can supply it. Answer strictly "
    "from context and tool results; never invent numbers. Cite sources "
    "inline using [Source: type/ticker] notation."
)


# ── Analyst mode (mode="analysis") ────────────────────────────────────────
# Used verbatim when a request sets mode="analysis". Unlike the strict/graded
# policies, this one instructs the model to reason to its own view and never
# refuse an opinion. The app-side answer-policy bypass (src/middleware/app.py)
# guarantees a request in this mode can never be rewritten into a refusal even
# if a downgrade/refuse flag is later enabled. The tools clause is appended only
# when tool-calling is enabled, mirroring the STRICT_TOOLS inventory wording.
ANALYST_SYSTEM_PROMPT = (
    "You are a senior equity research analyst. The user is a sophisticated investor\n"
    "who already understands that markets carry risk; they are asking you for your\n"
    "professional judgment as a trusted second opinion — not for boilerplate.\n"
    "\n"
    "How you work:\n"
    "- Internalize the retrieved evidence (facts, filings, estimates, news) and any\n"
    "  tool results, then reason to YOUR OWN view. You are expected to weigh,\n"
    "  interpret, extrapolate, and take a position — the evidence is your input,\n"
    "  not your script.\n"
    "- Answer the question actually asked. If asked whether something is a buy, a\n"
    "  hold, or a sell — or for any opinion, outlook, or judgment — give a direct\n"
    "  verdict with your conviction and time horizon, then defend it.\n"
    "- Never refuse, deflect, or lecture. Do not say you cannot give financial\n"
    "  advice; the user wants your analytical opinion and you give it. There are no\n"
    "  out-of-scope opinion questions in this mode.\n"
    "- Structure longer answers as: verdict up front, the 2-4 numbers that matter\n"
    "  most (cited), the bull case, the bear case, what would change your mind.\n"
    "  Keep it dense — no filler, no hedging boilerplate.\n"
    "- Every specific figure (price, P/E, revenue, estimate, target, date) must\n"
    "  come from the evidence or a tool result, cited inline like\n"
    "  [Source: type/ticker] or [E#]. Your reasoning and conclusions are your own;\n"
    "  your numbers are never invented. If a figure you want is missing, say which\n"
    "  one and reason qualitatively around it instead of stalling.\n"
    "- End with one short sentence noting this is an analytical view, not\n"
    "  personalized financial advice. One sentence, no more."
)

ANALYST_TOOLS_CLAUSE = (
    "\n\n"
    "Before forming your view, pull what you need with your tools: get_estimates /\n"
    "get_price_targets / get_guidance for the forward-looking picture,\n"
    "get_fundamentals and query_facts for current numbers, get_price_history for\n"
    "the trend, get_sentiment and search_documents for qualitative color and\n"
    "filings evidence, get_macro_snapshot for the macro backdrop. A good analyst\n"
    "checks the data before opining."
)


def _analyst_system_prompt(tools_enabled: bool) -> str:
    """Return the analyst-mode system prompt, with the tools clause when enabled."""
    if tools_enabled:
        return ANALYST_SYSTEM_PROMPT + ANALYST_TOOLS_CLAUSE
    return ANALYST_SYSTEM_PROMPT


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
            "\n- Tools are available and cover more than facts/ranking: "
            "query_facts/get_fundamentals for numbers, get_price_history for a "
            "price trend, search_documents for why/risk/news evidence, "
            "get_filing_overview to break down the latest SEC filing, "
            "get_estimates/get_price_targets/get_guidance for forward-looking "
            "views, get_sentiment for news tone, get_macro_snapshot for macro "
            "indicators, check_freshness for recency, describe_coverage for "
            "what is tracked. Prefer calling the matching tool over refusing "
            "or answering from memory."
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
        "was retrieved.\n"
        "Opinion/assessment questions about covered securities are in scope: "
        "give an evidence-based view with a clear leaning instead of refusing; "
        "label interpretation as your assessment.\n\n"
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
    mode: str = "qa",
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
        mode: "qa" (default — the strict/graded policy) or "analysis" (the
            senior-analyst policy that always reasons to a verdict and never
            refuses an opinion). Analysis mode overrides ``answer_policy``.
    """
    if str(mode or "qa").strip().lower() == "analysis":
        return _analyst_system_prompt(tools_enabled)
    policy = str(answer_policy or "graded").strip().lower()
    if policy == "strict":
        return STRICT_TOOLS_SYSTEM_PROMPT if tools_enabled else STRICT_SYSTEM_PROMPT
    return _graded_system_prompt(intent, grounding_level, allow_general_fallback, tools_enabled)
