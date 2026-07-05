"""
src/middleware/guardrails.py
Fail-soft guardrails for forward-looking projection answers.
"""

import logging
import re

logger = logging.getLogger(__name__)

_SUFFIX_MULTIPLIERS = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
    "usd": 1.0,
}
_DOLLAR_RE = re.compile(
    r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*"
    r"(k|m|b|bn|t|thousand|million|billion|trillion)?\b",
    re.IGNORECASE,
)
# Bare numbers only count as support when marked as currency ("usd", as
# rendered by the fact/consensus sections) or carrying a magnitude suffix —
# otherwise analyst counts, years, and scores would vouch for dollar figures.
_BARE_NUMBER_RE = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?)\s*"
    r"(k|m|b|bn|t|thousand|million|billion|trillion|usd)\b",
    re.IGNORECASE,
)


def _parse_number(value: str, suffix: str | None = None) -> float:
    base = float(value.replace(",", ""))
    multiplier = _SUFFIX_MULTIPLIERS.get((suffix or "").lower(), 1.0)
    return base * multiplier


def extract_dollar_figures(text) -> list[float]:
    """Extract dollar-prefixed figures, applying k/m/b/t suffix multipliers."""
    figures = []
    for match in _DOLLAR_RE.finditer(str(text or "")):
        figures.append(_parse_number(match.group(1), match.group(2)))
    return figures


def collect_context_numbers(context_text) -> list[float]:
    """Collect dollar figures and currency/suffix-marked numbers from context."""
    values = []
    context = str(context_text or "")
    for regex in (_DOLLAR_RE, _BARE_NUMBER_RE):
        for match in regex.finditer(context):
            values.append(_parse_number(match.group(1), match.group(2)))
    return values


def unsupported_figures(answer, context_text) -> list[str]:
    """Return dollar figures in the answer that are not supported by context."""
    context_numbers = collect_context_numbers(context_text)
    unsupported = []
    for match in _DOLLAR_RE.finditer(str(answer or "")):
        value = _parse_number(match.group(1), match.group(2))
        supported = any(
            abs(value - number) / max(abs(number), 1e-9) <= 0.005
            for number in context_numbers
        )
        if not supported:
            unsupported.append(match.group(0).strip())
    return unsupported


def apply_projection_guardrail(answer, context_text) -> tuple[str, list[str]]:
    """Append a caveat for unsupported dollar figures; never rewrite prose."""
    try:
        flagged = unsupported_figures(answer, context_text)
        if not flagged:
            return answer, []

        logger.warning("Projection answer included unsupported figures: %s", flagged)
        caveat = (
            "\n\nNote: The following figures could not be verified against the "
            f"retrieved analyst estimates: {', '.join(flagged)}."
        )
        return f"{answer}{caveat}", flagged
    except Exception as exc:  # noqa: BLE001
        logger.warning("Projection guardrail failed: %s", exc)
        return answer, []
