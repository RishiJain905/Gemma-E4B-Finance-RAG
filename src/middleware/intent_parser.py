"""
1.5.1 placeholder — minimal intent parsing stub.

Replaced by full IntentParser implementation in spec 1.5.2.
"""

from src.storage.store import Store


class IntentParser:
    """Skeleton intent parser for Phase 1.5.1 middleware wiring."""

    def parse(self, question: str, override_ticker: str | None = None) -> dict:
        ticker = override_ticker or Store._detect_ticker(question)
        return {
            "ticker": ticker,
            "metrics": [],
            "question_type": "general",
            "original_question": question,
        }
