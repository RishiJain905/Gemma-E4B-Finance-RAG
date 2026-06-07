"""
1.5.1 placeholder — minimal prompt assembly stub.

Replaced by full PromptAugmenter implementation in spec 1.5.4.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import MiddlewareConfig


class PromptAugmenter:
    """Skeleton prompt builder for Phase 1.5.1 middleware wiring."""

    def __init__(self, config: "MiddlewareConfig"):
        self.config = config

    def build_prompt(self, question: str, intent: dict, retrieval: dict) -> str:
        facts = retrieval.get("facts", [])
        documents = retrieval.get("documents", [])

        facts_text = (
            "\n".join(str(f) for f in facts) if facts else "No facts available."
        )
        docs_text = (
            "\n".join(d.get("text", str(d)) for d in documents)
            if documents
            else "No documents available."
        )

        return (
            f"Context:\n"
            f"Facts:\n{facts_text}\n\n"
            f"Documents:\n{docs_text}\n\n"
            f"Question: {question}"
        )
