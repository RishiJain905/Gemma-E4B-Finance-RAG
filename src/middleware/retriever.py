"""
1.5.1 placeholder — minimal retrieval stub.

Replaced by hybrid Retriever implementation in spec 1.5.3.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.storage.store import Store

    from .config import MiddlewareConfig


class Retriever:
    """Skeleton retriever that delegates to Store.search()."""

    def __init__(self, store: "Store", config: "MiddlewareConfig"):
        self.store = store
        self.config = config

    def retrieve(
        self,
        query: str,
        intent: dict,
        top_k_documents: int,
        top_k_facts: int,
    ) -> dict:
        ticker = intent.get("ticker")
        n_results = max(top_k_documents, top_k_facts)
        return self.store.search(query=query, n_results=n_results, ticker=ticker)
