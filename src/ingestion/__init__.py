"""
src/ingestion/__init__.py
Ingestion sub-package for Gemma-E4B-Finance-RAG.
"""

__all__ = ["YFinanceIngestor"]


def __getattr__(name: str):
    """Keep the legacy ingestor export without eager Store imports."""
    if name == "YFinanceIngestor":
        from .yfinance_ingestor import YFinanceIngestor

        return YFinanceIngestor
    raise AttributeError(name)
