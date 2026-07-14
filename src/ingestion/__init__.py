"""
src/ingestion/__init__.py
Ingestion sub-package for Gemma-E4B-Finance-RAG.
"""

__all__ = ["YFinanceIngestor", "FinnhubIngestor", "MassiveIngestor"]


def __getattr__(name: str):
    """Keep the legacy ingestor export without eager Store imports."""
    if name == "YFinanceIngestor":
        from .yfinance_ingestor import YFinanceIngestor

        return YFinanceIngestor
    if name == "FinnhubIngestor":
        from .finnhub_ingestor import FinnhubIngestor

        return FinnhubIngestor
    if name == "MassiveIngestor":
        from .massive_ingestor import MassiveIngestor

        return MassiveIngestor
    raise AttributeError(name)
