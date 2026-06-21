"""
src/ingestion/__init__.py
Ingestion sub-package for Gemma-E4B-Finance-RAG.
"""

from .yfinance_ingestor import YFinanceIngestor

__all__ = ["YFinanceIngestor"]
