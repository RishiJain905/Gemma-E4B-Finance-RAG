"""Macro-economic data ingestion (FRED, GDELT, etc.)."""
from .fred_ingestor import FREDIngestor
from .gdelt_ingestor import GDELTIngestor
from .ir_ingestor import IRIngestor

__all__ = ["FREDIngestor", "GDELTIngestor", "IRIngestor"]
