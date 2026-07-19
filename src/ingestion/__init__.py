"""
src/ingestion/__init__.py
Ingestion sub-package for Gemma-E4B-Finance-RAG.
"""

__all__ = [
    "YFinanceIngestor",
    "FinnhubIngestor",
    "MassiveIngestor",
    "FederalReserveIngestor",
    "TreasuryIngestor",
    "BLSIngestor",
    "BEAIngestor",
    "EIAIngestor",
    "NYFedIngestor",
    "CFTCIngestor",
    "OpenFDAIngestor",
    "NHTSAIngestor",
    "USAspendingIngestor",
]


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
    official = {
        "FederalReserveIngestor": ("official.federal_reserve", "FederalReserveIngestor"),
        "TreasuryIngestor": ("official.treasury", "TreasuryIngestor"),
        "BLSIngestor": ("official.bls", "BLSIngestor"),
        "BEAIngestor": ("official.bea", "BEAIngestor"),
        "EIAIngestor": ("official.eia", "EIAIngestor"),
        "NYFedIngestor": ("official.ny_fed", "NYFedIngestor"),
        "CFTCIngestor": ("official.cftc", "CFTCIngestor"),
        "OpenFDAIngestor": ("official.openfda", "OpenFDAIngestor"),
        "NHTSAIngestor": ("official.nhtsa", "NHTSAIngestor"),
        "USAspendingIngestor": ("official.usaspending", "USAspendingIngestor"),
    }
    if name in official:
        module_name, class_name = official[name]
        module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)
    raise AttributeError(name)
