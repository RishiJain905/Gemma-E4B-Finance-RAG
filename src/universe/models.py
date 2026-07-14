"""src/universe/models.py
Typed records and validation errors for security-universe refreshes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional


class SnapshotValidationError(ValueError):
    """Raised when a provider payload is unsafe to apply as a snapshot."""


def normalize_symbol(symbol: str) -> str:
    """Return the canonical comparison form for provider ticker symbols."""
    value = str(symbol or "").strip().upper()
    value = re.sub(r"[./]", "-", value)
    return re.sub(r"\s+", "", value)


@dataclass(frozen=True)
class UniverseRecord:
    """One provider security or index-constituent record."""

    symbol: str
    company_name: str
    source: str
    index_code: Optional[str] = None
    exchange: Optional[str] = None
    cik: Optional[str] = None
    security_type: str = "common_stock"
    share_class: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    source_url: Optional[str] = None

    def as_dict(self) -> dict:
        """Return a storage-ready dictionary without changing provider values."""
        return asdict(self)


@dataclass(frozen=True)
class UniverseRefreshResult:
    """Summary of one validated universe reconciliation run."""

    run_id: str
    source: str
    changed: bool
    securities_created: int = 0
    securities_updated: int = 0
    memberships_opened: int = 0
    memberships_closed: int = 0
    aliases_created: int = 0
    errors: int = 0
    revision: int = 0

    @classmethod
    def from_dict(cls, value: dict) -> "UniverseRefreshResult":
        """Build a typed result from the storage facade response."""
        return cls(**value)
