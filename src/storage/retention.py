"""src/storage/retention.py: Explicit narrative retention policy and cutoff helpers."""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "storage.yaml"
DEFAULT_PERMANENT_ITEM_TYPES = frozenset({
    "sec_filing",
    "sec_exhibit",
    "issuer_release",
    "official_release",
    "event",
    "corporate_action",
    "membership",
    "observation",
})


@dataclass(frozen=True)
class RetentionPolicy:
    """Configuration for explicit corpus maintenance runs."""

    company_news_months: int = 24
    retain_raw_network_payloads: bool = False
    permanent_item_types: frozenset[str] = DEFAULT_PERMANENT_ITEM_TYPES
    max_items_per_run: int = 1_000

    def __post_init__(self) -> None:
        if self.company_news_months < 1:
            raise ValueError("company_news_months must be positive")
        if self.max_items_per_run < 1 or self.max_items_per_run > 10_000:
            raise ValueError("max_items_per_run must be between 1 and 10000")

    def company_news_cutoff(self, as_of: Optional[str | date | datetime] = None) -> str:
        """Return the calendar-month cutoff as a canonical UTC timestamp."""
        current = _as_date(as_of)
        total_months = current.year * 12 + current.month - 1 - self.company_news_months
        year, month_index = divmod(total_months, 12)
        month = month_index + 1
        day = min(current.day, calendar.monthrange(year, month)[1])
        return f"{date(year, month, day).isoformat()}T00:00:00Z"


def _as_date(value: Optional[str | date | datetime]) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError("as_of must be an ISO date or timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError("as_of must be an ISO date or timestamp") from exc


def load_retention_policy(config_path: Optional[Path] = None) -> RetentionPolicy:
    """Load retention defaults from storage config, falling back safely."""
    path = config_path or DEFAULT_CONFIG_PATH
    block: dict = {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        candidate = loaded.get("retention") or {}
        if isinstance(candidate, dict):
            block = candidate
    except (OSError, yaml.YAMLError):
        logger.warning("Could not load retention config from %s; using defaults", path)

    permanent = block.get("permanent_item_types", DEFAULT_PERMANENT_ITEM_TYPES)
    if not isinstance(permanent, (list, tuple, set, frozenset)):
        permanent = DEFAULT_PERMANENT_ITEM_TYPES
    return RetentionPolicy(
        company_news_months=int(block.get("company_news_months", 24)),
        retain_raw_network_payloads=bool(block.get("retain_raw_network_payloads", False)),
        permanent_item_types=frozenset(str(value) for value in permanent),
        max_items_per_run=int(block.get("max_items_per_run", 1_000)),
    )
