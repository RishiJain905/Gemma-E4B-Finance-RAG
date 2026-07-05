"""
src/middleware/tools/sanity.py
Config-driven data-sanity rules (excluded symbols, plausible value ranges) for analytical tools.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs/analytics.yaml"
_cache: Optional[dict] = None


def load_analytics_config() -> dict:
    """Load configs/analytics.yaml, caching the result. Fails soft to empty defaults."""
    global _cache
    if _cache is not None:
        return _cache

    defaults = {"excluded_symbols": [], "equity_only_metrics": [], "metric_ranges": {}}
    try:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH) as f:
                data = yaml.safe_load(f) or {}
            defaults.update({k: v for k, v in data.items() if v is not None})
    except Exception:
        logger.exception("Failed to load analytics config from %s; using defaults", _CONFIG_PATH)

    _cache = defaults
    return _cache


def excluded_symbols(metric: str) -> list[str]:
    """Return the configured exclusion list when `metric` is equity-only, else []."""
    config = load_analytics_config()
    if metric in (config.get("equity_only_metrics") or []):
        return list(config.get("excluded_symbols") or [])
    return []


def sane_range(metric: str) -> Optional[tuple]:
    """Return (min, max) for `metric`, with None for an absent bound, or None if unconfigured."""
    config = load_analytics_config()
    ranges = config.get("metric_ranges") or {}
    metric_range = ranges.get(metric)
    if not metric_range:
        return None
    return (metric_range.get("min"), metric_range.get("max"))


def _reset_cache():
    """Clear the cached config — for tests that swap in a different analytics.yaml."""
    global _cache
    _cache = None
