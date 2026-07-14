"""src/universe/coverage.py
Deterministic, read-only coverage policy over the canonical security registry.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from src.storage.store import Store

from .models import normalize_symbol

logger = logging.getLogger(__name__)


class CoverageResolver:
    """Resolve source capabilities to canonical security coverage scopes."""

    CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "coverage.yaml"
    LEGACY_WATCHLIST_PATH = (
        Path(__file__).resolve().parents[2] / "configs" / "watchlist.yaml"
    )
    VALID_SCOPES = ("universe", "broad", "deep", "sector", "global")
    _SCOPE_ORDER = {scope: position for position, scope in enumerate(VALID_SCOPES)}
    _PAGE_SIZE = 100

    def __init__(
        self,
        store: Store,
        *,
        config_path: Optional[Path] = None,
        legacy_watchlist_path: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.config_path = Path(config_path) if config_path else self.CONFIG_PATH
        self.legacy_watchlist_path = (
            Path(legacy_watchlist_path)
            if legacy_watchlist_path
            else self.LEGACY_WATCHLIST_PATH
        )
        self.policy = self._load_policy()
        self.revision = str(self.policy.get("revision") or "unversioned")
        self._validate_policy()
        self._validate_explicit_tickers()

    @staticmethod
    def _default_sources() -> dict[str, dict]:
        """Return the one-release compatibility policy for a legacy watchlist."""
        return {
            "universe_ivv": {
                "enabled": True,
                "scopes": ["universe"],
                "capabilities": ["index_membership"],
            },
            "universe_nasdaq100": {
                "enabled": True,
                "scopes": ["universe"],
                "capabilities": ["index_membership"],
            },
            "universe_sec": {
                "enabled": True,
                "scopes": ["universe"],
                "capabilities": ["security_identity"],
            },
            "yfinance": {
                "enabled": True,
                "scopes": ["broad"],
                "capabilities": ["market_summary", "company_news"],
            },
            "yfinance_fundamentals": {
                "enabled": True,
                "scopes": ["broad"],
                "capabilities": ["market_summary"],
            },
            "yfinance_news": {
                "enabled": True,
                "scopes": ["broad"],
                "capabilities": ["company_news"],
            },
            "sec_filings": {
                "enabled": True,
                "scopes": ["broad"],
                "capabilities": ["sec_event_discovery"],
            },
            "sec_filing_text": {
                "enabled": True,
                "scopes": ["deep"],
                "capabilities": ["full_filing_text"],
            },
            "sec_companyfacts": {
                "enabled": True,
                "scopes": ["deep"],
                "capabilities": ["sec_companyfacts"],
            },
            "earnings_transcripts": {
                "enabled": True,
                "scopes": ["deep"],
                "capabilities": ["earnings_transcripts"],
            },
            "ir_pages": {
                "enabled": True,
                "scopes": ["deep"],
                "capabilities": ["investor_relations"],
            },
            "estimates": {
                "enabled": True,
                "scopes": ["deep"],
                "capabilities": ["analyst_estimates"],
            },
            "gdelt": {
                "enabled": True,
                "optional": True,
                "scopes": ["deep"],
                "capabilities": ["gdelt_news"],
            },
            "fred": {
                "enabled": True,
                "scopes": ["global"],
                "capabilities": ["official_macro"],
            },
            "openfda": {
                "enabled": False,
                "scopes": ["sector"],
                "sector_rules": ["health_care"],
                "capabilities": ["regulatory_events"],
            },
            "nhtsa": {
                "enabled": False,
                "scopes": ["sector"],
                "sector_rules": ["automotive"],
                "capabilities": ["vehicle_safety_events"],
            },
            "usaspending": {
                "enabled": False,
                "scopes": ["sector"],
                "sector_rules": ["government_contractors"],
                "capabilities": ["government_awards"],
            },
        }

    def _load_policy(self) -> dict:
        try:
            with open(self.config_path, encoding="utf-8") as config_file:
                loaded = yaml.safe_load(config_file) or {}
        except FileNotFoundError:
            if self.config_path != self.CONFIG_PATH or not self.legacy_watchlist_path.exists():
                raise ValueError(f"coverage configuration not found: {self.config_path}")
            with open(self.legacy_watchlist_path, encoding="utf-8") as config_file:
                loaded = yaml.safe_load(config_file) or {}

        if "core" in loaded and "sources" not in loaded:
            return self._legacy_policy(loaded)
        if not isinstance(loaded, dict):
            raise ValueError("coverage configuration must be a mapping")
        return loaded

    def _legacy_policy(self, watchlist: dict) -> dict:
        logger.warning(
            "Legacy watchlist coverage configuration is deprecated; "
            "migrate core/extended tickers to configs/coverage.yaml"
        )
        return {
            "revision": "legacy-watchlist-v1",
            "fallback_when_registry_empty": "deep",
            "deep": {
                "tickers": watchlist.get("core", []) or [],
                "allow_outside_indexes": True,
            },
            "broad": {"additions": watchlist.get("extended", []) or []},
            "sector": {
                "rules": {
                    "health_care": ["Health Care", "Healthcare"],
                    "automotive": ["Consumer Discretionary", "Automotive"],
                    "government_contractors": ["Industrials", "Aerospace & Defense"],
                }
            },
            "sources": self._default_sources(),
        }

    @staticmethod
    def _normalized_tickers(values: object, *, field: str) -> list[str]:
        if values is None:
            return []
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        normalized = [normalize_symbol(value) for value in values]
        if any(not value for value in normalized):
            raise ValueError(f"{field} contains an empty ticker")
        return sorted(set(normalized))

    def _validate_policy(self) -> None:
        if self.policy.get("fallback_when_registry_empty", "deep") not in {"deep", "empty"}:
            raise ValueError("fallback_when_registry_empty must be 'deep' or 'empty'")
        sources = self.policy.get("sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("coverage sources must be a non-empty mapping")
        sector_rules = (self.policy.get("sector") or {}).get("rules", {}) or {}
        if not isinstance(sector_rules, dict):
            raise ValueError("sector.rules must be a mapping")

        for name, config in sources.items():
            if not isinstance(config, dict):
                raise ValueError(f"coverage source {name} must be a mapping")
            scopes = config.get("scopes")
            if not isinstance(scopes, list) or not scopes:
                raise ValueError(f"coverage source {name} must declare scopes")
            invalid = [scope for scope in scopes if scope not in self.VALID_SCOPES]
            if invalid:
                raise ValueError(
                    f"invalid coverage scope for {name}: {', '.join(map(str, invalid))}"
                )
            rule_names = config.get("sector_rules", []) or []
            if "sector" in scopes:
                if not isinstance(rule_names, list) or not rule_names:
                    raise ValueError(f"sector source {name} must declare sector_rules")
                unknown = sorted(set(rule_names) - set(sector_rules))
                if unknown:
                    raise ValueError(
                        f"unknown sector rule for {name}: {', '.join(unknown)}"
                    )
            capabilities = config.get("capabilities", []) or []
            if not isinstance(capabilities, list):
                raise ValueError(f"capabilities for {name} must be a list")

        self._normalized_tickers(
            (self.policy.get("deep") or {}).get("tickers", []),
            field="deep.tickers",
        )
        self._normalized_tickers(
            (self.policy.get("broad") or {}).get("additions", []),
            field="broad.additions",
        )

    def _registry_is_empty(self) -> bool:
        rows = self.store.list_securities(active=None, limit=1, offset=0)
        return not isinstance(rows, list) or not rows

    def _deep_tickers(self) -> list[str]:
        return self._normalized_tickers(
            (self.policy.get("deep") or {}).get("tickers", []),
            field="deep.tickers",
        )

    def _broad_additions(self) -> list[str]:
        return self._normalized_tickers(
            (self.policy.get("broad") or {}).get("additions", []),
            field="broad.additions",
        )

    def _validate_explicit_tickers(self) -> None:
        if self._registry_is_empty():
            return
        deep = self._deep_tickers()
        broad = set(self._broad_tickers())
        outside = (self.policy.get("deep") or {}).get("allow_outside_indexes", [])
        allow_all = outside is True
        allowed = set()
        if isinstance(outside, list):
            allowed = set(self._normalized_tickers(outside, field="deep.allow_outside_indexes"))
        elif outside not in {True, False, None}:
            raise ValueError("deep.allow_outside_indexes must be a boolean or ticker list")
        unknown = [ticker for ticker in deep if ticker not in broad and not allow_all and ticker not in allowed]
        if unknown:
            raise ValueError(
                "deep tickers are outside the active index union without opt-in: "
                + ", ".join(unknown)
            )

    @staticmethod
    def _validate_as_of(as_of: Optional[str]) -> Optional[str]:
        if as_of is None:
            return None
        try:
            parsed = datetime.strptime(as_of, "%Y-%m-%d").date().isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("as_of must be an ISO date in YYYY-MM-DD format") from exc
        if parsed != as_of:
            raise ValueError("as_of must be an ISO date in YYYY-MM-DD format")
        return parsed

    def _broad_tickers(self, as_of: Optional[str] = None) -> list[str]:
        cutoff = self._validate_as_of(as_of)
        tickers: set[str] = set(self._broad_additions())
        membership_tickers: set[str] = set()
        if cutoff is None:
            for index_code in ("sp500", "nasdaq100"):
                for row in self.store.list_memberships(index_code=index_code, active=True):
                    ticker = normalize_symbol(row.get("ticker", ""))
                    if ticker:
                        membership_tickers.add(ticker)
        else:
            for index_code in ("sp500", "nasdaq100"):
                for row in self.store.list_memberships(index_code=index_code, active=None):
                    effective_from = str(row.get("effective_from") or "")
                    effective_to = row.get("effective_to")
                    if effective_from <= cutoff and (
                        not effective_to or str(effective_to) >= cutoff
                    ):
                        ticker = normalize_symbol(row.get("ticker", ""))
                        if ticker:
                            membership_tickers.add(ticker)
        if not membership_tickers and self._registry_is_empty():
            if self.policy.get("fallback_when_registry_empty", "deep") == "deep":
                return self._deep_tickers()
            return []
        tickers.update(membership_tickers)
        return sorted(tickers)

    def _all_active_securities(self) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while True:
            page = self.store.list_securities(
                active=True,
                limit=self._PAGE_SIZE,
                offset=offset,
            )
            rows.extend(page)
            if len(page) < self._PAGE_SIZE:
                break
            offset += self._PAGE_SIZE
        return rows

    def _sector_tickers(self, source_config: dict) -> list[str]:
        configured_rules = (self.policy.get("sector") or {}).get("rules", {}) or {}
        sectors: set[str] = set()
        for rule_name in source_config.get("sector_rules", []) or []:
            values = configured_rules.get(rule_name, []) or []
            if not isinstance(values, list):
                raise ValueError(f"sector rule {rule_name} must be a list")
            sectors.update(str(value) for value in values)
        return sorted(
            {
                normalize_symbol(row.get("ticker", ""))
                for row in self._all_active_securities()
                if row.get("sector") in sectors and row.get("ticker")
            }
        )

    def _source_config(self, source_name: str) -> dict:
        name = str(source_name or "").strip()
        config = (self.policy.get("sources") or {}).get(name)
        if config is None:
            raise ValueError(f"unknown coverage source: {name}")
        return config

    def tickers_for(self, source_name: str, as_of: Optional[str] = None) -> list[str]:
        """Return a deterministic, unique ticker list for one configured source."""
        config = self._source_config(source_name)
        if not bool(config.get("enabled", True)):
            return []
        tickers: set[str] = set()
        for scope in config.get("scopes", []):
            if scope == "broad":
                tickers.update(self._broad_tickers(as_of))
            elif scope == "deep":
                tickers.update(self._deep_tickers())
            elif scope == "sector":
                tickers.update(self._sector_tickers(config))
        return sorted(tickers)

    def scopes_for(self, security_id: str) -> list[str]:
        """Return every policy scope currently held by one canonical security."""
        security = self.store.get_security(security_id)
        if not security:
            return []
        ticker = normalize_symbol(security.get("ticker", ""))
        scopes = {"universe"}
        if ticker in set(self._broad_tickers()):
            scopes.add("broad")
        if ticker in set(self._deep_tickers()):
            scopes.add("deep")
        sector = security.get("sector")
        for values in ((self.policy.get("sector") or {}).get("rules", {}) or {}).values():
            if isinstance(values, list) and sector in values:
                scopes.add("sector")
                break
        return sorted(scopes, key=self._SCOPE_ORDER.__getitem__)

    def is_enabled(self, source_name: str, security_id: Optional[str] = None) -> bool:
        """Return whether a source is enabled globally or for one security."""
        config = self._source_config(source_name)
        if not bool(config.get("enabled", True)):
            return False
        if security_id is None:
            return True
        security = self.store.get_security(security_id)
        if not security:
            return False
        return normalize_symbol(security.get("ticker", "")) in set(
            self.tickers_for(source_name)
        )

    def _inclusion_reasons(self, config: dict, security: dict) -> list[str]:
        ticker = normalize_symbol(security.get("ticker", ""))
        reasons: list[str] = []
        scopes = config.get("scopes", [])
        if "broad" in scopes:
            memberships = self.store.list_memberships(
                security_id=security["security_id"], active=True
            )
            for index_code in sorted({row.get("index_code") for row in memberships}):
                if index_code:
                    reasons.append(f"active {index_code} membership")
            if ticker in set(self._broad_additions()):
                reasons.append("explicit broad addition")
            if not memberships and self._registry_is_empty() and ticker in set(self._deep_tickers()):
                reasons.append("empty-registry deep fallback")
        if "deep" in scopes and ticker in set(self._deep_tickers()):
            reasons.append("explicit deep ticker")
        if "sector" in scopes:
            sector = security.get("sector")
            rules = (self.policy.get("sector") or {}).get("rules", {}) or {}
            for rule_name in config.get("sector_rules", []) or []:
                if sector in (rules.get(rule_name, []) or []):
                    reasons.append(f"sector rule {rule_name}: {sector}")
        return list(dict.fromkeys(reasons))

    def explain(
        self,
        source_name: str,
        security_id: Optional[str] = None,
    ) -> dict:
        """Return a redacted explanation suitable for status and graph projections."""
        config = self._source_config(source_name)
        enabled = bool(config.get("enabled", True))
        scopes = sorted(set(config.get("scopes", [])), key=self._SCOPE_ORDER.__getitem__)
        capabilities = sorted(set(str(value) for value in config.get("capabilities", [])))
        if security_id is None:
            return {
                "source": source_name,
                "enabled": enabled,
                "coverage_scopes": scopes,
                "ticker_count": len(self.tickers_for(source_name)),
                "enabled_capabilities": capabilities if enabled else [],
                "policy_revision": self.revision,
            }

        security = self.store.get_security(security_id)
        ticker = normalize_symbol(security.get("ticker", "")) if security else None
        included = bool(security and enabled and ticker in set(self.tickers_for(source_name)))
        return {
            "source": source_name,
            "enabled": enabled,
            "included": included,
            "security_id": security_id,
            "ticker": ticker,
            "coverage_scopes": scopes,
            "inclusion_reasons": self._inclusion_reasons(config, security) if included else [],
            "enabled_capabilities": capabilities if enabled else [],
            "policy_revision": self.revision,
        }
