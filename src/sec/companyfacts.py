"""
src/sec/companyfacts.py
SEC CompanyFacts client and deterministic XBRL observation normalization.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Optional

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _utc_timestamp(value: datetime) -> str:
    """Return a UTC timestamp using the repository's compact Z form."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def derive_period_kind(
    period_start: str,
    period_end: str,
    fiscal_period: str,
    form: str,
) -> str:
    """Classify an SEC fact as instant, quarterly, YTD, or annual."""
    if not period_start:
        return "instant"

    start = date.fromisoformat(period_start)
    end = date.fromisoformat(period_end)
    duration_days = (end - start).days + 1
    base_form = form.removesuffix("/A")

    if fiscal_period == "FY" or (base_form == "10-K" and duration_days >= 270):
        return "annual"
    if duration_days <= 120:
        return "quarterly"
    return "ytd"


class SECCompanyFactsIngestor:
    """Fetch and persist authoritative SEC CompanyFacts observations."""

    DEFAULT_CONFIG_PATH = (
        Path(__file__).parent.parent.parent / "configs/sec_companyfacts.yaml"
    )

    def __init__(
        self,
        store=None,
        config_path: Optional[Path] = None,
        config: Optional[dict] = None,
        session: Optional[requests.Session] = None,
        cik_resolver=None,
        clock: Optional[Callable[[], datetime]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        from src.storage.store import Store
        from src.utils.env import load_env

        load_env()
        self.store = store or Store()
        self.sqlite = getattr(self.store, "sqlite", self.store)
        explicit_config = config is not None
        self.config = config if explicit_config else self._load_config(config_path)
        self.enabled = bool(self.config.get("enabled", False))
        configured_user_agent = str(self.config.get("user_agent", "")).strip()
        self.user_agent = (
            configured_user_agent
            if explicit_config
            else os.environ.get("SEC_EDGAR_USER_AGENT") or configured_user_agent
        )
        if self.enabled and not self.user_agent:
            raise ValueError("SEC CompanyFacts requires a descriptive User-Agent")

        self.timeout_seconds = float(self.config.get("timeout_seconds", 30))
        self.request_delay_seconds = float(
            self.config.get("request_delay_seconds", 0.2)
        )
        self.allowed_taxonomies = set(
            self.config.get("allowed_taxonomies", ["us-gaap"])
        )
        self.allowed_forms = set(
            self.config.get("allowed_forms", ["10-K", "10-K/A", "10-Q", "10-Q/A"])
        )
        self.session = session or self._build_session()
        self._cik_resolver = cik_resolver
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or time.sleep
        self.last_skip_reasons: dict[str, int] = {}

    def _load_config(self, config_path: Optional[Path]) -> dict:
        path = config_path or self.DEFAULT_CONFIG_PATH
        with open(path, encoding="utf-8") as config_file:
            return yaml.safe_load(config_file) or {}

    def _build_session(self) -> requests.Session:
        retries = Retry(
            total=max(0, int(self.config.get("retries", 3))),
            connect=max(0, int(self.config.get("retries", 3))),
            read=max(0, int(self.config.get("retries", 3))),
            backoff_factor=float(self.config.get("backoff_factor", 0.5)),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        session.headers.update({"User-Agent": self.user_agent})
        return session

    def _resolver(self):
        if self._cik_resolver is None:
            from src.sec.edgar_fetcher import SECEdgarFilingFetcher

            self._cik_resolver = SECEdgarFilingFetcher(
                store=self.store,
                user_agent=self.user_agent,
            )
        return self._cik_resolver

    def fetch_for_ticker(self, ticker: str) -> dict:
        """Fetch, normalize, and commit CompanyFacts for one ticker."""
        normalized_ticker = ticker.upper()
        accessed_at = _utc_timestamp(self._clock())
        summary = {
            "ticker": normalized_ticker,
            "cik": "",
            "facts_seen": 0,
            "facts_written": 0,
            "facts_skipped": 0,
            "errors": [],
            "source_accessed_at": accessed_at,
        }
        if not self.enabled:
            return summary

        try:
            raw_cik = self._resolver().resolve_cik(normalized_ticker)
            if not raw_cik:
                raise ValueError(f"CIK not found for {normalized_ticker}")
            cik = str(raw_cik).zfill(10)
            summary["cik"] = cik
            source_url = COMPANYFACTS_URL.format(cik=cik)
            if self.request_delay_seconds > 0:
                self._sleep(self.request_delay_seconds)
            response = self.session.get(
                source_url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = json.loads(
                response.text,
                parse_float=Decimal,
                parse_int=Decimal,
            )
            rows, facts_seen, skip_reasons = self._normalize_payload(
                payload,
                normalized_ticker,
                cik,
                source_url,
                accessed_at,
            )
            summary["facts_seen"] = facts_seen
            summary["facts_skipped"] = sum(skip_reasons.values())
            self.last_skip_reasons = dict(skip_reasons)
            write_result = self.sqlite.upsert_sec_companyfacts(rows)
            summary["facts_written"] = int(write_result["rows_written"])
            if skip_reasons:
                logger.debug(
                    "CompanyFacts skipped observations for %s: %s",
                    normalized_ticker,
                    dict(skip_reasons),
                )
        except Exception as exc:  # noqa: BLE001 - one ticker must fail soft
            logger.error("CompanyFacts ingestion failed for %s: %s", normalized_ticker, exc)
            summary["errors"].append(str(exc))
        return summary

    def _normalize_payload(
        self,
        payload: dict,
        ticker: str,
        cik: str,
        source_url: str,
        source_accessed_at: str,
    ) -> tuple[list[dict], int, Counter]:
        rows: list[dict] = []
        seen = 0
        skipped: Counter = Counter()
        facts = payload.get("facts")
        if not isinstance(facts, dict):
            raise ValueError("CompanyFacts response has no facts object")

        for taxonomy, concepts in facts.items():
            if taxonomy not in self.allowed_taxonomies or not isinstance(concepts, dict):
                continue
            for concept, concept_data in concepts.items():
                units = concept_data.get("units", {})
                if not isinstance(units, dict):
                    continue
                for unit, observations in units.items():
                    if not isinstance(observations, list):
                        continue
                    for observation in observations:
                        seen += 1
                        row, reason = self._normalize_observation(
                            ticker=ticker,
                            cik=cik,
                            taxonomy=taxonomy,
                            concept=concept,
                            concept_data=concept_data,
                            unit=unit,
                            observation=observation,
                            source_url=source_url,
                            source_accessed_at=source_accessed_at,
                        )
                        if row is None:
                            skipped[reason] += 1
                        else:
                            rows.append(row)
        return rows, seen, skipped

    def _normalize_observation(
        self,
        *,
        ticker: str,
        cik: str,
        taxonomy: str,
        concept: str,
        concept_data: dict,
        unit: str,
        observation: dict,
        source_url: str,
        source_accessed_at: str,
    ) -> tuple[Optional[dict], str]:
        form = str(observation.get("form", ""))
        if form not in self.allowed_forms:
            return None, "disallowed_form"
        accession = str(observation.get("accn", "")).strip()
        if not accession:
            return None, "missing_accession"
        period_end = str(observation.get("end", "")).strip()
        if not period_end:
            return None, "missing_period_end"
        filed_at = str(observation.get("filed", "")).strip()
        if not filed_at:
            return None, "missing_filed_at"
        try:
            date.fromisoformat(filed_at)
        except ValueError:
            return None, "invalid_filed_at"

        value = observation.get("val")
        if isinstance(value, bool):
            return None, "non_numeric"
        try:
            decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None, "non_numeric"
        if not decimal_value.is_finite():
            return None, "non_finite"
        try:
            numeric_value = float(decimal_value)
        except (OverflowError, ValueError):
            return None, "non_finite"
        if not math.isfinite(numeric_value):
            return None, "non_finite"

        period_start = str(observation.get("start", "")).strip()
        try:
            period_kind = derive_period_kind(
                period_start,
                period_end,
                str(observation.get("fp", "")),
                form,
            )
        except ValueError:
            return None, "invalid_period"

        fiscal_year = observation.get("fy")
        try:
            fiscal_year = int(fiscal_year) if fiscal_year is not None else None
        except (TypeError, ValueError):
            fiscal_year = None

        return {
            "ticker": ticker,
            "cik": cik,
            "taxonomy": taxonomy,
            "concept": concept,
            "label": concept_data.get("label"),
            "description": concept_data.get("description"),
            "value_text": format(decimal_value, "f"),
            "value_numeric": numeric_value,
            "unit": str(unit),
            "period_start": period_start,
            "period_end": period_end,
            "period_kind": period_kind,
            "fiscal_year": fiscal_year,
            "fiscal_period": observation.get("fp"),
            "form": form,
            "filed_at": filed_at,
            "accession": accession,
            "frame": str(observation.get("frame", "") or ""),
            "source_url": source_url,
            "source_accessed_at": source_accessed_at,
        }, ""
