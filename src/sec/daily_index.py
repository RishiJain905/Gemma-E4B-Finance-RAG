"""src/sec/daily_index.py
SEC daily-index discovery for the active broad-universe CIK set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
import json
import logging
import re
import time
from typing import Callable, Iterable, Optional
from zipfile import BadZipFile, ZipFile

import requests

from src.ingestion.errors import ErrorClass, ProviderError, safe_message
from src.storage.store import Store
from src.universe.coverage import CoverageResolver


logger = logging.getLogger(__name__)
_ARCHIVES = "https://www.sec.gov/Archives"
# 20-F is the annual report of a foreign private issuer (e.g. NBIS); treat it
# as a deep periodic form end-to-end so its full section text is indexed like a
# domestic 10-K. 6-K (foreign interim/event report) stays here for scope tagging
# but remains off the full-text index_forms set (event path).
_DEEP_PERIODIC_FORMS = frozenset({
    "10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "6-K", "6-K/A",
})


@dataclass(frozen=True)
class DailyIndexEntry:
    """One normalized row from an SEC master daily index."""

    cik: str
    company_name: str
    form: str
    filing_date: str
    filename: str
    accession: str
    source_url: str


def parse_daily_index(payload: str) -> list[DailyIndexEntry]:
    """Parse the pipe-delimited rows in an SEC master daily index."""
    rows: list[DailyIndexEntry] = []
    for line in payload.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        cik, company_name, form, filing_date, filename = parts
        name = filename.rsplit("/", 1)[-1]
        accession = re.sub(r"\.(?:txt|html?)$", "", name, flags=re.I)
        if not accession:
            continue
        rows.append(DailyIndexEntry(
            cik=cik.zfill(10),
            company_name=company_name,
            form=re.sub(r"\s+", " ", form.upper()),
            filing_date=filing_date,
            filename=filename,
            accession=accession,
            source_url=f"{_ARCHIVES}/{filename.lstrip('/')}",
        ))
    return rows


class SECDailyIndexDiscovery:
    """Download each new daily index once and atomically register filtered rows."""

    INDEX_BASE = "https://www.sec.gov/Archives/edgar/daily-index"
    SUBMISSIONS_BULK_URL = (
        "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
    )
    BOOTSTRAP_SOURCE = "sec_submissions_bootstrap"

    def __init__(
        self,
        store: Store,
        coverage_resolver: CoverageResolver,
        sec_config: dict,
        user_agent: str,
        request_delay: float = 0.5,
        http_get: Callable[..., object] = requests.get,
    ) -> None:
        if not str(user_agent or "").strip():
            raise ValueError("A declared SEC user agent is required")
        self.store = store
        self.coverage = coverage_resolver
        self.config = sec_config.get("sec", sec_config)
        self.user_agent = user_agent
        self.request_delay = max(float(request_delay), 0.0)
        self.http_get = http_get
        groups = self.config.get("forms", {})
        self.forms = {
            re.sub(r"\s+", " ", str(form).strip().upper())
            for values in groups.values()
            for form in (values or [])
        }

    def filter_entries(self, entries: Iterable[DailyIndexEntry]) -> list[DailyIndexEntry]:
        """Filter parsed rows locally using broad registry CIKs and configured forms."""
        securities = self._broad_securities_by_cik()
        return [entry for entry in entries if entry.cik in securities and entry.form in self.forms]

    def process_index(self, index_date: str, payload: str, source_url: str) -> dict[str, object]:
        """Register one index date atomically, checkpointing only after commit."""
        if self.store.get_sec_daily_index_status(index_date) == "processed":
            return {"discovered": 0, "registered": 0, "replayed": True}
        securities = self._broad_securities_by_cik()
        deep_tickers = set(self.coverage.tickers_for("sec_filing_text"))
        entries = [
            entry for entry in parse_daily_index(payload)
            if entry.cik in securities and entry.form in self.forms
        ]
        filings = []
        for entry in entries:
            security = securities[entry.cik]
            filings.append({
                "ticker": security["ticker"],
                "filing_type": entry.form,
                "filing_date": entry.filing_date,
                "period": "",
                "accession": entry.accession,
                "source_url": entry.source_url,
                "cik": entry.cik,
                "primary_document": None,
                "discovery_scope": (
                    "deep"
                    if security["ticker"] in deep_tickers and entry.form in _DEEP_PERIODIC_FORMS
                    else "broad"
                ),
            })
        registration = self.store.register_sec_daily_index(index_date, source_url, filings)
        return {
            "discovered": len(entries),
            "registered": int(registration["registered"]),
            "replayed": bool(registration["replayed"]),
        }

    def discover_dates(self, index_dates: Iterable[str]) -> dict[str, object]:
        """Download and process explicit dates, isolating failures by index date."""
        result: dict[str, object] = {"downloaded": 0, "registered": 0, "failed": 0, "errors": []}
        forbidden_dates: list[str] = []
        succeeded_dates: list[str] = []
        for index_date in index_dates:
            if self.store.get_sec_daily_index_status(index_date) in ("processed", "absent"):
                continue
            url = self.index_url(index_date)
            try:
                response = self.http_get(
                    url,
                    headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                    timeout=60,
                )
                response.raise_for_status()
                result["downloaded"] = int(result["downloaded"]) + 1
                processed = self.process_index(index_date, response.text, url)
                result["registered"] = int(result["registered"]) + int(processed["registered"])
                succeeded_dates.append(index_date)
                if self.request_delay:
                    time.sleep(self.request_delay)
            except Exception as exc:  # noqa: BLE001 - index dates fail independently
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code == 403:
                    forbidden_dates.append(index_date)
                normalized = (
                    exc
                    if isinstance(exc, ProviderError)
                    else ProviderError(
                        exc,
                        error_class=ErrorClass.TRANSIENT,
                        provider_wide=False,
                    )
                )
                logger.error(
                    "SEC daily index %s failed: %s",
                    index_date,
                    normalized.safe_message,
                )
                result["failed"] = int(result["failed"]) + 1
                errors = result["errors"]
                assert isinstance(errors, list)
                errors.append(f"{index_date}: {safe_message(normalized.safe_message)}")
                if result.get("error_class") is None or normalized.provider_wide:
                    result.update({
                        "error_class": normalized.error_class.value,
                        "retry_after": normalized.retry_after,
                        "reset_at": normalized.reset_at,
                        "provider_wide": normalized.provider_wide,
                        "circuit_open": normalized.circuit_open,
                    })
        # EDGAR never publishes a daily index for market holidays and its S3
        # answers 403 for the missing key forever. A 403 date is provably
        # absent — not rate limiting — only when a LATER date fetched fine in
        # the same pass (the provider was reachable, yet that key still 403s).
        # During a real block every date 403s, nothing is marked, and the
        # normal retry/backoff path stays in charge.
        if succeeded_dates:
            newest_success = max(succeeded_dates)
            absent = [d for d in forbidden_dates if d < newest_success]
            for index_date in absent:
                self.store.mark_sec_daily_index_absent(index_date, self.index_url(index_date))
                logger.info("SEC daily index %s marked absent (holiday/unpublished)", index_date)
            if absent:
                result["absent"] = absent
                result["failed"] = max(int(result["failed"]) - len(absent), 0)
                if int(result["failed"]) == 0:
                    for key in ("error_class", "retry_after", "reset_at",
                                "provider_wide", "circuit_open"):
                        result.pop(key, None)
        return result

    def discover(
        self,
        *,
        through: Optional[str] = None,
        allow_bootstrap: bool = True,
    ) -> dict[str, object]:
        """Refresh from the durable cursor, optionally allowing initial bulk bootstrap."""
        bootstrap: dict[str, object] = {
            "discovered": 0, "registered": 0, "replayed": True,
        }
        bootstrap_error: Optional[str] = None
        if allow_bootstrap and self.store.get_sec_daily_index_cursor() is None:
            try:
                bootstrap = self.bootstrap_from_submissions_bulk()
            except Exception as exc:  # noqa: BLE001 - daily refresh remains independent
                bootstrap_error = str(exc)
                logger.error("SEC submissions bulk bootstrap failed: %s", exc)
        end = date.fromisoformat(through) if through else datetime.now().date()
        cursor = self.store.get_sec_daily_index_cursor()
        overlap = max(int(self.config.get("daily_index_overlap_days", 3)), 0)
        lookback = max(int(self.config.get("daily_index_bootstrap_days", 7)), 1)
        start = date.fromisoformat(cursor) - timedelta(days=overlap) if cursor else end - timedelta(days=lookback - 1)
        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                dates.append(current.isoformat())
            current += timedelta(days=1)
        result = self.discover_dates(dates)
        result["bootstrap"] = bootstrap
        if bootstrap_error:
            result["failed"] = int(result["failed"]) + 1
            errors = result["errors"]
            assert isinstance(errors, list)
            errors.append(f"submissions_bulk: {bootstrap_error}")
        return result

    def bootstrap_from_submissions_bulk(self) -> dict[str, object]:
        """Bootstrap broad filings from one SEC submissions bulk download."""
        status = self.store.get_cache_status("SCHEDULER", self.BOOTSTRAP_SOURCE)
        if status and status.get("status") == "fresh":
            return {"discovered": 0, "registered": 0, "replayed": True}
        url = str(self.config.get("submissions_bulk_url") or self.SUBMISSIONS_BULK_URL)
        response = self.http_get(
            url,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=180,
        )
        response.raise_for_status()
        filings = self._parse_submissions_bulk(response.content)
        registered = self.store.register_sec_filings(filings)
        self.store.mark_cache_fresh("SCHEDULER", self.BOOTSTRAP_SOURCE, 24 * 365 * 100)
        if self.request_delay:
            time.sleep(self.request_delay)
        return {
            "discovered": len(filings),
            "registered": registered,
            "replayed": False,
        }

    def _parse_submissions_bulk(self, payload: bytes) -> list[dict]:
        """Filter SEC bulk-submission members by broad CIK and form policy."""
        securities = self._broad_securities_by_cik()
        deep_tickers = set(self.coverage.tickers_for("sec_filing_text"))
        filings: list[dict] = []
        try:
            archive = ZipFile(BytesIO(payload))
        except BadZipFile as exc:
            raise ValueError("SEC submissions bulk payload is not a valid ZIP archive") from exc
        with archive:
            members = set(archive.namelist())
            for cik, security in securities.items():
                member = f"CIK{cik}.json"
                if member not in members:
                    logger.warning("SEC submissions bulk archive has no member for CIK %s", cik)
                    continue
                submission = json.loads(archive.read(member))
                recent = (submission.get("filings") or {}).get("recent") or {}
                forms = recent.get("form") or []
                for index, raw_form in enumerate(forms):
                    form = re.sub(r"\s+", " ", str(raw_form).strip().upper())
                    if form not in self.forms:
                        continue
                    accession = self._array_value(recent, "accessionNumber", index)
                    primary_document = self._array_value(recent, "primaryDocument", index)
                    if not accession or not primary_document:
                        continue
                    items_text = self._array_value(recent, "items", index)
                    filings.append({
                        "ticker": security["ticker"],
                        "filing_type": form,
                        "filing_date": self._array_value(recent, "filingDate", index),
                        "period": self._array_value(recent, "reportDate", index),
                        "accession": accession,
                        "source_url": self._filing_document_url(
                            cik, accession, primary_document,
                        ),
                        "cik": cik,
                        "primary_document": primary_document,
                        "discovery_scope": (
                            "deep"
                            if security["ticker"] in deep_tickers
                            and form in _DEEP_PERIODIC_FORMS
                            else "broad"
                        ),
                        "items": re.findall(r"\b\d+\.\d{2}\b", items_text),
                    })
        return filings

    @staticmethod
    def _array_value(recent: dict, name: str, index: int) -> str:
        values = recent.get(name) or []
        return str(values[index]).strip() if index < len(values) else ""

    @staticmethod
    def _filing_document_url(cik: str, accession: str, document: str) -> str:
        return (
            f"{_ARCHIVES}/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{document}"
        )

    @classmethod
    def index_url(cls, index_date: str) -> str:
        """Build the canonical SEC master-index URL for an ISO date."""
        value = date.fromisoformat(index_date)
        quarter = (value.month - 1) // 3 + 1
        return f"{cls.INDEX_BASE}/{value.year}/QTR{quarter}/master.{value:%Y%m%d}.idx"

    def _broad_securities_by_cik(self) -> dict[str, dict]:
        securities: dict[str, dict] = {}
        for ticker in self.coverage.tickers_for("sec_filings"):
            security = self.store.resolve_security(ticker)
            if not security or not security.get("cik"):
                logger.warning("Skipping broad SEC ticker without registry CIK: %s", ticker)
                continue
            cik = str(security["cik"]).zfill(10)
            securities[cik] = security
        return securities


DailyIndexDiscovery = SECDailyIndexDiscovery
