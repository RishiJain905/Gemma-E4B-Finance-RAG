"""
src/sec/edgar_fetcher.py
SEC EDGAR filing discovery and download.

Filing Types:
  10-K  — Annual report (comprehensive financials)
  10-Q  — Quarterly report (condensed financials)
  8-K   — Current report (material events)

Ticker → CIK mapping is resolved from SEC's company_tickers.json and the
EdgarClient.get_submissions endpoint provides the per-company filing index.
"""

import html as html_lib
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
import yaml
from sec_edgar_api import EdgarClient

from src.storage.store import Store

logger = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def fetch_sec_company_tickers(user_agent: str) -> list[dict]:
    """Download SEC company_tickers.json rows normalized for symbol catalogs."""
    resp = requests.get(
        COMPANY_TICKERS_URL,
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for entry in data.values():
        symbol = str(entry.get("ticker", "")).upper()
        cik_raw = entry.get("cik_str", "")
        title = str(entry.get("title", "")).strip()
        if symbol and cik_raw != "":
            rows.append(
                {
                    "ticker": symbol,
                    "cik": str(cik_raw).zfill(10),
                    "title": title,
                }
            )
    return rows


class SECEdgarFilingFetcher:
    """Discovers and downloads SEC EDGAR filings for tracked tickers.

    Usage:
        fetcher = SECEdgarFilingFetcher()
        filings = fetcher.discover_filings("NVDA", filing_types=["10-K", "10-Q"])
        text = fetcher.download_filing_text(filings[0])
    """

    # ── Config ─────────────────────────────────────────

    # Rate limiting: SEC EDGAR requests are logged and throttled.
    REQUEST_DELAY_SECONDS = 0.5  # Be kind to SEC servers

    # Filing types we care about
    FINANCIAL_FILING_TYPES = ["10-K", "10-Q", "8-K"]

    # SEC endpoints not covered by sec-edgar-api
    _COMPANY_TICKERS_URL = COMPANY_TICKERS_URL
    _ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

    # Fallback when no user_agent is configured anywhere.
    _DEFAULT_USER_AGENT = "TraceAlchemy Research contact@tracealchemy.example.com"

    # ── Init ───────────────────────────────────────────

    def __init__(
        self,
        store: Optional[Store] = None,
        request_delay: float = REQUEST_DELAY_SECONDS,
        user_agent: Optional[str] = None,
        sec_config: Optional[dict] = None,
    ):
        from src.utils.env import load_env
        load_env()  # ensure .env credentials are available in os.environ

        self.store = store or Store()
        self.request_delay = request_delay
        # Resolution order: constructor param > env var > config default.
        self._user_agent = (
            user_agent
            or os.environ.get("SEC_EDGAR_USER_AGENT")
            or self._load_config_user_agent()
        )
        self._client = EdgarClient(user_agent=self._user_agent)
        self.sec_config = sec_config or self._load_sec_config()
        # Lazily-populated ticker -> 10-digit CIK map (company_tickers.json).
        self._ticker_cik_cache: Optional[dict] = None

    # ── Public API ─────────────────────────────────────

    def discover_filings(
        self,
        ticker: str,
        filing_types: Optional[list[str]] = None,
        count: int = 10,
    ) -> list[dict]:
        """Discover the most recent SEC filings for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "NVDA")
            filing_types: List of filing types (default: ["10-K", "10-Q"])
            count: Max filings to return per type

        Returns:
            List of filing dicts with keys:
                ticker, filing_type, filing_date, period, accession,
                source_url, cik
        """
        filing_types = filing_types or self.FINANCIAL_FILING_TYPES[:2]

        cik = self._resolve_cik(ticker)
        if not cik:
            logger.warning("Could not resolve CIK for ticker %s", ticker)
            return []

        try:
            submissions = self._client.get_submissions(cik=cik)
            self._respect_rate_limit()
        except Exception as e:
            logger.error("Failed to fetch submissions for %s (CIK %s): %s", ticker, cik, e)
            return []

        all_filings = []
        for filing_type in filing_types:
            try:
                raw_filings = self._parse_submissions(submissions, filing_type, count, cik)
                parsed = self._parse_filing_results(ticker, filing_type, raw_filings)
                all_filings.extend(parsed)
            except Exception as e:
                logger.error(
                    "Failed to parse %s filings for %s: %s",
                    filing_type, ticker, e,
                )

        return all_filings

    def discover_all_core_tickers(
        self,
        filing_types: Optional[list[str]] = None,
        count: int = 10,
    ) -> list[dict]:
        """Discover filings for ALL core tickers from the watchlist."""
        from src.ingestion.yfinance_ingestor import YFinanceIngestor

        ingestor = YFinanceIngestor(store=self.store)
        all_filings = []

        for ticker in ingestor.core_tickers:
            logger.info("Discovering SEC filings for %s...", ticker)
            filings = self.discover_filings(ticker, filing_types, count)
            all_filings.extend(filings)

        logger.info(
            "Discovered %d total filings across %d core tickers",
            len(all_filings), len(ingestor.core_tickers),
        )
        return all_filings

    def register_discovered_filings(self, ticker: str) -> int:
        """Discover AND register new filings in the filings table.

        Only registers filings that don't already exist (by accession).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Number of newly registered filings
        """
        filings = self.discover_filings(ticker)
        registered = 0

        for filing in filings:
            # Check if already registered (idempotent)
            existing = self._get_registered_filing(filing["accession"])
            if existing:
                logger.debug(
                    "Filing %s already registered, skipping", filing["accession"]
                )
                continue

            success = self.store.register_filing(
                ticker=filing["ticker"],
                filing_type=filing["filing_type"],
                filing_date=filing["filing_date"],
                period=filing["period"],
                accession=filing["accession"],
                source_url=filing["source_url"],
                cik=filing.get("cik"),
                primary_document=filing.get("primary_document"),
                discovery_scope="deep",
            )
            if success:
                registered += 1
                logger.info(
                    "Registered new filing: %s %s (%s)",
                    ticker, filing["filing_type"], filing["period"],
                )

        logger.info(
            "Ticker %s: %d new filings registered (from %d discovered)",
            ticker, registered, len(filings),
        )
        return registered

    def register_all_core_tickers(self) -> dict[str, int]:
        """Register filings for all core tickers. Returns {ticker: count}."""
        from src.ingestion.yfinance_ingestor import YFinanceIngestor

        ingestor = YFinanceIngestor(store=self.store)
        results = {}

        for ticker in ingestor.core_tickers:
            count = self.register_discovered_filings(ticker)
            results[ticker] = count

        return results

    def download_filing_text(self, filing_record: dict) -> Optional[str]:
        """Download the full filing text from EDGAR.

        Args:
            filing_record: A filing dict (from discover_filings or filings table)

        Returns:
            The full filing text as a string, or None on failure
        """
        accession = filing_record.get("accession")
        if not accession:
            logger.error("No accession number in filing record")
            return None

        try:
            url = self._resolve_archive_url(filing_record)
            if not url:
                logger.error("Could not build archive URL for filing %s", accession)
                return None

            resp = requests.get(
                url,
                headers={"User-Agent": self._user_agent},
                timeout=60,
            )
            resp.raise_for_status()
            self._respect_rate_limit()

            text = self._strip_html(resp.text)

            if not text or len(text.strip()) < 50:
                logger.warning(
                    "Filing %s returned empty or very short text (%d chars)",
                    accession, len(text or ""),
                )
                return None

            logger.info("Downloaded filing %s: %d chars", accession, len(text))
            return text

        except Exception as e:
            logger.error("Failed to download filing %s: %s", accession, e)
            return None

    def download_relevant_documents(self, filing_record: dict) -> list[dict]:
        """Fetch the primary document and only policy-selected exhibits."""
        accession = str(filing_record.get("accession") or "").strip()
        if not accession:
            logger.error("No accession number in filing record")
            return []

        rows = self._list_filing_documents(filing_record)
        form = str(filing_record.get("filing_type") or filing_record.get("form") or "").upper()
        primary_name = str(filing_record.get("primary_document") or "")
        primary = next(
            (
                row for row in rows
                if (primary_name and row["name"] == primary_name) or row["document_type"] == form
            ),
            rows[0] if rows else None,
        )
        if primary is None:
            url = self._resolve_archive_url(filing_record)
            if not url:
                return []
            if url.split("?", 1)[0].lower().endswith(".txt"):
                logger.warning(
                    "Primary document unresolved for %s; refusing complete submission package",
                    accession,
                )
                return []
            primary = {"name": url.rsplit("/", 1)[-1], "document_type": form, "url": url}

        primary_text = self._download_document(primary["url"], accession)
        if not primary_text:
            return []
        documents = [{
            "document_type": "PRIMARY",
            "source_url": primary["url"],
            "text": primary_text,
        }]
        items = set(re.findall(r"(?i)\bitem\s+(\d+\.\d{2})\b", primary_text))
        configured_items = {
            str(value) for value in (self.sec_config.get("exhibits", {}).get(
                "material_agreement_items", ["1.01", "2.01", "2.03"]
            ) or [])
        }
        allowed = {
            str(value).upper()
            for value in (self.sec_config.get("exhibits", {}).get("always", ["EX-99.1"]) or [])
        }
        if items & configured_items:
            allowed.add("EX-10")
        allowlist = self.sec_config.get("exhibits", {}).get("allowlist", {}) or {}
        if isinstance(allowlist, dict):
            for value in allowlist.get(form, []) or []:
                allowed.add(str(value).upper())
            by_form = allowlist.get("by_form", {}) or {}
            for value in by_form.get(form, []) or []:
                allowed.add(str(value).upper())
            by_item = allowlist.get("by_item", {}) or {}
            for item in items:
                for value in by_item.get(item, []) or []:
                    allowed.add(str(value).upper())

        for row in rows:
            if row is primary:
                continue
            document_type = row["document_type"].upper()
            selected = document_type in allowed or (
                "EX-10" in allowed and document_type.startswith("EX-10")
            )
            if not selected:
                continue
            text = self._download_document(row["url"], accession)
            if text:
                documents.append({
                    "document_type": document_type,
                    "source_url": row["url"],
                    "text": text,
                })
        return documents

    # ── Internal Helpers ───────────────────────────────

    def resolve_cik(self, ticker: str) -> Optional[str]:
        """Resolve a ticker symbol to its zero-padded 10-digit CIK.

        Fetches SEC's company_tickers.json once and caches it on the instance.
        Returns None if the ticker is unknown.
        """
        if self._ticker_cik_cache is None:
            self._ticker_cik_cache = self._load_ticker_cik_map()

        cik = self._ticker_cik_cache.get(ticker.upper())
        if not cik:
            logger.warning("Ticker %s not found in SEC company_tickers map", ticker)
        return cik

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        """Compatibility wrapper for callers using the former private API."""
        return self.resolve_cik(ticker)

    def _load_ticker_cik_map(self) -> dict:
        """Download and parse company_tickers.json into a ticker -> CIK map."""
        try:
            rows = fetch_sec_company_tickers(self._user_agent)
            self._respect_rate_limit()
        except Exception as e:
            logger.error("Failed to load SEC company_tickers.json: %s", e)
            return {}

        mapping = {}
        for entry in rows:
            symbol = entry.get("ticker", "")
            cik = entry.get("cik", "")
            if symbol and cik:
                mapping[symbol] = cik
        return mapping

    def _parse_submissions(
        self, submissions: dict, filing_type: str, count: int, cik: str,
    ) -> list[dict]:
        """Zip the filings.recent parallel arrays into per-filing dicts.

        Filters by form type and truncates to `count`. Captures the
        primaryDocument and CIK needed to build canonical archive URLs.
        """
        recent = (submissions or {}).get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        items = []
        for i, form in enumerate(forms):
            if form != filing_type:
                continue
            items.append(
                {
                    "accessionNumber": accession_numbers[i] if i < len(accession_numbers) else "",
                    "filingDate": filing_dates[i] if i < len(filing_dates) else "",
                    "periodOfReport": report_dates[i] if i < len(report_dates) else "",
                    "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
                    "cik": cik,
                }
            )
            if len(items) >= count:
                break
        return items

    def _parse_filing_results(
        self, ticker: str, filing_type: str, raw_filings,
    ) -> list[dict]:
        """Parse raw sec-edgar-api results into our filing dict format."""
        if not raw_filings:
            return []

        filings = []

        # sec-edgar-api returns filings in different formats depending on
        # the version. We handle both list and dict responses.
        items = raw_filings
        if isinstance(raw_filings, dict):
            items = raw_filings.get("filings", raw_filings.get("results", []))

        for item in items:
            if isinstance(item, dict):
                # Try common field names from sec-edgar-api
                accession = (
                    item.get("accessionNumber")
                    or item.get("accession")
                    or item.get("accession_number")
                )
                if not accession:
                    continue

                period = item.get("period") or item.get("periodOfReport", "")
                if not period:
                    # Derive period from filing date for 10-K (annual) / 10-Q (quarterly)
                    period = self._derive_period(
                        item.get("filingDate", ""), filing_type,
                    )

                cik = item.get("cik", "")
                primary_document = (
                    item.get("primaryDocument")
                    or item.get("primary_document")
                    or ""
                )

                filing = {
                    "ticker": ticker.upper(),
                    "filing_type": filing_type,
                    "filing_date": item.get("filingDate", item.get("filing_date", "")),
                    "period": period,
                    "accession": accession,
                    "source_url": self._build_archive_url(cik, accession, primary_document),
                    "cik": cik,
                    "primary_document": primary_document,
                }
                filings.append(filing)

        return filings

    def _build_archive_url(self, cik, accession: str, primary_document: str) -> str:
        """Build the canonical EDGAR archive URL for a filing's primary document.

        Format: https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{primary_document}
        The CIK in the archive path has no leading zeros.
        """
        if not cik or not accession:
            return ""
        try:
            cik_int = int(cik)
        except (TypeError, ValueError):
            cik_int = str(cik).lstrip("0") or "0"
        accession_nodashes = accession.replace("-", "")
        if primary_document:
            return f"{self._ARCHIVES_BASE}/{cik_int}/{accession_nodashes}/{primary_document}"
        # Fall back to the filing index page when the primary doc is unknown.
        return f"{self._ARCHIVES_BASE}/{cik_int}/{accession_nodashes}/"

    def _resolve_archive_url(self, filing_record: dict) -> Optional[str]:
        """Determine the document URL to download for a filing record.

        Prefers an existing canonical archive source_url; otherwise rebuilds it,
        re-resolving the primary document via get_submissions when the record
        (e.g. a row from the filings table) lacks it.
        """
        source_url = filing_record.get("source_url", "")
        # A usable canonical URL points at a concrete document (not an index dir).
        if (
            source_url
            and source_url.startswith(self._ARCHIVES_BASE)
            and not source_url.endswith("/")
        ):
            return source_url

        accession = filing_record.get("accession", "")
        cik = filing_record.get("cik") or self._resolve_cik(filing_record.get("ticker", ""))
        primary_document = (
            filing_record.get("primaryDocument")
            or filing_record.get("primary_document")
            or ""
        )

        if not primary_document and cik and accession:
            primary_document = self._resolve_primary_document(cik, accession)

        url = self._build_archive_url(cik, accession, primary_document)
        return url or None

    def _resolve_primary_document(self, cik, accession: str) -> str:
        """Look up the primaryDocument for an accession via get_submissions."""
        try:
            cik_padded = str(int(cik)).zfill(10)
            submissions = self._client.get_submissions(cik=cik_padded)
            self._respect_rate_limit()
        except Exception as e:
            logger.error("Failed to re-resolve primary document for %s: %s", accession, e)
            return ""

        recent = (submissions or {}).get("filings", {}).get("recent", {})
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        for i, acc in enumerate(accession_numbers):
            if acc == accession:
                return primary_docs[i] if i < len(primary_docs) else ""
        return ""

    def _list_filing_documents(self, filing_record: dict) -> list[dict]:
        """Read one filing index page and return document metadata without downloads."""
        accession = str(filing_record.get("accession") or "")
        cik = filing_record.get("cik") or self._resolve_cik(str(filing_record.get("ticker") or ""))
        if not cik or not accession:
            return []
        directory = self._build_archive_url(cik, accession, "")
        index_url = f"{directory}{accession}-index.html"
        try:
            response = requests.get(
                index_url,
                headers={"User-Agent": self._user_agent},
                timeout=60,
            )
            response.raise_for_status()
            self._respect_rate_limit()
        except Exception as exc:  # noqa: BLE001 - primary URL fallback remains available
            logger.warning("Could not load filing index for %s: %s", accession, exc)
            return []

        rows: list[dict] = []
        for table_row in re.findall(r"(?is)<tr\b[^>]*>(.*?)</tr>", response.text):
            cells = re.findall(r"(?is)<td\b[^>]*>(.*?)</td>", table_row)
            link = re.search(r"(?is)<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", table_row)
            if len(cells) < 4 or not link:
                continue
            href = html_lib.unescape(link.group(1)).strip()
            name = self._strip_html(link.group(2))
            document_type = self._strip_html(cells[3]).upper()
            url = href if href.startswith("http") else f"https://www.sec.gov/{href.lstrip('/')}"
            rows.append({"name": name, "document_type": document_type, "url": url})
        return rows

    def _download_document(self, url: str, accession: str) -> Optional[str]:
        """Download and normalize one already-selected SEC document."""
        try:
            response = requests.get(
                url,
                headers={"User-Agent": self._user_agent},
                timeout=60,
            )
            response.raise_for_status()
            self._respect_rate_limit()
            text = self._strip_html(response.text)
            if len(text) < 20:
                logger.warning("SEC document for %s was too short: %s", accession, url)
                return None
            return text
        except Exception as exc:  # noqa: BLE001 - document failures are retryable per accession
            logger.error("Failed SEC document for %s (%s): %s", accession, url, exc)
            return None

    @staticmethod
    def _strip_html(raw_html: str) -> str:
        """Strip HTML tags to plain text with a lightweight regex (no bs4)."""
        if not raw_html:
            return ""
        # Drop script/style blocks entirely.
        text = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", raw_html)
        # Remove all remaining tags.
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        # Unescape HTML entities and collapse whitespace.
        text = html_lib.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _derive_period(filing_date: str, filing_type: str) -> str:
        """Derive a period label from filing date and type.

        For 10-K: returns the previous fiscal year (e.g., filing 2026-03-15 → "2025")
        For 10-Q: returns the previous quarter (e.g., filing 2026-05-10 → "2026-Q1")
        """
        if not filing_date:
            return ""

        try:
            from dateutil import parser
            dt = parser.parse(filing_date)
        except Exception:
            return ""

        if filing_type == "10-K":
            # 10-K for fiscal year Y is filed in Q1 of year Y+1
            return str(dt.year - 1)
        elif filing_type == "10-Q":
            # 10-Q for quarter N is filed in quarter N+1
            q = (dt.month - 1) // 3  # 0-indexed quarter of filing date
            # Filing in Q2 means it covers Q1, etc.
            if q == 0:
                # Filed in January-March, covers previous year Q4
                return f"{dt.year - 1}-Q4"
            else:
                return f"{dt.year}-Q{q}"
        return ""

    def _get_registered_filing(self, accession: str) -> Optional[dict]:
        """Check if a filing is already registered by querying the filings table."""
        # The filings table uses accession as UNIQUE, so we query via SQLite directly
        sql = "SELECT * FROM filings WHERE accession = ?"
        with self.store.sqlite._connect() as conn:
            row = conn.execute(sql, (accession,)).fetchone()
            return dict(row) if row else None

    def _respect_rate_limit(self):
        """Pause between SEC EDGAR requests."""
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    @classmethod
    def _load_config_user_agent(cls) -> str:
        """Read the default SEC user_agent from configs/storage.yaml."""
        config_path = Path(__file__).parent.parent.parent / "configs/storage.yaml"
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            user_agent = (cfg.get("sec") or {}).get("user_agent")
            return user_agent or cls._DEFAULT_USER_AGENT
        except Exception:
            return cls._DEFAULT_USER_AGENT

    @staticmethod
    def _load_sec_config() -> dict:
        """Load SEC form/exhibit selection without reading credential files."""
        config_path = Path(__file__).parents[2] / "configs" / "sec.yaml"
        try:
            with open(config_path, encoding="utf-8") as config_file:
                loaded = yaml.safe_load(config_file) or {}
            return loaded.get("sec", loaded)
        except Exception as exc:  # noqa: BLE001 - safe exhibit defaults remain in code
            logger.warning("Could not load SEC exhibit config: %s", exc)
            return {}
