"""src/universe/providers.py
Pure adapters for Nasdaq-100, IVV, and SEC universe payloads.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Optional

import requests

from .models import SnapshotValidationError, UniverseRecord, normalize_symbol

logger = logging.getLogger(__name__)

NASDAQ100_URL = "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies"
IVV_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
)
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


class _TableParser(HTMLParser):
    """Small HTML table reader used for pinned Nasdaq payloads."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _read_text(payload: str | bytes | Path) -> str:
    if isinstance(payload, Path):
        return payload.read_text(encoding="utf-8-sig")
    if isinstance(payload, bytes):
        return payload.decode("utf-8-sig")
    return str(payload).lstrip("\ufeff")


def _clean_mapping(row: dict) -> dict[str, str]:
    return {
        str(key or "").strip().lower(): str(value or "").strip()
        for key, value in row.items()
    }


def _pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name.lower(), "")
        if value:
            return value
    return ""


def _validate_records(
    rows: list[UniverseRecord],
    *,
    minimum: int,
    source: str,
) -> list[UniverseRecord]:
    if len(rows) < minimum:
        raise SnapshotValidationError(
            f"{source} snapshot must contain at least {minimum} valid records; got {len(rows)}"
        )
    symbols = [normalize_symbol(row.symbol) for row in rows]
    if any(not symbol for symbol in symbols):
        raise SnapshotValidationError(f"{source} snapshot contains an empty symbol")
    duplicates = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
    if duplicates:
        raise SnapshotValidationError(
            f"{source} snapshot contains duplicate symbols: {', '.join(duplicates)}"
        )
    return rows


class Nasdaq100Provider:
    """Parse and optionally download Nasdaq's published Nasdaq-100 list."""

    source = "nasdaq"
    index_code = "nasdaq100"

    def __init__(
        self,
        *,
        source_url: str = NASDAQ100_URL,
        min_constituents: int = 90,
        timeout: float = 30.0,
    ) -> None:
        self.source_url = source_url
        self.min_constituents = min_constituents
        self.timeout = timeout

    def fetch(self) -> list[UniverseRecord]:
        """Download and parse the current Nasdaq-100 constituent payload."""
        response = requests.get(self.source_url, timeout=self.timeout)
        response.raise_for_status()
        return self.parse(response.text)

    def parse(self, payload: str | bytes | Path) -> list[UniverseRecord]:
        """Parse a pinned Nasdaq CSV or HTML table into records only."""
        text = _read_text(payload)
        mappings = self._mappings(text)
        rows = []
        for raw in mappings:
            row = _clean_mapping(raw)
            symbol = _pick(row, "symbol", "ticker")
            name = _pick(row, "company name", "company", "name")
            if not symbol or not name:
                raise SnapshotValidationError(
                    "nasdaq snapshot requires non-empty Symbol and Company Name fields"
                )
            rows.append(
                UniverseRecord(
                    symbol=symbol,
                    company_name=name,
                    source=self.source,
                    index_code=self.index_code,
                    exchange=_pick(row, "exchange") or "Nasdaq",
                    sector=_pick(row, "sector", "gics sector") or None,
                    industry=_pick(row, "industry", "gics sub-industry") or None,
                    source_url=self.source_url,
                )
            )
        return _validate_records(
            rows, minimum=self.min_constituents, source=self.source
        )

    @staticmethod
    def _mappings(text: str) -> Iterable[dict]:
        if "<table" not in text.lower():
            reader = csv.DictReader(io.StringIO(text))
            if not reader.fieldnames:
                raise SnapshotValidationError("nasdaq snapshot has no CSV header")
            return list(reader)
        parser = _TableParser()
        parser.feed(text)
        if len(parser.rows) < 2:
            raise SnapshotValidationError("nasdaq snapshot has no HTML table rows")
        headers = parser.rows[0]
        return [dict(zip(headers, values, strict=False)) for values in parser.rows[1:]]


class IVVHoldingsProvider:
    """Parse and optionally download IVV holdings used as the S&P 500 proxy."""

    source = "ivv"
    index_code = "sp500"
    _EQUITY_TYPES = {"equity", "stock"}

    def __init__(
        self,
        *,
        source_url: str = IVV_HOLDINGS_URL,
        min_constituents: int = 450,
        timeout: float = 30.0,
    ) -> None:
        self.source_url = source_url
        self.min_constituents = min_constituents
        self.timeout = timeout

    def fetch(self) -> list[UniverseRecord]:
        """Download and parse the current IVV holdings CSV."""
        response = requests.get(self.source_url, timeout=self.timeout)
        response.raise_for_status()
        return self.parse(response.content)

    def parse(self, payload: str | bytes | Path) -> list[UniverseRecord]:
        """Parse an iShares CSV, ignoring metadata and non-equity holdings."""
        text = _read_text(payload)
        lines = text.splitlines()
        header_index = None
        for index, line in enumerate(lines):
            fields = next(csv.reader([line]), [])
            if fields and fields[0].lstrip("\ufeff").strip().lower() == "ticker":
                header_index = index
                break
        if header_index is None:
            raise SnapshotValidationError("ivv snapshot is missing the Ticker header")
        reader = csv.DictReader(lines[header_index:])
        rows = []
        for raw in reader:
            row = _clean_mapping(raw)
            asset_class = _pick(row, "asset class").lower()
            if asset_class not in self._EQUITY_TYPES:
                continue
            symbol = _pick(row, "ticker")
            name = _pick(row, "name")
            if not symbol or not name:
                raise SnapshotValidationError(
                    "ivv equity row requires non-empty Ticker and Name fields"
                )
            rows.append(
                UniverseRecord(
                    symbol=symbol,
                    company_name=name,
                    source=self.source,
                    index_code=self.index_code,
                    exchange=_pick(row, "exchange", "location") or None,
                    sector=_pick(row, "sector") or None,
                    source_url=self.source_url,
                )
            )
        return _validate_records(rows, minimum=self.min_constituents, source=self.source)


class SECCompanyTickersProvider:
    """Parse and optionally download SEC ticker, exchange, and CIK mappings."""

    source = "sec"

    def __init__(
        self,
        *,
        source_url: str = SEC_COMPANY_TICKERS_URL,
        min_records: int = 1_000,
        timeout: float = 30.0,
        user_agent: Optional[str] = None,
    ) -> None:
        self.source_url = source_url
        self.min_records = min_records
        self.timeout = timeout
        self.user_agent = user_agent

    def fetch(self) -> list[UniverseRecord]:
        """Download and parse the SEC company ticker mapping."""
        headers = {"User-Agent": self.user_agent} if self.user_agent else {}
        response = requests.get(self.source_url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return self.parse(response.content)

    def parse(self, payload: str | bytes | Path) -> list[UniverseRecord]:
        """Parse either SEC company_tickers JSON representation into records."""
        try:
            data = json.loads(_read_text(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise SnapshotValidationError("sec snapshot is not valid JSON") from exc
        entries = self._entries(data)
        rows = []
        for raw in entries:
            row = _clean_mapping(raw)
            symbol = _pick(row, "ticker", "symbol")
            name = _pick(row, "title", "name", "company name")
            cik_raw = _pick(row, "cik_str", "cik")
            if not symbol or not name or not cik_raw:
                raise SnapshotValidationError(
                    "sec row requires non-empty ticker, company name, and CIK fields"
                )
            if not cik_raw.isdigit():
                raise SnapshotValidationError(f"sec row has invalid CIK: {cik_raw}")
            rows.append(
                UniverseRecord(
                    symbol=symbol,
                    company_name=name,
                    source=self.source,
                    exchange=_pick(row, "exchange") or None,
                    cik=cik_raw.zfill(10),
                    source_url=self.source_url,
                )
            )
        return _validate_records(rows, minimum=self.min_records, source=self.source)

    @staticmethod
    def _entries(data: object) -> list[dict]:
        if not isinstance(data, dict) or not data:
            raise SnapshotValidationError("sec snapshot must be a non-empty object")
        fields = data.get("fields")
        values = data.get("data")
        if isinstance(fields, list) and isinstance(values, list):
            return [dict(zip(fields, row, strict=False)) for row in values]
        entries = [value for value in data.values() if isinstance(value, dict)]
        if not entries:
            raise SnapshotValidationError("sec snapshot contains no ticker records")
        return entries
