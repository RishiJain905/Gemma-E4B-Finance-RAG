"""src/ingestion/openfigi.py
OpenFIGI ticker↔FIGI mapping helper and optional identity enrichment source.

OpenFIGI is free; an API key is optional but raises rate limits. Use the
helper functions from other adapters, or run the lightweight scheduler source
``openfigi`` to register FIGI aliases on coverage securities.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

import requests

from src.ingestion.vendor_common import (
    VendorProviderError,
    empty_result,
    finish_result,
    utc_now,
)
from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)

OPENFIGI_MAPPING_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_API_KEY_ENV = "OPENFIGI_API_KEY"


def map_tickers_to_figi(
    tickers: list[str],
    *,
    api_key: Optional[str] = None,
    exch_code: str = "US",
    http_post: Optional[Callable[..., Any]] = None,
    timeout: float = 30.0,
) -> dict[str, dict[str, str]]:
    """Map ticker symbols to OpenFIGI records.

    Returns ``{TICKER: {"figi": ..., "name": ..., "ticker": ..., "exchCode": ...}}``.
    Tickers with no mapping are omitted. Never raises for empty/missing hits.
    """
    key = str(
        api_key if api_key is not None else os.environ.get(OPENFIGI_API_KEY_ENV, "")
    ).strip()
    jobs = [
        {"idType": "TICKER", "idValue": str(ticker).strip().upper(), "exchCode": exch_code}
        for ticker in tickers
        if str(ticker or "").strip()
    ]
    if not jobs:
        return {}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["X-OPENFIGI-APIKEY"] = key
    post = http_post or requests.post
    response = post(OPENFIGI_MAPPING_URL, json=jobs, headers=headers, timeout=timeout)
    status_code = int(getattr(response, "status_code", 200))
    try:
        payload = response.json()
    except Exception as exc:
        raise VendorProviderError(
            "OpenFIGI returned malformed JSON",
            error_class="contract",
            status_code=status_code,
        ) from exc
    if status_code >= 400:
        message = str(payload)[:500]
        raise VendorProviderError(
            message or f"OpenFIGI HTTP {status_code}",
            error_class="rate_limited" if status_code == 429 else "contract",
            status_code=status_code,
        )
    if not isinstance(payload, list):
        raise VendorProviderError("OpenFIGI mapping response must be a list", error_class="contract")

    mapped: dict[str, dict[str, str]] = {}
    for job, block in zip(jobs, payload):
        ticker = str(job["idValue"]).upper()
        if not isinstance(block, dict):
            continue
        data = block.get("data")
        if not isinstance(data, list) or not data:
            continue
        first = data[0]
        if not isinstance(first, dict):
            continue
        figi = str(first.get("figi") or "").strip()
        if not figi:
            continue
        mapped[ticker] = {
            "figi": figi,
            "name": str(first.get("name") or "").strip(),
            "ticker": str(first.get("ticker") or ticker).strip().upper(),
            "exchCode": str(first.get("exchCode") or exch_code).strip(),
            "compositeFIGI": str(first.get("compositeFIGI") or "").strip(),
            "shareClassFIGI": str(first.get("shareClassFIGI") or "").strip(),
        }
    return mapped


def enrich_security_figi(
    store: Store,
    ticker: str,
    *,
    api_key: Optional[str] = None,
    http_post: Optional[Callable[..., Any]] = None,
) -> Optional[dict[str, str]]:
    """Resolve one ticker and register its FIGI as a vendor_symbol alias.

    Returns the OpenFIGI mapping dict on success, otherwise ``None``.
    """
    ticker = str(ticker or "").strip().upper()
    if not ticker:
        return None
    security = store.resolve_security(ticker, provider="openfigi")
    if not isinstance(security, dict) or not security.get("security_id"):
        logger.debug("OpenFIGI enrichment skipped for %s: no canonical security", ticker)
        return None
    mapped = map_tickers_to_figi([ticker], api_key=api_key, http_post=http_post).get(ticker)
    if not mapped:
        return None
    store.register_security_alias(
        str(security["security_id"]),
        mapped["figi"],
        alias_type="vendor_symbol",
        provider="openfigi",
        source="openfigi",
    )
    composite = mapped.get("compositeFIGI") or ""
    if composite and composite != mapped["figi"]:
        store.register_security_alias(
            str(security["security_id"]),
            composite,
            alias_type="vendor_symbol",
            provider="openfigi_composite",
            source="openfigi",
        )
    return mapped


class OpenFIGIIngestor:
    """Optional scheduler source that enriches coverage tickers with FIGI aliases."""

    SOURCE_NAME = "openfigi"
    COVERAGE_SOURCE = "openfigi"

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        coverage_resolver: Optional[CoverageResolver] = None,
        api_key: Optional[str] = None,
        http_post: Optional[Callable[..., Any]] = None,
        timeout: float = 30.0,
        batch_size: int = 10,
        now_fn: Callable[[], object] = utc_now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store or Store()
        self.coverage = coverage_resolver or CoverageResolver(self.store)
        self.api_key = str(
            api_key if api_key is not None else os.environ.get(OPENFIGI_API_KEY_ENV, "")
        ).strip()
        self.http_post = http_post or requests.post
        self.timeout = max(float(timeout), 0.1)
        self.batch_size = max(int(batch_size), 1)
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn

    def ingest(self, tickers: Optional[list[str]] = None) -> dict:
        """Enrich coverage tickers with OpenFIGI aliases (API key optional)."""
        selected = list(tickers) if tickers is not None else self.coverage.tickers_for(
            self.COVERAGE_SOURCE
        )
        result = empty_result(self.SOURCE_NAME)
        result["mapped"] = 0
        result["skipped"] = 0
        # OpenFIGI works without a key; missing key is not a hard disable.
        for offset in range(0, len(selected), self.batch_size):
            batch = [str(t).upper() for t in selected[offset : offset + self.batch_size]]
            result["tickers"] += len(batch)
            try:
                mapped = map_tickers_to_figi(
                    batch,
                    api_key=self.api_key or None,
                    http_post=self.http_post,
                    timeout=self.timeout,
                )
                result["requests"] += 1
                result["attempts"] += 1
            except VendorProviderError as exc:
                result["errors"].append(str(exc))
                status = "rate_limited" if exc.error_class == "rate_limited" else "error"
                return finish_result(result, status, error_class=exc.error_class)

            for ticker in batch:
                hit = mapped.get(ticker)
                if not hit:
                    result["skipped"] += 1
                    continue
                security = self.store.resolve_security(ticker, provider=self.SOURCE_NAME)
                security_id = security.get("security_id") if isinstance(security, dict) else None
                if not security_id:
                    result["skipped"] += 1
                    continue
                try:
                    created = self.store.register_security_alias(
                        str(security_id),
                        hit["figi"],
                        alias_type="vendor_symbol",
                        provider="openfigi",
                        source="openfigi",
                    )
                    result["mapped"] += 1
                    if created:
                        result["stored"] += 1
                    else:
                        result["duplicates"] += 1
                except Exception as exc:  # noqa: BLE001
                    result["errors"].append(f"{ticker}: alias store failed: {exc}")
            if offset + self.batch_size < len(selected):
                self.sleep_fn(0.2)
        status = "partial" if result["errors"] else "ok"
        return finish_result(result, status)
