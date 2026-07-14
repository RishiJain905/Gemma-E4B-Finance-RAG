"""tests/test_universe_providers.py
Offline contract tests for bounded universe provider adapters.
"""

from pathlib import Path

import pytest

from src.universe.models import SnapshotValidationError
from src.universe.providers import (
    IVVHoldingsProvider,
    Nasdaq100Provider,
    SECCompanyTickersProvider,
)


FIXTURES = Path(__file__).parent / "fixtures" / "universe"


def test_nasdaq_provider_parses_pinned_fixture() -> None:
    provider = Nasdaq100Provider(min_constituents=4)

    rows = provider.parse(FIXTURES / "nasdaq100.csv")

    assert [row.symbol for row in rows] == ["AAPL", "GOOG", "GOOGL", "MSFT"]
    assert all(row.index_code == "nasdaq100" for row in rows)
    assert rows[0].company_name == "Apple Inc."
    assert rows[0].sector == "Technology"


def test_ivv_provider_skips_cash_and_derivatives() -> None:
    provider = IVVHoldingsProvider(min_constituents=4)

    rows = provider.parse(FIXTURES / "ivv_holdings.csv")

    assert [row.symbol for row in rows] == ["AAPL", "BRK.B", "GOOGL", "MSFT"]
    assert all(row.index_code == "sp500" for row in rows)
    assert all(row.security_type == "common_stock" for row in rows)


def test_sec_provider_parses_ticker_cik_and_exchange() -> None:
    provider = SECCompanyTickersProvider(min_records=5)

    rows = provider.parse(FIXTURES / "sec_company_tickers.json")

    assert len(rows) == 6
    apple = next(row for row in rows if row.symbol == "AAPL")
    assert apple.cik == "0000320193"
    assert apple.exchange == "Nasdaq"
    assert apple.index_code is None


@pytest.mark.parametrize(
    ("provider", "payload"),
    [
        (Nasdaq100Provider(min_constituents=1), "Company Name\nApple Inc.\n"),
        (
            IVVHoldingsProvider(min_constituents=1),
            "Ticker,Name,Sector,Asset Class\n,Apple Inc.,Technology,Equity\n",
        ),
        (SECCompanyTickersProvider(min_records=1), "{}"),
    ],
)
def test_providers_reject_malformed_or_empty_identifiers(provider, payload: str) -> None:
    with pytest.raises(SnapshotValidationError):
        provider.parse(payload)


def test_provider_rejects_duplicate_symbols() -> None:
    payload = (
        "Symbol,Company Name,Sector,Industry\n"
        "AAPL,Apple Inc.,Technology,Hardware\n"
        "AAPL,Apple Inc.,Technology,Hardware\n"
    )

    with pytest.raises(SnapshotValidationError, match="duplicate"):
        Nasdaq100Provider(min_constituents=1).parse(payload)


def test_provider_rejects_implausibly_small_snapshot() -> None:
    payload = (
        "Symbol,Company Name,Sector,Industry\n"
        "AAPL,Apple Inc.,Technology,Hardware\n"
    )

    with pytest.raises(SnapshotValidationError, match="at least 2"):
        Nasdaq100Provider(min_constituents=2).parse(payload)
