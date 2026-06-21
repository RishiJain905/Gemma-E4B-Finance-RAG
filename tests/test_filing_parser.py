"""
tests/test_filing_parser.py
Test suite for TraceAlchemyFilingParser (spec 1.4.2).

Two flavours of test:

1. Deterministic unit tests (no network) exercising the pure-logic helpers:
   - _parse_model_response against a pure JSON array, a fenced ```json block,
     JSON embedded in surrounding prose, and malformed input (-> []).
   - _select_extraction_sections marker discovery + fallback.
   - _split_filing_sections header splitting.
   - _build_extraction_prompt field/ticker/type inclusion.
   - filing_type_from_period derivation.
   - period_type (annual vs quarterly) derivation and float coercion /
     skipping of non-numeric values.

2. Live-model tests that actually call llama-server on port 8087
   (OpenAI-compatible /v1/chat/completions). These are guarded so they
   pytest.skip cleanly when the model endpoint is unreachable, keeping the
   suite green when the model is down. The model IS expected to be up.
"""

import httpx
import pytest

from src.sec import SECEdgarFilingFetcher, TraceAlchemyFilingParser
from src.sec.filing_parser import filing_type_from_period


FACT_KEYS = {"metric", "value", "unit", "period", "period_type", "source_type"}

LIVE_ENDPOINT = TraceAlchemyFilingParser.DEFAULT_ENDPOINT
HEALTH_URL = LIVE_ENDPOINT.rsplit("/v1/", 1)[0] + "/health"

# A small synthetic income statement with an obvious revenue and net income.
SYNTHETIC_INCOME_STATEMENT = """
CONSOLIDATED STATEMENTS OF INCOME
(In millions, except per share data)

Three Months Ended

Revenue                                       $ 26,000
Cost of revenue                                  7,500
Gross profit                                    18,500
Research and development                         2,700
Sales, general and administrative                1,600
Operating income                                14,200
Income before income taxes                      14,500
Provision for income taxes                       1,900
Net income                                    $ 12,600

Earnings per share:
   Basic                                       $  5.10
   Diluted                                     $  5.00

Weighted average shares:
   Basic                                          2,470
   Diluted                                        2,520
"""


# ── Helpers ─────────────────────────────────────────

def _model_available() -> bool:
    """Return True if the llama-server chat endpoint appears reachable."""
    try:
        # Prefer the lightweight /health endpoint, fall back to the chat URL.
        resp = httpx.get(HEALTH_URL, timeout=5)
        if resp.status_code < 500:
            return True
    except httpx.HTTPError:
        pass
    try:
        resp = httpx.options(LIVE_ENDPOINT, timeout=5)
        return resp.status_code < 500
    except httpx.HTTPError:
        return False


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="TraceAlchemy llama-server not reachable on port 8087",
)


# ── Fixtures ────────────────────────────────────────

@pytest.fixture(scope="module")
def parser():
    p = TraceAlchemyFilingParser()
    yield p
    p.close()


# ── 1. _parse_model_response (deterministic) ────────

def test_parse_response_pure_json_array(parser):
    """A pure JSON array parses into well-formed fact dicts."""
    raw = '[{"metric": "total_revenue", "value": 26.0, "unit": "billion_usd"}]'
    facts = parser._parse_model_response(raw, "NVDA", "2026-Q1")

    assert len(facts) == 1
    fact = facts[0]
    assert FACT_KEYS.issubset(fact.keys())
    assert fact["metric"] == "total_revenue"
    assert fact["value"] == 26.0
    assert isinstance(fact["value"], float)
    assert fact["unit"] == "billion_usd"
    assert fact["period"] == "2026-Q1"


def test_parse_response_fenced_json_block(parser):
    """A ```json fenced code block is stripped before parsing."""
    raw = (
        "```json\n"
        '[{"metric": "net_income", "value": 12.6, "unit": "billion_usd"}]\n'
        "```"
    )
    facts = parser._parse_model_response(raw, "NVDA", "2026")

    assert len(facts) == 1
    assert facts[0]["metric"] == "net_income"
    assert facts[0]["value"] == 12.6


def test_parse_response_json_embedded_in_prose(parser):
    """JSON embedded in surrounding prose is recovered via regex fallback."""
    raw = (
        "Sure! Here are the metrics I extracted from the filing:\n"
        '[{"metric": "operating_income", "value": 14.2, "unit": "billion_usd"}]\n'
        "Let me know if you need anything else."
    )
    facts = parser._parse_model_response(raw, "NVDA", "2026-Q2")

    assert len(facts) == 1
    assert facts[0]["metric"] == "operating_income"
    assert facts[0]["value"] == 14.2


def test_parse_response_malformed_returns_empty(parser):
    """Malformed / non-JSON input yields an empty list."""
    assert parser._parse_model_response("not json at all", "NVDA", "2026") == []
    assert parser._parse_model_response("", "NVDA", "2026") == []
    assert parser._parse_model_response(None, "NVDA", "2026") == []
    # A JSON object (not an array) is rejected.
    assert parser._parse_model_response('{"metric": "x"}', "NVDA", "2026") == []


def test_parse_response_period_type_annual_vs_quarterly(parser):
    """period_type is derived from presence of 'Q' in the period."""
    raw = '[{"metric": "total_revenue", "value": 1.0, "unit": "billion_usd"}]'

    annual = parser._parse_model_response(raw, "NVDA", "2025")
    assert annual[0]["period_type"] == "annual"
    assert annual[0]["source_type"] == "sec_10-K"

    quarterly = parser._parse_model_response(raw, "NVDA", "2026-Q1")
    assert quarterly[0]["period_type"] == "quarterly"
    assert quarterly[0]["source_type"] == "sec_10-Q"

    # Empty period defaults to annual / 10-K.
    empty = parser._parse_model_response(raw, "NVDA", "")
    assert empty[0]["period_type"] == "annual"
    assert empty[0]["source_type"] == "sec_10-K"


def test_parse_response_float_coercion_and_skipping(parser):
    """String numerics are coerced; non-numeric / missing values are skipped."""
    raw = """[
        {"metric": "total_revenue", "value": "26.0", "unit": "billion_usd"},
        {"metric": "net_income", "value": "not_a_number", "unit": "billion_usd"},
        {"metric": "gross_profit", "value": null, "unit": "billion_usd"},
        {"metric": "", "value": 5.0, "unit": "billion_usd"},
        {"metric": "operating_income", "value": 14, "unit": "billion_usd"}
    ]"""
    facts = parser._parse_model_response(raw, "NVDA", "2026-Q1")

    by_metric = {f["metric"]: f for f in facts}
    # Coerced string -> float.
    assert by_metric["total_revenue"]["value"] == 26.0
    assert isinstance(by_metric["total_revenue"]["value"], float)
    # Int -> float.
    assert by_metric["operating_income"]["value"] == 14.0
    # Non-numeric, null, and empty-metric entries are skipped.
    assert "net_income" not in by_metric
    assert "gross_profit" not in by_metric
    assert "" not in by_metric
    assert len(facts) == 2


def test_parse_response_default_unit(parser):
    """A missing unit defaults to billion_usd."""
    raw = '[{"metric": "total_revenue", "value": 26.0}]'
    facts = parser._parse_model_response(raw, "NVDA", "2026-Q1")
    assert facts[0]["unit"] == "billion_usd"


def test_parse_response_recovers_truncated_array(parser):
    """A JSON array cut off by max_tokens still yields its complete objects."""
    # Closing ] is missing and the final object is half-written.
    raw = (
        '[{"metric": "total_revenue", "value": 111.184, "unit": "billion_usd"}, '
        '{"metric": "net_income", "value": 71.675, "unit": "billion_usd"}, '
        '{"metric": "operating_income", "value": 35.8'
    )
    facts = parser._parse_model_response(raw, "AAPL", "2026-Q1")

    by_metric = {f["metric"]: f for f in facts}
    # The two complete objects are recovered; the truncated tail is dropped.
    assert by_metric["total_revenue"]["value"] == 111.184
    assert by_metric["net_income"]["value"] == 71.675
    assert "operating_income" not in by_metric
    assert len(facts) == 2


# ── 1b. Operations-table parser (deterministic, no model) ──

AAPL_OPERATIONS_SNIPPET = """
CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)
(In millions, except per-share amounts)
Three Months Ended Six Months Ended
Total net sales 111,184 95,359 254,940 219,659
Gross margin 54,781 44,867 124,012 103,142
Research and development 11,419 8,550 22,306 16,818
Operating income 35,885 29,589 86,737 72,421
Net income $ 29,578 $ 24,780 $ 71,675 $ 61,110
Earnings per share: Basic $ 2.02 $ 1.65 Diluted $ 2.01 $ 1.65
"""


def test_operations_table_extraction_aapl(parser):
    """Deterministic parser extracts core metrics from a real-style operations table."""
    facts = parser._extract_facts_from_operations_table(
        AAPL_OPERATIONS_SNIPPET, "2026-Q1",
    )
    by_metric = {f["metric"]: f for f in facts}
    assert "total_revenue" in by_metric
    assert "net_income" in by_metric
    assert by_metric["total_revenue"]["value"] == pytest.approx(111.184, rel=1e-3)
    assert by_metric["net_income"]["value"] == pytest.approx(29.578, rel=1e-3)
    assert by_metric["eps_diluted"]["value"] == pytest.approx(2.01, rel=1e-3)


def test_extract_facts_uses_table_without_model(parser, monkeypatch):
    """extract_facts_from_filing returns table facts when core metrics are present."""
    monkeypatch.setattr(parser, "_call_model", lambda _prompt: None)
    facts = parser.extract_facts_from_filing(
        "AAPL", "10-Q", AAPL_OPERATIONS_SNIPPET, "2026-Q1",
    )
    assert parser._has_core_metrics(facts)
    assert facts[0]["source_type"] == "sec_10-Q"


# ── 2. _select_extraction_sections (deterministic) ──

def test_select_sections_finds_income_statement(parser):
    """An income-statement marker is located and balance-sheet content included."""
    text = (
        "Some preamble text describing the company.\n" * 5
        + "CONSOLIDATED STATEMENTS OF INCOME\n"
        + "Revenue 26000\nNet income 12600\n"
        + "CONSOLIDATED BALANCE SHEETS\n"
        + "Total assets 100000\n"
    )
    selected = parser._select_extraction_sections(text, "10-Q")

    assert "CONSOLIDATED STATEMENTS OF INCOME" in selected
    assert "Revenue 26000" in selected
    # Balance sheet content after the income statement is included.
    assert "CONSOLIDATED BALANCE SHEETS" in selected


def test_select_sections_prefers_operations_over_loose_income_prose(parser):
    """A real 'statements of operations' table wins over loose prose earlier on.

    Mirrors AAPL's 10-Q: the loose phrase 'Income Statement' appears in the
    notes BEFORE the operations table; the selector must still anchor on the
    operations table so the revenue/net-income figures are captured.
    """
    text = (
        "Disaggregation of Income Statement Expenses is discussed below.\n"
        + "filler prose line\n" * 50
        + "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS\n"
        + "Total net sales 111184\nNet income 29578\n"
        + "Earnings per share: Basic 2.02\n"
        + "CONSOLIDATED BALANCE SHEETS\n"
        + "Total assets 371082\n"
    )
    selected = parser._select_extraction_sections(text, "10-Q")

    assert "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS" in selected
    assert "Total net sales 111184" in selected
    assert "Net income 29578" in selected
    # The loose-prose region must NOT be what anchored the slice.
    assert "Disaggregation of Income Statement Expenses" not in selected


def test_select_sections_fallback_when_no_markers(parser):
    """With no recognizable markers, falls back to the first N chars."""
    text = "X" * 20000  # no section markers
    selected = parser._select_extraction_sections(text, "10-K")

    # Fallback caps at 12000 chars.
    assert len(selected) == 12000
    assert set(selected) == {"X"}


# ── 3. _split_filing_sections (deterministic) ───────

def test_split_filing_sections_by_headers(parser):
    """Filing text is split into logical sections keyed by known headers."""
    text = (
        "Intro line one\n"
        "Intro line two\n"
        "CONSOLIDATED STATEMENTS OF INCOME\n"
        "Revenue 26000\n"
        "CONSOLIDATED BALANCE SHEET\n"
        "Total assets 100000\n"
        "RISK FACTORS\n"
        "Markets are volatile.\n"
    )
    sections = parser._split_filing_sections(text)

    assert "preamble" in sections
    assert "Intro line one" in sections["preamble"]
    assert "consolidated_statements_of_income" in sections
    assert "Revenue 26000" in sections["consolidated_statements_of_income"]
    assert "consolidated_balance_sheet" in sections
    assert "risk_factors" in sections
    assert "Markets are volatile." in sections["risk_factors"]


# ── 4. _build_extraction_prompt (deterministic) ─────

def test_build_prompt_includes_fields_and_metadata(parser):
    """The prompt includes ticker, filing type, and field names."""
    prompt = parser._build_extraction_prompt("NVDA", "10-Q", "some filing text")

    assert "NVDA" in prompt
    assert "10-Q" in prompt
    # A sampling of the EXTRACTION_FIELDS names appears.
    assert "total_revenue" in prompt
    assert "net_income" in prompt
    assert "eps_diluted" in prompt
    assert "free_cash_flow" in prompt
    # The filing text is embedded.
    assert "some filing text" in prompt


def test_build_prompt_truncates_long_text(parser):
    """Filing text is truncated to MAX_PROMPT_TEXT_CHARS in the prompt.

    The prompt now embeds a tight, targeted slice rather than a large noisy
    blob, so the body cap was lowered from 12000 to MAX_PROMPT_TEXT_CHARS.
    """
    cap = parser.MAX_PROMPT_TEXT_CHARS
    long_text = "A" * (cap + 8000)
    prompt = parser._build_extraction_prompt("NVDA", "10-K", long_text)
    # Only `cap` chars of the body should appear.
    assert "A" * cap in prompt
    assert "A" * (cap + 1) not in prompt


# ── 5. filing_type_from_period (deterministic) ──────

def test_filing_type_from_period():
    """Period strings map to the correct filing-type suffix."""
    assert filing_type_from_period("2025") == "10-K"
    assert filing_type_from_period("2026-Q1") == "10-Q"
    assert filing_type_from_period("2026-Q2") == "10-Q"
    assert filing_type_from_period("") == "10-K"


# ── 6. Live-model extraction tests (port 8087) ──────

@requires_model
def test_extract_facts_live_synthetic(parser):
    """Live model extracts well-formed facts from a synthetic income statement."""
    facts = parser.extract_facts_from_filing(
        "NVDA", "10-Q", SYNTHETIC_INCOME_STATEMENT, period="2026-Q1",
    )

    assert isinstance(facts, list)
    assert len(facts) > 0, "expected at least one extracted fact"

    for fact in facts:
        assert FACT_KEYS.issubset(fact.keys()), fact
        assert isinstance(fact["value"], float)
        assert fact["period"] == "2026-Q1"
        assert fact["period_type"] == "quarterly"
        assert fact["source_type"] == "sec_10-Q"

    # Lenient about exactly which metrics appear, but an obvious one should.
    metrics = {f["metric"] for f in facts}
    assert "total_revenue" in metrics, f"total_revenue missing; got {metrics}"


@requires_model
def test_extract_facts_live_end_to_end(parser):
    """End-to-end: fetch a real recent 10-Q from EDGAR, then parse it live."""
    fetcher = SECEdgarFilingFetcher(request_delay=0.2)
    filings = fetcher.discover_filings("AAPL", ["10-Q"], count=1)
    assert filings, "expected at least one AAPL 10-Q from EDGAR"

    filing = filings[0]
    text = fetcher.download_filing_text(filing)
    assert text and len(text) > 50, "expected substantial filing text"

    facts = parser.extract_facts_from_filing(
        "AAPL", "10-Q", text, period=filing.get("period") or "2026-Q1",
    )

    assert isinstance(facts, list)
    # The model may not always find metrics, but any returned fact must be
    # well-formed with a numeric value.
    for fact in facts:
        assert FACT_KEYS.issubset(fact.keys()), fact
        assert isinstance(fact["value"], float)
        assert fact["source_type"].startswith("sec_")
