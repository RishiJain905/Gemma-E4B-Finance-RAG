"""
tests/test_sec_pipeline.py
Comprehensive pytest suite for Phase 1.4 SEC filing pipeline (spec 1.4.5).

Test coverage:
  1. SECEdgarFilingFetcher — discovery, download, registration
  2. TraceAlchemyFilingParser — prompt building, model calls, response parsing
  3. FilingProcessor — pipeline orchestration
  4. FilingScheduler — cache awareness, incremental discovery
  5. Integration — live E2E + Phase 1.3 coexistence
  6. FilingUtils — period → filing type helpers

Usage:
    pytest tests/test_sec_pipeline.py -v
    pytest tests/test_sec_pipeline.py -v -m live
"""

import json
from pathlib import Path
import socket
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import httpx
import pytest

from src.sec import (
    FilingProcessor,
    FilingScheduler,
    SECEdgarFilingFetcher,
    TraceAlchemyFilingParser,
)
from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash
from src.ingestion.records import NarrativeRecord
from src.storage.store import Store


EDGAR_HOST = "www.sec.gov"
MODEL_ENDPOINT = "http://127.0.0.1:8087/v1/embeddings"
DISCOVERY_SOURCE = FilingScheduler.DISCOVERY_SOURCE
EVENT_FIXTURES = Path(__file__).parent / "fixtures" / "sec" / "events"


# ── Shared helpers (from tests/test_filing_scheduler.py) ──


def _patch_core_tickers(tickers):
    """Patch YFinanceIngestor to return a fixed core ticker list."""
    mock_ingestor = MagicMock()
    mock_ingestor.core_tickers = tickers
    return patch(
        "src.ingestion.yfinance_ingestor.YFinanceIngestor",
        return_value=mock_ingestor,
    )


def _endpoint_reachable(url: str, timeout: float = 3.0) -> bool:
    """TCP-connect check for a host:port derived from a URL."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _model_alive(timeout: float = 5.0) -> bool:
    """Confirm the llama-server embeddings endpoint actually responds."""
    try:
        resp = httpx.post(
            MODEL_ENDPOINT,
            json={"input": "ping", "model": "tracealchemy"},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _edgar_reachable() -> bool:
    return _endpoint_reachable(f"https://{EDGAR_HOST}")


def _mock_submissions():
    """SEC get_submissions response with filings.recent parallel arrays."""
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-K"],
                "accessionNumber": [
                    "0000320193-26-000012",
                    "0000320193-26-000011",
                    "0000320193-26-000099",
                ],
                "filingDate": ["2026-03-15", "2026-05-01", "2026-03-15"],
                "reportDate": ["2025", "2026-03-31", "2025"],
                "primaryDocument": [
                    "nvda-10k.htm",
                    "nvda-10q.htm",
                    "nvda-10k2.htm",
                ],
            }
        }
    }


def _make_store(tmp_path):
    return Store(
        db_path=tmp_path / "test.db",
        chroma_path=tmp_path / "chroma",
    )


def _use_deterministic_embeddings(store):
    """Keep offline pipeline tests on the real Chroma path without llama-server."""

    def _post(_url, *, json):
        inputs = json["input"]
        if isinstance(inputs, str):
            inputs = [inputs]

        response = MagicMock()
        response.json.return_value = {
            "data": [
                {"embedding": [1.0, float((sum(map(ord, text)) % 97) + 1), 0.5]}
                for text in inputs
            ],
        }
        return response

    store.chroma.embedding_fn._client.post = MagicMock(side_effect=_post)
    return store


# ============================================================
# 1. SECEdgarFilingFetcher Tests
# ============================================================


class TestSECEdgarFilingFetcher:
    """Tests for the SEC EDGAR filing discovery and download module."""

    def test_import(self):
        assert SECEdgarFilingFetcher is not None

    def test_init_defaults(self):
        fetcher = SECEdgarFilingFetcher()
        assert fetcher.store is not None
        assert fetcher.request_delay == 0.5
        assert fetcher._client is not None

    def test_init_with_custom_store(self, tmp_path):
        store = _make_store(tmp_path)
        fetcher = SECEdgarFilingFetcher(store=store)
        assert fetcher.store is store

    def test_init_custom_delay(self):
        fetcher = SECEdgarFilingFetcher(request_delay=0.1)
        assert fetcher.request_delay == 0.1

    @patch("src.sec.edgar_fetcher.EdgarClient")
    def test_discover_filings_parses_results(self, mock_edgar_cls, tmp_path):
        store = _make_store(tmp_path)

        mock_client = MagicMock()
        mock_client.get_submissions.return_value = _mock_submissions()
        mock_edgar_cls.return_value = mock_client

        fetcher = SECEdgarFilingFetcher(store=store, request_delay=0.01)

        with patch.object(fetcher, "_resolve_cik", return_value="0000320193"):
            filings = fetcher.discover_filings(
                "NVDA", filing_types=["10-K", "10-Q"], count=5,
            )

        assert len(filings) >= 2
        assert filings[0]["ticker"] == "NVDA"
        assert filings[0]["accession"] == "0000320193-26-000012"
        assert filings[0]["cik"] == "0000320193"
        assert filings[0]["source_url"].startswith(
            "https://www.sec.gov/Archives/edgar/data/"
        )

    @patch("src.sec.edgar_fetcher.EdgarClient")
    def test_register_discovered_filings(self, mock_edgar_cls, tmp_path):
        store = _make_store(tmp_path)

        mock_client = MagicMock()
        submissions = {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "accessionNumber": ["0000320193-26-000099"],
                    "filingDate": ["2026-03-15"],
                    "reportDate": ["2025"],
                    "primaryDocument": ["nvda-10k.htm"],
                }
            }
        }
        mock_client.get_submissions.return_value = submissions
        mock_edgar_cls.return_value = mock_client

        fetcher = SECEdgarFilingFetcher(store=store, request_delay=0.01)

        with patch.object(fetcher, "_resolve_cik", return_value="0000320193"):
            count = fetcher.register_discovered_filings("NVDA")

        assert count == 1

        with store.sqlite._connect() as conn:
            row = conn.execute(
                "SELECT * FROM filings WHERE accession = ?",
                ("0000320193-26-000099",),
            ).fetchone()
        assert row is not None
        assert row["ticker"] == "NVDA"
        assert row["filing_type"] == "10-K"
        assert row["status"] == "unprocessed"
        assert row["cik"] == "0000320193"
        assert row["primary_document"] == "nvda-10k.htm"
        assert row["discovery_scope"] == "deep"

    @patch("src.sec.edgar_fetcher.EdgarClient")
    def test_register_discovered_filings_dedup(self, mock_edgar_cls, tmp_path):
        store = _make_store(tmp_path)

        mock_client = MagicMock()
        submissions = {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "accessionNumber": ["0000320193-26-000100"],
                    "filingDate": ["2026-03-15"],
                    "reportDate": ["2025"],
                    "primaryDocument": ["nvda-10k.htm"],
                }
            }
        }
        mock_client.get_submissions.return_value = submissions
        mock_edgar_cls.return_value = mock_client

        fetcher = SECEdgarFilingFetcher(store=store, request_delay=0.01)

        with patch.object(fetcher, "_resolve_cik", return_value="0000320193"):
            count1 = fetcher.register_discovered_filings("NVDA")
            count2 = fetcher.register_discovered_filings("NVDA")

        assert count1 == 1
        assert count2 == 0

    @patch("src.sec.edgar_fetcher.requests.get")
    @patch("src.sec.edgar_fetcher.EdgarClient")
    def test_download_filing_text_success(self, mock_edgar_cls, mock_get, tmp_path):
        store = _make_store(tmp_path)
        fetcher = SECEdgarFilingFetcher(store=store, request_delay=0.01)
        mock_edgar_cls.return_value = MagicMock()

        mock_resp = MagicMock()
        mock_resp.text = (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
            "WASHINGTON, D.C. 20549\n\n"
            "FORM 10-K\n\n"
            "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) "
            "OF THE SECURITIES EXCHANGE ACT OF 1934\n\n"
            "For the fiscal year ended January 25, 2025\n"
            "Commission File Number: 0-00000\n\n"
            "NVIDIA CORPORATION\n"
        )
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        filing = {
            "ticker": "NVDA",
            "accession": "0000320193-26-000100",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/320193/"
                "00032019326000100/nvda-10k.htm"
            ),
        }
        text = fetcher.download_filing_text(filing)

        assert text is not None
        assert "NVIDIA CORPORATION" in text
        assert len(text) > 100

    @patch("src.sec.edgar_fetcher.requests.get")
    @patch("src.sec.edgar_fetcher.EdgarClient")
    def test_download_filing_text_failure(self, mock_edgar_cls, mock_get, tmp_path):
        store = _make_store(tmp_path)
        fetcher = SECEdgarFilingFetcher(store=store, request_delay=0.01)
        mock_edgar_cls.return_value = MagicMock()

        mock_get.side_effect = RuntimeError("SEC server error")

        filing = {
            "ticker": "NVDA",
            "accession": "bad-123",
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/bad/nvda.htm",
        }
        text = fetcher.download_filing_text(filing)

        assert text is None

    @patch("src.sec.edgar_fetcher.requests.get")
    @patch("src.sec.edgar_fetcher.EdgarClient")
    def test_download_relevant_documents_never_fetches_every_exhibit(
        self, mock_edgar_cls, mock_get, tmp_path,
    ):
        store = _make_store(tmp_path)
        fetcher = SECEdgarFilingFetcher(
            store=store,
            request_delay=0,
            user_agent="Researcher test@example.com",
            sec_config={
                "exhibits": {
                    "always": ["EX-99.1"],
                    "material_agreement_items": ["1.01", "2.01", "2.03"],
                    "allowlist": {},
                }
            },
        )
        mock_edgar_cls.return_value = MagicMock()
        index_html = (EVENT_FIXTURES / "filing-index.html").read_text()

        def response(url, **_kwargs):
            body = {
                "-index.html": index_html,
                "orcl-8k.htm": "ITEM 2.03 Creation of a Direct Financial Obligation. Senior notes agreement.",
                "ex991.htm": "Press release announcing the senior notes offering.",
                "ex101.htm": "Indenture governing the senior notes.",
            }
            match = next((value for suffix, value in body.items() if url.endswith(suffix)), None)
            if match is None:
                raise AssertionError(f"unexpected exhibit fetch: {url}")
            result = MagicMock(text=match)
            result.raise_for_status.return_value = None
            return result

        mock_get.side_effect = response
        documents = fetcher.download_relevant_documents({
            "ticker": "ORCL", "cik": "0001341439", "filing_type": "8-K",
            "accession": "0001193125-26-188001",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1341439/0001193125-26-188001.txt",
        })

        assert [document["document_type"] for document in documents] == [
            "PRIMARY", "EX-99.1", "EX-10.1",
        ]
        assert all("ex211.htm" not in call.args[0] for call in mock_get.call_args_list)

    @patch("src.sec.edgar_fetcher.requests.get")
    @patch("src.sec.edgar_fetcher.EdgarClient")
    def test_index_failure_never_falls_back_to_complete_submission_package(
        self, mock_edgar_cls, mock_get, tmp_path,
    ):
        store = _make_store(tmp_path)
        fetcher = SECEdgarFilingFetcher(
            store=store, request_delay=0,
            user_agent="Researcher test@example.com", sec_config={"exhibits": {}},
        )
        mock_edgar_cls.return_value = MagicMock()
        mock_get.side_effect = RuntimeError("index unavailable")

        documents = fetcher.download_relevant_documents({
            "ticker": "ORCL", "cik": "0001341439", "filing_type": "8-K",
            "accession": "0001193125-26-188001",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1341439/0001193125-26-188001.txt",
        })

        assert documents == []
        assert mock_get.call_count == 1


# ============================================================
# 2. TraceAlchemyFilingParser Tests
# ============================================================


class TestTraceAlchemyFilingParser:
    """Tests for the TraceAlchemy filing parser (model-as-parser)."""

    def test_import(self):
        assert TraceAlchemyFilingParser is not None

    def test_init_defaults(self):
        parser = TraceAlchemyFilingParser()
        assert parser.endpoint == "http://127.0.0.1:8087/v1/chat/completions"
        assert parser.model == "tracealchemy"

    def test_extraction_fields_defined(self):
        field_names = [f[0] for f in TraceAlchemyFilingParser.EXTRACTION_FIELDS]
        assert "total_revenue" in field_names
        assert "net_income" in field_names
        assert "eps_diluted" in field_names
        assert "operating_income" in field_names
        assert "total_assets" in field_names
        assert len(field_names) >= 20

    def test_build_extraction_prompt(self):
        parser = TraceAlchemyFilingParser()
        prompt = parser._build_extraction_prompt(
            "NVDA", "10-K", "Test filing text for NVIDIA.",
        )

        assert "NVDA" in prompt
        assert "10-K" in prompt
        assert "total_revenue" in prompt
        assert "net_income" in prompt
        assert "billion_usd" in prompt
        assert "Return the JSON array now" in prompt

    def test_build_extraction_prompt_truncates_long_text(self):
        parser = TraceAlchemyFilingParser()
        long_text = "X" * 50000
        prompt = parser._build_extraction_prompt("NVDA", "10-Q", long_text)

        start = prompt.index("--- START OF")
        end = prompt.index("--- END OF")
        embedded = prompt[start:end]
        assert len(embedded) <= TraceAlchemyFilingParser.MAX_PROMPT_TEXT_CHARS + 200
        assert embedded.count("X") == TraceAlchemyFilingParser.MAX_PROMPT_TEXT_CHARS

    def test_select_extraction_sections_finds_income_statement(self):
        parser = TraceAlchemyFilingParser()
        text = (
            "filler preamble line\n" * 50
            + "CONSOLIDATED STATEMENTS OF OPERATIONS\n"
            + "(In millions)\n"
            + "Total net sales $ 26,000\n"
            + "Net income $ 7,800\n\n"
            + "CONSOLIDATED BALANCE SHEETS\n"
            + "Total Assets: $50,000,000,000\n"
        )

        selected = parser._select_extraction_sections(text, "10-Q")
        assert "Total net sales" in selected
        assert "CONSOLIDATED STATEMENTS OF OPERATIONS" in selected
        assert "Net income" in selected

    def test_parse_model_response_valid_json(self):
        parser = TraceAlchemyFilingParser()
        response = json.dumps([
            {"metric": "total_revenue", "value": 26.0, "unit": "billion_usd"},
            {"metric": "net_income", "value": 7.8, "unit": "billion_usd"},
            {"metric": "eps_diluted", "value": 3.12, "unit": "usd"},
        ])

        facts = parser._parse_model_response(response, "NVDA", "2026-Q1")

        assert len(facts) == 3
        assert facts[0]["metric"] == "total_revenue"
        assert facts[0]["value"] == 26.0
        assert facts[0]["period"] == "2026-Q1"
        assert facts[0]["period_type"] == "quarterly"
        assert facts[0]["source_type"] == "sec_10-Q"

    def test_parse_model_response_with_fences(self):
        parser = TraceAlchemyFilingParser()
        response = """```json
[
    {"metric": "total_revenue", "value": 100.0, "unit": "billion_usd"}
]
```"""

        facts = parser._parse_model_response(response, "NVDA", "2025")
        assert len(facts) == 1
        assert facts[0]["metric"] == "total_revenue"
        assert facts[0]["period_type"] == "annual"

    def test_parse_model_response_empty_returns_empty_list(self):
        parser = TraceAlchemyFilingParser()
        assert parser._parse_model_response(None, "NVDA", "") == []
        assert parser._parse_model_response("", "NVDA", "") == []

    def test_parse_model_response_invalid_json_returns_empty(self):
        parser = TraceAlchemyFilingParser()
        facts = parser._parse_model_response("this is not json", "NVDA", "")
        assert facts == []

    def test_extract_facts_mocked_model(self):
        parser = TraceAlchemyFilingParser(
            endpoint="http://127.0.0.1:99999/v1/chat/completions",
        )
        facts = parser.extract_facts_from_filing(
            "NVDA",
            "10-Q",
            "Plain filing prose without financial table markers.",
            period="2026-Q1",
        )
        assert facts == []

    def test_extract_facts_from_filing_with_mock(self):
        parser = TraceAlchemyFilingParser()
        model_json = json.dumps([
            {"metric": "total_revenue", "value": 26.0, "unit": "billion_usd"},
            {"metric": "net_income", "value": 7.8, "unit": "billion_usd"},
        ])

        with patch.object(parser, "_call_model", return_value=model_json):
            facts = parser.extract_facts_from_filing(
                "NVDA",
                "10-Q",
                "NVIDIA CORPORATION quarterly report narrative section.",
                period="2026-Q1",
            )

        assert len(facts) == 2
        assert facts[0]["metric"] == "total_revenue"


# ============================================================
# 3. FilingProcessor Tests
# ============================================================


class TestFilingProcessor:
    """Tests for the FilingProcessor pipeline orchestrator."""

    def test_import(self):
        assert FilingProcessor is not None

    def test_init(self):
        processor = FilingProcessor()
        assert processor.store is not None
        assert processor.fetcher is not None
        assert processor.parser is not None

    def test_process_pending_filings_no_unprocessed(self, tmp_path):
        store = _make_store(tmp_path)
        processor = FilingProcessor(store=store)

        result = processor.process_pending_filings()
        assert result["processed"] == 0
        assert result["failed"] == 0

    def test_process_pending_filings_with_mock(self, tmp_path):
        store = _use_deterministic_embeddings(_make_store(tmp_path))
        store.register_filing(
            ticker="NVDA",
            filing_type="10-Q",
            filing_date="2026-05-15",
            period="2026-Q1",
            accession="0000320193-26-000099",
            source_url="https://sec.gov/...",
        )

        processor = FilingProcessor(store=store)
        processor.fetcher.download_filing_text = MagicMock(
            return_value=(
                "NVIDIA CORPORATION\nRevenue: $26,000,000,000\n"
                "Net Income: $7,800,000,000"
            ),
        )
        processor.parser.extract_facts_from_filing = MagicMock(return_value=[
            {
                "metric": "total_revenue", "value": 26.0, "unit": "billion_usd",
                "period": "2026-Q1", "period_type": "quarterly", "source_type": "sec_10-q",
            },
            {
                "metric": "net_income", "value": 7.8, "unit": "billion_usd",
                "period": "2026-Q1", "period_type": "quarterly", "source_type": "sec_10-q",
            },
        ])

        result = processor.process_pending_filings(limit=5)
        assert result["processed"] == 1
        assert result["failed"] == 0

        with store.sqlite._connect() as conn:
            row = conn.execute(
                "SELECT status FROM filings WHERE accession = ?",
                ("0000320193-26-000099",),
            ).fetchone()
        assert row["status"] == "parsed"

        facts = store.get_fundamentals_batch("NVDA", metrics=["total_revenue", "net_income"])
        assert "total_revenue" in facts
        assert facts["total_revenue"] == 26.0

    def test_process_pending_filings_download_failure(self, tmp_path):
        store = _make_store(tmp_path)
        store.register_filing(
            ticker="NVDA", filing_type="10-Q", filing_date="2026-05-15",
            period="2026-Q1", accession="ACC-001", source_url="https://sec.gov/...",
        )

        processor = FilingProcessor(store=store)
        processor.fetcher.download_filing_text = MagicMock(return_value=None)

        result = processor.process_pending_filings()
        assert result["processed"] == 0
        assert result["failed"] == 1

        with store.sqlite._connect() as conn:
            row = conn.execute(
                "SELECT status FROM filings WHERE accession = ?", ("ACC-001",),
            ).fetchone()
        assert row["status"] == "unprocessed"

    @patch("src.storage.store.ChromaStore")
    def test_event_filing_uses_normalized_contract_and_replays_without_duplicates(
        self, chroma_class, tmp_path,
    ):
        chroma_class.return_value = MagicMock()
        store = _make_store(tmp_path)
        store.upsert_universe_snapshot("ivv", "2026-07-10T00:00:00Z", [{
            "symbol": "ORCL", "company_name": "Oracle Corporation", "source": "ivv",
            "index_code": "sp500", "exchange": "NYSE", "cik": "0001341439",
            "source_url": "https://example.test/ivv",
        }])
        store.register_filing(
            "ORCL", "8-K", "2026-07-10", "", "0001193125-26-188001",
            "https://www.sec.gov/Archives/edgar/data/1341439/000119312526188001/orcl-8k.htm",
            cik="0001341439", discovery_scope="broad", items=["2.02"],
        )
        fetcher = MagicMock()
        fetcher.download_relevant_documents.return_value = [{
            "document_type": "PRIMARY",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1341439/000119312526188001/orcl-8k.htm",
            "text": "Item 2.02 Results of Operations. Quarterly earnings release.",
        }]
        parser = MagicMock()
        processor = FilingProcessor(
            store=store, fetcher=fetcher, parser=parser,
            sec_config={"index_event_filings": False, "index_filing_text": False},
        )

        first = processor.process_pending_filings()
        with store.sqlite._connect() as conn:
            conn.execute(
                "UPDATE filings SET status='unprocessed' WHERE accession=?",
                ("0001193125-26-188001",),
            )
            conn.commit()
        replay = processor.process_pending_filings()

        assert first["processed"] == replay["processed"] == 1
        assert store.sqlite.count_corpus_items() == 1
        assert store.sqlite.count_events() == 1
        item = store.sqlite.get_corpus_item("sec:0001193125-26-188001")
        assert item["evidence_authority"] == "direct_sec"
        assert item["document_family"] == "sec:0001193125-26-188001"
        event = store.sqlite.get_event("sec:0001193125-26-188001:earnings_release:sec-events-v1")
        assert event["classifier_version"] == "sec-events-v1"
        assert event["source_corpus_item_ids"] == ["sec:0001193125-26-188001"]
        parser.extract_facts_from_filing.assert_not_called()

    @patch("src.storage.store.ChromaStore")
    def test_direct_sec_evidence_promotes_a_deduplicated_vendor_copy(
        self, chroma_class, tmp_path,
    ):
        chroma_class.return_value = MagicMock()
        store = _make_store(tmp_path)
        store.upsert_universe_snapshot("ivv", "2026-07-10T00:00:00Z", [{
            "symbol": "ORCL", "company_name": "Oracle Corporation", "source": "ivv",
            "index_code": "sp500", "exchange": "NYSE", "cik": "0001341439",
            "source_url": "https://example.test/ivv",
        }])
        text = "Item 2.02 Results of Operations. Quarterly earnings release."
        body = f"[PRIMARY]\n{text}"
        store.upsert_narrative(NarrativeRecord(
            corpus_item_id="vendor-copy",
            source_name="vendor",
            source_category="vendor_parsed_filing",
            provider_record_id="vendor-1",
            original_publisher="Vendor",
            item_type="sec_filing",
            title="Vendor copy of ORCL filing",
            body=body,
            published_at="2026-07-10T00:00:00Z",
            observed_at="2026-07-10T01:00:00Z",
            accessed_at="2026-07-10T01:00:00Z",
            ingested_at="2026-07-10T01:00:00Z",
            source_url="https://vendor.test/orcl-filing",
            canonical_url="https://vendor.test/orcl-filing",
            license_label="vendor_parsed",
            normalization_version=NORMALIZATION_VERSION,
            content_hash=content_hash(body),
            document_family="vendor-copy",
            evidence_authority="vendor_parsed",
        ))
        store.register_filing(
            "ORCL", "8-K", "2026-07-10", "", "0001193125-26-188001",
            "https://www.sec.gov/Archives/edgar/data/1341439/000119312526188001/orcl-8k.htm",
            cik="0001341439", discovery_scope="broad", items=["2.02"],
        )
        fetcher = MagicMock()
        fetcher.download_relevant_documents.return_value = [{
            "document_type": "PRIMARY",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1341439/000119312526188001/orcl-8k.htm",
            "text": text,
        }]
        processor = FilingProcessor(
            store=store, fetcher=fetcher, parser=MagicMock(),
            sec_config={"index_event_filings": False, "index_filing_text": False},
        )

        result = processor.process_pending_filings()

        assert result["processed"] == 1
        assert store.sqlite.count_corpus_items() == 1
        promoted = store.sqlite.get_corpus_item("vendor-copy")
        assert promoted["source"] == "sec"
        assert promoted["provider_record_id"] == "0001193125-26-188001"
        assert promoted["evidence_authority"] == "direct_sec"
        assert promoted["source_url"].startswith("https://www.sec.gov/Archives/")
        event = store.sqlite.get_event(
            "sec:0001193125-26-188001:earnings_release:sec-events-v1"
        )
        assert event["source_corpus_item_ids"] == ["vendor-copy"]

    @patch("src.storage.store.ChromaStore")
    def test_one_accession_document_failure_does_not_stop_the_next(
        self, chroma_class, tmp_path,
    ):
        chroma_class.return_value = MagicMock()
        store = _make_store(tmp_path)
        store.upsert_universe_snapshot("ivv", "2026-07-10T00:00:00Z", [{
            "symbol": "ORCL", "company_name": "Oracle Corporation", "source": "ivv",
            "index_code": "sp500", "exchange": "NYSE", "cik": "0001341439",
            "source_url": "https://example.test/ivv",
        }])
        for accession, filing_date in (("FAIL-1", "2026-07-11"), ("PASS-2", "2026-07-10")):
            store.register_filing(
                "ORCL", "8-K", filing_date, "", accession,
                f"https://www.sec.gov/Archives/{accession}.htm",
                cik="0001341439", discovery_scope="broad", items=["2.02"],
            )
        fetcher = MagicMock()
        fetcher.download_relevant_documents.side_effect = [[], [{
            "document_type": "PRIMARY", "source_url": "https://www.sec.gov/pass.htm",
            "text": "Item 2.02 Results of Operations. Earnings release.",
        }]]
        processor = FilingProcessor(
            store=store, fetcher=fetcher, parser=MagicMock(),
            sec_config={"index_event_filings": False, "index_filing_text": False},
        )

        result = processor.process_pending_filings(limit=10)

        assert result["processed"] == 1
        assert result["failed"] == 1
        with store.sqlite._connect() as conn:
            statuses = dict(conn.execute("SELECT accession, status FROM filings").fetchall())
        assert statuses == {"FAIL-1": "unprocessed", "PASS-2": "parsed"}

        fetcher.download_relevant_documents.side_effect = None
        fetcher.download_relevant_documents.return_value = [{
            "document_type": "PRIMARY", "source_url": "https://www.sec.gov/retry.htm",
            "text": "Item 2.02 Results of Operations. Earnings release after retry.",
        }]
        retry = processor.process_pending_filings(limit=10)
        assert retry["processed"] == 1
        with store.sqlite._connect() as conn:
            assert conn.execute(
                "SELECT status FROM filings WHERE accession='FAIL-1'"
            ).fetchone()[0] == "parsed"


# ============================================================
# 4. FilingScheduler Tests
# ============================================================


class TestFilingScheduler:
    """Tests for the FilingScheduler (cache-aware discovery)."""

    def test_import(self):
        assert FilingScheduler is not None

    def test_init_with_custom_store(self, tmp_path):
        store = _make_store(tmp_path)
        scheduler = FilingScheduler(store=store)
        assert scheduler.store is store

    def test_ttl_default_from_watchlist(self, tmp_path):
        store = _make_store(tmp_path)
        scheduler = FilingScheduler(store=store)
        assert scheduler.ttl_hours == 12

    def test_run_discovery_caches_freshness(self, tmp_path):
        store = _make_store(tmp_path)
        scheduler = FilingScheduler(store=store, processor=MagicMock())

        store.mark_cache_fresh("NVDA", DISCOVERY_SOURCE, 12)
        scheduler.processor.discover_new_filings = MagicMock(return_value=2)

        with _patch_core_tickers(["NVDA"]):
            result = scheduler.run_discovery()

        assert result["skipped"] == 1
        scheduler.processor.discover_new_filings.assert_not_called()

    def test_run_discovery_stale_checks_ticker(self, tmp_path):
        store = _make_store(tmp_path)
        scheduler = FilingScheduler(store=store, processor=MagicMock())

        store.upsert_cache_stale("NVDA", DISCOVERY_SOURCE)
        scheduler.processor.discover_new_filings = MagicMock(return_value=3)

        with _patch_core_tickers(["NVDA"]):
            result = scheduler.run_discovery()

        assert result["checked"] == 1
        scheduler.processor.discover_new_filings.assert_called_once_with("NVDA")

    def test_status_report(self, tmp_path):
        store = _make_store(tmp_path)
        scheduler = FilingScheduler(store=store, processor=MagicMock())
        scheduler.processor.status_report.return_value = {
            "total_unprocessed": 0,
            "total_parsed": 0,
            "filings_by_ticker": {},
        }

        with _patch_core_tickers(["NVDA", "AMD"]):
            report = scheduler.status_report()

        assert "discovery" in report
        assert "pipeline" in report
        assert len(report["discovery"]) >= 2


# ============================================================
# 5. Live Integration Tests
# ============================================================


@pytest.mark.live
class TestLiveSECPipeline:
    """End-to-end live tests — requires running model server and internet."""

    pytestmark = pytest.mark.skipif(
        not _edgar_reachable() or not _model_alive(),
        reason="Requires SEC EDGAR + llama-server :8087",
    )

    def test_discover_and_process_single_ticker(self, tmp_path):
        store = _make_store(tmp_path)
        processor = FilingProcessor(store=store)
        scheduler = FilingScheduler(store=store, processor=processor)

        with _patch_core_tickers(["AAPL"]):
            result = scheduler.run_full_pipeline(force=True)

        assert result["discovery"]["checked"] >= 1
        assert result["processing"]["processed"] >= 1

        with store.sqlite._connect() as conn:
            fact_count = conn.execute(
                "SELECT COUNT(*) AS n FROM fundamentals WHERE ticker = ?",
                ("AAPL",),
            ).fetchone()["n"]
            parsed_count = conn.execute(
                "SELECT COUNT(*) AS n FROM filings WHERE ticker = ? AND status = 'parsed'",
                ("AAPL",),
            ).fetchone()["n"]

        assert fact_count >= 1
        assert parsed_count >= 1

    def test_parse_real_filing_text(self, tmp_path):
        store = _make_store(tmp_path)
        processor = FilingProcessor(store=store)

        filings = processor.fetcher.discover_filings("AAPL", ["10-Q"], count=1)
        assert filings, "expected at least one 10-Q from EDGAR"

        text = processor.fetcher.download_filing_text(filings[0])
        assert text is not None

        facts = processor.parser.extract_facts_from_filing(
            "AAPL", "10-Q", text, period=filings[0].get("period", "2026-Q1"),
        )
        assert len(facts) >= 1
        metrics = {f["metric"] for f in facts}
        assert TraceAlchemyFilingParser.CORE_METRICS.intersection(metrics)


class TestPhase14Integration:
    """Phase 1.3 + 1.4 coexistence in a shared Store."""

    def test_sec_pipeline_coexists_with_yfinance_data(self, tmp_path):
        store = _use_deterministic_embeddings(_make_store(tmp_path))

        store.save_fundamental(
            ticker="NVDA",
            metric="revenue_q1",
            value=26.0,
            unit="billion_usd",
            period="2026-Q1",
            period_type="quarterly",
            source_type="yfinance",
        )

        store.register_filing(
            ticker="NVDA",
            filing_type="10-Q",
            filing_date="2026-05-15",
            period="2026-Q1",
            accession="0000320193-26-000099",
            source_url="https://sec.gov/...",
        )

        # Legacy whole-document coexistence path (section indexing is covered
        # separately); the synthetic text is too small to section.
        processor = FilingProcessor(
            store=store, sec_config={"index_filing_text": False},
        )
        processor.fetcher.download_filing_text = MagicMock(
            return_value="NVIDIA filing text for embedding storage.",
        )
        processor.parser.extract_facts_from_filing = MagicMock(return_value=[
            {
                "metric": "total_revenue", "value": 26.5, "unit": "billion_usd",
                "period": "2026-Q1", "period_type": "quarterly", "source_type": "sec_10-q",
            },
        ])

        result = processor.process_pending_filings(limit=5)
        assert result["processed"] == 1

        yf_fact = store.get_fundamental("NVDA", "revenue_q1", period="2026-Q1")
        assert yf_fact is not None
        assert yf_fact["source_type"] == "yfinance"

        sec_facts = store.get_fundamentals_batch("NVDA", metrics=["total_revenue"])
        assert "total_revenue" in sec_facts

        search_result = store.search("NVDA revenue")
        assert search_result["facts"] or search_result["documents"]


# ============================================================
# 6. Helper & Utility Tests
# ============================================================


class TestFilingUtils:
    """Tests for utility functions in the SEC module."""

    def test_filing_type_from_period_annual(self):
        from src.sec.filing_parser import filing_type_from_period

        assert filing_type_from_period("2025") == "10-K"
        assert filing_type_from_period("") == "10-K"

    def test_filing_type_from_period_quarterly(self):
        from src.sec.filing_parser import filing_type_from_period

        assert filing_type_from_period("2026-Q1") == "10-Q"
        assert filing_type_from_period("2026-Q4") == "10-Q"
