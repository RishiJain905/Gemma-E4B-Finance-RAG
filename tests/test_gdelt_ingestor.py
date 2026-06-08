"""
tests/test_gdelt_ingestor.py
Pytest suite for Phase 1.6.2 GDELT global news ingestion.

CI / automated runs: mock all GDELT HTTP (no live api.gdeltproject.org calls).
Manual live verification: ONE fetch only, then wait 5+ seconds before another.

Usage:
    pytest tests/test_gdelt_ingestor.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.storage.store import Store


@pytest.fixture(autouse=True)
def reset_gdelt_rate_limit_state():
    """Isolate module-level GDELT throttle state between tests."""
    import src.macros.gdelt_ingestor as gdelt_mod

    gdelt_mod._last_gdelt_request_at = 0.0
    yield
    gdelt_mod._last_gdelt_request_at = 0.0


@pytest.fixture
def mock_chroma():
    """Patch ChromaStore construction; yield the mock instance."""
    with patch("src.storage.store.ChromaStore") as mock_cls:
        instance = MagicMock()
        instance.heartbeat.return_value = True
        instance.count.return_value = 0
        instance.search.return_value = []
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def store(mock_chroma, tmp_path: Path):
    """Store with real SQLite (tmp_path) and mocked Chroma."""
    return Store(db_path=tmp_path / "test.db", chroma_path=tmp_path / "chroma")


class TestGDELTIngestor:
    """Tests for the GDELT news ingestion module."""

    def test_import(self):
        """GDELTIngestor imports successfully."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        assert GDELTIngestor is not None

    def test_init(self, store):
        """Default init with ticker mapping loaded."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)
        assert len(ingestor.TICKER_TO_QUERY) > 20
        assert "NVDA" in ingestor.TICKER_TO_QUERY

    def test_process_articles_dedup(self):
        """Duplicate URLs are removed during processing."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()

        articles = [
            {"url": "https://example.com/news1", "title": "News 1",
             "content": "Content 1", "tone": "1.5", "date": "2026-06-01"},
            {"url": "https://example.com/news1", "title": "News 1 (dup)",
             "content": "Content 1 dup", "tone": "2.0", "date": "2026-06-01"},
            {"url": "https://example.com/news2", "title": "News 2",
             "content": "Content 2", "tone": "-0.5", "date": "2026-06-02"},
        ]

        processed = ingestor._process_articles(articles, "NVDA")
        assert len(processed) == 2

    def test_process_articles_parses_tone(self):
        """Tone scores are parsed correctly as floats."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()

        articles = [
            {"url": "https://example.com/nvda1", "title": "NVDA Up",
             "content": "NVIDIA stock rises", "tone": "12.5",
             "date": "2026-06-01"},
            {"url": "https://example.com/nvda2", "title": "NVDA Down",
             "content": "NVIDIA stock falls", "tone": "-8.3",
             "date": "2026-06-01"},
        ]

        processed = ingestor._process_articles(articles, "NVDA")
        assert len(processed) == 2
        assert processed[0]["tone"] == 12.5
        assert processed[1]["tone"] == -8.3

    def test_process_articles_missing_fields(self):
        """Articles missing critical fields are filtered."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()

        articles = [
            {"url": "", "title": "No URL", "content": "Missing URL"},
            {"url": "https://example.com/valid", "title": "", "content": "",
             "tone": "0.0", "date": "2026-06-01"},
            {"url": "https://example.com/good", "title": "Good Article",
             "content": "Real content here", "tone": "3.0",
             "date": "2026-06-01"},
        ]

        processed = ingestor._process_articles(articles, "NVDA")
        assert len(processed) == 1

    def test_store_articles(self, store, mock_chroma):
        """Articles are stored in ChromaDB with correct metadata."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        articles = [
            {
                "title": "NVIDIA Beats Expectations",
                "content": "NVIDIA reported strong earnings...",
                "url": "https://example.com/nvda-earnings",
                "tone": 8.5,
                "date": "2026-06-01",
                "ticker": "NVDA",
                "source": "gdelt",
                "entities": "Jensen Huang",
            },
        ]

        stored = ingestor._store_articles(articles)
        assert stored == 1
        mock_chroma.add_document.assert_called_once()
        call_kwargs = mock_chroma.add_document.call_args.kwargs
        assert call_kwargs["ticker"] == "NVDA"
        assert call_kwargs["source"] == "gdelt"
        assert call_kwargs["metadata"]["tone"] == 8.5

    def test_sentiment_summary_empty(self, store):
        """Empty data returns zero-count sentiment summary."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        summary = ingestor.get_sentiment_summary("NVDA")
        assert summary["ticker"] == "NVDA"
        assert summary["article_count"] == 0
        assert summary["average_tone"] is None

    def test_sentiment_summary_with_data(self, store, mock_chroma):
        """Stored articles produce a valid sentiment summary."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        tones = [5.0, 3.0, -2.0, 1.0, -4.0]
        mock_chroma.search.return_value = [
            {"id": f"gdelt/NVDA/{i}", "document": f"News {i}",
             "metadata": {"tone": tone, "ticker": "NVDA", "source": "gdelt"},
             "distance": 0.1}
            for i, tone in enumerate(tones)
        ]
        mock_chroma.count.return_value = len(tones)

        summary = ingestor.get_sentiment_summary("NVDA")
        assert summary["ticker"] == "NVDA"
        assert summary["article_count"] == 5
        assert summary["average_tone"] == 0.6
        assert summary["positive_ratio"] == 0.4
        assert summary["negative_ratio"] == 0.4

    def test_build_doc_query_with_domains(self, store):
        """DOC query joins keyword and domain OR block without extra parens."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)
        ingestor.config["finance_domains"] = ["reuters.com", "bloomberg.com"]

        query = ingestor._build_doc_query("NVIDIA")
        assert query == "NVIDIA (domain:reuters.com OR domain:bloomberg.com)"
        assert not query.startswith("(NVIDIA)")

        phrase_query = ingestor._build_doc_query("Advanced Micro Devices")
        assert phrase_query == '"Advanced Micro Devices" (domain:reuters.com OR domain:bloomberg.com)'

    def test_build_doc_query_without_domains(self, store):
        """DOC query without domains is just the keyword."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)
        ingestor.config["finance_domains"] = []

        assert ingestor._build_doc_query("NVIDIA") == "NVIDIA"

    def test_build_doc_query_caps_domains(self, store):
        """Long finance_domains lists are capped by max_finance_domains."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)
        ingestor.config["finance_domains"] = [
            "reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com",
        ]
        ingestor.config["max_finance_domains"] = 2

        query = ingestor._build_doc_query("NVIDIA")
        assert query == "NVIDIA (domain:reuters.com OR domain:bloomberg.com)"

    def test_build_doc_query_caps_domain_count(self, store):
        """Long domain lists are capped to stay within GDELT query limits."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)
        ingestor.config["finance_domains"] = [
            "reuters.com", "bloomberg.com", "cnbc.com",
            "marketwatch.com", "seekingalpha.com", "finance.yahoo.com",
        ]
        ingestor.config["max_finance_domains"] = 3

        query = ingestor._build_doc_query("NVIDIA")
        assert query == "NVIDIA (domain:reuters.com OR domain:bloomberg.com OR domain:cnbc.com)"
        assert "seekingalpha" not in query

    def test_search_gdelt_falls_back_without_domains(self, store):
        """Empty domain-scoped result retries with bare keyword query."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)
        ingestor.config["finance_domains"] = ["reuters.com"]

        domain_response = MagicMock()
        domain_response.text = "Your query was too short or too long.\n"
        domain_response.headers = {"content-type": "text/plain"}

        json_response = MagicMock()
        json_response.text = (
            '{"articles": [{"url": "https://example.com/nvda", "title": "NVDA news"}]}'
        )
        json_response.headers = {"content-type": "application/json"}

        with patch.object(
            ingestor, "_doc_api_get", side_effect=[domain_response, json_response],
        ) as mock_get:
            articles = ingestor._search_gdelt("NVIDIA", max_records=5, lookback_days=7)

        assert len(articles) == 1
        assert articles[0]["title"] == "NVDA news"
        assert mock_get.call_count == 2
        assert mock_get.call_args_list[1].args[0]["query"] == "NVIDIA"

    def test_search_gdelt_falls_back_after_429(self, store):
        """429 exhaustion on domain query still retries bare keyword once."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)
        ingestor.config["finance_domains"] = ["reuters.com"]

        json_response = MagicMock()
        json_response.text = (
            '{"articles": [{"url": "https://example.com/nvda", "title": "NVDA news"}]}'
        )
        json_response.headers = {"content-type": "application/json"}

        with patch.object(
            ingestor, "_doc_api_get", side_effect=[None, json_response],
        ) as mock_get:
            articles = ingestor._search_gdelt("NVIDIA", max_records=5, lookback_days=7)

        assert len(articles) == 1
        assert mock_get.call_count == 2

    def test_parse_doc_api_articles_non_json(self, store):
        """Non-JSON DOC API bodies log a warning and return []."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        mock_response = MagicMock()
        mock_response.text = "<html><body>Invalid query</body></html>"
        mock_response.headers = {"content-type": "text/html"}

        articles = ingestor._parse_doc_api_articles(mock_response, "NVIDIA")
        assert articles == []

    def test_parse_doc_api_articles_valid_json(self, store):
        """Valid ArtList JSON returns article list."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        mock_response = MagicMock()
        mock_response.text = '{"articles": [{"url": "https://example.com/a", "title": "A"}]}'
        mock_response.headers = {"content-type": "application/json"}

        articles = ingestor._parse_doc_api_articles(mock_response, "NVIDIA")
        assert len(articles) == 1
        assert articles[0]["title"] == "A"

    def test_search_gdelt_failure(self, store):
        """GDELT search failure returns empty list."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        with patch("src.macros.gdelt_ingestor.httpx.get", side_effect=Exception("GDELT API error")):
            articles = ingestor._search_gdelt("NVDA", max_records=10, lookback_days=7)

        assert articles == []

    def test_fetch_news_for_ticker_dedup_across_terms(self, store):
        """Same URL from multiple query terms is deduplicated."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        shared_article = {
            "url": "https://example.com/nvda-shared",
            "title": "NVIDIA Soars",
            "content": "Shared article",
            "tone": "4.0",
            "date": "2026-06-01",
        }

        with patch.object(ingestor, "_search_gdelt", return_value=[shared_article]), \
             patch.object(ingestor, "_fetch_gkg_tone_map", return_value={}):
            articles = ingestor.fetch_news_for_ticker("NVDA", max_records=10)

        assert len(articles) == 1
        assert articles[0]["url"] == "https://example.com/nvda-shared"

    def test_respect_rate_limit_enforces_delay(self, store):
        """_respect_rate_limit sleeps when calls are closer than request_delay."""
        from src.macros.gdelt_ingestor import GDELTIngestor

        ingestor = GDELTIngestor(store=store)
        ingestor.config["request_delay"] = 5.0
        monotonic_values = iter([1000.0, 1000.0, 1002.0, 1002.0])

        with patch(
            "src.macros.gdelt_ingestor.time.monotonic",
            side_effect=lambda: next(monotonic_values),
        ), patch("src.macros.gdelt_ingestor.time.sleep") as mock_sleep:
            ingestor._respect_rate_limit()
            ingestor._respect_rate_limit()

        mock_sleep.assert_called_once()
        slept = mock_sleep.call_args[0][0]
        assert slept == pytest.approx(3.0)

    def test_fetch_news_serializes_doc_api_calls(self, store):
        """Each query term uses the throttled GDELT HTTP path (no back-to-back burst)."""
        from src.macros.gdelt_ingestor import GDELTIngestor

        ingestor = GDELTIngestor(store=store)
        ingestor.config["finance_domains"] = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"articles": []}'
        mock_response.headers = {"content-type": "application/json"}

        with patch.object(
            ingestor, "_gdelt_http_get", return_value=mock_response,
        ) as mock_http, patch.object(ingestor, "_fetch_gkg_tone_map", return_value={}):
            ingestor.fetch_news_for_ticker("NVDA", max_records=10)

        assert mock_http.call_count == len(ingestor.TICKER_TO_QUERY["NVDA"])

    def test_fetch_financial_news_searches_topics(self, store):
        """fetch_financial_news searches each configured topic."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        with patch.object(ingestor, "_search_gdelt", return_value=[]) as mock_search, \
             patch.object(ingestor, "_fetch_gkg_tone_map", return_value={}):
            ingestor.fetch_financial_news(max_records=70)

        topics = ingestor.config.get("financial_topics", [])
        assert mock_search.call_count == len(topics)

    def test_parse_v2tone(self):
        """V2Tone CSV field parses to average tone float."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()

        assert ingestor._parse_v2tone(
            "1.86,2.83,0.97,3.80,18.70,0.22,1165",
        ) == 1.86
        assert ingestor._parse_v2tone("") is None
        assert ingestor._parse_v2tone("not-a-number,1,2") is None

    def test_estimate_title_tone(self):
        """Title lexicon produces bounded sentiment scores."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()

        positive = ingestor._estimate_title_tone("NVIDIA stock soars on record earnings beat")
        negative = ingestor._estimate_title_tone("Tech stocks plunge amid sell-off fears")
        neutral = ingestor._estimate_title_tone("Company announces quarterly update")

        assert positive > 0
        assert negative < 0
        assert neutral == 0.0

    def test_enrich_articles_with_gkg_tone(self, store):
        """GKG URL matches populate article tone before processing."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        articles = [
            {
                "url": "https://example.com/nvda-gkg",
                "title": "NVIDIA update",
                "seendate": "20260608T034600Z",
            },
            {
                "url": "https://example.com/nvda-fallback",
                "title": "NVIDIA stock soars on beat",
                "seendate": "20260608T034600Z",
            },
        ]
        gkg_map = {"https://example.com/nvda-gkg": 3.5}

        with patch.object(ingestor, "_fetch_gkg_tone_map", return_value=gkg_map):
            ingestor._enrich_articles_with_tone(articles, ["NVIDIA"], lookback_days=1)

        assert articles[0]["tone"] == 3.5
        assert articles[1]["tone"] > 0

    def test_process_articles_without_doc_tone(self):
        """DOC ArtList articles get tone via enrichment or title fallback."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()

        articles = [
            {
                "url": "https://example.com/nvda1",
                "title": "NVIDIA stock soars",
                "tone": 4.2,
                "date": "2026-06-01",
            },
            {
                "url": "https://example.com/nvda2",
                "title": "NVIDIA quarterly update",
                "date": "2026-06-01",
            },
        ]

        processed = ingestor._process_articles(articles, "NVDA")
        assert processed[0]["tone"] == 4.2
        assert processed[1]["tone"] == 0.0

    def test_load_gkg_file_parses_zip(self, store):
        """GKG zip CSV rows map DocumentIdentifier to V2Tone."""
        import io
        import zipfile
        from src.macros.gdelt_ingestor import GDELTIngestor

        ingestor = GDELTIngestor(store=store)
        csv_body = (
            "id\t20260608040000\t1\tsource\thttps://example.com/nvda-gkg\t"
            + "\t" * 10
            + "2.5,1.0,1.5,0,0,0,100\n"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("20260608040000.gkg.csv", csv_body)
        mock_response = type("R", (), {"status_code": 200, "content": buf.getvalue()})()

        with patch.object(
            ingestor, "_gdelt_http_get", return_value=mock_response,
        ):
            rows = ingestor._load_gkg_file("20260608040000")

        assert rows["https://example.com/nvda-gkg"] == 2.5

    def test_fetch_news_for_ticker_enriches_tone(self, store):
        """fetch_news_for_ticker enriches DOC articles with tone metadata."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        doc_article = {
            "url": "https://example.com/nvda-live",
            "title": "NVIDIA CEO comments",
            "seendate": "20260608T034600Z",
        }

        with patch.object(ingestor, "_search_gdelt", return_value=[doc_article]), \
             patch.object(
                 ingestor,
                 "_fetch_gkg_tone_map",
                 return_value={"https://example.com/nvda-live": -1.2},
             ):
            articles = ingestor.fetch_news_for_ticker("NVDA", max_records=5)

        assert len(articles) == 1
        assert articles[0]["tone"] == -1.2

    def test_sentiment_summary_after_store_with_tone(self, store, mock_chroma):
        """Live fetch+store path supports get_sentiment_summary tone metadata."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        articles = [
            {
                "title": "NVIDIA beats",
                "content": "Strong quarter",
                "url": "https://example.com/nvda-earn",
                "tone": 6.0,
                "date": "2026-06-01",
                "ticker": "NVDA",
                "source": "gdelt",
                "entities": "",
            },
            {
                "title": "NVIDIA warning",
                "content": "Supply concerns",
                "url": "https://example.com/nvda-warn",
                "tone": -3.0,
                "date": "2026-06-02",
                "ticker": "NVDA",
                "source": "gdelt",
                "entities": "",
            },
        ]
        ingestor._store_articles(articles)

        mock_chroma.search.return_value = [
            {
                "id": "gdelt/NVDA/1",
                "document": "NVIDIA beats",
                "metadata": {"tone": 6.0, "ticker": "NVDA", "source": "gdelt"},
                "distance": 0.1,
            },
            {
                "id": "gdelt/NVDA/2",
                "document": "NVIDIA warning",
                "metadata": {"tone": -3.0, "ticker": "NVDA", "source": "gdelt"},
                "distance": 0.1,
            },
        ]

        summary = ingestor.get_sentiment_summary("NVDA")
        assert summary["article_count"] == 2
        assert summary["average_tone"] == 1.5
        assert summary["positive_ratio"] == 0.5
        assert summary["negative_ratio"] == 0.5
