"""Tests for the IntentParser module (Phase 1.5.2)."""

import pytest


class TestIntentParser:
    """Tests for the IntentParser module."""

    def test_import(self):
        """IntentParser imports successfully."""
        from src.middleware.intent_parser import IntentParser
        assert IntentParser is not None

    def test_init(self):
        """Default init creates parser with compiled regexes."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        assert parser._metric_regexes is not None
        assert len(parser._metric_regexes) > 10
        assert parser._type_regexes is not None
        assert len(parser._type_regexes) >= 7

    # ── Ticker Detection ───────────────────────────────

    @pytest.mark.parametrize("question,expected_ticker", [
        ("What is NVDA revenue?", "NVDA"),
        ("How is Apple doing?", "AAPL"),
        ("Compare AMD and NVDA", "AMD"),  # First match
        ("Tell me about Microsoft", "MSFT"),
        ("What is Meta's PE ratio?", "META"),
        ("CrowdStrike earnings report", "CRWD"),
        ("Palantir outlook", "PLTR"),
        ("Broadcom dividend yield", "AVGO"),
        ("What is the market cap of Tesla?", "TSLA"),
        ("How is Amazon's cloud business?", "AMZN"),
    ])
    def test_ticker_detection_symbols_and_names(self, question, expected_ticker):
        """Ticker is detected from both symbols and company names."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        assert result["ticker"] == expected_ticker, (
            f"Expected {expected_ticker} for '{question}', got {result['ticker']}"
        )

    @pytest.mark.parametrize("question", [
        "What is the weather today?",
        "How do I cook pasta?",
        "Tell me about the economy",
        "What time is it?",
    ])
    def test_no_ticker_detected(self, question):
        """Questions without ticker references return None."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        assert result["ticker"] is None

    def test_override_ticker(self):
        """Override ticker takes precedence over detected ticker."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("What is revenue?", override_ticker="CRWD")
        assert result["ticker"] == "CRWD"

    # ── Metric Extraction ──────────────────────────────

    @pytest.mark.parametrize("question,expected_metrics", [
        ("What is NVDA revenue?", ["total_revenue"]),
        ("What is the PE ratio and EPS for MSFT?", ["pe_ratio", "eps_diluted"]),
        ("Show me gross margin and operating margin", ["gross_margin_pct", "operating_margin_pct"]),
        ("What is free cash flow?", ["free_cash_flow"]),
        ("ROE and ROA for AMD", ["roe", "roa"]),
        ("Market cap and enterprise value", ["market_cap", "enterprise_value"]),
        ("What is the dividend yield?", ["dividend_yield"]),
        ("What's the price target for NVDA?", ["price_target_mean"]),
    ])
    def test_metric_extraction(self, question, expected_metrics):
        """Financial metrics are extracted from the question."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        for metric in expected_metrics:
            assert metric in result["metrics"], (
                f"Expected metric '{metric}' in {result['metrics']} for '{question}'"
            )

    def test_no_metrics_detected(self):
        """Questions without financial metrics return empty list."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("What is the outlook for NVDA?")
        assert result["metrics"] == []

    # ── Question Type Classification ───────────────────

    @pytest.mark.parametrize("question,expected_type", [
        ("What is NVDA revenue?", "fact_lookup"),
        ("How much did AMD earn?", "fact_lookup"),
        ("Compare NVDA and AMD", "comparison"),
        ("NVDA vs AMD which is better", "comparison"),
        ("What is the trend for NVDA revenue?", "trend"),
        ("How has NVDA performed over time?", "trend"),
        ("Why did NVDA stock drop?", "explanation"),
        ("Explain NVDA's competitive advantage", "explanation"),
        ("What is the market sentiment on AMD?", "sentiment"),
        ("Analyst outlook for META", "sentiment"),
        ("what's the price target for NVDA", "projection"),
        ("what should I expect next quarter", "projection"),
        ("What will NVIDIA's revenue be next year?", "projection"),
        ("What's the consensus outlook for NVDA next quarter?", "projection"),
        ("Any news on CRWD?", "news"),
        ("What happened with Palantir?", "news"),
        ("What are the risks for NVDA?", "risk"),
        ("NVDA risk factors", "risk"),
        ("Tell me about AI chips", "general"),
    ])
    def test_question_type_classification(self, question, expected_type):
        """Question type is correctly classified."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        assert result["question_type"] == expected_type, (
            f"Expected '{expected_type}' for '{question}', got '{result['question_type']}'"
        )

    # ── Timeframe Extraction ──────────────────────────

    @pytest.mark.parametrize("question,expected_tf", [
        ("What was revenue in Q1 2026?", "q1 2026"),
        ("Revenue for FY 2025", "fy 2025"),
        ("Latest quarter results", "latest quarter"),
        ("TTM revenue", "ttm"),
        ("YTD performance", "ytd"),
    ])
    def test_timeframe_extraction(self, question, expected_tf):
        """Timeframe is extracted from the question."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        assert result["timeframe"] is not None
        assert expected_tf in result["timeframe"].lower()

    def test_no_timeframe(self):
        """Questions without timeframe return None."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("What is NVDA's competitive advantage?")
        assert result["timeframe"] is None
