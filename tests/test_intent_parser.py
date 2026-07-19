"""Tests for the IntentParser module (Phase 1.5.2)."""

import json

import pytest


def _catalog(tmp_path, entries=()):
    """Write a temporary symbol catalog so plan tests stay offline/deterministic.

    Passing an empty ``entries`` disables catalog resolution entirely, leaving
    only IntentParser's local company map and known-ticker set — enough for the
    big-cap companies these tests reference, and it never touches the real
    10k-entry ``data/symbol_catalog.json``.
    """
    path = tmp_path / "symbol_catalog.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2099-01-01T00:00:00+00:00",
                "ttl_hours": 168,
                "entries": list(entries),
            }
        ),
        encoding="utf-8",
    )
    return path


def _plan_parser(tmp_path, entries=()):
    from src.middleware.intent_parser import IntentParser
    from src.middleware.symbol_resolver import SymbolResolver

    return IntentParser(resolver=SymbolResolver(catalog_path=_catalog(tmp_path, entries)))


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

    def test_intent_parser_uses_resolver(self):
        """Injected resolver can resolve companies absent from the local map."""
        from src.middleware.intent_parser import IntentParser
        from src.middleware.symbol_resolver import Resolution

        class StubResolver:
            def resolve(self, text: str) -> Resolution:
                assert text == "What is BlackBerry revenue?"
                return Resolution("BB", 0.95, "BlackBerry", "catalog_exact")

        parser = IntentParser(resolver=StubResolver())
        result = parser.parse("What is BlackBerry revenue?")

        assert result["ticker"] == "BB"
        assert result["ticker_confidence"] == 0.95
        assert result["resolved_name"] == "BlackBerry"
        assert result["ticker_source"] == "catalog_exact"

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

    @pytest.mark.parametrize(
        ("question", "topic", "event_type", "item_type"),
        [
            ("How did Oracle finance its latest debt raise?", "financing", "debt_raise", "filing"),
            ("What corporate actions did Apple announce?", "corporate_action", "buyback", "corporate_action"),
            ("What did Oracle acquire?", "corporate_action", "acquisition", "corporate_action"),
            ("Summarize the latest merger and divestiture", "corporate_action", "divestiture", "corporate_action"),
            ("Show recent beneficial ownership changes for Tesla", "ownership", "beneficial_ownership_change", "filing"),
            ("Are there new regulatory actions affecting Microsoft?", "regulatory", "enforcement_action", "regulatory_event"),
            ("What was in the latest inflation release?", "macro_release", "economic_release", "economic_release"),
            ("What is the latest company news about Nvidia?", "company_news", None, "news"),
        ],
    )
    def test_finance_evidence_intents_do_not_require_provider_names(
        self, question, topic, event_type, item_type
    ):
        from src.middleware.intent_parser import IntentParser

        result = IntentParser().parse(question)
        assert result["evidence_topic"] == topic
        assert result["evidence_filters"]["item_type"] == item_type
        if event_type is not None:
            assert event_type in result["evidence_filters"]["event_types"]
        assert not any(
            provider in str(result["evidence_filters"]).lower()
            for provider in ("finnhub", "massive", "gdelt")
        )

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


class TestQueryPlanParsing:
    """parse_plan() multi-entity/intent/metric/period contract (2.2.3.1).

    Uses an injected resolver over a controlled catalog so no network, real
    catalog, Chroma, middleware, or model is involved.
    """

    def test_raw_question_preserved_exactly(self, tmp_path):
        """Whitespace and punctuation survive verbatim in original_question."""
        parser = _plan_parser(tmp_path)
        raw = "  Compare  NVDA  and AMD's revenue?? \t"
        plan = parser.parse_plan(raw)
        assert plan.original_question == raw, "Raw question must be byte-identical"
        # Matching-only normalization is a separate, collapsed representation.
        assert plan.normalized_question != raw
        assert "  " not in plan.normalized_question

    def test_retrieval_query_is_separate_and_becomes_sq0(self, tmp_path):
        """A carried standalone query enriches retrieval without touching the raw."""
        parser = _plan_parser(tmp_path)
        raw = "What about AMD?"
        retrieval = "Compare NVDA and AMD revenue for FY 2025"
        plan = parser.parse_plan(raw, retrieval_query=retrieval)

        assert plan.original_question == raw, "Raw question stays unchanged"
        assert plan.retrieval_query == retrieval, "Retrieval query is stored separately"
        assert plan.subqueries[0].id == "sq0"
        assert plan.subqueries[0].text == retrieval, "sq0 is the retrieval query"
        # Entities/metrics/periods come from the enriched retrieval query.
        assert plan.tickers == ["NVDA", "AMD"]
        assert "total_revenue" in plan.metrics
        assert "fy 2025" in plan.periods

    def test_resolve_all_preserves_mention_order(self, tmp_path):
        """'AMD versus NVIDIA' resolves to [AMD, NVDA] by mention, not map order."""
        from src.middleware.symbol_resolver import SymbolResolver

        resolver = SymbolResolver(catalog_path=_catalog(tmp_path))
        results = resolver.resolve_all("AMD versus NVIDIA")
        assert [r.ticker for r in results] == ["AMD", "NVDA"]
        assert [r.start for r in results] == sorted(r.start for r in results)

    def test_override_keeps_comparison_entity(self, tmp_path):
        """An override plus another explicit ticker yields two entities."""
        parser = _plan_parser(tmp_path)
        plan = parser.parse_plan("Compare revenue with AMD", override_ticker="NVDA")
        assert [e.ticker for e in plan.entities] == ["NVDA", "AMD"]
        assert plan.entities[0].source == "override"
        assert plan.entities[1].source in {"local_map", "known_ticker", "catalog_exact"}

    def test_override_only_when_no_other_entity(self, tmp_path):
        """With no other entity mentioned, the override is the only entity."""
        parser = _plan_parser(tmp_path)
        plan = parser.parse_plan("What is the revenue?", override_ticker="CRWD")
        assert [e.ticker for e in plan.entities] == ["CRWD"]
        assert plan.entities[0].source == "override"

    def test_multi_intent_plan(self, tmp_path):
        """Revenue trend plus risks retains both trend and risk intents."""
        parser = _plan_parser(tmp_path)
        plan = parser.parse_plan("Show NVDA revenue trend and its key risks")
        assert "trend" in plan.intents
        assert "risk" in plan.intents

    def test_multiple_periods_retained(self, tmp_path):
        """A 2023-through-2025 range is not collapsed to a single year."""
        parser = _plan_parser(tmp_path)
        plan = parser.parse_plan("What was Apple revenue from 2023 through 2025?")
        assert "2023" in plan.periods
        assert "2025" in plan.periods

    def test_ambiguous_words_do_not_become_tickers(self, tmp_path):
        """Ordinary words 'target'/'gap' inside prose never resolve to tickers."""
        parser = _plan_parser(
            tmp_path,
            entries=[
                {"ticker": "TGT", "name": "Target Corp"},
                {"ticker": "GAP", "name": "Gap Inc"},
            ],
        )
        plan = parser.parse_plan(
            "what is the analyst price target and the gap between margins"
        )
        assert plan.tickers == [], f"Expected no tickers, got {plan.tickers}"

    def test_plan_rejects_derived_entity_drift(self, tmp_path):
        """A derived subquery introducing a new entity fails validation."""
        from src.middleware.query_plan import QueryPlanError, QuerySubquery

        parser = _plan_parser(tmp_path)
        plan = parser.parse_plan("What is NVDA revenue?")
        plan.subqueries.append(
            QuerySubquery(
                id="sq1", text="AMD revenue", entity_tickers=("AMD",),
                derived=True, parent_id="sq0",
            )
        )
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "derived_entity_drift" in exc.value.reason_codes

    @pytest.mark.parametrize("question", [
        "What is NVDA revenue?",
        "How is Apple doing?",
        "Compare AMD and NVDA",
        "What is Meta's PE ratio in Q1 2026?",
        "Broadcom dividend yield",
        "What is the trend for NVDA revenue?",
        "Why did NVDA stock drop?",
        "What is the weather today?",
    ])
    def test_legacy_adapter_matches_existing_parse_contract(self, tmp_path, question):
        """The legacy adapter preserves its established fields and values."""
        parser = _plan_parser(tmp_path)
        legacy = parser.parse_plan(question).to_legacy_intent()
        parsed = parser.parse(question)
        assert legacy == {key: parsed[key] for key in legacy}
