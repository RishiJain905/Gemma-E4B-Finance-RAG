import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.middleware.intent_parser import IntentParser
from src.middleware.prompt_augmenter import PromptAugmenter
from src.middleware.retriever import Retriever
from src.middleware.tools import data_tools
from src.storage.store import Store


@pytest.fixture
def mock_chroma():
    with patch("src.storage.store.ChromaStore") as mock_cls:
        instance = MagicMock()
        instance.heartbeat.return_value = True
        instance.count.return_value = 0
        instance.search.return_value = []
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def store(mock_chroma, tmp_path: Path):
    return Store(db_path=tmp_path / "projection.db", chroma_path=tmp_path / "chroma")


def _seed_metric(
    store,
    ticker,
    metric,
    value,
    period="FY2027E",
    unit="usd",
    period_type="estimate",
    source_type="estimates",
):
    store.save_fundamental(
        ticker=ticker,
        metric=metric,
        value=value,
        unit=unit,
        period=period,
        period_type=period_type,
        source_type=source_type,
    )


@pytest.mark.parametrize(
    "question",
    [
        "what's the price target for NVDA",
        "what should I expect next quarter",
        "What will NVIDIA's revenue be next year?",
        "What's the consensus outlook for NVDA next quarter?",
    ],
)
def test_intent_detects_projection(question):
    assert IntentParser().parse(question)["question_type"] == "projection"


def test_analyst_outlook_stays_sentiment():
    assert IntentParser().parse("Analyst outlook for META")["question_type"] == "sentiment"


def test_get_estimates_tool(store):
    _seed_metric(store, "NVDA", "estimate_revenue_next_y", 120.0, period="FY2027E")
    _seed_metric(store, "NVDA", "estimate_eps_next_q", 4.1, period="2026-Q3E")
    _seed_metric(
        store,
        "NVDA",
        "total_revenue",
        100.0,
        period="FY2026",
        period_type="annual",
        source_type="yfinance",
    )

    result = data_tools.get_estimates_handler(store, ticker="nvda")

    assert result["ticker"] == "NVDA"
    assert result["estimates"]["estimate_revenue_next_y"] == {
        "value": 120.0,
        "period": "FY2027E",
    }
    assert result["estimates"]["estimate_eps_next_q"] == {
        "value": 4.1,
        "period": "2026-Q3E",
    }
    assert result["growth_vs_realized"]["estimate_revenue_next_y"] == 0.2


def test_get_price_targets_tool(store):
    for metric, value in {
        "price_target_mean": 185.0,
        "price_target_high": 220.0,
        "price_target_low": 150.0,
        "num_analysts": 42.0,
        "recommendation_mean": 1.8,
    }.items():
        _seed_metric(store, "NVDA", metric, value, period="2027-07E")

    result = data_tools.get_price_targets_handler(store, ticker="nvda")

    assert result["ticker"] == "NVDA"
    assert result["price_targets"]["price_target_mean"] == {
        "value": 185.0,
        "period": "2027-07E",
    }
    assert result["price_targets"]["num_analysts"] == {
        "value": 42.0,
        "period": "2027-07E",
    }


def test_projection_prompt_has_caveat():
    prompt = PromptAugmenter().build_prompt(
        question="What's the consensus outlook for NVDA next quarter?",
        intent={"ticker": "NVDA", "question_type": "projection"},
        retrieval={
            "facts": [
                {
                    "ticker": "NVDA",
                    "metric": "estimate_revenue_next_q",
                    "value": 30100000000.0,
                    "unit": "usd",
                    "period": "2026-Q3E",
                    "source_type": "estimates",
                }
            ],
            "documents": [],
            "ticker": "NVDA",
        },
    )

    assert "Analyst Consensus" in prompt
    assert "not guarantees" in prompt
    # The projection instruction keeps the "analyst estimates, not guarantees"
    # framing but no longer appends "not financial advice" (opinion/analysis work).
    assert "not financial advice" not in prompt
    assert "Retrieved Financial Facts" not in prompt


def test_projection_prompt_labels_num_analysts():
    prompt = PromptAugmenter().build_prompt(
        question="What's the price target for NVDA?",
        intent={"ticker": "NVDA", "question_type": "projection"},
        retrieval={
            "facts": [
                {
                    "ticker": "NVDA",
                    "metric": "num_analysts",
                    "value": 42.0,
                    "unit": "count",
                    "period": "2027-07E",
                    "source_type": "estimates",
                },
            ],
            "documents": [],
            "ticker": "NVDA",
        },
    )

    assert "Based on 42 analysts" in prompt


def test_no_fabricated_target_guard():
    from src.middleware.guardrails import apply_projection_guardrail, unsupported_figures

    answer = "The consensus target is $500."
    context = "## Analyst Consensus\n- Price target (mean): 185.00 usd (2027-07E)"

    caveated, flagged = apply_projection_guardrail(answer, context)

    assert unsupported_figures(answer, context) == ["$500"]
    assert flagged == ["$500"]
    assert caveated.startswith(answer)
    assert "could not be verified" in caveated

    supported, flagged = apply_projection_guardrail("The mean target is $185.", context)
    assert supported == "The mean target is $185."
    assert flagged == []


def test_guard_formatting_tolerance():
    from src.middleware.guardrails import unsupported_figures

    context = "Revenue estimate: 30,100,000,000.00 usd. Mean target: 185.0 usd. EPS: 4.1 usd."

    assert unsupported_figures("Revenue is $30.1B.", context) == []
    assert unsupported_figures("Target is $185.", context) == []
    assert unsupported_figures("EPS is $4.10.", context) == []


def test_guard_rejects_scale_mismatch_and_unmarked_numbers():
    from src.middleware.guardrails import unsupported_figures

    # A bare context number must not vouch for a figure a thousand/billion
    # times larger — "$185B" is not supported by a 185.00 usd price target.
    context = "- Price target (mean): 185.00 usd (2027-07E)"
    assert unsupported_figures("The target is $185B.", context) == ["$185B"]

    # Numbers without a currency/magnitude marker (analyst counts, years,
    # scores) do not support dollar figures.
    context = "Based on 42 analysts. Analyst rating: 1.90 score."
    assert unsupported_figures("The target is $42.", context) == ["$42"]


def test_estimates_refresh_not_marked_fresh_on_failure(monkeypatch):
    from src.middleware import app as app_module

    fake_store = MagicMock()
    fake_store._schedule_ttls.return_value = {"estimates": 24}
    monkeypatch.setattr(app_module, "store", fake_store)

    failed = {"ticker": "NVDA", "status": "error", "facts_stored": 0, "errors": ["boom"]}
    with patch(
        "src.macros.estimates_ingestor.EstimatesIngestor.fetch_for_ticker",
        return_value=failed,
    ):
        with pytest.raises(RuntimeError):
            app_module._refresh_one_source_direct("NVDA", "estimates")
    fake_store.mark_source_fresh.assert_not_called()

    ok = {"ticker": "NVDA", "status": "success", "facts_stored": 13, "errors": []}
    with patch(
        "src.macros.estimates_ingestor.EstimatesIngestor.fetch_for_ticker",
        return_value=ok,
    ):
        app_module._refresh_one_source_direct("NVDA", "estimates")
    fake_store.mark_source_fresh.assert_called_once()


def test_retriever_projection_strategy(store):
    retriever = Retriever(store=store)

    assert retriever._select_strategy("projection", "NVDA", []) == "hybrid"
    assert retriever._select_strategy("projection", None, []) == "broad"


def test_projection_uses_analysis_params(store, monkeypatch):
    from src.middleware import app as middleware_app
    from src.middleware.models import QueryRequest

    monkeypatch.setattr(middleware_app, "store", store)
    monkeypatch.setattr(middleware_app, "config", MagicMock())
    middleware_app.config.top_k_documents = 5
    middleware_app.config.top_k_facts = 10
    middleware_app.config.default_temperature = 0.7
    middleware_app.config.max_tokens = 512
    middleware_app.config.enable_lexical = False
    middleware_app.config.enable_reranker = False
    middleware_app.config.enable_fetch_on_miss = False  # keep this test offline
    monkeypatch.setattr(middleware_app, "_task_params", lambda task: {"temperature": 0.5, "max_tokens": 2048})
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))

    captured = {}

    async def fake_call_model(prompt, temperature, max_tokens):
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return "The target is $185.", []

    monkeypatch.setattr(middleware_app, "_call_model", fake_call_model)

    response = asyncio.run(
        middleware_app.query(QueryRequest(question="What's the consensus outlook for NVDA next quarter?"))
    )

    assert response.detected_intent == "projection"
    assert captured == {"temperature": 0.5, "max_tokens": 2048}
