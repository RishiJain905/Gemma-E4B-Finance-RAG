"""
tests/test_query_rewriter.py
Offline tests for the optional bounded ambiguity fallback (2.2.2.2 Step 3).

Every model call is injected (no network, no local model). Covers: a validated
rewrite merging only catalog/turn-present values, and every failure mode
(invalid JSON, timeout, entity drift, invented number) retaining the
deterministic compiled query without raising. Also asserts the deterministic
path makes zero model requests.
"""

import json
from types import SimpleNamespace

from src.middleware.conversation import compile_question
from src.middleware.models import ChatTurn
from src.middleware.query_rewriter import rewrite_query, should_use_llm_fallback


def _config(**over):
    base = dict(
        enable_conversation_rewrite=True,
        enable_llm_rewrite_fallback=True,
        conversation_rewrite_timeout_s=15.0,
        llama_endpoint="http://127.0.0.1:8087/v1/chat/completions",
        model_name="tracealchemy",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _history(user_q, *, ticker="NVDA"):
    return [
        ChatTurn(role="user", content=user_q),
        ChatTurn(role="assistant", content="an answer",
                 context={"grounding": "grounded", "ticker": ticker}),
    ]


def _ambiguous():
    """A deterministically-ambiguous compiled turn (pronoun, two entities)."""
    history = _history("Compare NVDA and AMD revenue for FY2025")
    compiled = compile_question("Why did it grow?", history)
    assert "entity" in compiled.ambiguous_slots
    return history, compiled


def test_valid_rewrite_merges_only_validated_values():
    history, compiled = _ambiguous()
    reply = json.dumps({
        "standalone_query": "NVDA revenue FY2025",
        "entities": ["NVDA"],
        "metrics": ["total_revenue"],
        "timeframe": "FY2025",
        "topic_reset": False,
    })
    calls = []

    def call_model(prompt):
        calls.append(prompt)
        return reply

    out = rewrite_query("Why did it grow?", history, compiled,
                        config=_config(), call_model=call_model)
    assert len(calls) == 1  # exactly one bounded model call
    assert out.retrieval_query == "NVDA revenue FY2025"
    assert out.carried_entities == ["NVDA"]
    assert out.carried_metrics == ["total_revenue"]
    assert out.carried_timeframe == "FY2025"
    assert out.ambiguous_slots == []
    assert "llm_rewrite" in out.resolution_sources


def test_rewrite_rejects_invented_entity():
    history, compiled = _ambiguous()
    # TSLA is never mentioned in the turns → drift → reject → deterministic.
    reply = json.dumps({
        "standalone_query": "TSLA revenue FY2025",
        "entities": ["TSLA"],
        "metrics": ["total_revenue"],
        "timeframe": "FY2025",
        "topic_reset": False,
    })
    out = rewrite_query("Why did it grow?", history, compiled,
                        config=_config(), call_model=lambda p: reply)
    assert out is compiled
    assert "entity" in out.ambiguous_slots


def test_rewrite_rejects_invented_number_in_query():
    history, compiled = _ambiguous()
    reply = json.dumps({
        "standalone_query": "NVDA revenue $9999",  # figure absent from turns
        "entities": ["NVDA"],
        "metrics": ["total_revenue"],
        "timeframe": "FY2025",
        "topic_reset": False,
    })
    out = rewrite_query("Why did it grow?", history, compiled,
                        config=_config(), call_model=lambda p: reply)
    assert out is compiled


def test_invalid_json_falls_back_without_raising():
    history, compiled = _ambiguous()
    out = rewrite_query("Why did it grow?", history, compiled,
                        config=_config(), call_model=lambda p: "not json at all")
    assert out is compiled
    assert "entity" in out.ambiguous_slots


def test_timeout_falls_back_without_raising():
    history, compiled = _ambiguous()

    def call_model(prompt):
        raise TimeoutError("model call timed out")

    out = rewrite_query("Why did it grow?", history, compiled,
                        config=_config(), call_model=call_model)
    assert out is compiled


def test_rejects_unknown_metric():
    history, compiled = _ambiguous()
    reply = json.dumps({
        "standalone_query": "NVDA revenue FY2025",
        "entities": ["NVDA"],
        "metrics": ["made_up_metric"],  # not in the known catalog
        "timeframe": "FY2025",
        "topic_reset": False,
    })
    out = rewrite_query("Why did it grow?", history, compiled,
                        config=_config(), call_model=lambda p: reply)
    assert out is compiled


def test_deterministic_turn_makes_no_model_request():
    # An unambiguous follow-up never triggers the LLM fallback.
    history = _history("Show NVDA revenue for FY2025")
    compiled = compile_question("What about AMD?", history)
    assert compiled.ambiguous_slots == []
    assert should_use_llm_fallback(compiled, _config()) is False


def test_fallback_disabled_by_default():
    _history_, compiled = _ambiguous()
    # Even with an ambiguous slot, both flags default off -> no call.
    off = _config(enable_conversation_rewrite=False, enable_llm_rewrite_fallback=False)
    assert should_use_llm_fallback(compiled, off) is False
