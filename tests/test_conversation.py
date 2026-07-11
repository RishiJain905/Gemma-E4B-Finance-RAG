"""
tests/test_conversation.py
Offline tests for bounded client-owned history selection (2.2.2.1) and
deterministic follow-up rewriting / entity carryover (2.2.2.2).

Covers src/middleware/conversation.select_history, ConversationState,
compile_question, and the ChatTurn/QueryRequest validation contract. No
FastAPI/Chroma/model startup — pure model + helpers, no network.
"""

import pytest
from pydantic import ValidationError

from src.middleware.conversation import (
    CompiledQuestion,
    ConversationState,
    HistorySelection,
    compile_question,
    select_history,
)
from src.middleware.models import ChatTurn, QueryRequest


def _turns(*pairs):
    """Build a flat user/assistant turn list from (role, content) pairs."""
    return [ChatTurn(role=r, content=c) for r, c in pairs]


def test_history_keeps_complete_recent_turns_within_both_limits():
    turns = _turns(
        ("user", "oldest q"),
        ("assistant", "oldest a"),
        ("user", "middle q"),
        ("assistant", "middle a"),
        ("user", "newest q"),
        ("assistant", "newest a"),
    )

    # Turn budget: keep only the two most recent turns, in chronological order.
    sel = select_history(turns, max_turns=2, max_chars=10_000)
    assert isinstance(sel, HistorySelection)
    assert [t.content for t in sel.turns] == ["newest q", "newest a"]
    assert sel.turns_received == 6
    assert sel.turns_used == 2
    assert sel.truncated is True
    assert sel.topic_reset is False

    # Char budget: each content is 8 chars; a 20-char budget fits exactly two
    # whole turns (16 chars); the third would exceed it and is dropped whole.
    sel2 = select_history(turns, max_turns=8, max_chars=20)
    assert [t.content for t in sel2.turns] == ["newest q", "newest a"]
    assert sel2.truncated is True


def test_full_history_within_budget_is_not_truncated():
    turns = _turns(("user", "q1"), ("assistant", "a1"))
    sel = select_history(turns, max_turns=8, max_chars=8000)
    assert [t.content for t in sel.turns] == ["q1", "a1"]
    assert sel.turns_used == sel.turns_received == 2
    assert sel.truncated is False


def test_empty_history_selects_nothing():
    sel = select_history([], max_turns=8, max_chars=8000)
    assert sel.turns == []
    assert sel.turns_received == 0
    assert sel.turns_used == 0
    assert sel.truncated is False


def test_metadata_block_shape():
    sel = select_history(_turns(("user", "q"), ("assistant", "a")),
                         max_turns=1, max_chars=8000)
    assert sel.as_metadata() == {
        "history_turns_received": 2,
        "history_turns_used": 1,
        "history_truncated": True,
        "topic_reset": False,
    }


def test_structured_context_retained_only_on_assistant_turns():
    ctx = {"ticker": "NVDA", "intent": "fact_lookup"}
    turns = [
        ChatTurn(role="user", content="q", context={"ticker": "AMD"}),
        ChatTurn(role="assistant", content="a", context=ctx),
    ]
    sel = select_history(turns, max_turns=8, max_chars=8000)
    user_turn, assistant_turn = sel.turns
    # Context on a user turn does not belong to a response — it is dropped.
    assert user_turn.context is None
    # Assistant context is preserved for follow-up rewriting (2.2.2.2).
    assert assistant_turn.context == ctx


def test_empty_assistant_context_is_dropped():
    turns = [ChatTurn(role="assistant", content="a", context={})]
    sel = select_history(turns, max_turns=8, max_chars=8000)
    assert sel.turns[0].context is None


def test_invalid_role_or_blank_turn_is_rejected():
    # Malformed role -> validation error (surfaces as HTTP 422 at the endpoint).
    with pytest.raises(ValidationError):
        ChatTurn(role="system", content="hi")

    # Blank / whitespace-only content -> validation error.
    with pytest.raises(ValidationError):
        ChatTurn(role="user", content="   ")
    with pytest.raises(ValidationError):
        ChatTurn(role="assistant", content="")

    # A QueryRequest carrying a malformed turn is rejected as a whole.
    with pytest.raises(ValidationError):
        QueryRequest(question="q", history=[{"role": "bogus", "content": "x"}])
    with pytest.raises(ValidationError):
        QueryRequest(question="q", history=[{"role": "user", "content": " "}])


def test_selection_does_not_mutate_input_turns():
    ctx = {"ticker": "NVDA"}
    turns = [ChatTurn(role="assistant", content="a", context=ctx)]
    select_history(turns, max_turns=8, max_chars=8000)
    # model_copy is used for sanitization; the caller's turns are untouched.
    assert turns[0].context == ctx


# ── Follow-up rewriting & entity carryover (2.2.2.2) ───────────────────────

def _history(user_q, *, ticker="NVDA", grounding="grounded",
             timeframe=None, intent=None):
    """A one-round (user, grounded-assistant) history for carryover tests."""
    ctx = {"grounding": grounding}
    if ticker:
        ctx["ticker"] = ticker
    if timeframe:
        ctx["timeframe"] = timeframe
    if intent:
        ctx["intent"] = intent
    return [
        ChatTurn(role="user", content=user_q),
        ChatTurn(role="assistant", content="an answer", context=ctx),
    ]


def test_state_from_grounded_turn_recovers_slots():
    state = ConversationState.from_history(
        _history("Show NVDA revenue for FY2025"))
    assert state.active_tickers == ["NVDA"]
    assert "total_revenue" in state.active_metrics
    assert state.active_timeframe == "fy2025"
    assert state.primary_entity == "NVDA"


def test_state_ignores_ungrounded_prior_answer():
    # A refused prior answer is not authoritative evidence — no active slots.
    state = ConversationState.from_history(
        _history("Show NVDA revenue for FY2025", grounding="refused"))
    assert state.active_tickers == []
    assert state.active_metrics == []
    assert state.primary_entity is None


def test_entity_substitution_what_about():
    history = _history("Show NVDA revenue for FY2025")
    compiled = compile_question("What about AMD?", history)
    assert isinstance(compiled, CompiledQuestion)
    assert compiled.entity == "AMD"
    assert compiled.carried_metrics == ["total_revenue"]
    assert compiled.carried_timeframe == "fy2025"
    assert compiled.topic_reset is False
    # Compact standalone query from validated slots + untouched turn.
    assert compiled.retrieval_query == "AMD total revenue FY2025"


def test_timeframe_only_carry_same_period():
    history = _history("Show NVDA gross margin for FY2025")
    compiled = compile_question("same period", history)
    assert compiled.carried_timeframe == "fy2025"
    # "same period" carries only the named slot — not the metric.
    assert compiled.carried_metrics == []
    assert compiled.metrics == []


def test_metric_only_carry_same_metric():
    history = _history("Show NVDA revenue for FY2025")
    compiled = compile_question("same metric", history)
    assert compiled.carried_metrics == ["total_revenue"]
    # "same metric" carries only the named slot — not the timeframe.
    assert compiled.carried_timeframe is None


def test_pronoun_carries_single_active_entity():
    history = _history("Show NVDA revenue for FY2025")
    compiled = compile_question("Why did it grow?", history)
    assert compiled.carried_entities == ["NVDA"]
    assert compiled.entity == "NVDA"
    assert "entity" not in compiled.ambiguous_slots


def test_pronoun_refused_with_multiple_active_entities():
    history = _history("Compare NVDA and AMD revenue")
    compiled = compile_question("Why did it grow?", history)
    assert compiled.carried_entities == []
    assert compiled.entity is None
    assert "entity" in compiled.ambiguous_slots


def test_topic_shift_clears_incompatible_slots():
    history = _history("Show NVDA revenue for FY2025")
    compiled = compile_question("What are Apple's main risk factors?", history)
    assert compiled.entity == "AAPL"
    assert compiled.topic_reset is True
    assert compiled.carried_metrics == []
    assert compiled.carried_timeframe is None


def test_ticker_override_outranks_history():
    history = _history("Show NVDA revenue for FY2025")
    compiled = compile_question("What is the revenue?", history,
                                override_ticker="AMD")
    assert compiled.entity == "AMD"
    assert "override" in compiled.resolution_sources
    assert "NVDA" not in (compiled.carried_entities + [compiled.entity])


def test_raw_question_is_unchanged():
    raw = "What about AMD?"
    compiled = compile_question(raw, _history("Show NVDA revenue for FY2025"))
    assert compiled.raw_question == raw


def test_no_history_is_single_turn_query():
    compiled = compile_question("What is NVDA revenue for FY2025?", [])
    # No carry, no reset — the retrieval query is built from the current turn.
    assert compiled.carried_entities == []
    assert compiled.carried_metrics == []
    assert compiled.carried_timeframe is None
    assert compiled.topic_reset is False
    assert "NVDA" in compiled.retrieval_query
