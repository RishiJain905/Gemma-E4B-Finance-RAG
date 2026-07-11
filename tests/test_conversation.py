"""
tests/test_conversation.py
Offline tests for bounded client-owned history selection (2.2.2.1).

Covers src/middleware/conversation.select_history and the ChatTurn/QueryRequest
validation contract. No FastAPI/Chroma/model startup — pure model + helper.
"""

import pytest
from pydantic import ValidationError

from src.middleware.conversation import HistorySelection, select_history
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
