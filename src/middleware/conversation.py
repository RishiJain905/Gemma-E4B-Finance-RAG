"""
src/middleware/conversation.py
Deterministic, dependency-free bounded-history selection (2.2.2.1).

The middleware is stateless: ``scripts/chat.py`` owns each session's turns and
sends them with every request. This module picks which of those turns fit one
shared conversation budget before generation. It never summarizes history and
never truncates a turn's content — a turn is kept whole or dropped whole so a
finance entity or number can never be silently rewritten (see
docs/phase2.2/ARCHITECTURE-DECISION.md "Query contract").

Selection is newest-first (recency wins under budget pressure) but the returned
turns are restored to chronological order so downstream prompt assembly reads
oldest → newest. Structured ``context`` (from a prior response) is retained
only on assistant turns; anything else is dropped so follow-up rewriting
(2.2.2.2) never consumes context that does not belong to a real answer.
"""

from dataclasses import dataclass, field
from typing import Optional

from .models import ChatTurn

# Defaults mirror configs/middleware.yaml. Callers pass config-resolved values;
# these keep the helper usable standalone (tests, offline tooling).
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_CHARS = 8000


@dataclass
class HistorySelection:
    """Result of :func:`select_history`.

    ``turns`` is chronological and context-sanitized. The counts feed the
    response's ``conversation`` metadata block so a client can see exactly how
    much of what it sent was used.
    """

    turns: list[ChatTurn] = field(default_factory=list)
    turns_received: int = 0
    turns_used: int = 0
    truncated: bool = False
    topic_reset: bool = False

    def as_metadata(self) -> dict:
        """Render the QueryResponse.conversation metadata block (2.2.2.1)."""
        return {
            "history_turns_received": self.turns_received,
            "history_turns_used": self.turns_used,
            "history_truncated": self.truncated,
            "topic_reset": self.topic_reset,
        }


def _validated_context(turn: ChatTurn) -> Optional[dict]:
    """Return the turn's structured context only when it belongs to it.

    Context is carried from a prior *response*, so it is meaningful only on an
    assistant turn and only when it is a non-empty dict. Everything else is
    dropped (returned as ``None``) rather than passed through unchecked.
    """
    if turn.role != "assistant":
        return None
    ctx = turn.context
    if isinstance(ctx, dict) and ctx:
        return ctx
    return None


def _sanitized_turn(turn: ChatTurn) -> ChatTurn:
    """Return a copy of the turn with its context validated against it."""
    return turn.model_copy(update={"context": _validated_context(turn)})


def select_history(
    turns: list[ChatTurn],
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> HistorySelection:
    """Select the most recent complete turns that fit both budgets.

    Args:
        turns:     The client-sent history (already ChatTurn-validated by the
                   request model, which 422s on malformed roles/blank content).
        max_turns: Maximum number of turns to keep.
        max_chars: Maximum total ``content`` characters across kept turns.

    A turn is kept only if adding it exceeds neither budget; once a budget
    would be exceeded, selection stops (older turns are dropped, never split).
    Returns a :class:`HistorySelection` with chronological turns and the
    received/used/truncated counts for response metadata.
    """
    received = list(turns or [])
    max_turns = max(0, int(max_turns))
    max_chars = max(0, int(max_chars))

    kept_reversed: list[ChatTurn] = []
    used_chars = 0
    for turn in reversed(received):
        if len(kept_reversed) >= max_turns:
            break
        turn_chars = len(turn.content or "")
        if used_chars + turn_chars > max_chars:
            break
        kept_reversed.append(_sanitized_turn(turn))
        used_chars += turn_chars

    kept = list(reversed(kept_reversed))
    return HistorySelection(
        turns=kept,
        turns_received=len(received),
        turns_used=len(kept),
        truncated=len(kept) < len(received),
        topic_reset=False,  # topic-shift detection arrives in 2.2.2.2
    )
