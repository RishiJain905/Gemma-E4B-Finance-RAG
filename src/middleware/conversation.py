"""
src/middleware/conversation.py
Deterministic bounded-history selection (2.2.2.1) and follow-up rewriting
(2.2.2.2).

The middleware is stateless: ``scripts/chat.py`` owns each session's turns and
sends them with every request. ``select_history`` picks which of those turns
fit one shared conversation budget before generation; it never summarizes
history and never truncates a turn's content — a turn is kept whole or dropped
whole so a finance entity or number can never be silently rewritten (see
docs/phase2.2/ARCHITECTURE-DECISION.md "Query contract").

Selection is newest-first (recency wins under budget pressure) but the returned
turns are restored to chronological order so downstream prompt assembly reads
oldest → newest. Structured ``context`` (from a prior response) is retained
only on assistant turns; anything else is dropped so follow-up rewriting never
consumes context that does not belong to a real answer.

``ConversationState``/``compile_question`` (2.2.2.2) then compile the current
turn plus that bounded history into a *separate* standalone retrieval query
while leaving the raw question byte-for-byte unchanged. Carryover is
deterministic and conservative: explicit current-turn values always win, only
missing compatible slots are carried, and an explicit topic shift clears
incompatible state. Assistant prose is never treated as authoritative evidence
for a numeric value — only structured prior-turn context and values the
existing ``IntentParser``/symbol resolver can validate are ever recovered.
"""

import re
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
        topic_reset=False,  # populated by compile_question (2.2.2.2) when it runs
    )


# ── Follow-up rewriting & entity carryover (2.2.2.2) ───────────────────────

# Grounding modes whose prior answer may contribute carried facts. A refused
# or general (not-from-data) answer is not authoritative evidence, so it never
# seeds the active entity/metric/timeframe state.
_GROUNDED_MODES = ("grounded", "partial")

# Referential cues. These are deliberately conservative — a slot is carried
# only when the current turn signals a genuine follow-up, never merely because
# prior context exists.
_PRONOUN_RE = re.compile(
    r"\b(it|its|it's|they|them|their|theirs|that|this|those|these"
    r"|the company|the firm|the stock)\b",
    re.IGNORECASE,
)
_SAME_METRIC_RE = re.compile(r"\bsame\s+(metric|number|figure|measure|ratio)\b", re.IGNORECASE)
_SAME_PERIOD_RE = re.compile(
    r"\bsame\s+(period|timeframe|time\s+frame|quarter|year|fiscal\s+year)\b",
    re.IGNORECASE,
)
_SUB_RE = re.compile(r"\b(what|how)\s+about\b|\bwhat\s+if\b|\binstead\b", re.IGNORECASE)
_FOLLOWUP_RE = re.compile(r"^\s*(and|also|then|so|what about|how about)\b", re.IGNORECASE)
_INVENTORY_FOLLOWUP_RE = re.compile(
    r"\b(?:which|what)\s+of\s+(?:those|these)\b|\b(?:those|these)\s+(?:also|that)\b"
    r"|\b(?:that|the)\s+(?:previously\s+)?returned\s+set\b",
    re.IGNORECASE,
)
MAX_CARRIED_INVENTORY = 3

# Entity resolutions the current turn owns outright (a real name/symbol match),
# as opposed to the low-confidence "fallback" (any stray uppercase word).
_EXPLICIT_TICKER_SOURCES = ("local_map", "known_ticker", "catalog_exact", "fuzzy", "override")

# Words dropped when building the residual retrieval terms from the raw turn —
# referential/filler tokens that carry no finance content.
_FILLER = {
    "what", "about", "how", "and", "also", "then", "so", "the", "a", "an",
    "for", "of", "to", "in", "on", "is", "was", "are", "were", "do", "does",
    "did", "me", "give", "show", "tell", "please", "same", "it", "its", "that",
    "this", "they", "them", "their", "those", "these", "company", "firm",
    "stock", "much", "many", "s",
}

# Readable phrases for canonical metric names when rendering the retrieval
# query. Anything not listed falls back to a spaces-for-underscores rendering.
_METRIC_PHRASES = {
    "total_revenue": "total revenue",
    "gross_profit": "gross profit",
    "gross_margin_pct": "gross margin",
    "net_income": "net income",
    "eps_diluted": "EPS",
    "operating_income": "operating income",
    "cost_of_revenue": "cost of revenue",
    "research_development": "R&D",
    "free_cash_flow": "free cash flow",
    "operating_cash_flow": "operating cash flow",
    "operating_margin_pct": "operating margin",
    "net_margin_pct": "net margin",
    "pe_ratio": "P/E ratio",
    "forward_pe": "forward P/E",
    "market_cap": "market cap",
    "dividend_yield": "dividend yield",
    "price_target_mean": "price target",
}

# Forward-looking / estimate metrics the intent parser does not enumerate but
# the retriever/store use. Included in the catalog so a carried or rewritten
# estimate metric still validates.
_ESTIMATE_METRICS = frozenset({
    "estimate_revenue_current_q", "estimate_revenue_next_q",
    "estimate_revenue_current_y", "estimate_revenue_next_y",
    "estimate_eps_current_q", "estimate_eps_next_q",
    "estimate_eps_current_y", "estimate_eps_next_y",
    "price_target_mean", "price_target_high", "price_target_low",
    "num_analysts", "recommendation_mean",
})


def _build_known_metrics() -> frozenset[str]:
    """Canonical metric catalog used to validate recovered/rewritten metrics.

    Sourced from the same ``IntentParser`` patterns that produce a fact's
    metric name, so a carried metric can only ever be one the retriever can
    actually look up. Imported locally to keep this module's import light.
    """
    from .intent_parser import IntentParser

    base = {metric for _, metric in IntentParser.METRIC_PATTERNS}
    return frozenset(base | _ESTIMATE_METRICS)


KNOWN_METRICS = _build_known_metrics()


@dataclass
class ConversationState:
    """Deterministic view of what the conversation is currently 'about'.

    Built from the most recent *grounded* prior answer (2.2.2.2 Step 1):
    structured assistant ``context`` first, then values the intent parser /
    symbol resolver can recover from that turn's user question. Stale or
    ungrounded prior answers contribute nothing — they never seed a numeric
    slot. ``topic_id``/``resolution_sources`` are observability only.
    """

    active_tickers: list[str] = field(default_factory=list)
    active_metrics: list[str] = field(default_factory=list)
    active_timeframe: Optional[str] = None
    last_intent: Optional[str] = None
    last_question: Optional[str] = None
    last_grounding: Optional[str] = None
    topic_id: Optional[str] = None
    resolution_sources: list[str] = field(default_factory=list)
    active_inventory: list[str] = field(default_factory=list)
    inventory_total: int = 0

    @property
    def primary_entity(self) -> Optional[str]:
        """The single most salient active entity, or ``None``."""
        return self.active_tickers[0] if self.active_tickers else None

    @classmethod
    def from_history(
        cls,
        turns: Optional[list[ChatTurn]],
        *,
        parser=None,
    ) -> "ConversationState":
        """Build state from the most recent grounded (user, assistant) pair.

        ``turns`` is the chronological, context-sanitized selection from
        :func:`select_history`. The newest grounded answer wins; ungrounded or
        refused answers are skipped (they contribute no facts). Returns an empty
        state when no grounded prior answer exists.
        """
        pairs: list[tuple[str, dict]] = []
        last_user = ""
        for turn in turns or []:
            role = getattr(turn, "role", None)
            content = getattr(turn, "content", "") or ""
            if role == "user":
                last_user = content
            elif role == "assistant":
                ctx = getattr(turn, "context", None)
                pairs.append((last_user, ctx if isinstance(ctx, dict) else {}))
        for user_q, ctx in reversed(pairs):
            if _ctx_is_grounded(ctx):
                return cls._from_pair(user_q, ctx, parser=parser)
        return cls()

    @classmethod
    def _from_pair(cls, user_q: str, ctx: dict, *, parser=None) -> "ConversationState":
        """Assemble state from one grounded turn's question + structured context."""
        parsed = _parse(user_q, parser) if user_q else {}

        entities = _extract_entities(user_q)
        for candidate in (ctx.get("ticker"), parsed.get("ticker")):
            norm = _norm_ticker(candidate)
            if norm and norm not in entities:
                entities.append(norm)

        metrics = [str(m) for m in (parsed.get("metrics") or [])]
        timeframe = ctx.get("timeframe") or parsed.get("timeframe")
        intent = ctx.get("intent") or parsed.get("question_type")
        coverage = ctx.get("coverage_metadata")
        inventory: list[str] = []
        inventory_total = 0
        if isinstance(coverage, dict) and coverage.get("complete") is True:
            raw_inventory = coverage.get("securities") or []
            for raw in raw_inventory:
                candidate = raw.get("ticker") if isinstance(raw, dict) else raw
                norm = _norm_ticker(candidate)
                if norm and norm not in inventory:
                    inventory.append(norm)
            inventory_total = int(
                coverage.get("total_matching") or len(inventory)
            )
            if inventory_total > MAX_CARRIED_INVENTORY:
                inventory = []

        return cls(
            active_tickers=entities,
            active_metrics=metrics,
            active_timeframe=str(timeframe) if timeframe else None,
            last_intent=str(intent) if intent else None,
            last_question=user_q or None,
            last_grounding=str(ctx.get("grounding")) if ctx.get("grounding") else None,
            topic_id=(entities[0] if entities else None),
            resolution_sources=["history"],
            active_inventory=inventory,
            inventory_total=inventory_total,
        )


@dataclass
class CompiledQuestion:
    """The compiled current turn (2.2.2.2 Step 2).

    ``raw_question`` is byte-for-byte the user's turn (prompt/answer display).
    ``retrieval_query`` is a *separate* standalone search string built from
    validated slots and the untouched current turn — retrieval input only, never
    shown as though it were the user's wording. ``entity``/``metrics``/
    ``timeframe`` are the effective resolved slots (current-turn value, else a
    carried one); the ``carried_*`` fields record only what was pulled from
    history so a client/evaluator can see exactly what carried.
    """

    raw_question: str
    retrieval_query: str
    carried_entities: list[str] = field(default_factory=list)
    carried_metrics: list[str] = field(default_factory=list)
    carried_timeframe: Optional[str] = None
    topic_reset: bool = False
    ambiguous_slots: list[str] = field(default_factory=list)
    resolution_sources: list[str] = field(default_factory=list)
    # Effective resolved slots (not just the carried subset) — used to build the
    # retrieval intent and response metadata.
    entity: Optional[str] = None
    metrics: list[str] = field(default_factory=list)
    timeframe: Optional[str] = None
    inventory_scope: list[str] = field(default_factory=list)
    inventory_size: int = 0

    def as_metadata(self) -> dict:
        """Render the optional response ``carried_context`` block."""
        return {
            "entities": list(self.carried_entities),
            "metrics": list(self.carried_metrics),
            "timeframe": self.carried_timeframe,
            "topic_reset": self.topic_reset,
            "ambiguous_slots": list(self.ambiguous_slots),
            "resolution_sources": list(self.resolution_sources),
            "inventory_scope": list(self.inventory_scope),
            "inventory_size": self.inventory_size,
        }


def compile_question(
    raw_question: str,
    selected_history: Optional[list[ChatTurn]],
    override_ticker: Optional[str] = None,
    *,
    parser=None,
) -> CompiledQuestion:
    """Compile a raw turn + bounded history into a standalone retrieval query.

    Deterministic and dependency-free (no model call). Carryover rules
    (2.2.2.2 Step 1): explicit current-turn values always win; a ``/ticker``
    override outranks history; "what about X" replaces the primary entity and
    carries the compatible metric/period; "same period/metric" carries only the
    named slot; a pronoun carries an entity only when exactly one active entity
    exists; an explicit new topic clears incompatible carried slots.
    """
    parser = parser or _default_parser()
    current = _parse(raw_question, parser, override_ticker=override_ticker)
    state = ConversationState.from_history(selected_history, parser=parser)

    raw = raw_question or ""
    lower = raw.lower()
    override_norm = _norm_ticker(override_ticker)

    has_pronoun = bool(_PRONOUN_RE.search(raw))
    inventory_followup = bool(_INVENTORY_FOLLOWUP_RE.search(raw))
    same_metric = bool(_SAME_METRIC_RE.search(lower))
    same_period = bool(_SAME_PERIOD_RE.search(lower))
    current_entity_explicit = bool(override_norm) or _is_explicit_entity(current)
    # "what about X" — or a bare "AMD?" / "and AMD" — is an entity substitution
    # that still carries compatible slots, never a topic shift.
    is_sub = bool(_SUB_RE.search(lower)) or (
        _is_explicit_entity(current) and not override_norm and _word_count(raw) <= 3
    )

    resolution_sources: list[str] = []
    carried_entities: list[str] = []
    carried_metrics: list[str] = []
    carried_timeframe: Optional[str] = None
    ambiguous: list[str] = []
    topic_reset = False
    inventory_scope: list[str] = []

    # ── Entity ──
    entity: Optional[str] = None
    if inventory_followup and state.active_inventory:
        inventory_scope = list(state.active_inventory)
        carried_entities = list(inventory_scope)
        resolution_sources.append("history_catalog")
    elif override_norm:
        entity = override_norm
        resolution_sources.append("override")
    elif current_entity_explicit:
        entity = _norm_ticker(current.get("ticker"))
        resolution_sources.append("current_turn")
        if state.primary_entity and entity and entity != state.primary_entity and not is_sub:
            topic_reset = True  # explicit new entity, not a substitution → new topic
    else:
        referential = (
            has_pronoun or is_sub or same_metric or same_period
            or bool(_FOLLOWUP_RE.match(lower))
        )
        if referential and state.active_tickers:
            if len(state.active_tickers) == 1:
                entity = state.active_tickers[0]
                carried_entities = [entity]
                resolution_sources.append("history")
            else:
                ambiguous.append("entity")  # pronoun with >1 active entity is unsafe

    if inventory_followup and not inventory_scope:
        ambiguous.append("universe_scope")

    # ── Which slots may carry ──
    narrow_metric = same_metric and not same_period and not is_sub
    narrow_timeframe = same_period and not same_metric and not is_sub
    if topic_reset:
        want_metric = want_timeframe = False
    elif narrow_metric:
        want_metric, want_timeframe = True, False
    elif narrow_timeframe:
        want_metric, want_timeframe = False, True
    else:
        followup = bool(carried_entities) or is_sub or has_pronoun or (same_metric and same_period)
        want_metric = want_timeframe = followup

    # ── Metrics ── (explicit current wins)
    if current.get("metrics"):
        metrics = list(current["metrics"])
    else:
        metrics = []
        if want_metric and state.active_metrics:
            metrics = list(state.active_metrics)
            carried_metrics = list(metrics)
            _append_unique(resolution_sources, "history")
        elif narrow_metric and not state.active_metrics:
            ambiguous.append("metric")

    # ── Timeframe ── (explicit current wins)
    if current.get("timeframe"):
        timeframe = str(current["timeframe"])
    else:
        timeframe = None
        if want_timeframe and state.active_timeframe:
            timeframe = state.active_timeframe
            carried_timeframe = timeframe
            _append_unique(resolution_sources, "history")
        elif narrow_timeframe and not state.active_timeframe:
            ambiguous.append("timeframe")

    retrieval_query = _render_query(entity, metrics, timeframe, raw)
    if inventory_scope:
        # Preserve the referential inventory semantics after filler-word
        # removal so the rule-based intent compiler sees a catalog operation,
        # while the explicit bounded ticker set fixes the universe.
        retrieval_query = " ".join(
            [*inventory_scope, "the previously returned set", retrieval_query]
        ).strip()

    return CompiledQuestion(
        raw_question=raw,
        retrieval_query=retrieval_query,
        carried_entities=carried_entities,
        carried_metrics=carried_metrics,
        carried_timeframe=carried_timeframe,
        topic_reset=topic_reset,
        ambiguous_slots=list(dict.fromkeys(ambiguous)),
        resolution_sources=list(dict.fromkeys(resolution_sources)),
        entity=entity,
        metrics=metrics,
        timeframe=timeframe,
        inventory_scope=inventory_scope,
        inventory_size=state.inventory_total,
    )


# ── Internal helpers ───────────────────────────────────────────────────────

def _default_parser():
    from .intent_parser import IntentParser

    return IntentParser()


def _parse(question: str, parser, *, override_ticker: Optional[str] = None) -> dict:
    """Parse a question with the intent parser, tolerating a missing parser."""
    p = parser or _default_parser()
    return p.parse(question, override_ticker=override_ticker)


def _ctx_is_grounded(ctx: dict) -> bool:
    """True only when structured context confirms a grounded/partial answer."""
    if not isinstance(ctx, dict) or not ctx:
        return False
    grounding = str(ctx.get("grounding") or "").strip().lower()
    return grounding in _GROUNDED_MODES


def _extract_entities(text: str) -> list[str]:
    """Ordered, deduped tickers named in ``text`` (company names + symbols).

    Deterministic and offline — uses only the intent parser's hardcoded maps so
    a comparison turn ("Compare NVDA and AMD") yields both entities and a bare
    company name resolves. Ordered by first appearance so ``primary_entity`` is
    the leading mention.
    """
    from .intent_parser import IntentParser

    if not text:
        return []
    normalized = text.lower()
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for name, ticker in IntentParser.COMPANY_TO_TICKER.items():
        idx = normalized.find(name)
        if idx != -1 and ticker not in seen:
            found.append((idx, ticker))
            seen.add(ticker)
    for match in re.finditer(r"\b[A-Z]{1,5}\b", text):
        sym = match.group(0)
        if sym in IntentParser.KNOWN_TICKERS and sym not in seen:
            found.append((match.start(), sym))
            seen.add(sym)
    found.sort(key=lambda item: item[0])
    return [ticker for _, ticker in found]


def _is_explicit_entity(intent: dict) -> bool:
    """True when the current turn names its own entity (not a weak fallback)."""
    return bool(intent.get("ticker")) and intent.get("ticker_source") in _EXPLICIT_TICKER_SOURCES


def _norm_ticker(value) -> Optional[str]:
    if value is None:
        return None
    norm = str(value).strip().upper()
    return norm or None


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _metric_phrase(metric: str) -> str:
    return _METRIC_PHRASES.get(metric, str(metric).replace("_", " "))


def _display_timeframe(timeframe: str) -> str:
    """Render a timeframe token for the query (upper-case fiscal/quarter labels)."""
    text = str(timeframe).strip()
    if re.search(r"(?i)\b(fy|q[1-4])\b", text) or re.search(r"20\d{2}", text):
        return text.upper()
    return text


def _render_query(
    entity: Optional[str],
    metrics: list[str],
    timeframe: Optional[str],
    raw: str,
) -> str:
    """Compose a compact standalone retrieval query from validated slots + turn.

    Slots (entity, metrics, timeframe) lead; then any residual non-filler words
    from the raw turn are appended so genuinely new intent ("why did that
    change?") is not lost. Falls back to the raw turn when there are no slots.
    """
    parts: list[str] = []
    if entity:
        parts.append(entity)
    for metric in metrics or []:
        parts.append(_metric_phrase(metric))
    if timeframe:
        parts.append(_display_timeframe(timeframe))
    slot_query = " ".join(p for p in parts if p).strip()

    residual = _residual_terms(raw, slot_query)
    combined = f"{slot_query} {residual}".strip() if residual else slot_query
    return combined or (raw or "").strip()


def _residual_terms(raw: str, slot_query: str) -> str:
    """Non-filler current-turn words not already represented in the slot query."""
    have = set(re.findall(r"[a-z0-9]+", slot_query.lower()))
    out: list[str] = []
    for token in re.findall(r"[A-Za-z0-9%/&.$-]+", raw or ""):
        low = token.lower().strip(".,?!")
        if not low or low in _FILLER or low in have:
            continue
        out.append(low)
        have.add(low)
    return " ".join(out)
