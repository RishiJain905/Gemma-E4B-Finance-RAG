"""
scripts/chat.py
Interactive terminal chat client for the Gemma-E4B-Finance-RAG middleware.

One centralized loop:
  - Auto-starts the FastAPI middleware (:8000) if it isn't already running.
  - Sends your questions to POST /query and prints grounded answers + metadata.
  - Slash commands to run ingestion jobs, refresh a ticker, and check health.

Usage:
    python scripts/chat.py                # start middleware (if needed) + chat
    python scripts/chat.py --no-start     # don't auto-start; just connect
    python scripts/chat.py --port 8000

Prerequisite: the model server (llama-server) on :8087 with --embeddings
enabled. The script warns if it is unreachable (answers degrade gracefully).
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_HEALTH_URL = "http://127.0.0.1:8087/health"
STARTUP_HEALTH_PATH = "/health?details=false"
SCHEDULER_MODES = {"daily", "hourly", "weekly", "all", "status"}

# Local fallback for the current-question ceiling when the server does not
# advertise it in /health capabilities (older middleware). Mirrors
# src/middleware/models.MAX_QUESTION_CHARS; the client rejects an over-limit
# multiline buffer locally rather than shipping it and getting a 422.
DEFAULT_MAX_QUESTION_CHARS = 16000

# Eval harness entry points wrapped by the /eval command. The client only ever
# shells out to these — it never imports the pipeline or starts the model.
RUN_EVAL = PROJECT_ROOT / "eval" / "run_eval.py"
SCORE_EVAL = PROJECT_ROOT / "eval" / "score.py"

# Conversational (Phase 2.2) eval metrics surfaced by /eval conversations, and
# the score.py section header they live under. Kept in sync with
# eval/metrics.PHASE22_METRICS (carryover / paraphrase / subquestion coverage /
# staleness / hallucination / cross-session leakage).
_CONVERSATION_METRIC_HEADER = "=== Conversational (Phase 2.2) ==="
_CONVERSATION_METRIC_NAMES = (
    "entity_carryover_accuracy",
    "metric_carryover_accuracy",
    "timeframe_carryover_accuracy",
    "verbose_paraphrase_parity",
    "compound_subquestion_coverage",
    "stale_disclosure_rate",
    "unanswerable_numeric_hallucination_rate",
    "cross_session_leakage_rate",
)

# Enable ANSI escape processing on Windows consoles.
if os.name == "nt":
    os.system("")


class C:
    R = "\033[0m"
    B = "\033[1m"
    DIM = "\033[2m"
    CY = "\033[36m"
    GR = "\033[32m"
    YE = "\033[33m"
    RE = "\033[31m"
    MA = "\033[35m"


def col(s: str, c: str) -> str:
    return f"{c}{s}{C.R}"


def _tty_col(s: str, c: str) -> str:
    """Like col(), but only emits ANSI escapes when stdout is a real terminal."""
    return col(s, c) if sys.stdout.isatty() else s


# Grounding mode -> (color, short label) for the answer-renderer tag (2.1.7.1).
_GROUNDING_TAGS = {
    "grounded": (C.GR, "GROUNDED"),
    "partial": (C.YE, "PARTIAL"),
    "general": (C.MA, "GENERAL"),
    "refused": (C.RE, "REFUSED"),
}


def _grounding_tag(grounding: str | None) -> str:
    """Return a small colored [LABEL] tag for the response's grounding mode."""
    if not grounding:
        return ""
    color, label = _GROUNDING_TAGS.get(grounding, (None, str(grounding).upper()))
    tag = f"[{label}]"
    return _tty_col(tag, color) if color else tag


class _ElapsedSpinner:
    """Lightweight elapsed-time spinner for interactive non-streaming waits."""

    def __init__(self, label: str = "waiting") -> None:
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not sys.stdout.isatty():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1)
        sys.stdout.write("\r" + (" " * 40) + "\r")
        sys.stdout.flush()

    def _run(self) -> None:
        started = time.perf_counter()
        for frame in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                return
            elapsed = time.perf_counter() - started
            sys.stdout.write(f"\r  {frame} {self.label} {elapsed:0.1f}s")
            sys.stdout.flush()
            time.sleep(0.1)


def _new_session_id() -> str:
    """Generate a fresh local session id (tracing only, never persisted)."""
    return uuid.uuid4().hex


def _open_url(url: str, *, opener=webbrowser.open) -> bool:
    """Open a URL in the default browser via stdlib webbrowser (2.2.7.4).

    Never starts the middleware or model — only hands the URL to the OS browser.
    Returns True if the opener reported success; on any failure the caller keeps
    running and prints the URL so it can be opened manually. ``opener`` is
    injected in tests so nothing actually launches a browser."""
    try:
        return bool(opener(url))
    except Exception:
        return False


class MultilineBuffer:
    """Pure, dependency-free accumulator for a multi-line question (2.2.2.3).

    Holds the exact lines the user typed and joins them with ``\\n`` on demand.
    No I/O, no truncation, no punctuation rewriting — what you add is exactly
    what ``text()`` returns, so a structured research prompt reaches the
    middleware contract intact. The composer loop and tests both drive this
    helper; keeping it side-effect-free is what makes it trivial to test.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str) -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines)

    def char_count(self) -> int:
        return len(self.text())

    def is_empty(self) -> bool:
        return not self.text().strip()


def _over_limit_ratio(count: int, limit: int) -> bool:
    """True when a character count is past 80% of its limit (usage warning)."""
    return limit > 0 and count > 0.8 * limit


def compose_multiline(
    session: "ChatSession",
    ticker: str | None,
    autorefresh: bool,
    *,
    input_fn=input,
) -> bool:
    """Run the ``/ask`` multiline composer and submit the result once.

    Reads lines via ``input_fn`` until a composer command:
      - ``/send``    — join with ``\\n``, validate locally, submit exactly once;
      - ``/cancel``  — discard the draft, leaving history untouched;
      - ``/preview`` — show the character count and the exact buffered text;
      - EOF / Ctrl+C — cancel the *buffer* (not the session).

    Everything else is appended verbatim. Returns True only when a question was
    actually submitted (so callers/tests can assert exactly-once submission);
    cancel, interrupt, empty, and over-limit drafts return False and never call
    ``session.query`` — an over-limit buffer is rejected here without a request.
    """
    max_chars = getattr(session, "max_question_chars", DEFAULT_MAX_QUESTION_CHARS)
    buf = MultilineBuffer()
    print(col("  multiline mode — end with /send, discard with /cancel, "
              "inspect with /preview.", C.DIM))
    while True:
        try:
            line = input_fn(col("... ", C.DIM))
        except (EOFError, KeyboardInterrupt):
            print(col("\n  draft cancelled (history unchanged).", C.DIM))
            return False

        command = line.strip().lower()
        if command == "/send":
            question = buf.text()
            if buf.is_empty():
                print(col("  nothing to send yet — type some text, or /cancel.", C.YE))
                continue
            count = buf.char_count()
            if count > max_chars:
                print(col(f"  too long to send: {count}/{max_chars} chars. "
                          "Trim the draft (or /cancel); nothing was sent.", C.RE))
                continue
            if _over_limit_ratio(count, max_chars):
                print(col(f"  {count}/{max_chars} chars", C.YE))
            session.query(question, ticker, autorefresh)
            return True
        if command == "/cancel":
            print(col("  draft cancelled (history unchanged).", C.DIM))
            return False
        if command == "/preview":
            count = buf.char_count()
            print(col(f"  preview — {count}/{max_chars} chars:", C.DIM))
            body = buf.text()
            print(body if body else col("  (empty)", C.DIM))
            continue

        buf.add(line)
        if _over_limit_ratio(buf.char_count(), max_chars):
            print(col(f"  {buf.char_count()}/{max_chars} chars", C.YE))


def _payload(
    question: str,
    ticker: str | None,
    refresh: bool,
    answer_policy: str | None = None,
    *,
    history: list[dict] | None = None,
    session_id: str | None = None,
    mode: str | None = None,
) -> dict:
    payload = {"question": question, "refresh": refresh}
    if ticker:
        payload["ticker"] = ticker
    if answer_policy:
        payload["answer_policy"] = answer_policy
    if history:
        payload["history"] = history
    if session_id:
        payload["session_id"] = session_id
    # Only send mode when it's the non-default analysis mode, so an older server
    # that doesn't know the field is never sent an unexpected key.
    if mode == "analysis":
        payload["mode"] = mode
    return payload


class _StreamProgress:
    """Restrained in-place progress renderer for SSE stage/tool events (2.2.6.1).

    Consumes the redacted progress events (``stage``, ``tool_started``,
    ``tool_completed``) and renders the pipeline as a single line updated in
    place — e.g. ``resolve -> hybrid retrieval -> query_facts (12 rows) -> answer``.

    Rendering is TTY-only: on a non-terminal stdout every method is a no-op, so
    piped/redirected output carries only the terminal answer and metadata (and no
    ANSI escapes are ever emitted). Under ``/verbose`` the single line is replaced
    by a per-event dim log that also surfaces stage timings and corrective
    reasons. Unknown event types never reach here (the caller filters them), and
    an unrecognized stage name falls back to its raw label rather than raising.
    """

    _STAGE_LABELS = {
        "compile": "resolve",
        "route": "route",
        "retrieve": "retrieval",
        "grade": "grade",
        "correct": "correct",
        "pack": "pack",
        "generate": "answer",
        "validate": "validate",
    }

    def __init__(self, *, tty: bool, verbose: bool) -> None:
        self.tty = tty
        self.verbose = verbose
        self.steps: list[str] = []
        self._width = 0

    def _label(self, name) -> str:
        return self._STAGE_LABELS.get(name, str(name or "?"))

    def handle(self, event: str, data: dict) -> None:
        if not self.tty:
            return
        if event == "stage":
            self._stage(data)
        elif event == "tool_completed":
            self._tool(data)
        # tool_started carries no count; the matching tool_completed renders it.
        # query_started / unknown -> nothing to show.

    def _stage(self, data: dict) -> None:
        name = data.get("stage")
        phase = data.get("phase")
        if self.verbose:
            if phase in ("completed", "fallback"):
                extra = ""
                if data.get("elapsed_ms") is not None:
                    extra += f" {data['elapsed_ms']}ms"
                if data.get("reason"):
                    extra += f" ({data['reason']})"
                if phase == "fallback":
                    extra += " [fallback]"
                print(col(f"  · {self._label(name)}{extra}", C.DIM))
            return
        if phase == "started":
            self.steps.append(self._label(name))
            self._render()
        elif phase == "fallback":
            self.steps.append(f"{self._label(name)}!")
            self._render()

    def _tool(self, data: dict) -> None:
        tool = data.get("tool")
        count = data.get("count")
        label = f"{tool} ({count} rows)" if count is not None else str(tool)
        if self.verbose:
            status = data.get("status")
            print(col(f"  · {label} [{status}]", C.DIM))
            return
        self.steps.append(label)
        self._render()

    def _render(self) -> None:
        if not self.tty or self.verbose:
            return
        line = "  " + " -> ".join(self.steps)
        pad = max(0, self._width - len(line))
        sys.stdout.write("\r" + line + (" " * pad))
        sys.stdout.flush()
        self._width = max(self._width, len(line))

    def finish(self) -> None:
        """Clear the in-place progress line (before the answer starts)."""
        if self.tty and not self.verbose and self._width:
            sys.stdout.write("\r" + (" " * self._width) + "\r")
            sys.stdout.flush()
        self._width = 0


def _iter_sse_events(lines) -> object:
    event = "message"
    data_parts: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if not line:
            if data_parts:
                yield event, "\n".join(data_parts)
            event = "message"
            data_parts = []
            continue
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_parts.append(line.removeprefix("data:").strip())
    if data_parts:
        yield event, "\n".join(data_parts)


def _format_timings(timings: dict | None) -> str:
    if not timings:
        return ""
    parts: list[str] = []
    for key, value in timings.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                parts.append(f"{key}.{sub_key}={sub_value}ms")
        else:
            parts.append(f"{key}={value}ms")
    return "  timings: " + " ".join(parts)


def _format_validation_detail(detail) -> list[str]:
    """Format a FastAPI 422 ``detail`` payload into full, unclipped lines.

    FastAPI reports validation errors as a list of ``{loc, msg, type}`` dicts.
    We print every field of every error in full — a clipped error hides exactly
    the field/limit the user needs (e.g. an over-limit ``question``)."""
    lines: list[str] = []
    if isinstance(detail, list):
        for err in detail:
            if isinstance(err, dict):
                loc = ".".join(str(p) for p in (err.get("loc") or [])) or "(request)"
                msg = err.get("msg", "")
                etype = err.get("type", "")
                suffix = f" [{etype}]" if etype else ""
                lines.append(f"    - {loc}: {msg}{suffix}")
            else:
                lines.append(f"    - {err}")
    elif detail is not None:
        lines.append(f"    {detail}")
    return lines


def _print_http_error(resp) -> None:
    """Print an error response. A 422 gets its full structured validation
    detail (never clipped); other statuses show a short body preview."""
    status = resp.status_code
    if status == 422:
        try:
            body = resp.json()
        except Exception:
            body = None
        print(col("  HTTP 422 — request validation failed:", C.RE))
        detail_lines = _format_validation_detail(body.get("detail") if isinstance(body, dict) else body)
        if detail_lines:
            for line in detail_lines:
                print(col(line, C.RE))
        else:
            # No structured detail — fall back to the full raw body, unclipped.
            print(col(f"    {resp.text}", C.RE))
        return
    print(col(f"  HTTP {status}: {resp.text[:200]}", C.RE))


def _render_metadata(data: dict, *, verbose: bool) -> None:
    """Render the full answer metadata block for a /query response.

    Single shared renderer for both the streaming (terminal metadata event)
    and non-streaming paths. Every field is optional so a minimal response
    from an older middleware still renders cleanly with no KeyErrors.
    """
    if data.get("mode") == "analysis":
        print(col("  [analyst] analytical view — not personalized financial advice", C.CY))
    tag = _grounding_tag(data.get("grounding"))
    if tag:
        print(f"  {tag}")

    meta = (
        f"  ticker={data.get('detected_ticker')} "
        f"intent={data.get('detected_intent')} "
        f"grounding={data.get('grounding')} "
        f"facts={data.get('facts_used')} docs={data.get('documents_used')} "
        f"model_available={data.get('model_available')} "
        f"latency={data.get('latency_ms')}ms "
        f"retrieval={data.get('retrieval_strategy')}"
    )
    print(col(meta, C.DIM))

    tools_used = data.get("tools_used") or []
    if tools_used:
        print(col(f"  used: {', '.join(tools_used)}", C.DIM))

    # Local query-graph trace pointer (2.2.7.4). Present only when the server's
    # graph observer is enabled; a short id keeps the terminal restrained while
    # letting the user open it with /graph trace.
    trace_id = data.get("graph_trace_id")
    if trace_id:
        print(col(f"  trace={str(trace_id)[:8]} (/graph trace to open)", C.DIM))

    resolved = data.get("resolved_ticker") or {}
    if resolved.get("name"):
        ticker = data.get("detected_ticker") or resolved["name"]
        print(col(f"  interpreting as {resolved['name']} / {ticker}", C.CY))

    fresh = data.get("freshness") or {}
    if fresh.get("fetched_on_miss"):
        print(col(f"  fetched live data for {', '.join(fresh['fetched_on_miss'])}", C.CY))
    if fresh.get("warning"):
        print(col(f"  warning: {fresh['warning']}", C.YE))
    if fresh.get("refreshed_during_query"):
        print(col(f"  refreshed: {', '.join(fresh['refreshed_during_query'])}", C.CY))

    _render_conversation_state(data, verbose=verbose)

    # Adaptive-RAG route (2.2.3.4). A fallback is always worth surfacing (the
    # adaptive layer demoted to the legacy path); the full lane/budget line is
    # verbose-only to keep normal output restrained.
    orch = data.get("orchestration") or {}
    fallback = orch.get("fallback_reason")
    if fallback:
        print(col(f"  adaptive fallback: {fallback}", C.YE))
    if verbose and orch:
        lane = orch.get("lane")
        line = (
            f"  lane={lane} "
            f"subqueries={orch.get('subqueries_executed')} "
            f"rounds={orch.get('retrieval_rounds')} "
            f"rerank={orch.get('reranker_calls')} "
            f"planning={orch.get('planning_calls')}"
        )
        tools = orch.get("deterministic_tools") or []
        if tools:
            line += f" tools={', '.join(tools)}"
        print(col(line, C.DIM))

    if verbose:
        timings = _format_timings(data.get("timings"))
        if timings:
            print(col(timings, C.DIM))


def _carried_context_summary(data: dict) -> str:
    """Compact ``entities · metrics · timeframe`` line from carryover metadata.

    Prefers the follow-up ``carried_context`` block (2.2.2.2); falls back to the
    effective ``resolved_*`` fields. Returns '' when nothing was carried."""
    carried = data.get("carried_context") or {}
    entities = carried.get("entities") or data.get("resolved_tickers") or []
    metrics = carried.get("metrics") or data.get("resolved_metrics") or []
    timeframe = carried.get("timeframe") or data.get("resolved_timeframe")

    parts: list[str] = []
    if entities:
        parts.append(", ".join(str(e) for e in entities))
    if metrics:
        parts.append(", ".join(str(m) for m in metrics))
    if timeframe:
        parts.append(str(timeframe))
    return " · ".join(parts)


def _render_conversation_state(data: dict, *, verbose: bool) -> None:
    """Render conversation-aware indicators for a /query response (2.2.2.3).

    Truncation and topic-reset are shown always (they change what the answer
    could see); carried context and the standalone retrieval query are verbose
    diagnostics — the retrieval query is never echoed as if it were the user's
    wording. Every field is optional so older responses render nothing extra.
    """
    conversation = data.get("conversation") or {}
    carried = data.get("carried_context") or {}

    if conversation.get("history_truncated"):
        used = conversation.get("history_turns_used")
        received = conversation.get("history_turns_received")
        detail = f" (kept {used} of {received} turns)" if used is not None else ""
        print(col(f"  history truncated to fit the server's limit{detail}", C.YE))

    if conversation.get("topic_reset") or carried.get("topic_reset"):
        print(col("  topic reset detected — earlier context was not carried into "
                  "this question.", C.YE))

    if verbose:
        summary = _carried_context_summary(data)
        if summary:
            print(col(f"  context: {summary}", C.CY))
        # Diagnostic only — the standalone retrieval query is never echoed as
        # the user's wording; it surfaces solely under /verbose when rewriting
        # (2.2.2.2) produced one.
        retrieval_query = data.get("retrieval_query")
        if retrieval_query:
            print(col(f"  retrieval query: {retrieval_query}", C.DIM))


class ChatSession:
    """Persistent HTTP session for middleware chat operations.

    Owns this conversation's bounded memory (2.2.2.1): a flat, chronological
    list of ChatTurn dicts and a local ``session_id``. The middleware stays
    stateless — each request carries the selected history and the id (tracing
    only). Turns are recorded only after a request is accepted and produces a
    terminal answer; failed/cancelled/validation-error/incomplete requests
    never mutate history. History is never persisted to disk.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 240,
        stream_enabled: bool = True,
    ) -> None:
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        self.client = httpx.Client(base_url=base_url, timeout=timeout, limits=limits)
        self.stream_enabled = stream_enabled
        self.stream_unavailable = not stream_enabled
        self.verbose = False
        self.answer_policy: str | None = None  # per-session /grounding override
        # Sticky analyst mode (/analysis on|off). When on, every plain-typed
        # question is sent with mode="analysis"; a one-shot /analysis <question>
        # overrides for a single request regardless of this flag.
        self.analysis_mode: bool = False
        # Conversation memory — instance-scoped so two ChatSessions never share
        # turns. No module-level history / mutable default / global last-ticker.
        self.history: list[dict] = []
        self.session_id: str = _new_session_id()
        self.history_enabled: bool = True
        # Effective limits/capabilities — refreshed from /health at startup via
        # load_capabilities(); these defaults keep the client usable against an
        # older middleware that doesn't advertise them.
        self.max_question_chars: int = DEFAULT_MAX_QUESTION_CHARS
        self.history_capable: bool = True
        self.multiline_capable: bool = True
        self.conversation_max_turns: int | None = None
        self.conversation_max_history_chars: int | None = None
        # Local query-graph observer (2.2.7.4). Advertised by /health only when
        # the server has ENABLE_GRAPH_OBSERVER on; older/observer-off servers
        # leave these defaults so /graph reports the feature is unavailable.
        self.graph_observer: bool = False
        self.graph_url: str | None = None
        self.graph_observer_limits: dict | None = None
        # Short id of the most recent query's graph trace, for /graph trace.
        self.last_trace_id: str | None = None

    def close(self) -> None:
        self.client.close()

    def load_capabilities(self) -> dict | None:
        """Read effective limits/capabilities from /health once at startup.

        Populates the multiline composer's character ceiling and the history
        capability flags from the server's advertised ``capabilities`` block.
        Missing or partial blocks (older middleware) leave the local defaults
        in place, so the client stays usable. Returns the block for display."""
        try:
            caps = self.client.get(STARTUP_HEALTH_PATH).json().get("capabilities")
        except Exception:
            return None
        if not isinstance(caps, dict):
            return None
        max_chars = caps.get("max_question_chars")
        if isinstance(max_chars, int) and max_chars > 0:
            self.max_question_chars = max_chars
        if "history" in caps:
            self.history_capable = bool(caps.get("history"))
        if "multiline" in caps:
            self.multiline_capable = bool(caps.get("multiline"))
        if isinstance(caps.get("conversation_max_turns"), int):
            self.conversation_max_turns = caps["conversation_max_turns"]
        if isinstance(caps.get("conversation_max_history_chars"), int):
            self.conversation_max_history_chars = caps["conversation_max_history_chars"]
        # Graph observer capabilities (2.2.7.4) — present only when enabled.
        self.graph_observer = bool(caps.get("graph_observer"))
        graph_url = caps.get("graph_url")
        if isinstance(graph_url, str) and graph_url:
            self.graph_url = graph_url
        limits = caps.get("graph_observer_limits")
        if isinstance(limits, dict):
            self.graph_observer_limits = limits
        return caps

    def prompt_suffix(self) -> str:
        """Compact session/turn indicator for the input prompt (2.2.2.3)."""
        short = self.session_id[:4]
        analyst = " · analyst" if self.analysis_mode else ""
        if not self.history_enabled:
            return f"session {short} · history off{analyst}"
        return f"session {short} · {len(self.history)} turns{analyst}"

    def middleware_up(self) -> bool:
        try:
            return self.client.get(STARTUP_HEALTH_PATH).status_code == 200
        except Exception:
            return False

    # ── Conversation memory ────────────────────────────

    def new_session(self) -> None:
        """Clear turns and rotate the local session id for a fresh conversation.

        Explicit CLI settings (/grounding answer_policy, /verbose, /history
        on|off) are deliberately preserved — only the conversation is reset.
        """
        self.history = []
        self.session_id = _new_session_id()

    def _outgoing_history(self) -> list[dict] | None:
        """The history to send with the next request (None when disabled/empty)."""
        if not self.history_enabled or not self.history:
            return None
        return list(self.history)

    def _assistant_context(self, metadata: dict) -> dict | None:
        """Distil the response fields a follow-up (2.2.2.2) needs to resolve
        references: detected/resolved tickers, intent, grounding, timeframe."""
        ctx: dict = {}
        ticker = metadata.get("detected_ticker")
        if ticker:
            ctx["ticker"] = ticker
        resolved = metadata.get("resolved_ticker")
        if resolved:
            ctx["resolved_ticker"] = resolved
        intent = metadata.get("detected_intent")
        if intent:
            ctx["intent"] = intent
        grounding = metadata.get("grounding")
        if grounding:
            ctx["grounding"] = grounding
        timeframe = metadata.get("timeframe")
        if timeframe:
            ctx["timeframe"] = timeframe
        coverage = metadata.get("coverage_metadata")
        if isinstance(coverage, dict) and coverage:
            ctx["coverage_metadata"] = dict(coverage)
        return ctx or None

    def _record_turn(self, question: str, result: dict) -> None:
        """Append the completed user+assistant turns after a terminal answer.

        Called only when a request was accepted and produced a non-blank answer,
        so incomplete/failed requests never enter history.
        """
        if not self.history_enabled:
            return
        answer = (result.get("answer") or "").strip()
        if not question.strip() or not answer:
            return
        metadata = result.get("metadata") or {}
        self.history.append({"role": "user", "content": question})
        assistant_turn: dict = {"role": "assistant", "content": answer}
        ctx = self._assistant_context(metadata)
        if ctx:
            assistant_turn["context"] = ctx
        self.history.append(assistant_turn)

    def print_history(self) -> None:
        """Local preview of the conversation — no API request (2.2.2.1 Step 4)."""
        if not self.history:
            print(col("  (no conversation history)", C.DIM))
            return
        state = "on" if self.history_enabled else "off"
        print(col(f"  session {self.session_id[:8]} — history {state}, "
                  f"{len(self.history)} turns", C.DIM))
        for i, turn in enumerate(self.history):
            role = turn.get("role", "?")
            content = (turn.get("content") or "").replace("\n", " ")
            preview = content[:60] + ("..." if len(content) > 60 else "")
            print(col(f"  {i}. {role}: {preview}", C.DIM))

    # ── Query ──────────────────────────────────────────

    def query(
        self, question: str, ticker: str | None, refresh: bool,
        mode: str | None = None,
    ) -> None:
        # Effective mode: an explicit one-shot mode wins; otherwise the sticky
        # session flag decides. qa is the default and sends no mode key.
        effective_mode = mode if mode is not None else (
            "analysis" if self.analysis_mode else None
        )
        payload = _payload(
            question, ticker, refresh, getattr(self, "answer_policy", None),
            history=self._outgoing_history(),
            session_id=self.session_id if self.history_enabled else None,
            mode=effective_mode,
        )
        result: dict | None = None
        if self.stream_enabled and not self.stream_unavailable:
            handled, result, disable_stream = self._query_stream(payload)
            # Only a documented 404/405 capability response disables streaming
            # for the rest of the session. A per-request rejection (400/401/403/
            # 409/422) must not poison streaming — we fall back to POST /query
            # for this one request and keep streaming enabled for later ones.
            if disable_stream:
                self.stream_unavailable = True
            if not handled:
                result = self._query_non_stream(payload)
        else:
            result = self._query_non_stream(payload)
        if result is not None:
            self._record_turn(question, result)
            trace_id = (result.get("metadata") or {}).get("graph_trace_id")
            if trace_id:
                self.last_trace_id = trace_id

    # ── Live graph (2.2.7.4) ───────────────────────────

    def effective_graph_url(self) -> str:
        """Return the graph UI URL: the server-advertised value or a local fallback.

        Prefers the same-origin ``graph_url`` the server reports in /health
        capabilities; falls back to this client's own base URL + ``/graph`` so
        the command still resolves a correct localhost URL against an older
        server that does not advertise it."""
        if self.graph_url:
            return self.graph_url
        base = str(self.client.base_url).rstrip("/")
        return f"{base}/graph"

    def graph(self, arg: str, *, opener=webbrowser.open) -> None:
        """Handle ``/graph``, ``/graph url``, and ``/graph trace`` (2.2.7.4).

        Opens (or, for ``url``, only prints) the effective local graph URL using
        the standard-library browser opener. Never starts the middleware or model
        process; a failed browser launch prints the URL and continues."""
        if not self.graph_observer:
            print(col("  graph observer is disabled on this server. Restart the "
                      "middleware with ENABLE_GRAPH_OBSERVER=1 to use /graph.", C.YE))
            return
        sub = (arg or "").strip().lower()
        url = self.effective_graph_url()
        if sub == "trace":
            if self.last_trace_id:
                url = f"{url}#trace={self.last_trace_id}"
            else:
                print(col("  no query trace yet — ask a question first; opening the "
                          "live view.", C.DIM))
        elif sub == "url":
            print(col(f"  {url}", C.CY))
            return
        elif sub not in ("", "open"):
            print(col("  usage: /graph [url|trace]", C.YE))
            return
        if _open_url(url, opener=opener):
            print(col(f"  opened {url}", C.DIM))
        else:
            print(col(f"  couldn't open a browser automatically — open: {url}", C.YE))

    def refresh(self, arg: str) -> None:
        do_refresh(self.client, arg)

    def health(self) -> None:
        do_health(self.client)

    def tools(self) -> None:
        do_tools(self.client)

    def _query_stream(self, payload: dict) -> tuple[bool, dict | None, bool]:
        """Stream a query. Returns ``(handled, result, disable_stream)``.

        ``handled`` is False only when streaming was not usable, so ``query``
        can fall back to POST /query. ``result`` is ``{"answer", "metadata"}``
        only on a complete terminal response (tokens + metadata event); None for
        an incomplete stream, so an incomplete request is never recorded.

        ``disable_stream`` is True *only* for a documented 404/405 capability
        response (the stream endpoint genuinely isn't there). A per-request
        rejection (400/401/403/409/422) returns ``(False, None, False)`` — we
        fall back for this request but must not disable streaming, or one bad
        input would silently degrade every later query to non-streaming.
        """
        metadata: dict | None = None
        printed_token = False
        answer_started = False
        answer_parts: list[str] = []
        progress = _StreamProgress(tty=sys.stdout.isatty(), verbose=self.verbose)
        try:
            with self.client.stream("POST", "/query/stream", json=payload) as resp:
                if resp.status_code != 200:
                    # 404/405 = endpoint/method absent -> disable streaming.
                    # Any other status is a per-request error; fall back once
                    # (POST /query surfaces the structured detail) but keep
                    # streaming available for subsequent questions.
                    return False, None, resp.status_code in (404, 405)
                for event, raw_data in _iter_sse_events(resp.iter_lines()):
                    if event == "token":
                        data = json.loads(raw_data)
                        token = data.get("token") or ""
                        if token:
                            if not answer_started:
                                progress.finish()
                                print()
                                answer_started = True
                            printed_token = True
                            answer_parts.append(token)
                            print(token, end="", flush=True)
                    elif event == "metadata":
                        metadata = json.loads(raw_data)
                    elif event == "error":
                        progress.finish()
                        data = json.loads(raw_data) if raw_data else {}
                        message = data.get("message") if isinstance(data, dict) else None
                        if message:
                            print(col(f"\n  stream error: {message}", C.YE))
                        return printed_token, None, False
                    elif event in ("stage", "tool_started", "tool_completed", "query_started"):
                        # Redacted progress events (2.2.6.1). Ignored on non-TTY;
                        # unknown future event types fall through and are ignored.
                        try:
                            progress.handle(event, json.loads(raw_data))
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            progress.finish()
            if printed_token:
                print(col(f"\n  stream ended early: {e}", C.YE))
                return True, None, False
            # Pre-token transport error: fall back for this request without
            # permanently disabling streaming (the failure may be transient).
            return False, None, False

        if printed_token:
            print()
        if metadata is not None:
            _render_metadata(metadata, verbose=self.verbose)
            print()
            return True, {"answer": "".join(answer_parts), "metadata": metadata}, False
        # Tokens printed but no terminal metadata: handled (avoid a double
        # answer from a fallback) but incomplete, so nothing is recorded.
        return printed_token, None, False

    def _query_non_stream(self, payload: dict) -> dict | None:
        """POST /query. Returns ``{"answer", "metadata"}`` on a 200 with a
        non-blank answer, else None (failed/empty requests are not recorded)."""
        spinner = _ElapsedSpinner()
        spinner.start()
        try:
            resp = self.client.post("/query", json=payload)
        except Exception as e:
            spinner.stop()
            print(col(f"  request failed: {e}", C.RE))
            return None
        finally:
            spinner.stop()

        if resp.status_code != 200:
            _print_http_error(resp)
            return None

        data = resp.json()
        answer = (data.get("answer") or "").strip()
        print()
        if answer:
            print(col(answer, C.B))
        else:
            print(col("(model returned an empty completion)", C.YE))

        _render_metadata(data, verbose=self.verbose)
        print()
        if not answer:
            return None
        return {"answer": data.get("answer") or "", "metadata": data}


# ── Middleware lifecycle ───────────────────────────────

def middleware_up(base: str) -> bool:
    try:
        with httpx.Client(base_url=base, timeout=3) as client:
            return client.get(STARTUP_HEALTH_PATH).status_code == 200
    except Exception:
        return False


def start_middleware(port: int):
    """Launch uvicorn as a subprocess; return the Popen handle once healthy."""
    PROJECT_ROOT.joinpath("logs").mkdir(exist_ok=True)
    log = open(PROJECT_ROOT / "logs" / "middleware.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.middleware.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(PROJECT_ROOT), stdout=log, stderr=log,
    )
    base = f"http://127.0.0.1:{port}"
    print(col(f"Starting middleware on :{port} ...", C.DIM), end="", flush=True)
    for _ in range(40):  # ~40s
        if proc.poll() is not None:
            print(col(" failed (process exited). See logs/middleware.log", C.RE))
            return None
        if middleware_up(base):
            print(col(" ready.", C.GR))
            return proc
        print(col(".", C.DIM), end="", flush=True)
        time.sleep(1)
    print(col(" timed out waiting for /health.", C.RE))
    return proc


def check_model(client: httpx.Client) -> None:
    try:
        ok = client.get(MODEL_HEALTH_URL).status_code == 200
    except Exception:
        ok = False
    if ok:
        print(col("Model server (:8087) reachable.", C.GR))
    else:
        print(col("Model server (:8087) NOT reachable — answers will run in "
                  "degraded mode (raw retrieved data). Start llama-server with "
                  "--embeddings.", C.YE))


# ── Actions ────────────────────────────────────────────

def do_refresh(client: httpx.Client, arg: str) -> None:
    """Bare/mode arg -> run the scheduler; a ticker -> hit /refresh/{ticker}."""
    arg = (arg or "").strip()
    mode = arg.lower() if arg else "all"

    if not arg or mode in SCHEDULER_MODES:
        run_scheduler(mode)
        return

    ticker = arg.upper()
    print(col(f"Refreshing {ticker} via API ...", C.DIM))
    try:
        resp = client.post(f"/refresh/{ticker}", json={}, timeout=600)
        data = resp.json()
        print(col(f"  refreshed: {data.get('refreshed')}", C.GR))
        if data.get("errors"):
            print(col(f"  errors: {data.get('errors')}", C.YE))
        print(col(f"  duration: {data.get('duration_s')}s", C.DIM))
    except Exception as e:
        print(col(f"  refresh failed: {e}", C.RE))


def run_scheduler(mode: str):
    """Run `python -m src.scheduler <mode>` and stream its output."""
    cmd = [sys.executable, "-m", "src.scheduler", mode]
    if mode != "status":
        cmd.append("--force")
    print(col(f"Running scheduler '{mode}' (this can take a few minutes) ...", C.MA))
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(col("  " + line.rstrip(), C.DIM))
        proc.wait()
        print(col(f"scheduler '{mode}' finished (exit {proc.returncode}).", C.GR))
    except Exception as e:
        print(col(f"  scheduler failed: {e}", C.RE))


def do_health(client: httpx.Client) -> None:
    try:
        data = client.get("/health").json()
    except Exception as e:
        print(col(f"  health request failed: {e}", C.RE))
        return
    print(col(f"  status={data.get('status')} model_available={data.get('model_available')}", C.GR))
    fresh = data.get("freshness") or {}
    if fresh:
        print(col("  freshness: " + ", ".join(f"{k}={v}" for k, v in fresh.items()), C.DIM))
    storage = data.get("storage") or {}
    print(col(f"  storage: sqlite={storage.get('sqlite')} chroma={storage.get('chroma')} "
              f"docs={storage.get('chroma_doc_count')}", C.DIM))


def do_tools(client: httpx.Client) -> None:
    try:
        resp = client.get("/tools")
    except Exception as e:
        print(col(f"  tools request failed: {e}", C.RE))
        return
    if resp.status_code != 200:
        print(col(f"  tools endpoint unavailable (HTTP {resp.status_code}) — "
                  "server may not support tool-calling.", C.YE))
        return
    data = resp.json()

    enabled = data.get("enabled")
    allow_write = data.get("allow_write_tools")
    print(col(f"  enabled={enabled} allow_write_tools={allow_write}", C.GR))
    for tool in data.get("tools", []):
        mode = "W" if tool.get("write") else "R"
        desc = (tool.get("description") or "").replace("\n", " ")
        if len(desc) > 70:
            desc = desc[:67].rstrip() + "..."
        print(col(f"  {tool.get('name', ''):<22} {mode}  {desc}", C.DIM))


def _default_eval_runner(argv: list[str]) -> tuple[int, str]:
    """Shell out to a python eval script and return ``(returncode, output)``.

    The single boundary the /eval command crosses — no in-process import of the
    pipeline, no model load/start here. Injected in tests so nothing runs."""
    proc = subprocess.run(
        [sys.executable, *argv], cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=1800,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _print_eval_tail(output: str, *, n: int = 15) -> None:
    lines = [line for line in output.splitlines() if line.strip()]
    for line in lines[-n:]:
        print(col("  " + line, C.DIM))


def _print_conversation_metrics(output: str) -> None:
    """Echo the conversational (Phase 2.2) metric lines from score.py output.

    Best-effort: prints the metric rows under score.py's Conversational header,
    or any line naming a known carryover/coverage/leakage metric. Prints a note
    when none are present (e.g. the run scored no conversation fixtures)."""
    lines = output.splitlines()
    shown: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == _CONVERSATION_METRIC_HEADER:
            in_block = True
            continue
        if in_block:
            if not stripped or stripped.startswith("==="):
                in_block = False
                continue
            shown.append(stripped)
        elif any(name in line for name in _CONVERSATION_METRIC_NAMES):
            shown.append(stripped)

    if shown:
        print(col("  conversational metrics:", C.MA))
        for line in shown:
            print(col("    " + line, C.DIM))
    else:
        print(col("  (no conversational metrics in this run — did it score any "
                  "conversation fixtures?)", C.YE))


def do_eval(limit: int = 5, *, conversations: bool = False, runner=None) -> None:
    """Run the eval harness against the running server and print a summary.

    - Single-turn (default): drive N single-turn golden cases only
      (``run_eval.py --limit N --no-conversations``).
    - ``conversations=True``: include the multi-turn conversation fixtures
      (``run_eval.py --limit N``), then score deterministically
      (``score.py --no-judge``) and surface the carryover / topic-reset /
      subquestion-coverage / leakage metrics.

    Never loads or starts the model: it only shells out via ``runner`` (default
    :func:`_default_eval_runner`), which tests replace to assert the argv."""
    runner = runner or _default_eval_runner
    if conversations:
        run_argv = [str(RUN_EVAL), "--limit", str(limit)]
        score_argv = [str(SCORE_EVAL), "--no-judge"]
        print(col(f"Running conversation eval (limit={limit} single-turn "
                  "+ all conversation fixtures) ...", C.MA))
        try:
            run_rc, run_out = runner(run_argv)
            _print_eval_tail(run_out)
            score_rc, score_out = runner(score_argv)
        except Exception as e:
            print(col(f"  eval failed: {e}", C.RE))
            return
        _print_conversation_metrics(score_out)
        rc = run_rc or score_rc
    else:
        run_argv = [str(RUN_EVAL), "--limit", str(limit), "--no-conversations"]
        print(col(f"Running single-turn eval (limit={limit}) ...", C.MA))
        try:
            rc, run_out = runner(run_argv)
        except Exception as e:
            print(col(f"  eval failed: {e}", C.RE))
            return
        _print_eval_tail(run_out)
    status_color = C.GR if rc == 0 else C.RE
    print(col(f"eval finished (exit {rc}).", status_color))


def print_capabilities(client: httpx.Client) -> None:
    """Fetch /health once at startup and show the active deployment capabilities."""
    try:
        data = client.get(STARTUP_HEALTH_PATH).json()
    except Exception:
        return
    caps = data.get("capabilities")
    if not isinstance(caps, dict):
        return
    parts = []
    if "tools" in caps:
        parts.append(f"tools={'on' if caps.get('tools') else 'off'}")
    if "streaming" in caps:
        parts.append(f"streaming={'on' if caps.get('streaming') else 'off'}")
    if "answer_policy" in caps:
        parts.append(f"answer_policy={caps.get('answer_policy')}")
    if "history" in caps:
        parts.append(f"history={'on' if caps.get('history') else 'off'}")
    if "max_question_chars" in caps:
        parts.append(f"max_question_chars={caps.get('max_question_chars')}")
    if caps.get("graph_observer"):
        parts.append("graph=on")
    if parts:
        print(col("Capabilities: " + " ".join(parts), C.DIM))


HELP = f"""
{C.B}Commands{C.R}
  {C.CY}<just type a question>{C.R}   ask the RAG (POST /query)
  {C.CY}/analysis <question>{C.R}     one-shot analyst-mode answer (verdict, no refusals)
  {C.CY}/analysis on|off{C.R}         toggle sticky analyst mode for every question
  {C.CY}/ask{C.R}                     compose a multiline question (/send, /cancel, /preview)
  {C.CY}/refresh{C.R}                 run ALL ingestion jobs (scheduler all --force)
  {C.CY}/refresh daily|hourly|weekly|all|status{C.R}   run that scheduler mode
  {C.CY}/refresh NVDA{C.R}            refresh one ticker via the API
  {C.CY}/ticker NVDA{C.R}             pin a ticker override for following questions
  {C.CY}/ticker clear{C.R}            clear the pinned ticker
  {C.CY}/autorefresh on|off{C.R}      toggle auto-refresh of stale data per query
  {C.CY}/verbose on|off{C.R}          toggle server timings under answers
  {C.CY}/grounding strict|graded{C.R} set the answer policy sent with each query
  {C.CY}/grounding clear{C.R}         use the server's default answer policy
  {C.CY}/new{C.R} or {C.CY}/clear{C.R}           start a fresh conversation (clear history, new session id)
  {C.CY}/history{C.R}                 preview this conversation's turns (no API call)
  {C.CY}/history off|on{C.R}          stop/resume sending & recording history
  {C.CY}/health{C.R}                  show middleware health summary
  {C.CY}/tools{C.R}                   show model-callable tools
  {C.CY}/eval [N]{C.R}                run the single-turn eval (default 5 cases) against this server
  {C.CY}/eval conversations [N]{C.R}  run the conversation fixtures + carryover/leakage metrics
  {C.CY}/graph{C.R}                   open the live retrieval graph in your browser (if enabled)
  {C.CY}/graph url{C.R}               print the graph URL without opening it
  {C.CY}/graph trace{C.R}             open the graph focused on the latest query trace
  {C.CY}/help{C.R}                    show this help
  {C.CY}/quit{C.R} or {C.CY}/exit{C.R}            leave (stops the middleware if this script started it)
"""


# ── REPL ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive RAG chat client")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-start", action="store_true",
                        help="Do not auto-start the middleware")
    parser.add_argument("--timeout", type=float, default=240,
                        help="HTTP request timeout in seconds")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming and use POST /query")
    parser.add_argument("--open-graph", action="store_true",
                        help="Open the live retrieval graph once at startup "
                             "(only if the server has the graph observer enabled)")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    started_proc = None
    session = ChatSession(
        base,
        timeout=args.timeout,
        stream_enabled=not args.no_stream,
    )
    model_health_client = httpx.Client(timeout=3)

    print(col("\n=== Gemma-E4B-Finance-RAG chat ===", C.B))
    check_model(model_health_client)

    if session.middleware_up():
        print(col(f"Middleware already running on :{args.port}.", C.GR))
    elif args.no_start:
        print(col(f"Middleware not running on :{args.port} and --no-start set. "
                  f"Start it first.", C.RE))
        session.close()
        model_health_client.close()
        return
    else:
        started_proc = start_middleware(args.port)
        if not session.middleware_up():
            print(col("Could not reach the middleware; exiting.", C.RE))
            if started_proc:
                started_proc.terminate()
            session.close()
            model_health_client.close()
            return

    ticker = None
    autorefresh = False
    session.load_capabilities()
    print_capabilities(session.client)
    if args.open_graph:
        # Open once after capabilities are known (2.2.7.4). No-op with a friendly
        # note when the server has the observer disabled; never starts a process.
        session.graph("")
    print(HELP)

    try:
        while True:
            prefix = f"[{ticker}] " if ticker else ""
            header = col(f"{prefix}{session.prompt_suffix()}", C.DIM)
            try:
                line = input(f"\n{header}\n{col('you> ', C.GR)}").strip()
            except EOFError:
                break
            if not line:
                continue

            if line.startswith("/"):
                parts = line[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                rest = parts[1] if len(parts) > 1 else ""

                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "ask":
                    compose_multiline(session, ticker, autorefresh)
                elif cmd == "help":
                    print(HELP)
                elif cmd == "health":
                    session.health()
                elif cmd == "tools":
                    session.tools()
                elif cmd == "graph":
                    session.graph(rest)
                elif cmd == "refresh":
                    session.refresh(rest)
                elif cmd == "ticker":
                    if rest.lower() in ("", "clear", "none"):
                        ticker = None
                        print(col("  ticker override cleared.", C.DIM))
                    else:
                        ticker = rest.upper()
                        print(col(f"  ticker pinned to {ticker}.", C.DIM))
                elif cmd == "autorefresh":
                    autorefresh = rest.lower() in ("on", "true", "1", "yes")
                    print(col(f"  autorefresh = {autorefresh}", C.DIM))
                elif cmd == "verbose":
                    session.verbose = rest.lower() in ("on", "true", "1", "yes")
                    print(col(f"  verbose = {session.verbose}", C.DIM))
                elif cmd == "analysis":
                    val = rest.strip().lower()
                    if val == "on":
                        session.analysis_mode = True
                        print(col("  analyst mode = on (every question runs as analysis)", C.DIM))
                    elif val == "off":
                        session.analysis_mode = False
                        print(col("  analyst mode = off", C.DIM))
                    elif rest.strip():
                        # One-shot analysis query; keeps pinned ticker + refresh state.
                        session.query(rest.strip(), ticker, autorefresh, mode="analysis")
                    else:
                        state = "on" if session.analysis_mode else "off"
                        print(col(f"  usage: /analysis <question> | /analysis on|off "
                                  f"(currently {state})", C.YE))
                elif cmd == "grounding":
                    val = rest.strip().lower()
                    if val in ("strict", "graded"):
                        session.answer_policy = val
                        print(col(f"  grounding policy = {val}", C.DIM))
                    elif val in ("", "clear", "default", "none"):
                        session.answer_policy = None
                        print(col("  grounding policy = server default", C.DIM))
                    else:
                        print(col("  usage: /grounding strict|graded|clear", C.YE))
                elif cmd in ("new", "clear"):
                    session.new_session()
                    print(col("  started a new conversation (history cleared, "
                              f"session {session.session_id[:8]}).", C.DIM))
                elif cmd == "history":
                    val = rest.strip().lower()
                    if val == "off":
                        session.history_enabled = False
                        print(col("  history = off (not sending or recording turns)", C.DIM))
                    elif val == "on":
                        session.history_enabled = True
                        print(col("  history = on", C.DIM))
                    elif val == "":
                        session.print_history()
                    else:
                        print(col("  usage: /history [off|on]", C.YE))
                elif cmd == "eval":
                    tokens = rest.split()
                    if tokens and tokens[0].lower() in (
                            "conversations", "conversation", "convo", "convos"):
                        limit = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 5
                        do_eval(limit, conversations=True)
                    else:
                        limit = int(tokens[0]) if tokens and tokens[0].isdigit() else 5
                        do_eval(limit)
                else:
                    print(col(f"  unknown command: /{cmd} (try /help)", C.YE))
                continue

            session.query(line, ticker, autorefresh)
    except KeyboardInterrupt:
        print()
    finally:
        if started_proc is not None:
            print(col("Stopping middleware (started by this script) ...", C.DIM))
            started_proc.terminate()
            try:
                started_proc.wait(timeout=10)
            except Exception:
                started_proc.kill()
        session.close()
        model_health_client.close()
        print(col("Bye.", C.B))


if __name__ == "__main__":
    main()
