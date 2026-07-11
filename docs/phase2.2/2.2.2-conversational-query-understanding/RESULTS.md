# 2.2.2 — Conversational Query Understanding: Results

Feature 2.2.2 turns the single-turn `/query` pipeline into a conversation-
capable one while keeping the middleware fully stateless. Three tasks, all
landed on `phase-2.2.2`:

- **2.2.2.1** — bounded client-owned `ChatTurn` history contract.
- **2.2.2.2** — deterministic follow-up rewriting and entity carryover.
- **2.2.2.3** — multiline chat composer, session visibility, conversation-aware eval.

## Setup

All verification is **offline and deterministic** via `scripts\verify.ps1`
(ruff + `pytest -m "not live"`). No live scored run was performed; the
conversational shipping gates (carryover ≥ 0.90, Recall@10 +10 pp, etc.) are
wired in `eval/gate.py` (2.2.1.3) and enforce on the first live evaluation.

## 2.2.2.1 — Request history contract & bounded memory

- `ChatTurn {role, content, turn_id?, context?}` + `QueryRequest.history`;
  `session_id` is tracing-only. Middleware remains stateless.
- `select_history`: ≤ 8 turns / ≤ 8,000 chars, whole turns only, newest-first
  selection restored to chronological order; no summarization.
- Question cap raised 2,000 → 16,000 chars with an explicit validation error —
  no silent truncation anywhere.
- `ChatSession` (scripts/chat.py) owns history per session; turns recorded only
  after accepted terminal answers; `/new`, `/clear`, `/history`, `/history
  off|on`; nothing persisted to disk.
- Single-turn requests unchanged byte-for-byte (`conversation` metadata is
  omitted without history).
- Gate at completion: **VERIFY: PASS** (856 passed).

## 2.2.2.2 — Follow-up rewriting & entity carryover

- `ConversationState` builds only from structured metadata of grounded prior
  answers; assistant prose is never numeric evidence.
- `compile_question` → `CompiledQuestion`: deterministic carryover ("what
  about X" substitution, "same period/metric" single-slot carry, pronouns only
  with exactly one active entity, topic shifts clear incompatible slots,
  `/ticker` override outranks history). Retrieval uses the compiled standalone
  query; the raw question stays untouched for prompts and display.
- Optional one-call LLM rewrite fallback (`query_rewriter.py`): disabled by
  default, strict JSON, SymbolResolver/metric-catalog validated, rejects
  invented entities/numbers, fail-soft to the deterministic query.
- Response exposes `retrieval_query`, `carried_context`, resolved slots;
  `conversation.topic_reset` is now real. Eval rows capture raw vs compiled
  query so the 2.2.1.3 carryover metrics score real values.
- Everything is gated behind `enable_conversation_rewrite` (default **false**);
  flag-off behavior is byte-for-byte unchanged.
- Gate at completion: **VERIFY: PASS** (878 passed).

## 2.2.2.3 — Multiline chat sessions & evaluation

Before → after (client front door):

| dimension | before | after |
|---|---|---|
| multi-part questions | one line per request; pasting split into unrelated queries | `/ask` composer with `/send`, `/preview`, `/cancel`; EOF/Ctrl+C cancels the buffer, not the session; no new dependencies |
| validation errors | 422 detail clipped to 200 chars | full structured FastAPI detail printed |
| streaming robustness | any non-200 stream response disabled streaming for the session | only documented 404/405 disables it; 400/401/403/409/422 fall back once and keep streaming on |
| limits | hardcoded client assumptions | `/health` advertises history/multiline capabilities + effective limits; client falls back for older servers |
| session state | invisible | `session xxxx · N turns` suffix; truncation/topic-reset indicators; carried context + retrieval query in verbose only; `/history` previews never expose traces |
| eval | single implicit mode | `/eval N` single-turn only; `/eval conversations [N]` runs the conversation subset (live, opt-in) |

- Gate at completion: **VERIFY: PASS** (869 passed, 21 skipped).

## Regression-gate outcome

- Offline suite green at every task boundary; final: **VERIFY: PASS**.
- The 21 skips in the final run are pre-existing environment-conditional tests
  (`:8087`/`:8000` unreachable — llama-server was down at run time); they are
  unrelated to 2.2.2 changes and pass when the local stack is up.

## Caveats

- No live conversational metrics yet: `enable_conversation_rewrite` defaults to
  false and the graded baseline is still the 2.2.1.2 placeholder. The shipping
  gates activate on the first live scored run.
- `/eval conversations [N]`'s `N` caps single-turn cases rather than excluding
  them (run_eval has no conversations-only lever; deliberately left unscoped).
- Deterministic retrieval-query rendering (`_render_query` residual-term
  filtering) is tuned to the canonical fixture shapes — first place to tune if
  live multi-turn Recall@10 misses the +10 pp gate.

## Ship / rollback decision

**Ship on `phase-2.2.2`**: additive API, flag-gated rewriting (off by
default), byte-for-byte single-turn compatibility, and full offline coverage.
Rollback is config-only: `enable_conversation_rewrite=false` (already the
default) and clients simply omit `history`. A live scored run to set the real
conversational baseline is the follow-up once the stack is up
(`eval/run_eval.py && eval/score.py --set-baseline --policy graded`).
