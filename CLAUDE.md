# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Picking the right models for workflows and subagents

Rankings, higher = better. Cost reflects what I actually pay (OpenAI has really generous limits), not list price. Intelligence is how hard a problem you can handle the model unsupervised. Taste covers UI/UX, code quality, API design, and copy.

| model | cost | intelligence | taste |
|-------|------|--------------|-------|
| gpt-5.5 | 9 | 8 | 5 |
| sonnet-5 | 5 | 5 | 7 |
| opus-4.8 | 4 | 7 | 8 |
| fable-5 | 2 | 9 | 9 |

How to apply:
- These are defaults, not limits. You have standing permission to override them: if a cheaper model's output doesn't meet the bar, rerun or redo the work with a smarter model without asking. Judge the output, not the price tag. Escalating costs less than shipping mediocre work.
- Cost is a tie-breaker only; when axes conflict for anything that ships, intelligence > taste > cost.
- Bulk/mechanical work (clear-spec implementation, data analysis, migrations): gpt-5.5 — it's effectively free.
- Anything user-facing (UI, copy, API design) needs taste ≥ 7.
- Reviews of plans/implementations: fable-5 or opus-4.8, optionally gpt-5.5 as an extra independent perspective.
- Never use Haiku.
- Mechanics: gpt-5.5 is accessed from Claude Code through the Codex plugin. For implementation, debugging, investigation, data analysis, or other delegated work, use `/codex:rescue --model gpt-5.5 --effort high <task>`. Add `--background` for longer-running work, then use `/codex:status` and `/codex:result` to monitor it and retrieve the result. For reviews, use `/codex:review` or `/codex:adversarial-review`; these use the model selected by the Codex configuration, so gpt-5.5 should be configured as the default model in `~/.codex/config.toml` or the repository's `.codex/config.toml`.
- Claude models (sonnet-5, opus-4.8, fable-5) run via the Agent/Workflow model parameter.

Using gpt-5.5 inside workflows and subagents:
- The Agent/Workflow `model` parameter only accepts Claude models. To delegate work to gpt-5.5, use the Codex plugin's bundled `codex:codex-rescue` subagent rather than creating a custom Claude wrapper. If a wrapper is needed and is a must only then spawn a Claude wrapper agent with `model: 'sonnet', effort: 'low'` whose prompt instructs it to write a self-contained codex prompt. Plugin is priority and first target as it is setup with this intent and workflow in mind. 
- For implementation or investigation, invoke `/codex:rescue --model gpt-5.5 --effort high --background <self-contained task>`.
- Use `/codex:status` to check progress and `/codex:result` to retrieve the completed response.
- For an independent code review, run `/codex:review --background`.
- For a review focused on challenging design decisions, assumptions, or specific risk areas, run `/codex:adversarial-review --background <focus>`.
- Claude may also delegate naturally by being instructed to ask Codex to complete a task.

When Using Plan mode:
- Inherited / current model the user is using will be the model that is used to create the plan for the task at hand. This will likely be Fable 5 or Opus 4.8
- Once Fable 5 or Opus 4.8 has thought of a plan, spawn a subagent who will use `model: 'sonnet 5'` and the thinking effort will be based on complexity of task. This sonnet 5 model will create a HTML file using the frontend design skill. This HTML file should outline the entire plan and be presented to me (user).
- Instead of the typical MD file that is shown as the plan outline before the user (me) clicks proceed to implement, this HTML file will replace it. Make sure the Artifact HTML created is opened for the user when you are ready to show the plan and HTML file. 
- The objective is to visualize the plan prior to implementation so that its easier to optimize the plan before any code is written. 
- All subagents launched in Plan Mode will use `'model: 'sonnet 5'`. Effort level can be your choice based on complexity of task given to the model. This includes `Explore` Agents. The only Exception is the `plan` Agent who can use the `Model: 'Opus 4.8'` as the plan-agent default when specs are detailed and exploration ran first; `Model: 'Fable 5'` for open-ended or high ambigutiy design.

## What this is

Hybrid finance RAG system: six data sources (SEC EDGAR, Yahoo Finance, FRED, GDELT, earnings transcripts, IR pages) are ingested into dual stores — SQLite (`data/finance.db`, structured facts/filings/freshness) and ChromaDB (`data/chroma`, document embeddings) — and served through a FastAPI middleware (`:8000`) that does intent parsing → hybrid retrieval → prompt augmentation → a locally-served fine-tuned Gemma model ("TraceAlchemy") on `llama-server` (`:8087`, chat + embeddings from the same server).

## Commands

```bash
pytest tests/ -v -m "not live"        # standard run (offline, deterministic)
pytest tests/ -v                      # full suite
pytest tests/test_retriever.py -v     # single file
pytest tests/test_retriever.py::TestClass::test_name -v   # single test
pytest tests/ --live                  # opt-in live tests (need llama-server on :8087 + network)
pytest tests/ -v --cov=src --cov-report=term-missing      # coverage
python scripts/validate_setup.py      # environment sanity check
python -m src.scheduler daily --force # run ingestion (modes: daily/hourly/weekly/all/status)
python scripts/chat.py                # interactive client (auto-starts middleware)
uvicorn src.middleware.app:app --port 8000                # middleware directly
ruff check .                          # lint
```

Windows stack scripts: `scripts/serve_model.ps1 start` (llama-server), `scripts/start_stack.ps1` / `stop_stack.ps1` (middleware). `.sh` equivalents exist for Unix.

`--strict-markers` is enabled; registered markers are `live`, `slow`, `integration`, `regression`, `e2e`, `network` (see `pytest.ini`). Network/integration tests skip cleanly when `:8087` or upstream services are unavailable — keep new tests offline-safe by mocking external services.

## Architecture (big picture)

`docs/ARCHITECTURE.md` is the authoritative reference (components, schema, query flow, failure modes). Key structural points that span multiple files:

- **Storage facade** — `src/storage/store.py` (`Store`) unifies `SQLiteStore` and `ChromaStore`; middleware and ingestors go through it. Freshness/TTL state lives in the `cache_meta` table; scheduler cadence uses the synthetic ticker `"SCHEDULER"`.
- **Query pipeline** — `src/middleware/app.py` orchestrates `IntentParser` → `Retriever` (strategy chosen from intent: facts_only / hybrid / documents_only / comparison / macro / broad) → `PromptAugmenter` → model call. If the model is down, `/query` returns a **degraded** answer from raw retrieval instead of failing.
- **Retrieval (Phase 2.1.2)** — optional hybrid pipeline: ChromaDB vector search + BM25 (`src/middleware/lexical_index.py`) → RRF fusion → optional cross-encoder/LLM re-rank (`src/middleware/reranker.py`). Each stage is toggleable in `configs/middleware.yaml` or via env (`ENABLE_LEXICAL`, `ENABLE_RERANKER`, `RERANKER_BACKEND`). Every stage fails soft — a query never errors because a retrieval stage failed.
- **Ingestion** — `UnifiedScheduler` (`src/scheduler/__init__.py`) drives all sources with TTL gating, staggered execution, and per-source failure isolation (log + mark stale + dead-letter, continue). Resilience primitives (retry/backoff, `CircuitBreaker`, `DeadLetterQueue`) live in `src/utils/resilience.py`.
- **Embeddings** — `TraceAlchemyEmbeddingFunction` in `chroma_store.py` POSTs to llama-server's `/v1/embeddings` (requires `--embeddings --pooling mean` on the server). Documents >1000 chars are chunked with 150-char overlap and stored as `"{id}#{i}"` with parent metadata.

## Configuration

- Runtime config is YAML under `configs/` (documented in `docs/CONFIGURATION.md`); secrets go in `.env` (`FRED_API_KEY`, `SEC_EDGAR_USER_AGENT`).
- Machine-specific model paths go in `configs/model.local.yaml` (gitignored; copy from `configs/model.example.yaml`) or env vars `MAIN_MODEL_PATH` / `LLAMA_BUILD_DIR` / `DRAFT_MODEL_PATH`. Path resolution is in `src/utils/model_config.py` / `scripts/resolve_model_paths.py`.
- Never commit `.env`, `configs/model.local.yaml`, or anything under `data/`.

## Conventions

- **Branching**: the active development branch is `Rishi-Ghost` — branch from and PR into it, not `main`.
- Module docstrings start with the file path and a one-line purpose; annotate public signatures.
- Lazy local imports inside functions in hot paths (e.g. the middleware pipeline) are intentional — don't hoist them.
- `logger = logging.getLogger(__name__)` in library code; `print` only in `scripts/`.
- Per-source/per-item failures are swallowed and recorded, not propagated — preserve this isolation pattern in scheduler/refresh paths.
