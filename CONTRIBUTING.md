# Contributing

Thanks for contributing to Gemma-E4B-Finance-RAG. This guide covers local
setup, code style, testing, and the pull-request process.

---

## Development Setup

1. **Clone and create a virtual environment:**

   ```bash
   git clone <repo-url> Gemma-E4B-Finance-RAG
   cd Gemma-E4B-Finance-RAG
   python -m venv .venv
   ```

2. **Activate it:**

   ```powershell
   # Windows (PowerShell)
   .\.venv\Scripts\activate
   ```

   ```bash
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies** (runtime + test tooling):

   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```

   `requirements.txt` already pins `pytest`; `pytest-cov` is needed for the
   coverage commands below.

> **One-time model download (Phase 2.1.2 re-ranker):** enabling the
> cross-encoder re-ranker (`enable_reranker: true` with
> `reranker_backend: "cross-encoder"`) lazy-loads
> `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` on first
> use — a ~90 MB download from Hugging Face cached under
> `~/.cache/huggingface/`. This happens automatically the first time a query
> runs with the re-ranker on; it is **not** needed for the default config
> (`enable_reranker: false`) or the `llm` backend (which reuses the
> TraceAlchemy model on `:8087`). If the download fails (offline), the
> re-ranker falls back to the fused order and logs a warning.

4. **Configure secrets** — create a `.env` in the project root:

   ```
   FRED_API_KEY=your_fred_api_key_here
   SEC_EDGAR_USER_AGENT=Your Name your.email@example.com
   ```

   See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for details.

5. **Sanity-check the environment:**

   ```bash
   python scripts/validate_setup.py
   ```

Most unit tests run without a live model or network. Running the full stack
end-to-end additionally requires `llama-server` on `:8087` (see the
[README](README.md) Quick Start).

---

## Code Style

- **PEP 8** — follow standard Python style. Keep lines reasonable (~88–100
  chars) and match the formatting of surrounding code.
- **Type hints** — annotate public function signatures and return types, as the
  existing modules do (e.g. `def get_freshness_report(self, ticker: str) -> dict`).
- **Docstrings** — every module, class, and public method has a docstring.
  Module docstrings start with the file path and a one-line purpose; method
  docstrings describe args/returns. Match this convention.
- **Imports** — group standard library, third-party, then local (`src.*`)
  imports. Local lazy imports inside functions are used intentionally in some
  hot paths (e.g. the middleware pipeline) — preserve that pattern where it
  exists rather than hoisting.
- **Logging** — use the module-level `logger = logging.getLogger(__name__)`
  pattern; do not use bare `print` in library code (scripts may print).
- **Error handling** — keep per-source/per-item failures isolated where the
  existing code does (the scheduler and refresh paths swallow and record
  failures rather than aborting the whole run).

The repository includes a `.ruff_cache/`, so running `ruff` locally is
encouraged for linting.

---

## Testing

The suite lives under `tests/` and is configured by `pytest.ini`
(`pythonpath = .`, `testpaths = tests`).

Run everything:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Markers

`pytest.ini` declares these markers (`--strict-markers` is enabled, so any
marker used must be registered here first):

- `live` — requires the SEC EDGAR network and a running `llama-server` on `:8087`
  (opt-in via `--live`).
- `slow` — slow tests (real ingestion / full pipeline).
- `integration` — cross-module tests that need external services.
- `regression` — Phase 1.1–1.7 feature regression checks.
- `e2e` — end-to-end pipeline tests.
- `network` — tests that make network calls (model/embedding/upstream APIs).

Common runs:

```bash
pytest tests/ -v                       # full suite
pytest tests/ -v -m "not slow"         # skip slow ingestion tests
pytest tests/ -v -m regression         # prior-phase regression only
pytest tests/ -v -m integration        # cross-module integration
pytest tests/ --live                   # include opt-in live model/SEC tests
```

> Network/integration tests skip cleanly when the embedding endpoint (`:8087`)
> or an upstream service is unavailable, so the suite stays runnable offline.

### Guidelines

- Add or update tests for any behavior change; mirror the existing
  `tests/test_*.py` naming.
- Keep unit tests offline — mock external services (yfinance, FRED, GDELT, SEC,
  llama-server) so the default `-m "not live"` run stays deterministic.
- Run the suite before opening a PR and confirm it passes (don't claim green
  without the output).

---

## Pull Request Process

1. **Branch from `Rishi-Ghost`** (the active development branch), not `main`:

   ```bash
   git checkout Rishi-Ghost
   git pull
   git checkout -b feature/<short-description>
   ```

2. **Make focused commits** with clear messages describing the change.

3. **Verify before pushing:**

   ```bash
   pytest tests/ -v -m "not live"
   ```

   Run the relevant `live` tests too if your change touches the model or
   ingestion paths and you have the stack available.

4. **Open a PR targeting `Rishi-Ghost`.** Describe what changed and why, link
   any related task, and note how you tested it.

5. **Keep the PR scoped** — avoid mixing unrelated changes. Do not commit
   secrets, the `.env` file, or generated data under `data/`.
