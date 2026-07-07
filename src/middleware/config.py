"""
src/middleware/config.py
Middleware configuration — loaded from configs/ or environment.
"""

from pathlib import Path
from typing import Optional

import yaml


class MiddlewareConfig:
    """Configuration for the FastAPI middleware layer."""

    def __init__(self, config_path: Optional[Path] = None):
        config_path = config_path or (
            Path(__file__).parent.parent.parent / "configs/middleware.yaml"
        )

        self.llama_endpoint: str = "http://127.0.0.1:8087/v1/chat/completions"
        self.embedding_endpoint: str = "http://127.0.0.1:8087/v1/embeddings"
        self.model_name: str = "tracealchemy"
        self.default_temperature: float = 0.3
        self.max_tokens: int = 2048
        self.top_k_documents: int = 5
        self.top_k_facts: int = 10
        self.enable_citations: bool = True

        # Phase 2.1.4 — analytical tool-calling controls.
        self.enable_tools: bool = False
        self.max_tool_iterations: int = 3
        self.allow_write_tools: bool = False
        self.max_refreshes_per_query: int = 2

        # Phase 2.1.6 — fetch-on-miss ingestion controls.
        self.enable_fetch_on_miss: bool = True
        self.fetch_on_miss_timeout_s: float = 10.0
        self.fetch_on_miss_per_query: int = 1

        # Phase 2.1.2 — hybrid retrieval & re-ranking.
        # Lexical (BM25) channel + RRF fusion (2.1.2.1).
        self.enable_lexical: bool = True
        self.rrf_k: int = 60
        # Cross-encoder re-ranker (2.1.2.2). Opt-in by default — the
        # cross-encoder backend downloads a model on first use.
        self.enable_reranker: bool = False
        self.reranker_backend: str = "cross-encoder"  # "cross-encoder" | "llm"
        self.reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.rerank_candidates: int = 30   # broad retrieve, then re-rank
        self.rerank_top_n: int = 5         # final docs after re-rank

        if config_path.exists():
            self._load_from_file(config_path)
        self._apply_env_overrides()

    def _load_from_file(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _apply_env_overrides(self):
        """Let the 2.1.2 retrieval knobs be overridden by env vars (A/B eval)."""
        import os

        def _bool(env_key: str, attr: str):
            v = os.environ.get(env_key)
            if v is not None:
                setattr(self, attr, v.strip().lower() in ("1", "true", "yes", "on"))

        def _str(env_key: str, attr: str):
            v = os.environ.get(env_key)
            if v is not None:
                setattr(self, attr, v.strip())

        def _int(env_key: str, attr: str):
            v = os.environ.get(env_key)
            if v is not None:
                try:
                    setattr(self, attr, int(v))
                except ValueError:
                    pass

        def _float(env_key: str, attr: str):
            v = os.environ.get(env_key)
            if v is not None:
                try:
                    setattr(self, attr, float(v))
                except ValueError:
                    pass

        _bool("ENABLE_LEXICAL", "enable_lexical")
        _bool("ENABLE_RERANKER", "enable_reranker")
        _str("RERANKER_BACKEND", "reranker_backend")
        _str("RERANKER_MODEL", "reranker_model")
        _int("RERANK_CANDIDATES", "rerank_candidates")
        _int("RERANK_TOP_N", "rerank_top_n")
        _bool("ENABLE_TOOLS", "enable_tools")
        _int("MAX_TOOL_ITERATIONS", "max_tool_iterations")
        _bool("ALLOW_WRITE_TOOLS", "allow_write_tools")
        _int("MAX_REFRESHES_PER_QUERY", "max_refreshes_per_query")
        _bool("ENABLE_FETCH_ON_MISS", "enable_fetch_on_miss")
        _float("FETCH_ON_MISS_TIMEOUT_S", "fetch_on_miss_timeout_s")
        _int("FETCH_ON_MISS_PER_QUERY", "fetch_on_miss_per_query")
