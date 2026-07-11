"""
src/middleware/config.py
Middleware configuration — loaded from configs/ or environment.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Documented safe maxima for the conversation budget (2.2.2.1). Configured
# values above these are clamped (with one warning) so a misconfiguration can
# never let an unbounded history/question payload through. The question ceiling
# matches models.MAX_QUESTION_CHARS — the pydantic-enforced hard cap.
_MAX_CONVERSATION_TURNS_CEILING = 50
_MAX_HISTORY_CHARS_CEILING = 32000
_MAX_QUESTION_CHARS_CEILING = 16000


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
        self.answer_policy: str = "graded"
        self.allow_general_fallback: bool = True
        self.return_timings: bool = True
        self.enable_streaming: bool = True
        self.embedding_cache_size: int = 256

        # Phase 2.1.4 — analytical tool-calling controls.
        self.enable_tools: bool = False
        self.max_tool_iterations: int = 3
        self.allow_write_tools: bool = False
        self.max_refreshes_per_query: int = 2

        # Phase 2.1.6 — fetch-on-miss ingestion controls.
        self.enable_fetch_on_miss: bool = True
        self.fetch_on_miss_timeout_s: float = 10.0
        self.fetch_on_miss_per_query: int = 1

        # Phase 2.2.2.1 — bounded client-owned conversation history. The
        # middleware stays stateless; these cap how much of the client-sent
        # history/question one request may use (see conversation.select_history).
        self.conversation_max_turns: int = 8
        self.conversation_max_history_chars: int = 8000
        self.conversation_max_question_chars: int = 16000

        # Phase 2.2.2.2 — follow-up rewriting & entity carryover. When
        # enable_conversation_rewrite is on and history is present, the current
        # turn + bounded history are compiled into a separate retrieval query
        # (the raw question is never changed). Off by default so behavior is
        # unchanged until the conversational gates pass. The LLM ambiguity
        # fallback is a further opt-in and makes at most one bounded model call.
        self.enable_conversation_rewrite: bool = False
        self.enable_llm_rewrite_fallback: bool = False
        self.conversation_rewrite_timeout_s: float = 15.0

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
        self._clamp_conversation_limits()

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
        _str("ANSWER_POLICY", "answer_policy")
        _bool("ALLOW_GENERAL_FALLBACK", "allow_general_fallback")
        _bool("RETURN_TIMINGS", "return_timings")
        _bool("ENABLE_STREAMING", "enable_streaming")
        _int("EMBEDDING_CACHE_SIZE", "embedding_cache_size")
        _int("CONVERSATION_MAX_TURNS", "conversation_max_turns")
        _int("CONVERSATION_MAX_HISTORY_CHARS", "conversation_max_history_chars")
        _int("CONVERSATION_MAX_QUESTION_CHARS", "conversation_max_question_chars")
        _bool("ENABLE_CONVERSATION_REWRITE", "enable_conversation_rewrite")
        _bool("ENABLE_LLM_REWRITE_FALLBACK", "enable_llm_rewrite_fallback")
        _float("CONVERSATION_REWRITE_TIMEOUT_S", "conversation_rewrite_timeout_s")

    def _clamp_conversation_limits(self) -> None:
        """Clamp conversation budgets to documented safe maxima (2.2.2.1).

        A single warning is logged if any value was out of range, rather than
        accepting an unbounded history/question payload.
        """
        clamped: list[str] = []

        def _clamp(attr: str, lo: int, hi: int) -> None:
            try:
                val = int(getattr(self, attr))
            except (TypeError, ValueError):
                val = hi
            bounded = max(lo, min(hi, val))
            if bounded != val:
                clamped.append(f"{attr}={val}->{bounded}")
            setattr(self, attr, bounded)

        _clamp("conversation_max_turns", 0, _MAX_CONVERSATION_TURNS_CEILING)
        _clamp("conversation_max_history_chars", 0, _MAX_HISTORY_CHARS_CEILING)
        _clamp("conversation_max_question_chars", 1, _MAX_QUESTION_CHARS_CEILING)

        if clamped:
            logger.warning(
                "Clamped conversation limits to safe maxima: %s", ", ".join(clamped))
