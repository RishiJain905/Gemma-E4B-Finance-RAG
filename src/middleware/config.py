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

# Documented safe bounds for the adaptive-RAG orchestrator (2.2.3.3). Values
# outside these ranges are clamped (with one warning) at load so a
# misconfiguration can never grant an unbounded subquery / retrieval-round /
# planning budget or an oversized context window.
_ADAPTIVE_MAX_SUBQUERIES_RANGE = (1, 3)
_ADAPTIVE_MAX_RETRIEVAL_ROUNDS_RANGE = (1, 2)
_ADAPTIVE_MAX_PLANNING_CALLS_RANGE = (0, 1)
_ADAPTIVE_MAX_CONTEXT_CHARS_RANGE = (1000, 64000)


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

        # Phase 2.2.3.2 — deterministic finance tool routing. When
        # enable_deterministic_tool_routing is on, safe analytical/comparison/
        # projection/calculation requests are routed to the existing read tools
        # before any model call (see deterministic_router.route). Off by default
        # so 2.2.3.4 can compare the route against the current tool loop first.
        # enable_deterministic_answers additionally lets a fully-covered route
        # skip model generation and answer from a template.
        self.enable_deterministic_tool_routing: bool = False
        self.max_deterministic_tools_per_query: int = 3
        self.enable_deterministic_answers: bool = False

        # Phase 2.2.3.3 — bounded adaptive-RAG orchestration. When
        # enable_adaptive_rag is on, 2.2.3.4 routes a request through
        # adaptive_orchestrator.orchestrate (fast/standard/complex lanes) under
        # one shared execution budget + one context budget, with conditional
        # re-ranking. Off by default; every adaptive stage falls soft to the
        # existing single-query Retriever.retrieve(). The optional pre-answer
        # planning model call is a further opt-in (adaptive_enable_planning_call).
        # Invalid limits are clamped to the documented safe range with one
        # warning (see _clamp_adaptive_limits). Env overrides mirror the pattern
        # below (ENABLE_ADAPTIVE_RAG, ADAPTIVE_*).
        self.enable_adaptive_rag: bool = False
        self.adaptive_enable_planning_call: bool = False
        self.adaptive_max_subqueries: int = 3          # clamp 1–3 (incl. sq0)
        self.adaptive_max_retrieval_rounds: int = 2    # clamp 1–2
        self.adaptive_max_planning_calls: int = 1      # clamp 0–1
        self.adaptive_max_context_chars: int = 18000   # hard context ceiling
        self.adaptive_conditional_rerank: bool = True

        # Phase 2.2.4.1 — deterministic evidence sufficiency and one bounded
        # internal corrective round. Both switches default off so the legacy
        # count-based answer path remains byte-identical until promotion.
        self.enable_evidence_sufficiency: bool = False
        self.enable_corrective_retry: bool = False
        self.max_corrective_retries: int = 1  # hard clamp 0–1

        # Phase 2.2.4.3 — citation provenance & deterministic numerical
        # validation. answer_validation selects the policy:
        #   off     — legacy behavior, byte-identical (no evidence ids, no
        #             validator, no extra response metadata);
        #   report  — render [E#] evidence ids, run the deterministic validator,
        #             attach validation metadata + log unsupported claims (no
        #             answer/grounding change);
        #   enforce — additionally downgrade grounded->partial (or refuse a
        #             wholly-unsupported answer) with a short support warning.
        # require_evidence_ids tightens enforcement during the rollout window so
        # a specific-figure answer with no resolving [E#] is treated as a
        # violation. The validator never makes a model call and always fails
        # soft to report_unavailable. Env: ANSWER_VALIDATION, REQUIRE_EVIDENCE_IDS.
        self.answer_validation: str = "report"  # off | report | enforce
        self.require_evidence_ids: bool = False

        # Phase 2.2.4.2 — selective query decomposition & weighted fusion. When
        # on (and the complex lane is reached), a genuinely compound / low-
        # coverage plan is decomposed into at most two derived, drift-validated
        # subqueries whose specialized evidence is retrieved in the bounded
        # corrective seam and fused with the original query as the strongest
        # signal. Off by default: a simple query never decomposes and the
        # RUN_DERIVED_SUBQUERIES corrective action stays the 2.2.4.1 deferred
        # placeholder, so legacy behavior is byte-identical until promotion.
        self.enable_query_decomposition: bool = False

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
        self._clamp_adaptive_limits()
        self._clamp_corrective_limits()
        self._normalize_answer_validation()

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
        _bool("ENABLE_DETERMINISTIC_TOOL_ROUTING", "enable_deterministic_tool_routing")
        _int("MAX_DETERMINISTIC_TOOLS_PER_QUERY", "max_deterministic_tools_per_query")
        _bool("ENABLE_DETERMINISTIC_ANSWERS", "enable_deterministic_answers")
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
        _bool("ENABLE_ADAPTIVE_RAG", "enable_adaptive_rag")
        _bool("ADAPTIVE_ENABLE_PLANNING_CALL", "adaptive_enable_planning_call")
        _int("ADAPTIVE_MAX_SUBQUERIES", "adaptive_max_subqueries")
        _int("ADAPTIVE_MAX_RETRIEVAL_ROUNDS", "adaptive_max_retrieval_rounds")
        _int("ADAPTIVE_MAX_PLANNING_CALLS", "adaptive_max_planning_calls")
        _int("ADAPTIVE_MAX_CONTEXT_CHARS", "adaptive_max_context_chars")
        _bool("ADAPTIVE_CONDITIONAL_RERANK", "adaptive_conditional_rerank")
        _bool("ENABLE_EVIDENCE_SUFFICIENCY", "enable_evidence_sufficiency")
        _bool("ENABLE_CORRECTIVE_RETRY", "enable_corrective_retry")
        _int("MAX_CORRECTIVE_RETRIES", "max_corrective_retries")
        _bool("ENABLE_QUERY_DECOMPOSITION", "enable_query_decomposition")
        _str("ANSWER_VALIDATION", "answer_validation")
        _bool("REQUIRE_EVIDENCE_IDS", "require_evidence_ids")

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

    def _clamp_adaptive_limits(self) -> None:
        """Clamp adaptive-RAG budgets to their documented safe ranges (2.2.3.3).

        A single warning is logged if any value was out of range, so a bad
        config can never grant an unbounded subquery/round/planning budget or an
        oversized context window. An invalid limit clamps toward the safe bound.
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

        _clamp("adaptive_max_subqueries", *_ADAPTIVE_MAX_SUBQUERIES_RANGE)
        _clamp("adaptive_max_retrieval_rounds", *_ADAPTIVE_MAX_RETRIEVAL_ROUNDS_RANGE)
        _clamp("adaptive_max_planning_calls", *_ADAPTIVE_MAX_PLANNING_CALLS_RANGE)
        _clamp("adaptive_max_context_chars", *_ADAPTIVE_MAX_CONTEXT_CHARS_RANGE)

        if clamped:
            logger.warning(
                "Clamped adaptive limits to safe ranges: %s", ", ".join(clamped))

    def _normalize_answer_validation(self) -> None:
        """Coerce answer_validation to off|report|enforce (invalid -> off)."""
        value = str(getattr(self, "answer_validation", "report") or "").strip().lower()
        if value not in ("off", "report", "enforce"):
            logger.warning(
                "Unknown answer_validation=%r; falling back to 'off'",
                self.answer_validation)
            value = "off"
        self.answer_validation = value
        self.require_evidence_ids = bool(getattr(self, "require_evidence_ids", False))

    def _clamp_corrective_limits(self) -> None:
        """Hard-clamp corrective retries to the documented zero-or-one range."""
        try:
            value = int(self.max_corrective_retries)
        except (TypeError, ValueError):
            value = 1
        bounded = max(0, min(1, value))
        if bounded != value:
            logger.warning(
                "Clamped max_corrective_retries to safe range: %s->%s",
                value, bounded,
            )
        self.max_corrective_retries = bounded
