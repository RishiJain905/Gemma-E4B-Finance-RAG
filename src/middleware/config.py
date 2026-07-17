"""
src/middleware/config.py
Middleware configuration — loaded from configs/ or environment.
"""

import logging
import os
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

# Safe query-graph observer budgets (2.2.7.4). The in-memory TraceHub also
# clamps these internally; enforcing them here keeps a bad config from ever
# requesting an unbounded trace/element/excerpt footprint for the local UI.
_GRAPH_TRACE_LIMIT_RANGE = (1, 1000)
_GRAPH_ELEMENT_LIMIT_RANGE = (1, 50000)
_GRAPH_TRACE_TTL_RANGE = (1, 86400)
_GRAPH_EXCERPT_CHARS_RANGE = (0, 10000)
_GRAPH_QUESTION_PREVIEW_CHARS_RANGE = (0, 2000)

# Safe corpus explorer budgets (2.2.7.2). These caps are also enforced by the
# API/projector so a bad configuration cannot overload a local graph client.
_CORPUS_PAGE_LIMIT_RANGE = (1, 200)
_CORPUS_DEFAULT_PAGE_LIMIT_RANGE = (1, 100)
_CORPUS_ELEMENT_LIMIT_RANGE = (1, 2000)
_CORPUS_VISIBLE_NODE_TARGET_RANGE = (1, 499)
_CORPUS_OVERVIEW_TTL_RANGE = (0.1, 60.0)
_CORPUS_OPAQUE_ID_TTL_RANGE = (1.0, 3600.0)
_CORPUS_INSPECTOR_EXCERPT_BYTES_RANGE = (0, 4000)
_CORPUS_INSPECTOR_METADATA_BYTES_RANGE = (0, 16000)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "middleware.yaml"
PROFILE_DIR = DEFAULT_CONFIG_PATH.parent / "profiles"
PROFILE_NAMES = ("legacy", "recommended", "evaluation")

# Feature flags are deliberately enumerated here so the configuration, profile,
# documentation, and test parity checks share one small source of truth. Numeric
# budgets remain ordinary settings and are not part of this inventory.
MIDDLEWARE_FEATURE_FLAGS = (
    "enable_citations",
    "allow_general_fallback",
    "enable_streaming",
    "enable_phase2_3_retrieval",
    "enable_phase2_3_corpus_projection",
    "enable_tool_final_streaming",
    "enable_stream_progress_events",
    "stream_progress_include_counts",
    "enable_graph_observer",
    "enable_tools",
    "allow_write_tools",
    "enable_deterministic_tool_routing",
    "enable_deterministic_answers",
    "enable_adaptive_rag",
    "adaptive_enable_planning_call",
    "adaptive_conditional_rerank",
    "enable_evidence_sufficiency",
    "enable_corrective_retry",
    "enable_query_decomposition",
    "answer_validation",
    "require_evidence_ids",
    "enable_hierarchical_retrieval",
    "enable_retrieval_cache",
    "llama_cache_prompt",
    "enable_fetch_on_miss",
    "enable_conversation_rewrite",
    "enable_llm_rewrite_fallback",
    "enable_lexical",
    "enable_reranker",
    "enable_evidence_taxonomy",
    "enable_authority_ranking",
    "enable_duplicate_coverage_packing",
)

MIDDLEWARE_ENV_OVERRIDES = {
    "enable_citations": "ENABLE_CITATIONS",
    "allow_general_fallback": "ALLOW_GENERAL_FALLBACK",
    "enable_streaming": "ENABLE_STREAMING",
    "enable_phase2_3_retrieval": "ENABLE_PHASE2_3_RETRIEVAL",
    "enable_phase2_3_corpus_projection": "ENABLE_PHASE2_3_CORPUS_PROJECTION",
    "enable_tool_final_streaming": "ENABLE_TOOL_FINAL_STREAMING",
    "enable_stream_progress_events": "ENABLE_STREAM_PROGRESS_EVENTS",
    "stream_progress_include_counts": "STREAM_PROGRESS_INCLUDE_COUNTS",
    "enable_graph_observer": "ENABLE_GRAPH_OBSERVER",
    "enable_tools": "ENABLE_TOOLS",
    "allow_write_tools": "ALLOW_WRITE_TOOLS",
    "enable_deterministic_tool_routing": "ENABLE_DETERMINISTIC_TOOL_ROUTING",
    "enable_deterministic_answers": "ENABLE_DETERMINISTIC_ANSWERS",
    "enable_adaptive_rag": "ENABLE_ADAPTIVE_RAG",
    "adaptive_enable_planning_call": "ADAPTIVE_ENABLE_PLANNING_CALL",
    "adaptive_conditional_rerank": "ADAPTIVE_CONDITIONAL_RERANK",
    "enable_evidence_sufficiency": "ENABLE_EVIDENCE_SUFFICIENCY",
    "enable_corrective_retry": "ENABLE_CORRECTIVE_RETRY",
    "enable_query_decomposition": "ENABLE_QUERY_DECOMPOSITION",
    "answer_validation": "ANSWER_VALIDATION",
    "require_evidence_ids": "REQUIRE_EVIDENCE_IDS",
    "enable_hierarchical_retrieval": "ENABLE_HIERARCHICAL_RETRIEVAL",
    "enable_retrieval_cache": "ENABLE_RETRIEVAL_CACHE",
    "llama_cache_prompt": "LLAMA_CACHE_PROMPT",
    "enable_fetch_on_miss": "ENABLE_FETCH_ON_MISS",
    "enable_conversation_rewrite": "ENABLE_CONVERSATION_REWRITE",
    "enable_llm_rewrite_fallback": "ENABLE_LLM_REWRITE_FALLBACK",
    "enable_lexical": "ENABLE_LEXICAL",
    "enable_reranker": "ENABLE_RERANKER",
    "enable_evidence_taxonomy": "ENABLE_EVIDENCE_TAXONOMY",
    "enable_authority_ranking": "ENABLE_AUTHORITY_RANKING",
    "enable_duplicate_coverage_packing": "ENABLE_DUPLICATE_COVERAGE_PACKING",
}


class MiddlewareConfig:
    """Configuration for the FastAPI middleware layer."""

    def __init__(self, config_path: Optional[Path] = None,
                 profile: Optional[str] = None):
        """Load defaults, an optional profile, explicit YAML, then env values.

        ``MIDDLEWARE_PROFILE`` selects a complete profile runtime source when no
        explicit ``config_path`` is supplied. A ``profile:`` key in an explicit
        YAML file loads that profile first, allowing the file to override it.
        Environment values always have the final say. Profile dependency
        violations are hard errors; post-profile overrides are clamped with a
        warning so A/B experiments cannot create an invalid runtime.
        """
        config_path = Path(config_path) if config_path is not None else None

        self.profile: Optional[str] = None
        self.profile_metadata: dict[str, object] = {}

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

        # Phase 2.3 additive rollout gates. Off preserves the Phase 2.2 query
        # and Corpus Explorer paths; stored evidence and schema remain intact.
        self.enable_phase2_3_retrieval: bool = False
        self.enable_phase2_3_corpus_projection: bool = False

        # Phase 2.2.6.1 — tool-aware streaming & progress events. Both feature
        # flags default off so behavior is byte-identical to the pre-2.2.6
        # streaming path until promotion:
        #   enable_tool_final_streaming  — allow /query/stream to serve a
        #       tools-enabled request by running the bounded tool/planning
        #       rounds non-streaming, then streaming ONLY the final answer.
        #       Off -> /query/stream still 404s while tools are enabled.
        #   enable_stream_progress_events — emit versioned, redacted pipeline
        #       stage / tool progress events on the SSE stream (query_started,
        #       stage, tool_started, tool_completed, error). Off -> only the
        #       legacy token/metadata events are sent.
        #   stream_progress_include_counts — include row/item counts on
        #       retrieve/tool_completed progress events (default on).
        # Env: ENABLE_TOOL_FINAL_STREAMING, ENABLE_STREAM_PROGRESS_EVENTS,
        # STREAM_PROGRESS_INCLUDE_COUNTS.
        self.enable_tool_final_streaming: bool = False
        self.enable_stream_progress_events: bool = False
        self.stream_progress_include_counts: bool = True

        # Phase 2.2.7.1 — local, read-only live retrieval graph. Disabled by
        # default. The in-memory TraceHub is bounded and never persists traces.
        # Question text is stored only as this bounded preview plus a SHA-256
        # digest. Env: ENABLE_GRAPH_OBSERVER, GRAPH_*.
        self.enable_graph_observer: bool = False
        self.graph_trace_limit: int = 100
        self.graph_element_limit: int = 5000
        self.graph_trace_ttl_s: int = 3600
        self.graph_excerpt_chars: int = 1000
        self.graph_question_preview_chars: int = 200

        # Phase 2.2.7.2 — Store-backed, read-only corpus explorer budgets.
        self.corpus_page_limit: int = 100
        self.corpus_default_page_limit: int = 50
        self.corpus_element_limit: int = 2000
        self.corpus_visible_node_target: int = 450
        self.corpus_overview_cache_ttl_s: float = 2.0
        self.corpus_opaque_id_ttl_s: float = 300.0
        # Phase 2.3.5.3 — inspector detail byte caps enforced server-side.
        self.corpus_inspector_excerpt_bytes: int = 1000
        self.corpus_inspector_metadata_bytes: int = 4000

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

        # Phase 2.2.5.3 — bounded hierarchical (filing → section → child) retrieval.
        # When enable_hierarchical_retrieval is on, a precise SEC child hit is
        # expanded with at most hierarchy_max_siblings same-section neighbors (to
        # complete a sentence/table) and at most hierarchy_max_adjacent_sections
        # adjacent sections (only on an obligation-heading match or a grader
        # missing-context signal) under the shared context character budget. An
        # entire filing parent is never injected. Off by default: the corrective
        # EXPAND_PARENT_SECTION seam keeps its 2.2.4.1 sibling-only behavior until
        # the long-document eval gates pass. hierarchy_max_expanded_items caps the
        # total added neighbors regardless of budget (0 = bounded only by the
        # context packer). Env: ENABLE_HIERARCHICAL_RETRIEVAL, HIERARCHY_MAX_*.
        self.enable_hierarchical_retrieval: bool = False
        self.hierarchy_max_siblings: int = 2
        self.hierarchy_max_adjacent_sections: int = 1
        self.hierarchy_max_expanded_items: int = 12

        # Phase 2.2.6.2 — versioned retrieval cache & prompt efficiency.
        #   enable_retrieval_cache — reuse a request's normalized PRE-PROMPT
        #     evidence (facts/documents + retrieval metadata, never a final
        #     answer) when an identical planned request recurs at the SAME store
        #     data revision. Keyed on the compiled query + validated plan +
        #     config/model fingerprint + store revision, so any ingestion write
        #     (including a same-count document replacement) invalidates it. Off by
        #     default -> byte-identical behavior; a cache miss/failure never fails
        #     a query. NO semantic final-answer cache exists — volatile finance
        #     answers (prices/news/estimates/"latest") must never be reused.
        #   retrieval_cache_ttl_s is a SECONDARY staleness bound below the
        #     revision key; retrieval_cache_max_value_chars refuses to store a
        #     single oversized evidence snapshot so memory stays bounded.
        #   llama_cache_prompt — when on AND the backend supports it, send
        #     llama-server's official prompt-reuse field (cache_prompt: true) on
        #     the final answer request and capture reused-token timings; on
        #     backend rejection it disables for the process and retries plain once
        #     (fail-soft capability behavior). Off -> the request JSON is unchanged.
        # Env: ENABLE_RETRIEVAL_CACHE, RETRIEVAL_CACHE_MAX_ENTRIES,
        # RETRIEVAL_CACHE_TTL_S, RETRIEVAL_CACHE_MAX_VALUE_CHARS, LLAMA_CACHE_PROMPT.
        self.enable_retrieval_cache: bool = False
        self.retrieval_cache_max_entries: int = 256
        self.retrieval_cache_ttl_s: float = 300.0
        self.retrieval_cache_max_value_chars: int = 200_000
        self.llama_cache_prompt: bool = False

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
        self.lexical_backend: str = "fts5"
        self.rrf_k: int = 60
        # Cross-encoder re-ranker (2.1.2.2). Opt-in by default — the
        # cross-encoder backend downloads a model on first use.
        self.enable_reranker: bool = False
        self.reranker_backend: str = "cross-encoder"  # "cross-encoder" | "llm"
        self.reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.rerank_candidates: int = 30   # broad retrieve, then re-rank
        self.rerank_top_n: int = 5         # final docs after re-rank

        # Phase 2.3.3.3 — additive stable evidence metadata plus bounded
        # post-relevance policy ranking/packing. Taxonomy normalization is
        # additive. Authority can add at most 0.025 and duplicate packing keeps
        # one primary plus at most two materially different secondaries.
        self.enable_evidence_taxonomy: bool = True
        self.enable_authority_ranking: bool = True
        self.enable_duplicate_coverage_packing: bool = True
        self.authority_max_boost: float = 0.025
        self.max_secondary_per_event: int = 2

        self._load_selected_sources(config_path, profile)
        self._apply_env_overrides()
        if self.lexical_backend not in {"fts5", "memory"}:
            logger.warning(
                "Invalid lexical_backend=%r; using fts5", self.lexical_backend
            )
            self.lexical_backend = "fts5"
        self._clamp_conversation_limits()
        self._clamp_adaptive_limits()
        self._clamp_corrective_limits()
        self._clamp_hierarchy_limits()
        self._clamp_graph_limits()
        self._clamp_corpus_limits()
        self._clamp_evidence_policy()
        self._normalize_answer_validation()
        self._validate_dependencies(strict=False)

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        with path.open(encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Middleware config must be a mapping: {path}")
        return data

    @staticmethod
    def _resolve_profile_path(profile: str | Path) -> Path:
        """Resolve a named profile or an explicit YAML path."""
        value = str(profile).strip()
        if not value:
            raise ValueError("Middleware profile name cannot be blank")

        candidate = Path(value)
        if candidate.is_absolute() or candidate.suffix in (".yaml", ".yml"):
            path = candidate if candidate.is_absolute() else Path.cwd() / candidate
        else:
            if value not in PROFILE_NAMES:
                raise ValueError(
                    f"Unknown middleware profile {value!r}; expected one of "
                    f"{', '.join(PROFILE_NAMES)}"
                )
            path = PROFILE_DIR / f"{value}.yaml"
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Middleware profile not found: {path}")
        return path

    def _apply_mapping(self, data: dict) -> None:
        metadata = data.get("profile_metadata")
        if isinstance(metadata, dict):
            self.profile_metadata.update(metadata)
        for key, value in data.items():
            if key in ("profile", "profile_metadata"):
                continue
            if hasattr(self, key):
                setattr(self, key, value)

    def _load_selected_sources(
        self, config_path: Optional[Path], requested_profile: Optional[str]
    ) -> None:
        """Apply profile values before an explicit YAML source."""
        explicit_path = config_path or DEFAULT_CONFIG_PATH
        explicit_data = (
            self._read_yaml(explicit_path) if explicit_path.exists() else {}
        )
        env_profile = os.environ.get("MIDDLEWARE_PROFILE")

        # Passing a file from configs/profiles means that file itself is the
        # profile source, even when it carries its own descriptive `profile:`
        # key. This keeps direct profile loading deterministic in tests/tools.
        is_profile_file = (
            config_path is not None
            and explicit_path.parent.resolve() == PROFILE_DIR.resolve()
        )
        profile_spec = requested_profile or env_profile
        if profile_spec is None:
            profile_spec = explicit_path if is_profile_file else explicit_data.get("profile")

        profile_path: Optional[Path] = None
        if profile_spec:
            profile_path = self._resolve_profile_path(profile_spec)
            profile_data = self._read_yaml(profile_path)
            self.profile = str(profile_data.get("profile") or profile_path.stem)
            self._apply_mapping(profile_data)
            # A profile is an operator-reviewed bundle. It must be internally
            # coherent before an explicit YAML/env experiment can override it.
            self._validate_dependencies(strict=True)

        # MIDDLEWARE_PROFILE with the default constructor is a complete runtime
        # selection. An explicitly supplied config_path remains an overlay, as
        # required for controlled tests and operator-local overrides.
        skip_default_for_selected_profile = (
            config_path is None and (env_profile is not None or requested_profile is not None)
        )
        if explicit_path.exists() and not skip_default_for_selected_profile:
            if profile_path is None or explicit_path.resolve() != profile_path.resolve():
                self._apply_mapping(explicit_data)

    def _load_from_file(self, path: Path):
        """Backward-compatible helper for callers that load one YAML file."""
        self._apply_mapping(self._read_yaml(path))

    def _sec_filing_text_indexed(self) -> bool:
        """Return whether SEC filing sections are configured for indexing."""
        env_value = os.environ.get("SEC_INDEX_FILING_TEXT")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "on")

        path = Path(os.environ.get(
            "SEC_CONFIG_PATH",
            str(DEFAULT_CONFIG_PATH.parent / "sec.yaml"),
        ))
        if not path.exists():
            return False
        try:
            data = self._read_yaml(path)
        except (OSError, ValueError, yaml.YAMLError):
            return False
        return bool((data.get("sec") or {}).get("index_filing_text", False))

    def _validate_dependencies(self, *, strict: bool) -> None:
        """Validate capability prerequisites, hard-failing profiles or clamping.

        The returned runtime always satisfies these relationships. Explicit
        profiles fail before overrides are applied; env/config experiments clamp
        only the dependent capability and emit a warning.
        """
        violations: list[tuple[str, str]] = []
        if self.enable_deterministic_answers:
            if not self.enable_deterministic_tool_routing:
                violations.append((
                    "enable_deterministic_answers",
                    "enable_deterministic_tool_routing",
                ))
            if not self.enable_adaptive_rag:
                violations.append(("enable_deterministic_answers", "enable_adaptive_rag"))
        if self.enable_corrective_retry and not self.enable_evidence_sufficiency:
            violations.append(("enable_corrective_retry", "enable_evidence_sufficiency"))
        if self.enable_hierarchical_retrieval and not self._sec_filing_text_indexed():
            violations.append((
                "enable_hierarchical_retrieval",
                "sec.index_filing_text",
            ))
        if self.adaptive_enable_planning_call and not self.enable_adaptive_rag:
            violations.append(("adaptive_enable_planning_call", "enable_adaptive_rag"))
        if self.enable_llm_rewrite_fallback and not self.enable_conversation_rewrite:
            violations.append((
                "enable_llm_rewrite_fallback",
                "enable_conversation_rewrite",
            ))

        if not violations:
            return

        details = "; ".join(f"{flag} requires {dependency}" for flag, dependency in violations)
        if strict:
            raise ValueError(f"Invalid middleware profile dependencies: {details}")

        for flag, dependency in violations:
            setattr(self, flag, False)
            logger.warning(
                "Clamped dependency violation: %s requires %s; setting %s=false",
                flag,
                dependency,
                flag,
            )

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
        _str("LEXICAL_BACKEND", "lexical_backend")
        _bool("ENABLE_CITATIONS", "enable_citations")
        _bool("ENABLE_RERANKER", "enable_reranker")
        _str("RERANKER_BACKEND", "reranker_backend")
        _str("RERANKER_MODEL", "reranker_model")
        _int("RERANK_CANDIDATES", "rerank_candidates")
        _int("RERANK_TOP_N", "rerank_top_n")
        _bool("ENABLE_EVIDENCE_TAXONOMY", "enable_evidence_taxonomy")
        _bool("ENABLE_AUTHORITY_RANKING", "enable_authority_ranking")
        _bool(
            "ENABLE_DUPLICATE_COVERAGE_PACKING",
            "enable_duplicate_coverage_packing",
        )
        _float("AUTHORITY_MAX_BOOST", "authority_max_boost")
        _int("MAX_SECONDARY_PER_EVENT", "max_secondary_per_event")
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
        _bool("ENABLE_TOOL_FINAL_STREAMING", "enable_tool_final_streaming")
        _bool("ENABLE_STREAM_PROGRESS_EVENTS", "enable_stream_progress_events")
        _bool("STREAM_PROGRESS_INCLUDE_COUNTS", "stream_progress_include_counts")
        _bool("ENABLE_GRAPH_OBSERVER", "enable_graph_observer")
        _bool("ENABLE_PHASE2_3_RETRIEVAL", "enable_phase2_3_retrieval")
        _bool(
            "ENABLE_PHASE2_3_CORPUS_PROJECTION",
            "enable_phase2_3_corpus_projection",
        )
        _int("GRAPH_TRACE_LIMIT", "graph_trace_limit")
        _int("GRAPH_ELEMENT_LIMIT", "graph_element_limit")
        _int("GRAPH_TRACE_TTL_S", "graph_trace_ttl_s")
        _int("GRAPH_EXCERPT_CHARS", "graph_excerpt_chars")
        _int("GRAPH_QUESTION_PREVIEW_CHARS", "graph_question_preview_chars")
        _int("CORPUS_PAGE_LIMIT", "corpus_page_limit")
        _int("CORPUS_DEFAULT_PAGE_LIMIT", "corpus_default_page_limit")
        _int("CORPUS_ELEMENT_LIMIT", "corpus_element_limit")
        _int("CORPUS_VISIBLE_NODE_TARGET", "corpus_visible_node_target")
        _int("CORPUS_INSPECTOR_EXCERPT_BYTES", "corpus_inspector_excerpt_bytes")
        _int("CORPUS_INSPECTOR_METADATA_BYTES", "corpus_inspector_metadata_bytes")
        _float("CORPUS_OVERVIEW_CACHE_TTL_S", "corpus_overview_cache_ttl_s")
        _float("CORPUS_OPAQUE_ID_TTL_S", "corpus_opaque_id_ttl_s")
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
        _bool("ENABLE_HIERARCHICAL_RETRIEVAL", "enable_hierarchical_retrieval")
        _int("HIERARCHY_MAX_SIBLINGS", "hierarchy_max_siblings")
        _int("HIERARCHY_MAX_ADJACENT_SECTIONS", "hierarchy_max_adjacent_sections")
        _int("HIERARCHY_MAX_EXPANDED_ITEMS", "hierarchy_max_expanded_items")
        _str("ANSWER_VALIDATION", "answer_validation")
        _bool("REQUIRE_EVIDENCE_IDS", "require_evidence_ids")
        _bool("ENABLE_RETRIEVAL_CACHE", "enable_retrieval_cache")
        _int("RETRIEVAL_CACHE_MAX_ENTRIES", "retrieval_cache_max_entries")
        _float("RETRIEVAL_CACHE_TTL_S", "retrieval_cache_ttl_s")
        _int("RETRIEVAL_CACHE_MAX_VALUE_CHARS", "retrieval_cache_max_value_chars")
        _bool("LLAMA_CACHE_PROMPT", "llama_cache_prompt")

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

    def _clamp_hierarchy_limits(self) -> None:
        """Clamp hierarchical-expansion caps to documented safe ranges (2.2.5.3).

        Bounds the per-hit sibling / adjacent-section fan-out and the total
        expanded-item cap so a misconfiguration can never reconstruct an entire
        filing. An invalid value clamps toward the safe bound with one warning.
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

        _clamp("hierarchy_max_siblings", 0, 4)
        _clamp("hierarchy_max_adjacent_sections", 0, 3)
        _clamp("hierarchy_max_expanded_items", 0, 50)

        if clamped:
            logger.warning(
                "Clamped hierarchy limits to safe ranges: %s", ", ".join(clamped))

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

    def _clamp_graph_limits(self) -> None:
        """Clamp query-graph observer budgets to documented safe ranges (2.2.7.4).

        A single warning is logged if any value was out of range so a bad config
        can never request an unbounded trace/element/excerpt footprint for the
        local read-only UI. The TraceHub clamps again defensively at construction.
        """
        clamped: list[str] = []

        def _clamp(attr: str, bounds: tuple[int, int]) -> None:
            lo, hi = bounds
            try:
                value = int(getattr(self, attr))
            except (TypeError, ValueError):
                value = hi
            bounded = max(lo, min(hi, value))
            if bounded != value:
                clamped.append(f"{attr}={value}->{bounded}")
            setattr(self, attr, bounded)

        _clamp("graph_trace_limit", _GRAPH_TRACE_LIMIT_RANGE)
        _clamp("graph_element_limit", _GRAPH_ELEMENT_LIMIT_RANGE)
        _clamp("graph_trace_ttl_s", _GRAPH_TRACE_TTL_RANGE)
        _clamp("graph_excerpt_chars", _GRAPH_EXCERPT_CHARS_RANGE)
        _clamp("graph_question_preview_chars", _GRAPH_QUESTION_PREVIEW_CHARS_RANGE)
        if clamped:
            logger.warning(
                "Clamped graph observer limits to safe ranges: %s", ", ".join(clamped))

    def _clamp_corpus_limits(self) -> None:
        """Clamp corpus explorer budgets to documented safe ranges."""
        clamped: list[str] = []

        def _clamp_int(attr: str, bounds: tuple[int, int]) -> None:
            lo, hi = bounds
            try:
                value = int(getattr(self, attr))
            except (TypeError, ValueError):
                value = hi
            bounded = max(lo, min(hi, value))
            if bounded != value:
                clamped.append(f"{attr}={value}->{bounded}")
            setattr(self, attr, bounded)

        def _clamp_float(attr: str, bounds: tuple[float, float]) -> None:
            lo, hi = bounds
            try:
                value = float(getattr(self, attr))
            except (TypeError, ValueError):
                value = hi
            bounded = max(lo, min(hi, value))
            if bounded != value:
                clamped.append(f"{attr}={value}->{bounded}")
            setattr(self, attr, bounded)

        _clamp_int("corpus_page_limit", _CORPUS_PAGE_LIMIT_RANGE)
        _clamp_int("corpus_default_page_limit", _CORPUS_DEFAULT_PAGE_LIMIT_RANGE)
        _clamp_int("corpus_element_limit", _CORPUS_ELEMENT_LIMIT_RANGE)
        _clamp_int("corpus_visible_node_target", _CORPUS_VISIBLE_NODE_TARGET_RANGE)
        _clamp_int(
            "corpus_inspector_excerpt_bytes", _CORPUS_INSPECTOR_EXCERPT_BYTES_RANGE)
        _clamp_int(
            "corpus_inspector_metadata_bytes", _CORPUS_INSPECTOR_METADATA_BYTES_RANGE)
        _clamp_float("corpus_overview_cache_ttl_s", _CORPUS_OVERVIEW_TTL_RANGE)
        _clamp_float("corpus_opaque_id_ttl_s", _CORPUS_OPAQUE_ID_TTL_RANGE)
        # The default page cannot exceed the hard maximum page size.
        if self.corpus_default_page_limit > self.corpus_page_limit:
            self.corpus_default_page_limit = self.corpus_page_limit
        if clamped:
            logger.warning(
                "Clamped corpus explorer limits to safe ranges: %s", ", ".join(clamped))

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

    def _clamp_evidence_policy(self) -> None:
        """Clamp Phase 2.3 authority and event-packing policy bounds."""
        try:
            authority = float(self.authority_max_boost)
        except (TypeError, ValueError):
            authority = 0.025
        self.authority_max_boost = max(0.0, min(0.025, authority))
        try:
            secondary = int(self.max_secondary_per_event)
        except (TypeError, ValueError):
            secondary = 2
        self.max_secondary_per_event = max(0, min(5, secondary))
