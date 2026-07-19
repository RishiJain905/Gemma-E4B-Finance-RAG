"""Offline contracts for Phase 2.3.7.4 middleware profiles."""

from __future__ import annotations

import builtins
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.middleware.config import MiddlewareConfig
from src.middleware.retriever import Retriever


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "configs" / "profiles"

PROFILE_FLAGS = (
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


CURRENT_SAFE_DEFAULTS = {
    "enable_citations": True,
    "allow_general_fallback": True,
    "enable_streaming": True,
    "enable_phase2_3_retrieval": True,
    "enable_phase2_3_corpus_projection": True,
    "enable_tool_final_streaming": True,
    "enable_stream_progress_events": True,
    "stream_progress_include_counts": True,
    "enable_graph_observer": True,
    "enable_tools": True,
    "allow_write_tools": True,
    # Promoted by the measured 2.3.7.4 arm1 comparison (2026-07-16).
    "enable_deterministic_tool_routing": True,
    "enable_deterministic_answers": True,
    "enable_adaptive_rag": True,
    "adaptive_enable_planning_call": False,
    "adaptive_conditional_rerank": True,
    "enable_evidence_sufficiency": False,
    "enable_corrective_retry": False,
    "enable_query_decomposition": False,
    "answer_validation": "report",
    "require_evidence_ids": False,
    "enable_hierarchical_retrieval": False,
    "enable_retrieval_cache": False,
    "llama_cache_prompt": False,
    "enable_fetch_on_miss": True,
    "enable_conversation_rewrite": False,
    "enable_llm_rewrite_fallback": False,
    "enable_lexical": True,
    "enable_reranker": False,
    "enable_evidence_taxonomy": True,
    "enable_authority_ranking": True,
    "enable_duplicate_coverage_packing": True,
}


def _profile_values(config: MiddlewareConfig) -> dict[str, object]:
    return {key: getattr(config, key) for key in PROFILE_FLAGS}


@pytest.fixture(autouse=True)
def _clear_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIDDLEWARE_PROFILE", raising=False)
    monkeypatch.delenv("SEC_INDEX_FILING_TEXT", raising=False)
    monkeypatch.delenv("SEC_CONFIG_PATH", raising=False)
    for key in (
        "ENABLE_DETERMINISTIC_ANSWERS",
        "ENABLE_DETERMINISTIC_TOOL_ROUTING",
        "ENABLE_ADAPTIVE_RAG",
        "ADAPTIVE_ENABLE_PLANNING_CALL",
        "ENABLE_EVIDENCE_SUFFICIENCY",
        "ENABLE_CORRECTIVE_RETRY",
        "ENABLE_HIERARCHICAL_RETRIEVAL",
        "ENABLE_CONVERSATION_REWRITE",
        "ENABLE_LLM_REWRITE_FALLBACK",
    ):
        monkeypatch.delenv(key, raising=False)


def test_all_profiles_have_exact_validated_effective_flags() -> None:
    expected = {
        "recommended": CURRENT_SAFE_DEFAULTS,
        "evaluation": {**CURRENT_SAFE_DEFAULTS},
        "legacy": {
            **CURRENT_SAFE_DEFAULTS,
            "enable_deterministic_tool_routing": False,
            "enable_deterministic_answers": False,
            "enable_adaptive_rag": False,
            "enable_phase2_3_retrieval": False,
            "enable_phase2_3_corpus_projection": False,
            "enable_tool_final_streaming": False,
            "enable_stream_progress_events": False,
            "enable_graph_observer": False,
            "adaptive_conditional_rerank": False,
            "answer_validation": "off",
            "enable_evidence_taxonomy": False,
            "enable_authority_ranking": False,
            "enable_duplicate_coverage_packing": False,
        },
    }

    for name, values in expected.items():
        config = MiddlewareConfig(config_path=PROFILE_DIR / f"{name}.yaml")
        assert config.profile == name
        assert _profile_values(config) == values


def test_committed_middleware_source_selects_recommended_profile() -> None:
    config = MiddlewareConfig()

    assert config.profile == "recommended"
    assert _profile_values(config) == CURRENT_SAFE_DEFAULTS


def test_evaluation_profile_adds_trace_metadata_without_promoting_features() -> None:
    config = MiddlewareConfig(config_path=PROFILE_DIR / "evaluation.yaml")

    assert config.profile_metadata == {
        "config_label": "evaluation",
        "include_evidence_trace": True,
        "include_progress_events": True,
    }
    assert _profile_values(config) == CURRENT_SAFE_DEFAULTS


def test_profile_key_loads_first_and_explicit_yaml_overrides_it(tmp_path: Path) -> None:
    path = tmp_path / "middleware.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "profile": "legacy",
                "enable_graph_observer": True,
            }
        ),
        encoding="utf-8",
    )

    config = MiddlewareConfig(config_path=path)

    assert config.profile == "legacy"
    assert config.enable_phase2_3_retrieval is False
    assert config.enable_graph_observer is True


def test_environment_profile_is_overridable_by_explicit_env_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIDDLEWARE_PROFILE", "legacy")
    monkeypatch.setenv("ENABLE_STREAM_PROGRESS_EVENTS", "1")

    config = MiddlewareConfig(config_path=Path("missing-middleware.yaml"))

    assert config.profile == "legacy"
    assert config.enable_stream_progress_events is True


def test_explicit_profile_argument_selects_complete_profile() -> None:
    config = MiddlewareConfig(profile="legacy")

    assert config.profile == "legacy"
    assert config.enable_phase2_3_retrieval is False
    assert config.enable_stream_progress_events is False


@pytest.mark.parametrize(
    ("key", "values", "clamped"),
    [
        (
            "enable_deterministic_answers",
            {"enable_deterministic_answers": True},
            "enable_deterministic_answers",
        ),
        (
            "enable_corrective_retry",
            {"enable_corrective_retry": True},
            "enable_corrective_retry",
        ),
        (
            "enable_hierarchical_retrieval",
            {"enable_hierarchical_retrieval": True},
            "enable_hierarchical_retrieval",
        ),
        (
            "adaptive_enable_planning_call",
            {"adaptive_enable_planning_call": True},
            "adaptive_enable_planning_call",
        ),
        (
            "enable_llm_rewrite_fallback",
            {"enable_llm_rewrite_fallback": True},
            "enable_llm_rewrite_fallback",
        ),
    ],
)
def test_invalid_explicit_yaml_dependencies_clamp_with_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    key: str,
    values: dict[str, object],
    clamped: str,
) -> None:
    path = tmp_path / "middleware.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="src.middleware.config"):
        config = MiddlewareConfig(config_path=path)

    assert getattr(config, key) is False
    assert clamped in caplog.text
    assert "dependency" in caplog.text.lower()


def test_invalid_profile_dependency_is_a_hard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "invalid-profile.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "profile": "invalid-profile",
                "enable_deterministic_answers": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MIDDLEWARE_PROFILE", str(path))

    with pytest.raises(ValueError, match="enable_deterministic_answers"):
        MiddlewareConfig(config_path=Path("missing-middleware.yaml"))


def test_env_dependency_violations_clamp_and_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ENABLE_DETERMINISTIC_ANSWERS", "1")
    monkeypatch.setenv("ENABLE_CORRECTIVE_RETRY", "1")
    monkeypatch.setenv("ADAPTIVE_ENABLE_PLANNING_CALL", "1")
    monkeypatch.setenv("ENABLE_LLM_REWRITE_FALLBACK", "1")
    monkeypatch.setenv("ENABLE_HIERARCHICAL_RETRIEVAL", "1")

    with caplog.at_level(logging.WARNING, logger="src.middleware.config"):
        config = MiddlewareConfig(config_path=Path("missing-middleware.yaml"))

    assert config.enable_deterministic_answers is False
    assert config.enable_corrective_retry is False
    assert config.adaptive_enable_planning_call is False
    assert config.enable_llm_rewrite_fallback is False
    assert config.enable_hierarchical_retrieval is False
    assert caplog.text.lower().count("dependency") >= 5


def test_recommended_profile_does_not_import_reranker_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    imported: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            imported.append(name)
            raise AssertionError("cross-encoder dependency imported while disabled")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    config = MiddlewareConfig(config_path=PROFILE_DIR / "recommended.yaml")
    Retriever(store=object(), config=config)

    assert config.enable_reranker is False
    assert imported == []
    assert "sentence_transformers" not in sys.modules


def test_citation_flag_controls_citation_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.middleware.app as middleware_app

    monkeypatch.setattr(
        middleware_app, "config", SimpleNamespace(enable_citations=False)
    )
    assert middleware_app._extract_citations("[Source: sec/NVDA]") == []

    monkeypatch.setattr(
        middleware_app, "config", SimpleNamespace(enable_citations=True)
    )
    assert len(middleware_app._extract_citations("[Source: sec/NVDA]")) == 1


def test_phase23_retrieval_flag_controls_postprocessing() -> None:
    config = SimpleNamespace(enable_phase2_3_retrieval=False)
    retriever = Retriever(store=object(), config=config)
    documents = [
        {"id": "second", "document": "same event", "metadata": {}},
        {"id": "first", "document": "same event", "metadata": {}},
    ]

    assert retriever._postprocess_documents(
        "event", documents, {}, limit=1
    ) == documents[:1]
