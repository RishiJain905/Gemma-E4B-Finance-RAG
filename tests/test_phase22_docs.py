"""tests/test_phase22_docs.py
Static, offline validation of the Phase 2.2 documentation set (task 2.2.6.3).

These tests parse Markdown / YAML / the middleware config module only — they
start no services and touch no network, so they run under `-m "not live"` like
the rest of the offline suite. Their job is to keep the authoritative docs in
lockstep with the implemented Phase 2.2 behavior:

1. every Phase 2.2 task spec has the required section skeleton;
2. every local Markdown link inside `docs/phase2.2/` resolves;
3. every Phase 2.2.4-2.2.6 feature flag exists in the config source AND is
   documented in `docs/CONFIGURATION.md`, and vice versa (a flag documented in
   the middleware config section must be a real `MiddlewareConfig` attribute);
4. the new `configs/sec.yaml` / `configs/sec_companyfacts.yaml` keys are both
   present in their YAML and documented.
"""

import re
from pathlib import Path

import yaml

from src.middleware.config import MiddlewareConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE22_DIR = REPO_ROOT / "docs" / "phase2.2"
CONFIG_MD = REPO_ROOT / "docs" / "CONFIGURATION.md"

# A numbered task spec, e.g. "2.2.4.1-evidence-sufficiency-and-bounded-retry.md".
_TASK_FILE_RE = re.compile(r"^2\.2\.\d+\.\d+-.*\.md$")

# Required section headings every task spec must carry (Steps is matched
# separately because specs use "## Step 1: ..." headings).
_REQUIRED_SECTIONS = ("Objective", "Why This Matters", "Testing",
                      "Verification Checklist")

# Phase 2.2.4-2.2.6 feature flags that MUST be real MiddlewareConfig attributes
# and MUST be documented in docs/CONFIGURATION.md (task 2.2.6.3 contract).
PHASE_2246_FLAGS = (
    # 2.2.4.1 / 2.2.4.2 — evidence sufficiency, corrective retry, decomposition
    "enable_evidence_sufficiency",
    "enable_corrective_retry",
    "max_corrective_retries",
    "enable_query_decomposition",
    # 2.2.4.3 — citation provenance & numeric validation
    "answer_validation",
    "require_evidence_ids",
    # 2.2.5.3 — bounded hierarchical retrieval
    "enable_hierarchical_retrieval",
    "hierarchy_max_siblings",
    "hierarchy_max_adjacent_sections",
    "hierarchy_max_expanded_items",
    # 2.2.6.1 — tool-aware streaming & progress events
    "enable_tool_final_streaming",
    "enable_stream_progress_events",
    "stream_progress_include_counts",
    # 2.2.6.2 — versioned retrieval cache & prompt efficiency
    "enable_retrieval_cache",
    "retrieval_cache_max_entries",
    "retrieval_cache_ttl_s",
    "retrieval_cache_max_value_chars",
    "llama_cache_prompt",
)

# configs/sec.yaml keys (under `sec:`) that must be present + documented.
SEC_YAML_KEYS = ("index_filing_text", "max_sections_per_filing",
                 "max_section_chars", "index_forms")

# configs/sec_companyfacts.yaml top-level keys that must be present + documented.
SEC_COMPANYFACTS_KEYS = ("enabled", "user_agent", "allowed_taxonomies",
                         "allowed_forms", "timeout_seconds", "retries",
                         "backoff_factor", "request_delay_seconds", "metrics")


def _task_files() -> list[Path]:
    return sorted(p for p in PHASE22_DIR.rglob("*.md")
                  if _TASK_FILE_RE.match(p.name))


def _config_md_text() -> str:
    return CONFIG_MD.read_text(encoding="utf-8")


def _documented(md_text: str, token: str) -> bool:
    """True when ``token`` appears backtick-wrapped in the docs, allowing an
    optional dotted config prefix (e.g. ```sec.index_filing_text```)."""
    return re.search(r"`(?:[a-z_]+\.)?" + re.escape(token) + "`", md_text) is not None


def _iter_yaml_blocks(md_text: str, start: int, end: int):
    """Yield parsed ```yaml fenced blocks between char offsets [start, end)."""
    section = md_text[start:end]
    for match in re.finditer(r"```ya?ml\n(.*?)```", section, re.DOTALL):
        data = yaml.safe_load(match.group(1))
        if isinstance(data, dict):
            yield data


# ── 1. Task-spec structure ────────────────────────────────────────────

def test_phase22_task_files_exist():
    files = _task_files()
    # 2.2.1 – 2.2.7 each have their numbered specs; guard against an empty glob
    # (a moved directory would otherwise make the structure checks vacuous).
    assert len(files) >= 20, f"expected the full Phase 2.2 spec set, found {len(files)}"


def test_every_task_spec_has_required_sections():
    problems: list[str] = []
    for path in _task_files():
        text = path.read_text(encoding="utf-8")
        headings = re.findall(r"^#{2,}\s+(.*)$", text, re.MULTILINE)
        rel = path.relative_to(REPO_ROOT)
        for section in _REQUIRED_SECTIONS:
            if not any(h.strip() == section for h in headings):
                problems.append(f"{rel}: missing '## {section}' heading")
        # At least one Step heading ("## Step 1: ..." or "## Steps").
        if not re.search(r"(?im)^#{2,}\s+steps?\b", text):
            problems.append(f"{rel}: missing a Step heading")
    assert not problems, "task spec section gaps:\n" + "\n".join(problems)


# ── 2. Local Markdown links resolve ───────────────────────────────────

def test_local_markdown_links_resolve():
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    problems: list[str] = []
    for path in PHASE22_DIR.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in link_re.findall(text):
            target = target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            # Drop any anchor fragment before resolving the file.
            file_part = target.split("#", 1)[0]
            if not file_part:
                continue
            resolved = (path.parent / file_part).resolve()
            if not resolved.exists():
                problems.append(
                    f"{path.relative_to(REPO_ROOT)} -> {target} (missing)")
    assert not problems, "unresolved local Markdown links:\n" + "\n".join(problems)


# ── 3. Config-flag coverage (both directions) ─────────────────────────

def test_phase_2246_flags_are_real_config_attributes():
    config = MiddlewareConfig()
    missing = [f for f in PHASE_2246_FLAGS if not hasattr(config, f)]
    assert not missing, f"documented flags absent from MiddlewareConfig: {missing}"


def test_phase_2246_flags_are_documented():
    md = _config_md_text()
    undocumented = [f for f in PHASE_2246_FLAGS if not _documented(md, f)]
    assert not undocumented, (
        "Phase 2.2.4-2.2.6 flags missing from docs/CONFIGURATION.md: "
        f"{undocumented}")


def test_middleware_yaml_doc_keys_are_real_attributes():
    """Reverse guard: every key in a fenced YAML block under the
    `configs/middleware.yaml` doc section must be a real MiddlewareConfig
    attribute (catches a documented-but-nonexistent flag / a typo)."""
    md = _config_md_text()
    start = md.find("## `configs/middleware.yaml`")
    assert start != -1, "middleware.yaml section heading not found in CONFIGURATION.md"
    # End at the next top-level '## ' heading after the section starts.
    next_heading = re.search(r"\n## ", md[start + 3:])
    end = (start + 3 + next_heading.start()) if next_heading else len(md)

    config = MiddlewareConfig()
    problems: list[str] = []
    for block in _iter_yaml_blocks(md, start, end):
        for key in block:
            if not hasattr(config, key):
                problems.append(key)
    assert not problems, (
        "middleware.yaml doc keys with no MiddlewareConfig attribute: "
        f"{problems}")


# ── 4. New SEC config files: present in YAML AND documented ────────────

def test_sec_yaml_keys_present_and_documented():
    data = yaml.safe_load((REPO_ROOT / "configs" / "sec.yaml").read_text())
    sec = data.get("sec", {})
    md = _config_md_text()
    missing_yaml = [k for k in SEC_YAML_KEYS if k not in sec]
    undocumented = [k for k in SEC_YAML_KEYS if not _documented(md, k)]
    assert not missing_yaml, f"configs/sec.yaml missing keys: {missing_yaml}"
    assert not undocumented, f"sec.yaml keys undocumented: {undocumented}"


def test_sec_companyfacts_yaml_keys_present_and_documented():
    data = yaml.safe_load(
        (REPO_ROOT / "configs" / "sec_companyfacts.yaml").read_text())
    md = _config_md_text()
    missing_yaml = [k for k in SEC_COMPANYFACTS_KEYS if k not in data]
    undocumented = [k for k in SEC_COMPANYFACTS_KEYS if not _documented(md, k)]
    assert not missing_yaml, (
        f"configs/sec_companyfacts.yaml missing keys: {missing_yaml}")
    assert not undocumented, (
        f"sec_companyfacts.yaml keys undocumented: {undocumented}")
