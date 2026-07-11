"""
src/middleware/evidence_trace.py
Serializable evidence trace for evaluated answers — the exact system/user
prompts, usable facts/documents, and tool results shown to the model for one
request (2.2.1.2). Opt-in via ``QueryRequest.include_evidence_trace``; carries
no secrets, HTTP headers, or tool schemas — only what was model-visible.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass
class EvidenceTrace:
    """The exact evidence and prompts delivered to the model for one answer.

    ``facts``/``documents`` are exactly the usable retrieval rows (see
    ``src/middleware/evidence.py``) — full provenance, untruncated document
    bodies, matching what ``evidence_counts()`` reports for the same
    retrieval. ``tool_results`` records each dispatched tool call, in the
    order it was appended as a ``role=tool`` message: name, the validated
    (schema-filtered) arguments actually passed to the handler, and the exact
    JSON-serializable result.
    """

    answer_policy: str
    grounding_level: str
    raw_question: str
    retrieval_query: str
    system_prompt: str
    user_prompt: str
    facts: list[dict] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def is_complete(self) -> bool:
        """Structural completeness: a real prompt was captured.

        Facts/documents/tool_results may legitimately be empty (no usable
        evidence retrieved, no tools called) — only a missing system/user
        prompt marks an incomplete trace.
        """
        return bool(self.system_prompt) and bool(self.user_prompt)


class EvidenceTraceCollector:
    """Request-scoped builder for one :class:`EvidenceTrace`.

    Created once per request (in ``_build_query_context``, after evidence
    normalization) and threaded through the model-call / tool-loop via a
    context-local variable — never shared across concurrent requests. The
    prompt is recorded only for the path that actually produces the returned
    answer: a plain call, a tool-loop call, or the tools-unsupported
    fallback's final plain messages. ``record_prompt`` may be called more
    than once (e.g. a tool loop spanning several iterations); the last call
    before the answer is returned wins, so ``finalize()`` reflects exactly
    what the model saw for the successful path.
    """

    def __init__(
        self,
        *,
        answer_policy: str,
        grounding_level: str,
        raw_question: str,
        retrieval_query: str,
        facts: list[dict],
        documents: list[dict],
    ) -> None:
        self._answer_policy = answer_policy
        self._grounding_level = grounding_level
        self._raw_question = raw_question
        self._retrieval_query = retrieval_query
        self._facts = list(facts)
        self._documents = list(documents)
        self._system_prompt: Optional[str] = None
        self._user_prompt: Optional[str] = None
        self._tool_results: list[dict] = []

    def record_prompt(self, *, system_prompt: str, user_prompt: str) -> None:
        """Record the exact system/user messages for the successful answer path."""
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt

    def record_tool_result(self, name: str, arguments: dict, result: dict) -> None:
        """Append one dispatched tool call, in the order it was resolved."""
        self._tool_results.append({
            "name": name,
            "arguments": dict(arguments or {}),
            "result": result,
        })

    def discard_tool_results(self) -> None:
        """Drop any tool results recorded so far.

        Used when a tool-mode attempt is abandoned for a plain-call fallback
        (tools unsupported / empty tool-mode response) — the abandoned
        attempt's tool calls were never part of the successful answer path.
        """
        self._tool_results = []

    def finalize(self) -> Optional[EvidenceTrace]:
        """Return the completed trace, or ``None`` if no model call ever
        recorded a prompt (e.g. the degraded model-unavailable path, which
        must not claim model-visible evidence)."""
        if self._system_prompt is None or self._user_prompt is None:
            return None
        return EvidenceTrace(
            answer_policy=self._answer_policy,
            grounding_level=self._grounding_level,
            raw_question=self._raw_question,
            retrieval_query=self._retrieval_query,
            system_prompt=self._system_prompt,
            user_prompt=self._user_prompt,
            facts=list(self._facts),
            documents=list(self._documents),
            tool_results=list(self._tool_results),
        )
