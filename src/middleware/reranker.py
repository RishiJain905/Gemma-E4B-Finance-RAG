"""
src/middleware/reranker.py
Cross-encoder / LLM re-ranker — re-scores retrieved documents against
the query and returns the top-N by relevance score.

Two backends are supported:
  - ``cross-encoder`` (default): a local SentenceTransformers ``CrossEncoder``
    (``cross-encoder/ms-marco-MiniLM-L-6-v2``). The model downloads on first
    use, so construction is cheap and loading is lazy.
  - ``llm``: an OpenAI-compatible chat endpoint (the local llama-server) is
    asked to score each (query, doc) pair 0.0–1.0.

The model is never loaded at construction time — ``_load_model`` runs on first
access via the ``model`` property, and any failure there propagates to
``rerank``, which falls back to the incoming (RRF-fused) order rather than
raising.

Usage:
    reranker = Reranker(config=config)
    top = reranker.rerank(query="What is NVDA revenue?", docs=docs, top_n=5)
    # Each returned doc carries a "rerank_score" key (float, or None on fallback).
"""

import logging
import re
import time
from typing import Optional

from .config import MiddlewareConfig

logger = logging.getLogger(__name__)

# System prompt for the LLM-judge backend. The model is a *thinking* model, so
# we give it ample max_tokens and ask it to end with an explicit ``Score:`` line
# that ``_parse_score`` can extract reliably.
_LLM_SYSTEM = (
    "You are a relevance judge. Score 0.0-1.0 how relevant the document is to "
    "the query. End with: Score: <float>"
)

# Extract an explicit ``Score: X`` (preferred) else fall back to the last
# number in the text. Values in [0, 1] are used directly; values in [2, 10] are
# treated as a 0–10 scale and divided by 10. Values in (1, 2) or > 10 are
# ambiguous and rejected (None). Mirrors eval/metrics._parse_score but kept
# standalone so the middleware does not depend on the eval package.
_SCORE_TAG_RE = re.compile(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[0-9]*\.?[0-9]+")


def _parse_score(text: Optional[str]) -> float | None:
    """Extract a 0–1 float from a judge reply, or None if unparseable.

    Accepts an explicit ``Score: X`` (preferred) or falls back to the last
    number in the text. Values in [0, 1] are used directly; values in [2, 10]
    are treated as a 0–10 scale and divided by 10. Values in (1, 2) are
    ambiguous (a 0–1 violation vs. a low 0–10 score) and rejected; values > 10
    are also rejected. The result is clamped to [0, 1].
    """
    if not text:
        return None
    m = _SCORE_TAG_RE.search(text)
    if m:
        raw = m.group(1)
    else:
        nums = _NUMBER_RE.findall(text)
        if not nums:
            return None
        raw = nums[-1]
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return None
    if 0.0 <= val <= 1.0:
        result = val
    elif 2.0 <= val <= 10.0:
        result = val / 10.0  # 0–10 scale
    else:
        return None  # (1, 2) ambiguous, or > 10
    return max(0.0, min(1.0, result))  # clamp to [0, 1]


class Reranker:
    """Cross-encoder / LLM re-ranker for the finance-RAG middleware.

    Construction is cheap (no model download). The cross-encoder loads lazily
    on first ``rerank`` via the ``model`` property; any load or scoring failure
    makes ``rerank`` fall back to the incoming document order with
    ``rerank_score=None`` rather than raising.
    """

    def __init__(self, config: Optional[MiddlewareConfig] = None, *,
                 backend: Optional[str] = None,
                 model_id: Optional[str] = None,
                 llm_endpoint: Optional[str] = None,
                 llm_model: str = "tracealchemy",
                 timeout: float = 60.0):
        """Configure the re-ranker.

        If ``config`` (a MiddlewareConfig) is given, backend / model_id /
        llm_endpoint / llm_model are read from it. Explicit non-None
        ``backend`` / ``model_id`` / ``llm_endpoint`` override the config
        values. Without a config, explicit args fall back to defaults:
        backend="cross-encoder",
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_endpoint="http://127.0.0.1:8087/v1/chat/completions".
        """
        if config is not None:
            self.backend = backend if backend is not None else config.reranker_backend
            self.model_id = model_id if model_id is not None else config.reranker_model
            self.llm_endpoint = (
                llm_endpoint if llm_endpoint is not None else config.llama_endpoint
            )
            self.llm_model = config.model_name
        else:
            self.backend = backend if backend is not None else "cross-encoder"
            self.model_id = (
                model_id if model_id is not None
                else "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            self.llm_endpoint = (
                llm_endpoint if llm_endpoint is not None
                else "http://127.0.0.1:8087/v1/chat/completions"
            )
            self.llm_model = llm_model  # "tracealchemy" by default

        self.backend = self.backend.lower()  # normalize
        self.timeout = timeout
        self._model = None  # lazy — loaded on first use

    # ── Model loading ─────────────────────────────────

    def _load_model(self):
        """Lazy-load the cross-encoder. RAISE on any failure so rerank can
        fall back. This is the seam tests patch to raise."""
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_id)

    @property
    def model(self):
        """The loaded cross-encoder, loading on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    # ── Scoring ───────────────────────────────────────

    def _score(self, query: str, docs: list[dict]) -> list[float]:
        """Backend scoring, returns floats aligned with ``docs``.

        This is the seam tests patch to return fixed scores.
        """
        if self.backend == "llm":
            return self._score_llm(query, docs)
        # cross-encoder (default)
        pairs = [(query, self._doc_text(d)) for d in docs]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

    def _score_llm(self, query: str, docs: list[dict]) -> list[float]:
        """Score each doc via the OpenAI-compatible chat endpoint.

        On any call failure (network / HTTP / parse), the doc gets 0.0 so a
        single bad request does not sink the whole rerank.
        """
        import httpx
        scores: list[float] = []
        for d in docs:
            text = self._doc_text(d)
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user", "content": f"Query: {query}\n\nDocument: {text}"},
                ],
                "temperature": 0.0,
                "max_tokens": 1024,  # thinking model — give it room to reason
            }
            try:
                r = httpx.post(self.llm_endpoint, json=payload, timeout=self.timeout)
                r.raise_for_status()
                content = (
                    r.json().get("choices", [{}])[0]
                    .get("message", {}).get("content", "")
                )
                s = _parse_score(content)
                scores.append(s if s is not None else 0.0)
            except Exception:
                scores.append(0.0)
        return scores

    # ── Public API ────────────────────────────────────

    def rerank(self, query: str, docs: list[dict],
               top_n: int = 5) -> list[dict]:
        """Re-rank ``docs`` by relevance to ``query``; return the top-N.

        Each returned doc is mutated in place to carry
        ``rerank_score`` (float). On any scoring failure, the incoming order
        is preserved (truncated to ``top_n``) with ``rerank_score=None``.
        NEVER raises.
        """
        if not docs:
            return []
        start = time.perf_counter()
        try:
            scores = self._score(query, docs)
            paired = list(zip(docs, scores))
            paired.sort(key=lambda pair: pair[1], reverse=True)
            result = []
            for d, s in paired[:top_n]:
                d["rerank_score"] = float(s)
                result.append(d)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info("rerank scored %d docs in %dms", len(docs), elapsed_ms)
            return result
        except Exception as e:
            logger.warning("reranker failed, using fused order: %s", e)
            for d in docs[:top_n]:
                d["rerank_score"] = None
            return docs[:top_n]

    def refresh(self):
        """Force the cross-encoder to reload on next use."""
        self._model = None

    # ── Helpers ───────────────────────────────────────

    @staticmethod
    def _doc_text(d: dict) -> str:
        """Return the document text from a doc dict.

        Tries ``document`` then ``text`` then ``content``; ``""`` if none.
        """
        return d.get("document") or d.get("text") or d.get("content") or ""