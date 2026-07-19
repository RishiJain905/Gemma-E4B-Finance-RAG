"""src/middleware/lexical_index.py
Persistent FTS5 lexical retrieval with a selectable legacy memory backend.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
MAX_QUERY_TERMS = 32
MAX_TERM_CHARS = 64
MAX_COMPILED_QUERY_CHARS = 2048


class _FallbackBM25:
    """Small dependency fallback used only by the explicit memory backend."""

    def __init__(self, corpus: list[list[str]]):
        self.documents = [Counter(document) for document in corpus]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        frequencies: Counter[str] = Counter()
        for document in self.documents:
            frequencies.update(document.keys())
        total = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores: list[float] = []
        for document, length in zip(self.documents, self.lengths):
            score = 0.0
            for term in query:
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * length / max(1.0, self.average_length)
                )
                score += self.idf.get(term, 0.0) * frequency * 2.5 / denominator
            scores.append(score)
        return scores


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumerics, keeping tickers intact."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def compile_fts_query(text: str) -> Optional[str]:
    """Compile normalized terms into bounded FTS syntax without raw fragments."""
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokenize(text):
        token = token[:MAX_TERM_CHARS]
        if not token or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= MAX_QUERY_TERMS:
            break
    if not terms:
        return None
    compiled = " OR ".join(f'"{term}"' for term in terms)
    return compiled[:MAX_COMPILED_QUERY_CHARS]


class LexicalIndex:
    """Persistent FTS5 or legacy in-memory lexical retrieval channel."""

    def __init__(self, store, backend: str = "memory"):
        if backend not in {"fts5", "memory"}:
            raise ValueError("lexical backend must be 'fts5' or 'memory'")
        self.store = store
        self.backend = backend
        self.mode = backend
        self.last_status: dict[str, object] = {
            "mode": backend, "degraded": False, "reason": None,
        }
        self._bm25 = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._token_sets: list[set[str]] = []
        self._count_at_build: Optional[int] = None

    # -- legacy memory backend ------------------------------------------------

    def _build(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            BM25Okapi = _FallbackBM25

        ids, texts, metas = self.store.chroma.iter_documents()
        self._ids, self._texts, self._metas = ids, texts, metas
        tokenized = [tokenize(text) for text in texts]
        self._token_sets = [set(tokens) for tokens in tokenized]
        if not tokenized or not any(tokenized):
            self._bm25 = None
            self._count_at_build = 0
            logger.warning("LexicalIndex: empty/blank corpus, BM25 disabled")
            return
        self._bm25 = BM25Okapi(tokenized)
        try:
            self._count_at_build = self.store.chroma.count()
        except Exception:  # noqa: BLE001
            self._count_at_build = len(ids)
        logger.info("Legacy lexical index built over %d documents", len(ids))

    def _ensure(self) -> None:
        if self._bm25 is None:
            self._build()
            return
        try:
            current = self.store.chroma.count()
        except Exception:  # noqa: BLE001
            return
        if self._count_at_build is None or current != self._count_at_build:
            self._build()

    def refresh(self) -> None:
        """Refresh only the selected backend; FTS5 is maintained transactionally."""
        if self.backend == "fts5":
            self.warm()
            return
        self._bm25 = None
        self._build()

    def warm(self) -> None:
        """Probe FTS5 state or build the explicitly selected memory backend."""
        if self.backend == "memory":
            self._ensure()
            self.mode = "memory"
            return
        try:
            if not self.store.sqlite.fts5_available():
                self.mode = "degraded"
                self.last_status = {
                    "mode": "degraded", "degraded": True,
                    "reason": "fts5_unavailable",
                }
                return
            self.store.sqlite.get_lexical_index_state()
            self.mode = "fts5"
            self.last_status = {"mode": "fts5", "degraded": False, "reason": None}
        except Exception:  # noqa: BLE001 - startup must remain available
            self.mode = "degraded"
            self.last_status = {
                "mode": "degraded", "degraded": True,
                "reason": "fts5_unavailable",
            }

    @property
    def ready(self) -> bool:
        return self.mode == "fts5" or self._bm25 is not None

    # -- persistent FTS5 backend ---------------------------------------------

    def _chroma_revision(self) -> Optional[int]:
        getter = getattr(self.store.chroma, "corpus_revision", None)
        if callable(getter):
            value = getter()
            return None if value is None else int(value)
        collection = getattr(self.store.chroma, "collection", None)
        metadata = getattr(collection, "metadata", None) or {}
        value = metadata.get("corpus_revision") if isinstance(metadata, dict) else None
        return None if value is None else int(value)

    def _search_fts5(
        self,
        query: str,
        k: int,
        where: Optional[dict],
        filters: Optional[dict],
        corpus_revision: Optional[int],
    ) -> list[dict]:
        compiled = compile_fts_query(query)
        if compiled is None:
            return []
        revision_reader = getattr(self.store, "corpus_revision", None)
        if not callable(revision_reader):
            revision_reader = getattr(self.store, "retrieval_revision")
        revision = (
            int(corpus_revision)
            if corpus_revision is not None else int(revision_reader())
        )
        try:
            state = self.store.sqlite.get_lexical_index_state()
            chroma_revision = self._chroma_revision()
            if (
                int(state.get("indexed_revision", 0)) != revision
                or chroma_revision != revision
            ):
                self.last_status = {
                    "mode": "degraded", "degraded": True,
                    "reason": "revision_mismatch", "corpus_revision": revision,
                    "indexed_revision": int(state.get("indexed_revision", 0)),
                    "chroma_revision": chroma_revision,
                }
                return []
            ranked = self.store.sqlite.search_lexical(
                compiled, limit=k, revision=revision, where=where, filters=filters,
            )
            ids = [row["chunk_id"] for row in ranked]
            if not ids:
                self.last_status = {
                    "mode": "fts5", "degraded": False, "reason": None,
                    "corpus_revision": revision,
                }
                return []
            hydrated = self.store.chroma.get_documents(ids)
            by_id = {str(row.get("id")): row for row in hydrated}
            results: list[dict] = []
            for ranked_row in ranked:
                row = by_id.get(str(ranked_row["chunk_id"]))
                if row is None:
                    continue
                item = dict(row)
                item["score"] = float(ranked_row["score"])
                results.append(item)
            self.last_status = {
                "mode": "fts5", "degraded": False, "reason": None,
                "corpus_revision": revision,
            }
            return results
        except Exception as exc:  # noqa: BLE001 - lexical never fails a query
            logger.warning("FTS5 lexical query failed; using dense results: %s", exc)
            self.mode = "degraded"
            self.last_status = {
                "mode": "degraded", "degraded": True,
                "reason": "fts_query_error", "corpus_revision": revision,
            }
            return []

    # -- public search --------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        where: Optional[dict] = None,
        filters: Optional[dict] = None,
        corpus_revision: Optional[int] = None,
    ) -> list[dict]:
        """Return hard-bounded lexical hits shaped like Chroma search results."""
        if k < 1:
            return []
        k = min(int(k), 200)
        if self.backend == "fts5":
            if self.mode != "fts5":
                self.warm()
            if self.mode != "fts5":
                return []
            return self._search_fts5(query, k, where, filters, corpus_revision)

        self._ensure()
        if self._bm25 is None or not self._ids:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(list(query_tokens))
        ranked = sorted(enumerate(scores), key=lambda item: -item[1])
        output: list[dict] = []
        for index, score in ranked:
            document_tokens = (
                self._token_sets[index] if index < len(self._token_sets) else set()
            )
            if not (document_tokens & query_tokens):
                continue
            metadata = self._metas[index] if index < len(self._metas) else {}
            if where and not all(metadata.get(key) == value for key, value in where.items()):
                continue
            row = {
                "id": self._ids[index],
                "document": self._texts[index] if index < len(self._texts) else "",
                "metadata": metadata,
                "score": float(score),
            }
            if filters:
                try:
                    from .evidence_taxonomy import evidence_matches_filters, normalize_evidence

                    if not evidence_matches_filters(row, filters):
                        continue
                    row = normalize_evidence(row)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Evidence filter failed in lexical search: %s", exc)
            output.append(row)
            if len(output) >= k:
                break
        return output
