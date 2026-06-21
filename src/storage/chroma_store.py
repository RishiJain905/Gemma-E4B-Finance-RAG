"""
src/storage/chroma_store.py
ChromaDB vector store for document embeddings.
"""

import logging
from pathlib import Path
from typing import Optional
import httpx
import numpy as np
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class TraceAlchemyEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that calls your local llama-server
    /v1/embeddings endpoint with --pooling mean enabled.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:8087/v1/embeddings",
                 model: str = "tracealchemy",
                 batch_size: int = 10,
                 timeout: int = 60):
        self.endpoint = endpoint
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB calls this with a list of text documents.
        Returns a list of embedding vectors (list of floats).
        """
        # Process in batches to avoid overloading the server
        all_embeddings = []

        for i in range(0, len(input), self.batch_size):
            batch = input[i:i + self.batch_size]

            payload = {
                "input": batch if len(batch) > 1 else batch[0],
                "model": self.model,
            }

            response = self._client.post(self.endpoint, json=payload)
            response.raise_for_status()

            data = response.json()

            # llama-server returns: {"data": [{"embedding": [...], ...}, ...]}
            for item in data["data"]:
                all_embeddings.append(item["embedding"])

        return all_embeddings

    def __del__(self):
        if hasattr(self, "_client"):
            self._client.close()


class ChromaStore:
    """
    Wraps ChromaDB collection operations.
    Provides add_document, search, and delete methods.
    """

    DEFAULT_PATH = Path(__file__).parent.parent.parent / "data/chroma"

    # Documents longer than CHUNK_CHARS are split into overlapping windows so
    # each embedded sequence stays well under the llama-server batch limit.
    # With the default embedding batch_size of 10, a 1000-char window is
    # ~630 tokens worst case (~1.58 chars/token for dense filings), so a full
    # request is ~6.3K tokens — comfortably below the 8192 ubatch ceiling.
    DEFAULT_CHUNK_CHARS = 1000
    DEFAULT_CHUNK_OVERLAP = 150

    def __init__(self,
                 persist_directory: Optional[Path] = None,
                 collection_name: str = "tracealchemy_docs",
                 embedding_endpoint: str = "http://127.0.0.1:8087/v1/embeddings",
                 chunk_chars: int = DEFAULT_CHUNK_CHARS,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):

        self.persist_directory = persist_directory or self.DEFAULT_PATH
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap

        # Create embedding function
        self.embedding_fn = TraceAlchemyEmbeddingFunction(
            endpoint=embedding_endpoint
        )

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for finance
        )

    # ── CRUD Operations ──────────────────────────────

    def add_document(self,
                     document_id: str,
                     text: str,
                     metadata: Optional[dict] = None,
                     ticker: Optional[str] = None,
                     source: Optional[str] = None,
                     date: Optional[str] = None):
        """
        Add a single document to the vector store.

        Args:
            document_id: Unique identifier (e.g., 'filings/NVDA/10-Q-2026-Q1')
            text: The document content to embed and store
            metadata: Additional metadata dict
            ticker: Stock ticker (added to metadata)
            source: Source type (sec, yfinance, gdelt, etc.)
            date: Document date (ISO format)
        """
        meta = metadata or {}
        if ticker:
            meta["ticker"] = ticker.upper()
        if source:
            meta["source"] = source
        if date:
            meta["date"] = date

        chunks = self._chunk_text(text, self.chunk_chars, self.chunk_overlap)

        # Short documents are stored as a single entry under their original id,
        # preserving the existing id scheme and avoiding the embedder's batch limit.
        if len(chunks) <= 1:
            self.collection.add(
                documents=[text],
                metadatas=[meta],
                ids=[document_id]
            )
            return

        # Long documents are split so each embedded chunk fits the batch limit
        # and retrieval stays granular. Each chunk is its own entry: "{id}#{i}".
        total = len(chunks)
        ids = [f"{document_id}#{i}" for i in range(total)]
        metadatas = []
        for i in range(total):
            chunk_meta = dict(meta)
            chunk_meta["parent_id"] = document_id
            chunk_meta["chunk_index"] = i
            chunk_meta["chunk_count"] = total
            metadatas.append(chunk_meta)

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    @staticmethod
    def _chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
        """Split text into overlapping windows, breaking on whitespace when possible.

        Returns a single-element list for text at or below the chunk size,
        and an empty list for empty input.
        """
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= chunk_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_chars, n)
            # Prefer a whitespace boundary near the window end to avoid mid-word cuts.
            if end < n:
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            nxt = max(end - overlap, start + 1)
            # Snap the overlap start forward to a word boundary so the next
            # chunk doesn't begin mid-word.
            if nxt < n and not text[nxt - 1].isspace():
                space = text.find(" ", nxt)
                if space == -1:
                    break
                nxt = space + 1
            start = nxt
        return chunks

    def add_documents_batch(self,
                            ids: list[str],
                            texts: list[str],
                            metadatas: Optional[list[dict]] = None):
        """
        Add multiple documents at once (more efficient).

        Args:
            ids: List of unique document IDs
            texts: List of document texts
            metadatas: List of metadata dicts (same length as ids)
        """
        self.collection.add(
            documents=texts,
            metadatas=metadatas or [{}] * len(ids),
            ids=ids
        )

    def search(self, query: str, n_results: int = 5,
               filter_dict: Optional[dict] = None) -> list[dict]:
        """
        Search for documents semantically similar to the query.

        Args:
            query: Natural language query string
            n_results: Number of results to return
            filter_dict: Optional metadata filter (e.g., {"ticker": "NVDA"})

        Returns:
            List of dicts with: id, document, metadata, distance
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )

        # Format results into a cleaner structure
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })

        return formatted

    def get_document(self, document_id: str) -> Optional[dict]:
        """Retrieve a document by its ID."""
        results = self.collection.get(ids=[document_id])
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0] if results["documents"] else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else None,
            }
        return None

    def delete_document(self, document_id: str):
        """Remove a document from the collection."""
        self.collection.delete(ids=[document_id])

    def count(self) -> int:
        """How many documents are in the collection?"""
        return self.collection.count()

    def iter_documents(self, where: Optional[dict] = None,
                       limit: Optional[int] = None) -> tuple[list[str], list[str], list[dict]]:
        """Return the full corpus as (ids, texts, metadatas).

        Used by the BM25 lexical index (Phase 2.1.2.1) to build a keyword index
        over the same documents ChromaDB stores. ``where`` filters by metadata
        (e.g. {"ticker": "NVDA"}); ``limit`` caps the result count.
        """
        kwargs: dict = {"include": ["documents", "metadatas"]}
        if where is not None:
            kwargs["where"] = where
        if limit is not None:
            kwargs["limit"] = limit
        results = self.collection.get(**kwargs)
        ids = list(results.get("ids") or [])
        texts = list(results.get("documents") or [])
        metas = list(results.get("metadatas") or [])
        # Guard against a missing documents/metadata slot (shouldn't happen, but
        # keep the three lists aligned in length).
        n = len(ids)
        if len(texts) < n:
            texts += [""] * (n - len(texts))
        if len(metas) < n:
            metas += [{}] * (n - len(metas))
        return ids, texts, metas

    # ── Ticker-Specific Operations ────────────────────

    def search_by_ticker(self, query: str, ticker: str,
                         n_results: int = 5) -> list[dict]:
        """Search only documents related to a specific ticker."""
        return self.search(
            query=query,
            n_results=n_results,
            filter_dict={"ticker": ticker.upper()}
        )

    def get_ticker_documents(self, ticker: str,
                             source: Optional[str] = None,
                             limit: int = 20) -> list[dict]:
        """Get all documents for a ticker, optionally filtered by source."""
        where: dict = {"ticker": ticker.upper()}
        if source:
            where = {"$and": [{"ticker": ticker.upper()}, {"source": source}]}

        results = self.collection.get(where=where, limit=limit)
        formatted = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                formatted.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i] if results["documents"] else None,
                    "metadata": results["metadatas"][i] if results["metadatas"] else None,
                })
        return formatted

    # ── Collection Management ─────────────────────────

    def reset_collection(self):
        """Drop and recreate the collection (for testing)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def heartbeat(self) -> bool:
        """Check if ChromaDB is responsive."""
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.error("ChromaDB heartbeat failed: %s", e)
            return False
