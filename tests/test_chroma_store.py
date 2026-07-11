# tests/test_chroma_store.py
# Mocked tests for TraceAlchemyEmbeddingFunction and ChromaStore.
# Phase 1.2.3, Section 1.
#
# These tests do NOT require a live llama-server or a real ChromaDB instance.
# All external dependencies (httpx.Client, chromadb.PersistentClient)
# are replaced with unittest.mock objects.

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── chromadb may not be installed in this environment ──
if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.storage.chroma_store import (
    ChromaStore,
    TraceAlchemyEmbeddingFunction,
)


EMBEDDING_DIM = 2048


# ── Fixtures ────────────────────────────────────────────

@pytest.fixture
def mock_embedding_response():
    """Return a mock httpx response object with a valid embedding payload."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{"embedding": [0.0123] * EMBEDDING_DIM}]
    }
    return response


@pytest.fixture
def chroma_store():
    """Return a ChromaStore instance with fully mocked dependencies."""
    with patch("src.storage.chroma_store.httpx.Client") as mock_http_cls, \
         patch("src.storage.chroma_store.chromadb.PersistentClient") as mock_pc_cls:
        mock_http_instance = MagicMock()
        mock_http_cls.return_value = mock_http_instance
        mock_pc_instance = MagicMock()
        mock_pc_cls.return_value = mock_pc_instance
        mock_collection = MagicMock()
        mock_collection.name = "tracealchemy_docs"
        mock_pc_instance.get_or_create_collection.return_value = mock_collection
        store = ChromaStore()
        yield store


# ── TraceAlchemyEmbeddingFunction tests ───────────────

def test_embedding_function_call(mock_embedding_response):
    with patch("src.storage.chroma_store.httpx.Client") as mock_http_cls:
        mock_http_instance = MagicMock()
        mock_http_instance.post.return_value = mock_embedding_response
        mock_http_cls.return_value = mock_http_instance

        fn = TraceAlchemyEmbeddingFunction()
        result = fn(["test text"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM
        mock_http_instance.post.assert_called_once()


def test_embedding_function_batching():
    def _make_response(batch_size):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [{"embedding": [0.0123] * EMBEDDING_DIM} for _ in range(batch_size)]
        }
        return response

    with patch("src.storage.chroma_store.httpx.Client") as mock_http_cls:
        mock_http_instance = MagicMock()
        # Return 10 embeddings for first call, 5 for second
        mock_http_instance.post.side_effect = [
            _make_response(10),
            _make_response(5),
        ]
        mock_http_cls.return_value = mock_http_instance

        fn = TraceAlchemyEmbeddingFunction(batch_size=10)
        texts = [f"text {i}" for i in range(15)]
        result = fn(texts)

        assert len(result) == 15
        assert mock_http_instance.post.call_count == 2

        call1 = mock_http_instance.post.call_args_list[0]
        call2 = mock_http_instance.post.call_args_list[1]

        # First batch: 10 items → passed as list
        assert len(call1.kwargs["json"]["input"]) == 10
        # Second batch: 5 items → passed as list
        assert len(call2.kwargs["json"]["input"]) == 5


def test_embedding_function_del():
    with patch("src.storage.chroma_store.httpx.Client") as mock_http_cls:
        mock_http_instance = MagicMock()
        mock_http_cls.return_value = mock_http_instance

        fn = TraceAlchemyEmbeddingFunction()
        fn.__del__()

        mock_http_instance.close.assert_called_once()


# ── ChromaStore initialization ────────────────────────

def test_chroma_store_init():
    with patch("src.storage.chroma_store.httpx.Client") as mock_http_cls, \
         patch("src.storage.chroma_store.chromadb.PersistentClient") as mock_pc_cls:
        mock_http_instance = MagicMock()
        mock_http_cls.return_value = mock_http_instance
        mock_pc_instance = MagicMock()
        mock_pc_cls.return_value = mock_pc_instance
        mock_collection = MagicMock()
        mock_collection.name = "tracealchemy_docs"
        mock_pc_instance.get_or_create_collection.return_value = mock_collection

        store = ChromaStore()

        assert store.collection is mock_collection
        mock_pc_instance.get_or_create_collection.assert_called_once()
        call_kwargs = mock_pc_instance.get_or_create_collection.call_args.kwargs
        assert call_kwargs["name"] == "tracealchemy_docs"
        assert call_kwargs["metadata"] == {"hnsw:space": "cosine"}


# ── CRUD tests ─────────────────────────────────────────

def test_add_document(chroma_store):
    chroma_store.add_document(
        document_id="test-1",
        text="hello",
        ticker="nvda",
        source="sec",
    )
    chroma_store.collection.add.assert_called_once_with(
        documents=["hello"],
        metadatas=[{"ticker": "NVDA", "source": "sec"}],
        ids=["test-1"],
    )


def test_add_documents_batch(chroma_store):
    ids = ["a1", "a2", "a3"]
    texts = ["t1", "t2", "t3"]
    metas = [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]
    chroma_store.add_documents_batch(ids=ids, texts=texts, metadatas=metas)
    chroma_store.collection.add.assert_called_once_with(
        documents=texts,
        metadatas=metas,
        ids=ids,
    )


def test_search(chroma_store):
    chroma_store.collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"ticker": "NVDA"}, {"ticker": "AMD"}]],
        "distances": [[0.1, 0.2]],
    }

    results = chroma_store.search("query", n_results=2)

    assert len(results) == 2
    assert list(results[0].keys()) == ["id", "document", "metadata", "distance"]
    assert results[0] == {
        "id": "id1",
        "document": "doc1",
        "metadata": {"ticker": "NVDA"},
        "distance": 0.1,
    }
    assert results[1] == {
        "id": "id2",
        "document": "doc2",
        "metadata": {"ticker": "AMD"},
        "distance": 0.2,
    }


def test_get_document_found(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": ["test-1"],
        "documents": ["hello"],
        "metadatas": [{"ticker": "NVDA"}],
    }

    result = chroma_store.get_document("test-1")
    assert result == {
        "id": "test-1",
        "document": "hello",
        "metadata": {"ticker": "NVDA"},
    }


def test_get_document_not_found(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }

    result = chroma_store.get_document("test-1")
    assert result is None


def test_delete_document(chroma_store):
    chroma_store.delete_document("test-1")
    chroma_store.collection.delete.assert_called_once_with(ids=["test-1"])


def test_count(chroma_store):
    chroma_store.collection.count.return_value = 42
    assert chroma_store.count() == 42


# ── Ticker-specific tests ────────────────────────────

def test_search_by_ticker(chroma_store):
    with patch.object(chroma_store, "search") as mock_search:
        mock_search.return_value = []
        chroma_store.search_by_ticker("query text", "nvda", n_results=3)
        mock_search.assert_called_once_with(
            query="query text",
            n_results=3,
            filter_dict={"ticker": "NVDA"},
        )


def test_get_ticker_documents(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": ["doc1", "doc2"],
        "metadatas": [{"ticker": "NVDA"}, {"ticker": "NVDA"}],
    }

    results = chroma_store.get_ticker_documents("nvda", source="sec", limit=10)

    chroma_store.collection.get.assert_called_once_with(
        where={"$and": [{"ticker": "NVDA"}, {"source": "sec"}]},
        limit=10,
    )
    assert len(results) == 2
    assert results[0]["id"] == "d1"


# ── Collection management tests ──────────────────────

def test_reset_collection(chroma_store):
    chroma_store.reset_collection()

    chroma_store.client.delete_collection.assert_called_once_with("tracealchemy_docs")
    chroma_store.client.create_collection.assert_called_once()
    call_kwargs = chroma_store.client.create_collection.call_args.kwargs
    assert call_kwargs["name"] == "tracealchemy_docs"
    assert call_kwargs["metadata"] == {"hnsw:space": "cosine"}


def test_heartbeat_success(chroma_store):
    chroma_store.client.heartbeat.return_value = True
    assert chroma_store.heartbeat() is True
    chroma_store.client.heartbeat.assert_called_once()


def test_heartbeat_failure(chroma_store):
    chroma_store.client.heartbeat.side_effect = Exception("Connection failed")
    assert chroma_store.heartbeat() is False


# ── Chunking tests ───────────────────────────────────

def test_chunk_text_short_returns_single():
    assert ChromaStore._chunk_text("hello", 1000, 150) == ["hello"]


def test_chunk_text_empty_returns_empty():
    assert ChromaStore._chunk_text("", 1000, 150) == []
    assert ChromaStore._chunk_text("   ", 1000, 150) == []


def test_chunk_text_splits_long_with_overlap():
    # 50 space-separated words, each "wordNN" -> well over the small chunk size
    text = " ".join(f"word{i:02d}" for i in range(50))
    chunks = ChromaStore._chunk_text(text, chunk_chars=60, overlap=15)

    assert len(chunks) > 1
    # No chunk exceeds the window size.
    assert all(len(c) <= 60 for c in chunks)
    # No mid-word cuts: every token is a clean "wordNN".
    for c in chunks:
        for token in c.split():
            assert token.startswith("word")
    # Overlap means consecutive chunks share at least one token.
    first_tokens = set(chunks[0].split())
    second_tokens = set(chunks[1].split())
    assert first_tokens & second_tokens


def test_add_document_short_unchanged(chroma_store):
    chroma_store.add_document(document_id="doc-1", text="short text", ticker="nvda")
    chroma_store.collection.add.assert_called_once_with(
        documents=["short text"],
        metadatas=[{"ticker": "NVDA"}],
        ids=["doc-1"],
    )


def test_add_document_chunks_long_text(chroma_store):
    # Sentence-punctuated long text: the structural chunker packs sentences into
    # chunks <= max_chars (no mid-sentence cuts). Repeats to exceed chunk_chars.
    base = ("NVIDIA reported strong revenue growth this quarter. "
            "Datacenter sales surged across all regions. "
            "Gross margins improved meaningfully year over year. "
            "Free cash flow reached a record high. "
            "The company guided next quarter above consensus. ")
    long_text = base * 20  # ~4000 chars, ~100 sentences
    chroma_store.add_document(
        document_id="sec/NVDA/10-Q-2026-Q1",
        text=long_text,
        ticker="nvda",
        source="sec",
    )

    chroma_store.collection.add.assert_called_once()
    kwargs = chroma_store.collection.add.call_args.kwargs
    ids = kwargs["ids"]
    docs = kwargs["documents"]
    metas = kwargs["metadatas"]

    assert len(ids) > 1
    assert len(ids) == len(docs) == len(metas)
    # Ids are suffixed "#0", "#1", ... off the parent id.
    assert ids[0] == "sec/NVDA/10-Q-2026-Q1#0"
    assert ids[1] == "sec/NVDA/10-Q-2026-Q1#1"
    # Each chunk carries parent + index metadata alongside the base metadata.
    assert metas[0]["parent_id"] == "sec/NVDA/10-Q-2026-Q1"
    assert metas[0]["ticker"] == "NVDA"
    assert metas[0]["source"] == "sec"
    assert metas[0]["chunk_index"] == 0
    assert metas[0]["chunk_count"] == len(ids)
    # Each chunk respects the configured window size.
    assert all(len(d) <= chroma_store.chunk_chars for d in docs)


def test_sec_filing_family_uses_deterministic_child_ids_and_full_metadata(chroma_store):
    metadata = {
        "accession": "ACC-1", "form": "10-K", "filing_date": "2025-01-15",
        "report_period": "2024-09-28", "section_key": "item_1",
        "section_heading": "Item 1. Business", "section_index": 0,
        "parent_id": "sec:ACC-1:item_1", "source_url": "https://sec.example",
    }
    chroma_store.add_document(
        document_id="sec:ACC-1:item_1", text="Item 1. Business\nRevenue was $42.",
        ticker="AAPL", source="sec_filing", date="2025-01-15", metadata=metadata,
    )

    call = chroma_store.collection.add.call_args.kwargs
    assert call["ids"] == ["sec:ACC-1:item_1#0"]
    assert call["documents"][0].startswith("Item 1. Business")
    assert call["metadatas"][0] == {
        **metadata, "ticker": "AAPL", "source": "sec_filing", "date": "2025-01-15",
        "chunk_index": 0, "chunk_count": 1, "section": "Item 1. Business",
        "document_id": "sec:ACC-1:item_1#0",
    }


def test_get_section_chunks_is_filtered_and_bounded(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": ["p#1"], "documents": ["body"],
        "metadatas": [{"parent_id": "p", "chunk_index": 1}],
    }

    result = chroma_store.get_section_chunks("p", limit=10, offset=2)

    chroma_store.collection.get.assert_called_once_with(
        where={"parent_id": "p"}, limit=10, offset=2,
        include=["documents", "metadatas"],
    )
    assert result == [{
        "id": "p#1", "document": "body",
        "metadata": {"parent_id": "p", "chunk_index": 1},
    }]


def test_get_section_chunks_rejects_unbounded_pages(chroma_store):
    with pytest.raises(ValueError):
        chroma_store.get_section_chunks("p", limit=ChromaStore.MAX_READ_LIMIT + 1)
    with pytest.raises(ValueError):
        chroma_store.get_section_chunks("p", limit=1, offset=-1)
    chroma_store.collection.get.assert_not_called()


def test_adjacent_sections_filters_accession_not_corpus(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": ["a#0", "b#0"], "documents": ["A", "B"],
        "metadatas": [
            {"accession": "ACC-1", "section_index": 2, "chunk_index": 0},
            {"accession": "ACC-1", "section_index": 3, "chunk_index": 0},
        ],
    }

    result = chroma_store.get_adjacent_sections("ACC-1", 3, before=1, after=0)

    kwargs = chroma_store.collection.get.call_args.kwargs
    assert kwargs["where"] == {"$and": [
        {"source": "sec_filing"}, {"accession": "ACC-1"},
        {"section_index": {"$gte": 2}}, {"section_index": {"$lte": 3}},
    ]}
    assert kwargs["limit"] == ChromaStore.MAX_READ_LIMIT
    assert [row["id"] for row in result] == ["a#0", "b#0"]


def test_count_and_delete_filing_sections_use_metadata_filters(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": ["a#0", "a#1", "b#0"],
        "metadatas": [
            {"parent_id": "a"}, {"parent_id": "a"}, {"parent_id": "b"},
        ],
    }

    assert chroma_store.count_filing_sections("ACC-1") == 2
    chroma_store.collection.get.assert_called_once_with(
        where={"$and": [{"source": "sec_filing"}, {"accession": "ACC-1"}]},
        include=["metadatas"],
    )
    chroma_store.delete_filing_section_family("a")
    chroma_store.collection.delete.assert_called_once_with(where={"parent_id": "a"})


def test_count_filing_section_chunks_filters_parent_without_documents(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": ["a#0", "a#1"], "metadatas": [{}, {}],
    }

    assert chroma_store.count_filing_section_chunks("a") == 2
    chroma_store.collection.get.assert_called_once_with(
        where={"parent_id": "a"}, include=["metadatas"],
    )
