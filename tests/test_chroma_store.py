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


def test_search_document_families_uses_metadata_only_filters(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": ["ir/nvda/release#0", "ir/nvda/release#1", "sec/a#0"],
        "metadatas": [
            {"parent_id": "ir/nvda/release", "source": "ir", "ticker": "NVDA",
             "date": "2026-05-15", "chunk_count": 2},
            {"parent_id": "ir/nvda/release", "source": "ir", "ticker": "NVDA",
             "date": "2026-05-15", "chunk_count": 2},
            {"parent_id": "sec/a", "source": "sec_filing", "ticker": "NVDA",
             "date": "2026-05-14", "chunk_count": 1},
        ],
    }

    families = chroma_store.search_document_families(
        query="release", source="ir", ticker="NVDA", limit=10, offset=0,
    )

    chroma_store.collection.get.assert_called_once_with(
        where={"$and": [{"source": "ir"}, {"ticker": "NVDA"}]},
        limit=11, offset=0, include=["metadatas"],
    )
    assert len(families) == 1
    assert families[0]["id"] == "ir/nvda/release"
    assert families[0]["chunk_count"] == 2


def test_filing_section_families_are_metadata_only_and_bounded(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": ["p#0", "p#1", "q#0"],
        "metadatas": [
            {"parent_id": "p", "accession": "ACC-1", "source": "sec_filing",
             "section_index": 0, "section_heading": "Business", "chunk_count": 2},
            {"parent_id": "p", "accession": "ACC-1", "source": "sec_filing",
             "section_index": 0, "section_heading": "Business", "chunk_count": 2},
            {"parent_id": "q", "accession": "ACC-1", "source": "sec_filing",
             "section_index": 1, "section_heading": "Risk", "chunk_count": 1},
        ],
    }

    sections = chroma_store.get_filing_section_families("ACC-1", limit=10, offset=0)

    assert len(sections) == 2
    assert sections[0]["chunk_count"] == 2
    call = chroma_store.collection.get.call_args.kwargs
    assert call["include"] == ["metadatas"]
    assert "documents" not in call


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
        "chunk_ordinal": 0, "child_chunk_id": "sec:ACC-1:item_1#0",
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


def test_narrative_family_replay_uses_stable_chunk_ids(chroma_store):
    text = "First sentence about revenue. " * 80
    metadata = {
        "document_family_id": "news-1",
        "corpus_item_id": "news-1",
        "source_category": "news_vendor",
        "source_name": "finnhub",
        "provider_record_id": "provider-1",
        "security_ids": "security-1",
        "tickers": "ACME",
        "index_memberships": "sp500",
        "sectors": "Industrials",
        "industries": "Machinery",
        "item_type": "news",
        "published_at": "2026-07-14T00:00:00Z",
        "content_hash": "a" * 64,
        "normalization_version": "1",
        "authority_tier": "provider",
    }
    chroma_store.collection.get.return_value = {"ids": [], "metadatas": []}

    chroma_store.add_document(
        "news-1", text, source="finnhub", metadata=metadata, replace_family=True,
    )
    first_ids = chroma_store.collection.upsert.call_args.kwargs["ids"]
    chroma_store.collection.reset_mock()
    chroma_store.collection.get.return_value = {
        "ids": first_ids, "metadatas": metadata,
    }

    chroma_store.add_document(
        "news-1", text, source="finnhub", metadata=metadata, replace_family=True,
    )

    assert chroma_store.collection.upsert.call_args.kwargs["ids"] == first_ids
    chroma_store.collection.delete.assert_not_called()
    chunk_metadata = chroma_store.collection.upsert.call_args.kwargs["metadatas"]
    assert all(row["parent_id"] == "news-1" for row in chunk_metadata)
    assert [row["chunk_ordinal"] for row in chunk_metadata] == list(range(len(first_ids)))
    assert [row["child_chunk_id"] for row in chunk_metadata] == first_ids


def test_family_replacement_accepts_new_chunks_before_deleting_orphans(chroma_store):
    chroma_store.collection.get.return_value = {
        "ids": ["release-1#0", "release-1#1", "release-1#2"],
        "metadatas": [{"document_family_id": "release-1"}] * 3,
    }

    chroma_store.add_document(
        "release-1",
        "Updated release text.",
        source="treasury",
        metadata={"document_family_id": "release-1", "corpus_item_id": "release-1"},
        replace_family=True,
    )

    assert chroma_store.collection.upsert.call_args.kwargs["ids"] == ["release-1#0"]
    chroma_store.collection.get.assert_called_once_with(
        where={"$or": [
            {"document_family_id": "release-1"},
            {"corpus_item_id": "release-1"},
            {"parent_id": "release-1"},
        ]},
        include=["metadatas"],
    )
    chroma_store.collection.delete.assert_called_once_with(
        ids=["release-1#1", "release-1#2"],
    )
    calls = [call[0] for call in chroma_store.collection.mock_calls]
    assert calls.index("upsert") < calls.index("delete")


def test_delete_document_family_targets_current_and_legacy_metadata(chroma_store):
    chroma_store.delete_document_family("news-1")

    chroma_store.collection.delete.assert_called_once_with(where={"$or": [
        {"document_family_id": "news-1"},
        {"corpus_item_id": "news-1"},
        {"parent_id": "news-1"},
    ]})


def test_markdown_table_keeps_heading_and_units_in_its_chunk():
    from src.storage.chunking import chunk_document

    text = "## Revenue (USD millions)\n| Quarter | Revenue |\n| Q1 | 42 |\n| Q2 | 45 |"
    chunks = chunk_document(text, source="issuer_release", max_chars=40)

    assert len(chunks) == 1
    assert "Revenue (USD millions)" in chunks[0]["text"]
    assert "| Quarter | Revenue |" in chunks[0]["text"]


def test_mark_corpus_revision_strips_index_config_keys(chroma_store):
    """Re-sending hnsw:* keys makes chromadb reject the whole modify call
    ("Changing the distance function ... is not supported"), which left the
    Chroma-visible revision permanently stale and silently disabled lexical
    fusion via the revision-consistency guard (found live 2026-07-17)."""
    chroma_store.collection.metadata = {"hnsw:space": "cosine", "corpus_revision": 3}

    def _reject_hnsw(metadata):
        if any(k.startswith("hnsw:") for k in metadata):
            raise ValueError(
                "Changing the distance function of a collection once it is "
                "created is not supported currently.")

    chroma_store.collection.modify.side_effect = (
        lambda metadata: _reject_hnsw(metadata))

    chroma_store.mark_corpus_revision(7)

    chroma_store.collection.modify.assert_called_once()
    sent = chroma_store.collection.modify.call_args.kwargs["metadata"]
    assert sent["corpus_revision"] == 7
    assert not any(k.startswith("hnsw:") for k in sent)


def test_mark_corpus_revision_never_regresses(chroma_store):
    chroma_store.collection.metadata = {"corpus_revision": 9}

    chroma_store.mark_corpus_revision(7)

    chroma_store.collection.modify.assert_not_called()
