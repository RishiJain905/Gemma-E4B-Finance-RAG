"""One-off live test for ChromaStore (requires llama-server embeddings on :8087)."""

from src.storage.chroma_store import ChromaStore

store = ChromaStore()

store.add_document(
    document_id="test/nvda-001",
    text="NVIDIA reported record datacenter revenue of $26 billion in Q1 2026, driven by demand for Blackwell GPUs.",
    ticker="NVDA",
    source="sec_filing",
    date="2026-05-15",
)

store.add_document(
    document_id="test/nvda-002",
    text="AMD announced the MI400 AI accelerator, positioning it as a direct competitor to NVIDIA Blackwell.",
    ticker="AMD",
    source="news",
    date="2026-05-14",
)

store.add_document(
    document_id="test/macro-001",
    text="The Federal Reserve kept interest rates unchanged at 4.5%, citing persistent inflation concerns.",
    ticker=None,
    source="fred",
    date="2026-05-10",
)

print(f"Documents stored: {store.count()}")

results = store.search("What is NVIDIA doing in the data center space?")
print('\nSearch results for "data center":')
for r in results:
    print(f'  [{r["id"]}] (dist: {r["distance"]:.3f}) {r["document"][:80]}...')

nvda_results = store.search_by_ticker("What is happening?", "NVDA")
print(f"\nTicker-filtered results for NVDA: {len(nvda_results)}")
for r in nvda_results:
    print(f'  [{r["id"]}] {r["document"][:80]}...')

store.delete_document("test/nvda-001")
store.delete_document("test/nvda-002")
store.delete_document("test/macro-001")
print(f"\nAfter cleanup: {store.count()} documents")
