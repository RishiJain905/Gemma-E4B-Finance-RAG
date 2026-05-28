"""Live smoke test for unified Store (requires llama-server embeddings on :8087)."""

from src.storage.store import Store

store = Store()

health = store.heartbeat()
print("Health:", health)

store.save_fundamental("NVDA", "revenue_q1_2026", 26.0, "usd", "2026-Q1")
fact = store.get_fundamental("NVDA", "revenue_q1_2026")
print(f"Fact: {fact}")

doc_id = store.save_document(
    document_id="test/nvda-earnings",
    text=(
        "NVIDIA reported record datacenter revenue of $26 billion in Q1 2026. "
        "Data center revenue grew 42% year-over-year driven by Blackwell GPU demand."
    ),
    ticker="NVDA",
    source="earnings_call",
    date="2026-05-15",
)
print(f"Document saved: {doc_id}")

results = store.search("What was NVIDIA datacenter revenue?")
print("\nHybrid search results:")
print(f"  Ticker detected: {results['ticker']}")
print(f"  Documents found: {len(results['documents'])}")
print(f"  Facts found: {len(results['facts'])}")
for d in results["documents"]:
    print(f"  [{d['id']}] (dist: {d['distance']:.3f})")
for f in results["facts"]:
    print(f"  {f['metric']} = {f['value']}")

store.chroma.delete_document("test/nvda-earnings")
with store.sqlite._connect() as conn:
    conn.execute(
        "DELETE FROM fundamentals WHERE ticker=? AND metric=?",
        ("NVDA", "revenue_q1_2026"),
    )
    conn.commit()

print(f"\nAfter cleanup: {store.heartbeat()}")
