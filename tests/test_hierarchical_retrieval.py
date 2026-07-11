"""
tests/test_hierarchical_retrieval.py
Offline tests for Phase 2.2.5.3 — the bounded hierarchical filing expander.

Everything runs without ChromaDB, the model, or the network: a small
``FakeSectionStore`` returns deterministic section families and adjacent-section
windows, so each test exercises the expansion contract (sentence/table
completion, obligation-gated adjacent sections, cross-hit deduplication, budget
enforcement, and provenance) in isolation.
"""

from dataclasses import dataclass

from src.middleware.hierarchical_retrieval import expand_filing_hits


# ── Fakes / builders ──────────────────────────────────────


@dataclass
class Obligation:
    """Minimal stand-in for evidence_grader.EvidenceObligation (heading match)."""

    metrics: tuple = ()
    operations: tuple = ()


def _chunk(accession, section_key, section_index, chunk_index, body,
           heading="", ticker="NVDA"):
    parent = f"sec:{accession}:{section_key}"
    return {
        "id": f"{parent}#{chunk_index}",
        "document": body,
        "metadata": {
            "accession": accession,
            "parent_id": parent,
            "section_key": section_key,
            "section_heading": heading or section_key,
            "section_index": section_index,
            "chunk_index": chunk_index,
            "ticker": ticker,
            "source": "sec_filing",
        },
    }


class FakeSectionStore:
    """Deterministic section-family / adjacent-section reader.

    Built from a flat list of chunk dicts. ``get_section_chunks`` returns one
    parent's children sorted by chunk_index; ``get_adjacent_sections`` returns the
    section-index window for one filing.
    """

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.section_calls = 0
        self.adjacent_calls = 0

    def get_section_chunks(self, parent_id, *, limit, offset=0):
        self.section_calls += 1
        rows = [c for c in self.chunks
                if (c["metadata"].get("parent_id")) == parent_id]
        rows = sorted(rows, key=lambda c: c["metadata"].get("chunk_index", 0))
        return [dict(c) for c in rows[offset:offset + limit]]

    def get_adjacent_sections(self, accession, section_index, *, before=1, after=1):
        self.adjacent_calls += 1
        lo, hi = section_index - before, section_index + after
        rows = [
            c for c in self.chunks
            if c["metadata"].get("accession") == accession
            and lo <= c["metadata"].get("section_index", -999) <= hi
        ]
        rows = sorted(rows, key=lambda c: (
            c["metadata"].get("section_index", 0),
            c["metadata"].get("chunk_index", 0)))
        return [dict(c) for c in rows]


def _reasons(result):
    return {d.get("expansion_reason") for d in result}


def _ids(result):
    return [d["id"] for d in result]


# ── Root preservation + provenance ─────────────────────────


def test_root_hit_preserved_and_annotated():
    hit = _chunk("acc1", "item_7", 3, 1, "Complete sentence one. Sentence two.",
                 heading="Item 7. MD&A")
    hit["fusion_score"] = 0.42
    store = FakeSectionStore([hit])

    out = expand_filing_hits([hit], store, (), context_budget=10000)

    assert len(out) == 1
    root = out[0]
    assert root["expansion_reason"] == "root_hit"
    assert root["root_hit_id"] == hit["id"]
    # Section heading provenance survives on the root.
    assert root["metadata"]["section_heading"] == "Item 7. MD&A"
    # No sibling read was needed (sentence is complete on both ends).
    assert store.section_calls == 0


def test_expanded_items_inherit_root_score():
    hit = _chunk("acc1", "item_7", 3, 1, "continues the prior clause and then",
                 heading="Item 7")
    hit["rerank_score"] = 0.9
    pre = _chunk("acc1", "item_7", 3, 0, "The revenue trend")
    post = _chunk("acc1", "item_7", 3, 2, "reached a new high. Done.")
    store = FakeSectionStore([pre, hit, post])

    out = expand_filing_hits([hit], store, (), context_budget=10000, max_siblings=2)

    siblings = [d for d in out if d["expansion_reason"] != "root_hit"]
    assert siblings, "expected sibling expansion"
    for sib in siblings:
        assert sib["root_hit_id"] == hit["id"]
        assert sib["score"] == 0.9


# ── Sentence / table completion ────────────────────────────


def test_needs_preceding_sibling_added():
    hit = _chunk("acc1", "item_7", 3, 1, "and margins expanded further. End.")
    pre = _chunk("acc1", "item_7", 3, 0, "Revenue grew 20%")
    store = FakeSectionStore([pre, hit])

    out = expand_filing_hits([hit], store, (), context_budget=10000)

    assert "preceding_sibling" in _reasons(out)
    assert pre["id"] in _ids(out)


def test_needs_following_sibling_added_for_table_row():
    # Ends on a digit (table row) → needs the following chunk.
    hit = _chunk("acc1", "item_8", 4, 0, "Revenue 2025 26,000")
    post = _chunk("acc1", "item_8", 4, 1, "27,000 2026. Totals shown above.")
    store = FakeSectionStore([hit, post])

    out = expand_filing_hits([hit], store, (), context_budget=10000)

    assert "following_sibling" in _reasons(out)


def test_complete_sentence_adds_no_sibling():
    hit = _chunk("acc1", "item_7", 3, 1, "A wholly complete standalone sentence.")
    pre = _chunk("acc1", "item_7", 3, 0, "Earlier text.")
    post = _chunk("acc1", "item_7", 3, 2, "Later text.")
    store = FakeSectionStore([pre, hit, post])

    out = expand_filing_hits([hit], store, (), context_budget=10000)

    assert _reasons(out) == {"root_hit"}


# ── Adjacent sections ──────────────────────────────────────


def test_adjacent_section_added_on_obligation_heading_match():
    hit = _chunk("acc1", "item_7", 3, 0, "Discussion of results.",
                 heading="Item 7. MD&A")
    # Adjacent section (index 4) whose heading matches a requested metric token.
    adj = _chunk("acc1", "item_7a", 4, 0,
                 "Market risk from interest rate changes.",
                 heading="Item 7A. Quantitative Disclosures About Interest Rate Risk")
    store = FakeSectionStore([hit, adj])
    obligations = [Obligation(metrics=("interest_rate",))]

    out = expand_filing_hits([hit], store, obligations, context_budget=10000,
                             max_adjacent_sections=1)

    assert "adjacent_section_obligation" in _reasons(out)
    assert adj["id"] in _ids(out)


def test_adjacent_section_skipped_without_match_or_signal():
    hit = _chunk("acc1", "item_7", 3, 0, "Discussion of results.",
                 heading="Item 7")
    adj = _chunk("acc1", "item_8", 4, 0, "Financial statements.",
                 heading="Item 8. Financial Statements")
    store = FakeSectionStore([hit, adj])

    out = expand_filing_hits([hit], store, (), context_budget=10000,
                             max_adjacent_sections=1)

    assert "adjacent_section_obligation" not in _reasons(out)
    assert "adjacent_section_missing_context" not in _reasons(out)


def test_adjacent_section_added_on_missing_local_context():
    hit = _chunk("acc1", "item_7", 3, 0, "Discussion of results.", heading="Item 7")
    adj = _chunk("acc1", "item_8", 4, 0, "Financial statements.",
                 heading="Item 8")
    store = FakeSectionStore([hit, adj])

    out = expand_filing_hits([hit], store, (), context_budget=10000,
                             max_adjacent_sections=1, missing_local_context=True)

    assert "adjacent_section_missing_context" in _reasons(out)


def test_adjacent_section_never_returns_whole_parent():
    # A multi-chunk adjacent section only contributes its first child, not all.
    hit = _chunk("acc1", "item_7", 3, 0, "Discussion.", heading="Item 7")
    adj0 = _chunk("acc1", "item_8", 4, 0, "First risk chunk.", heading="Item 8 Risk")
    adj1 = _chunk("acc1", "item_8", 4, 1, "Second risk chunk continues.",
                  heading="Item 8 Risk")
    store = FakeSectionStore([hit, adj0, adj1])

    out = expand_filing_hits([hit], store, (), context_budget=10000,
                             max_adjacent_sections=1, missing_local_context=True)

    ids = _ids(out)
    assert adj0["id"] in ids
    assert adj1["id"] not in ids  # only the representative first chunk


# ── Deduplication ──────────────────────────────────────────


def test_shared_sibling_deduped_across_hits():
    # Two hits in the same section whose expansions overlap on chunk 1.
    h0 = _chunk("acc1", "item_7", 3, 0, "Opening clause that runs on")
    shared = _chunk("acc1", "item_7", 3, 1, "into the middle and keeps going")
    h2 = _chunk("acc1", "item_7", 3, 2, "then it finally concludes here.")
    store = FakeSectionStore([h0, shared, h2])

    # Both h0 (needs following → chunk 1) and h2 (needs preceding → chunk 1)
    # would pull the shared middle chunk; it must appear at most once.
    out = expand_filing_hits([h0, h2], store, (), context_budget=10000,
                             max_siblings=2)

    assert _ids(out).count(shared["id"]) <= 1


def test_hit_that_is_also_a_sibling_not_duplicated():
    h0 = _chunk("acc1", "item_7", 3, 0, "Opening clause that runs on")
    h1 = _chunk("acc1", "item_7", 3, 1, "the following chunk, already a hit.")
    store = FakeSectionStore([h0, h1])

    out = expand_filing_hits([h0, h1], store, (), context_budget=10000)

    # h1 is a root hit; h0's following-sibling read must not re-add it.
    assert _ids(out).count(h1["id"]) == 1
    assert {d["expansion_reason"] for d in out if d["id"] == h1["id"]} == {"root_hit"}


# ── Budget + caps ──────────────────────────────────────────


def test_budget_stops_expansion():
    hit = _chunk("acc1", "item_7", 3, 1, "runs into the next" + " x" * 5)
    pre = _chunk("acc1", "item_7", 3, 0, "L" * 500)
    post = _chunk("acc1", "item_7", 3, 2, "R" * 500 + ".")
    store = FakeSectionStore([pre, hit, post])

    # Budget only covers the root hit; no sibling should fit.
    tight = len(hit["document"]) + 10
    out = expand_filing_hits([hit], store, (), context_budget=tight, max_siblings=2)

    assert _reasons(out) == {"root_hit"}


def test_max_expanded_items_caps_total():
    hit = _chunk("acc1", "item_7", 3, 1, "continues onward and")
    pre = _chunk("acc1", "item_7", 3, 0, "Before")
    post = _chunk("acc1", "item_7", 3, 2, "after. Done.")
    store = FakeSectionStore([pre, hit, post])

    out = expand_filing_hits([hit], store, (), context_budget=100000,
                             max_siblings=2, max_expanded_items=1)

    siblings = [d for d in out if d["expansion_reason"] != "root_hit"]
    assert len(siblings) == 1


def test_max_siblings_zero_disables_sibling_expansion():
    hit = _chunk("acc1", "item_7", 3, 1, "continues onward and")
    pre = _chunk("acc1", "item_7", 3, 0, "Before")
    store = FakeSectionStore([pre, hit])

    out = expand_filing_hits([hit], store, (), context_budget=100000,
                             max_siblings=0)

    assert _reasons(out) == {"root_hit"}


# ── Non-filing + fail-soft ─────────────────────────────────


def test_non_filing_hit_passes_through():
    hit = {"id": "yf/NVDA/news-1", "document": "some news blurb",
           "metadata": {"ticker": "NVDA", "source": "yfinance_news"}}
    store = FakeSectionStore([])

    out = expand_filing_hits([hit], store, (), context_budget=10000)

    assert len(out) == 1
    assert out[0]["expansion_reason"] == "root_hit"
    assert store.section_calls == 0
    assert store.adjacent_calls == 0


def test_store_failure_is_fail_soft():
    class BrokenStore:
        def get_section_chunks(self, *a, **k):
            raise RuntimeError("chroma down")

        def get_adjacent_sections(self, *a, **k):
            raise RuntimeError("chroma down")

    hit = _chunk("acc1", "item_7", 3, 1, "needs a preceding clause and")
    out = expand_filing_hits([hit], BrokenStore(), (), context_budget=10000,
                             missing_local_context=True)

    # Root is still returned; no expansion, no raise.
    assert _reasons(out) == {"root_hit"}


def test_empty_hits_returns_empty():
    assert expand_filing_hits([], FakeSectionStore([]), (), context_budget=10000) == []
