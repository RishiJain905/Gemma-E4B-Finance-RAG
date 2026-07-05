"""
tests/test_chunking.py
Phase 2.1.3.1 — structure & sentence-aware chunker (pure-function tests, no network).
"""

from __future__ import annotations

import pytest

from src.storage.chunking import chunk_document, _split_sentences, _split_sections


# ── Basic shape ────────────────────────────────────────────────────────

def test_short_doc_single_chunk():
    text = "This is a short single-paragraph document about NVIDIA revenue."
    chunks = chunk_document(text, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["section"] == ""
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["chunk_count"] == 1


def test_empty_text():
    assert chunk_document("") == []
    assert chunk_document("   ") == []
    assert chunk_document("\n\n") == []


def test_chunk_metadata():
    text = ". ".join(f"Sentence number {i}" for i in range(12)) + "."
    chunks = chunk_document(text, max_chars=80, overlap_sentences=0)
    assert len(chunks) > 1
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["chunk_count"] == len(chunks) for c in chunks)
    assert all("section" in c for c in chunks)


# ── Sentence-aware packing ─────────────────────────────────────────────

def test_no_midsentence_cuts():
    text = "Revenue grew strongly. Datacenter sales surged. Margins improved. Guidance raised. Cash flow climbed."
    chunks = chunk_document(text, max_chars=60, overlap_sentences=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c["text"][-1] in ".!?", f"chunk does not end on sentence punctuation: {c['text']!r}"


def test_max_chars_respected():
    text = ". ".join(f"Short sentence number {i}" for i in range(30)) + "."
    chunks = chunk_document(text, max_chars=80, overlap_sentences=0)
    assert all(len(c["text"]) <= 80 for c in chunks)


def test_sentence_overlap():
    text = ". ".join(f"{w} sentence here" for w in
                     ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]) + "."
    chunks = chunk_document(text, max_chars=60, overlap_sentences=1)
    assert len(chunks) > 1
    # The last sentence of chunk 0 should be the first sentence of chunk 1.
    last = _split_sentences(chunks[0]["text"])[-1]
    assert chunks[1]["text"].startswith(last), (
        f"expected chunk[1] to start with overlap {last!r}, got {chunks[1]['text']!r}")


def test_no_overlap_when_zero():
    text = ". ".join(f"{w} sentence here" for w in
                     ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]) + "."
    chunks = chunk_document(text, max_chars=60, overlap_sentences=0)
    # No chunk starts with the previous chunk's last sentence.
    for i in range(1, len(chunks)):
        prev_last = _split_sentences(chunks[i - 1]["text"])[-1]
        assert not chunks[i]["text"].startswith(prev_last)


def test_oversize_sentence_is_not_dropped():
    long_sent = "this is one very long sentence with no internal punctuation " * 8
    chunks = chunk_document(long_sent, max_chars=100)
    assert len(chunks) >= 1
    assert all(c["text"] for c in chunks)  # nothing dropped


def test_split_sentences_abbreviation_merge():
    text = "Apple Inc. reported earnings. The U.S. market rose. Profit climbed."
    sents = _split_sentences(text)
    # "Inc." and "U.S." must NOT split a sentence.
    assert any("Apple Inc." in s for s in sents)
    assert any("U.S." in s for s in sents)
    assert len(sents) == 3


# ── Structural sections ────────────────────────────────────────────────

def test_section_split_sec_filing():
    sec_text = (
        "Item 1A. Risk Factors\n" + "We face competition. " * 80 + "\n\n"
        "Item 7. Management's Discussion\n" + "Revenue grew strongly. " * 80
    )
    chunks = chunk_document(sec_text, source="sec_10k", max_chars=200, overlap_sentences=0)
    sections = {c["section"] for c in chunks}
    assert any("Item 1A." in s for s in sections), sections
    assert any("Item 7." in s for s in sections), sections
    # Every chunk carries a section label (preamble or an Item).
    assert all(c["section"] for c in chunks)


def test_section_split_markdown_headings():
    md_text = (
        "## Overview\n" + "Overview point one. " * 40 + "\n"
        "## Risks\n" + "Risk point one. " * 40
    )
    chunks = chunk_document(md_text, source="ir", max_chars=200, overlap_sentences=0)
    sections = {c["section"] for c in chunks}
    assert "## Overview" in sections
    assert "## Risks" in sections


def test_no_headings_one_section():
    text = "Just a plain paragraph. With two sentences. No headings at all."
    sections = _split_sections(text, source="yfinance_news")
    assert sections == [("", text)]


def test_markdown_heading_content_is_preserved():
    """A news doc whose first line is a '# Headline' must keep the headline in
    the chunk text — it is content, not just a section label."""
    text = ("# S&P 500 Jumps For Ninth Week, DELL Leading Nasdaq To Record Highs\n\n"
            "President Trump said he was ready to extend the ceasefire. "
            "Markets rallied on the news.")
    chunks = chunk_document(text, source="yfinance_news", max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("# S&P 500 Jumps")
    # The body is present too.
    assert "ceasefire" in chunks[0]["text"]
    # The section is labelled with the heading.
    assert "S&P 500 Jumps" in chunks[0]["section"]


# ── Fixed strategy matches legacy ──────────────────────────────────────

def test_fixed_strategy_matches_legacy():
    from src.storage.chroma_store import ChromaStore
    text = " ".join(f"word{i:02d}" for i in range(50))
    fixed = chunk_document(text, strategy="fixed", max_chars=60, fixed_overlap_chars=15)
    legacy = ChromaStore._chunk_text(text, 60, 15)
    assert [c["text"] for c in fixed] == legacy
    # And the metadata is set.
    assert [c["chunk_index"] for c in fixed] == list(range(len(fixed)))


def test_fixed_strategy_short_single():
    fixed = chunk_document("short text", strategy="fixed", max_chars=1000)
    assert [c["text"] for c in fixed] == ["short text"]