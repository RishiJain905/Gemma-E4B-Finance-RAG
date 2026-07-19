"""
src/storage/chunking.py
Structure & sentence-aware document chunker (Phase 2.1.3.1).

Replaces the fixed ~max_chars sliding window in ChromaStore with a chunker that
splits on document structure first (SEC section markers / markdown headings),
then packs sentences into chunks up to ``max_chars`` — never breaking a
sentence — with sentence-based overlap between consecutive chunks.

Pure-function (only ``re`` from the stdlib); no network, no heavy deps.
"""

from __future__ import annotations

import re
from typing import Optional

# SEC "Item N." / "Item NA." headings at the start of a line.
_SEC_ITEM = re.compile(r"^Item\s+\d+[A-Z]?\.?", re.IGNORECASE | re.MULTILINE)
# Markdown headings (## ..., # ..., up to six #) at the start of a line.
_MD_HEADING = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)
# Sentence boundary: ., !, ? followed by whitespace.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Tokens whose trailing period is NOT a sentence end. Keys are the token with
# dots stripped, lowercased. Keep "item"/"sec" OUT — those are handled by the
# structural section split.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "sr", "jr", "st", "vs", "etc", "no", "fig",
    "vol", "pp", "al", "inc", "corp", "ltd", "co", "dept", "est", "approx",
    "us", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sept", "oct",
    "nov", "dec",
}


def _is_sec_source(source: Optional[str]) -> bool:
    s = (source or "").lower()
    return s.startswith("sec") or any(k in s for k in ("filing", "10-k", "10-q", "10k", "10q"))


def _split_sections(text: str, source: Optional[str]) -> list[tuple[str, str]]:
    """Split into (section_name, section_text) by structural boundaries.

    SEC sources split on ``Item N.`` headings; other sources split on markdown
    headings. Text before the first heading is a preamble section. If no
    structural boundary is found the whole document is one section named "".
    Blank-line paragraphs are NOT section boundaries (sentences pack across
    them within a section) so short docs are not over-split.
    """
    text = (text or "").strip()
    if not text:
        return []

    pattern = _SEC_ITEM if _is_sec_source(source) else _MD_HEADING
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(("(preamble)" if _is_sec_source(source) else "", pre))

    for i, m in enumerate(matches):
        line_end = text.find("\n", m.start())
        if line_end == -1:
            line_end = len(text)
        name = text[m.start():line_end].strip()
        # Include the heading line in the section text — it is content (e.g. a
        # news headline "# S&P 500 Jumps..."), not just a label, so dropping it
        # would lose information.
        sec_start = m.start()
        sec_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text = text[sec_start:sec_end].strip()
        if sec_text:
            sections.append((name, sec_text))
    return sections


def _ends_with_abbrev(sentence: str) -> bool:
    """True if the sentence ends with a known abbreviation + period."""
    m = re.search(r"([A-Za-z](?:\.?[A-Za-z])*)\.$", sentence.rstrip())
    if not m:
        return False
    return m.group(1).replace(".", "").lower() in _ABBREV


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the ending punctuation.

    Splits on ``.!?`` followed by whitespace, then merges back pieces that were
    split at a known abbreviation (``U.S.``, ``Inc.``, ``Mr.``, …). Whitespace-
    only sentences are dropped.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    merged: list[str] = []
    for part in parts:
        if merged and _ends_with_abbrev(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return [p.strip() for p in merged if p.strip()]


def _pack_sentences(section_text: str, max_chars: int,
                    overlap_sentences: int) -> list[str]:
    """Greedily pack sentences into chunks ≤ max_chars, never breaking a
    sentence. A single sentence longer than max_chars becomes its own chunk.

    ``overlap_sentences`` trailing sentences of the previous chunk are carried
    into the next chunk (0 = no overlap).
    """
    sentences = _split_sentences(section_text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        added = len(sent) + (1 if current else 0)  # +1 for the joining space
        if current and current_len + added > max_chars:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(overlap) + [sent]
            current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
        else:
            current.append(sent)
            current_len += added

    if current:
        chunks.append(" ".join(current))
    return chunks


def _fixed_chunk(text: str, chunk_chars: int, overlap: int = 150) -> list[str]:
    """Legacy fixed-size sliding window (char overlap).

    Replicates ``ChromaStore._chunk_text`` so ``strategy="fixed"`` reproduces
    the pre-2.1.3 chunking exactly.
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
        if nxt < n and not text[nxt - 1].isspace():
            space = text.find(" ", nxt)
            if space == -1:
                break
            nxt = space + 1
        start = nxt
    return chunks


def chunk_document(text: str, *, source: Optional[str] = None,
                   max_chars: int = 1000, overlap_sentences: int = 1,
                   strategy: str = "structural",
                   fixed_overlap_chars: int = 150) -> list[dict]:
    """Chunk a document into semantically coherent pieces.

    Args:
        text: the document text.
        source: source type (e.g. "sec", "sec_10k", "yfinance_news"); used to
            detect SEC ``Item`` markers. None → markdown-heading split only.
        max_chars: target maximum chunk length (chars). Sentences are never
            broken; a single sentence longer than this becomes its own chunk.
        overlap_sentences: number of trailing sentences carried into the next
            chunk (structural strategy only).
        strategy: "structural" (default) or "fixed" (legacy sliding window).
        fixed_overlap_chars: char overlap used by the "fixed" strategy
            (defaults to the legacy 150).

    Returns:
        ``[{"text": str, "section": str, "chunk_index": int,
        "chunk_ordinal": int, "chunk_count": int}, ...]``.
        Empty/whitespace text → ``[]``.
    """
    text = (text or "").strip()
    if not text:
        return []

    if strategy == "fixed":
        bodies = _fixed_chunk(text, max_chars, fixed_overlap_chars)
        out = [{"text": b, "section": ""} for b in bodies]
    else:
        out = []
        for section_name, section_text in _split_sections(text, source):
            for body in _pack_sentences(section_text, max_chars, overlap_sentences):
                out.append({"text": body, "section": section_name})

    # Drop empty chunks, assign global index/count.
    out = [c for c in out if c["text"]]
    for i, c in enumerate(out):
        c["chunk_index"] = i
        c["chunk_ordinal"] = i
        c["chunk_count"] = len(out)
    return out
