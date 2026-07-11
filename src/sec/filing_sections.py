"""
src/sec/filing_sections.py
Deterministic SEC filing section extraction and provenance contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from .filing_parser import SEC_ITEM_HEADING_RE, SECTION_HEADINGS


_UNKNOWN_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 &'(),./:\-]{3,159}$")
_PAGE_NUMBER_RE = re.compile(r"^PAGE\s+\d{1,4}$", re.IGNORECASE)
_TOC_ENTRY_RE = re.compile(r"^ITEM\s+\d{1,2}[A-Z]?\..*\.{2,}\s*\d+\s*$", re.IGNORECASE)
_NAV_LINES = frozenset({"TABLE OF CONTENTS", "INDEX", "PART I", "PART II", "PART III", "PART IV"})


@dataclass(frozen=True)
class FilingSection:
    """One canonical SEC filing section ready for structural chunking."""

    accession: str
    ticker: str
    form: str
    filing_date: str
    report_period: str
    section_key: str
    section_heading: str
    section_index: int
    text: str
    source_url: str
    parsed_path: str

    @property
    def document_id(self) -> str:
        """Stable parent identifier for this filing section."""
        return f"sec:{self.accession}:{self.section_key}"


def _known_heading(line: str) -> tuple[str, str] | None:
    item = SEC_ITEM_HEADING_RE.match(line)
    if item:
        item_key = item.group("item").lower().replace(".", "_")
        return f"item_{item_key}", line.strip()

    upper = line.strip().upper()
    for heading in SECTION_HEADINGS:
        if upper == heading:
            key = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
            return key, line.strip()
    return None


def _is_unknown_heading(line: str) -> bool:
    stripped = line.strip()
    if stripped in _NAV_LINES or _PAGE_NUMBER_RE.fullmatch(stripped):
        return False
    if _TOC_ENTRY_RE.fullmatch(stripped):
        return False
    return bool(_UNKNOWN_HEADING_RE.fullmatch(stripped)) and any(c.isalpha() for c in stripped)


def _normalise_body(lines: list[str], heading: str) -> str:
    cleaned: list[str] = []
    heading_seen = False
    for raw in lines:
        line = raw.rstrip().strip()
        if not line or line in _NAV_LINES or _PAGE_NUMBER_RE.fullmatch(line):
            continue
        if _TOC_ENTRY_RE.fullmatch(line):
            continue
        if line.casefold() == heading.casefold():
            if heading_seen:
                continue
            heading_seen = True
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def split_filing_sections(
    parsed_text: str,
    filing_metadata: Mapping[str, object],
) -> list[FilingSection]:
    """Split parsed filing text using parser-owned headings and SEC markers.

    Empty/navigation-only sections are removed. Numeric and table-like lines
    are retained verbatim apart from surrounding whitespace normalization.
    """
    candidates: list[tuple[str, str, list[str]]] = []
    current_key = ""
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_heading:
            candidates.append((current_key, current_heading, current_lines))
        current_lines = []

    for raw in (parsed_text or "").splitlines():
        line = raw.strip()
        known = _known_heading(line)
        if known:
            # TOC entries can match the Item regex; reject them explicitly.
            if _TOC_ENTRY_RE.fullmatch(line):
                continue
            flush()
            current_key, current_heading = known
            current_lines = [raw]
        elif _is_unknown_heading(line):
            flush()
            current_key = "unknown"
            current_heading = line
            current_lines = [raw]
        elif current_heading:
            current_lines.append(raw)
    flush()

    accession = str(filing_metadata.get("accession") or "").strip()
    ticker = str(filing_metadata.get("ticker") or "").strip().upper()
    form = str(
        filing_metadata.get("form") or filing_metadata.get("filing_type") or ""
    ).strip()
    filing_date = str(filing_metadata.get("filing_date") or "").strip()
    report_period = str(
        filing_metadata.get("report_period") or filing_metadata.get("period") or ""
    ).strip()
    source_url = str(filing_metadata.get("source_url") or "").strip()
    parsed_path = str(
        filing_metadata.get("parsed_path") or filing_metadata.get("file_path") or ""
    ).strip()

    sections: list[FilingSection] = []
    used_keys: dict[str, int] = {}
    unknown_ordinal = 0
    for key, heading, lines in candidates:
        body = _normalise_body(lines, heading)
        # A heading without any substantive body is not a usable section.
        if not body or body.casefold() == heading.casefold():
            continue
        if key == "unknown":
            unknown_ordinal += 1
            key = f"unknown_{unknown_ordinal:03d}"
        occurrence = used_keys.get(key, 0)
        used_keys[key] = occurrence + 1
        canonical_key = key if occurrence == 0 else f"{key}_{occurrence + 1:03d}"
        sections.append(
            FilingSection(
                accession=accession,
                ticker=ticker,
                form=form,
                filing_date=filing_date,
                report_period=report_period,
                section_key=canonical_key,
                section_heading=heading,
                section_index=len(sections),
                text=body,
                source_url=source_url,
                parsed_path=parsed_path,
            )
        )
    return sections
