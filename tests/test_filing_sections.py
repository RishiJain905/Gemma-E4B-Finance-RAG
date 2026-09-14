"""tests/test_filing_sections.py
Deterministic SEC filing-section contract tests using local text fixtures.
"""

from src.sec.filing_sections import FilingSection, split_filing_sections


def _metadata() -> dict:
    return {
        "accession": "0000320193-25-000001",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "filing_date": "2025-01-15",
        "period": "2024-09-28",
        "source_url": "https://www.sec.gov/example.htm",
        "file_path": "data/sec/parsed/0000320193-25-000001.txt",
    }


def test_filing_section_document_id_is_stable() -> None:
    section = FilingSection(
        accession="ACC-1",
        ticker="NVDA",
        form="10-Q",
        filing_date="2026-05-15",
        report_period="2026-03-31",
        section_key="item_2",
        section_heading="Item 2. Management's Discussion and Analysis",
        section_index=3,
        text="Item 2. Management's Discussion and Analysis\nRevenue increased.",
        source_url="https://www.sec.gov/filing.htm",
        parsed_path="data/sec/parsed/ACC-1.txt",
    )

    assert section.document_id == "sec:ACC-1:item_2"


def test_split_preserves_headings_tables_and_filters_boilerplate() -> None:
    parsed_text = """TABLE OF CONTENTS
Item 1. Business ................................ 3

ACME CORPORATION 2024 FORM 10-K
Item 1. Business
Item 1. Business
We sell useful products worldwide.

Item 7. Management's Discussion and Analysis
Revenue increased by 12% in 2024.
Year    Revenue    Net income
2024    $12,400    $1,250
2023    $11,071    $980

Item 7A. Quantitative and Qualitative Disclosures About Market Risk

CUSTOM OPERATING METRICS
Customers increased to 42,000.
"""

    sections = split_filing_sections(parsed_text, _metadata())

    assert [section.section_key for section in sections] == [
        "item_1",
        "item_7",
        "unknown_001",
    ]
    assert sections[0].text.startswith("Item 1. Business\n")
    assert sections[0].text.count("Item 1. Business") == 1
    assert "Year    Revenue    Net income" in sections[1].text
    assert "2024    $12,400    $1,250" in sections[1].text
    assert sections[2].section_heading == "CUSTOM OPERATING METRICS"
    assert all(section.text.strip() for section in sections)
    assert all("TABLE OF CONTENTS" not in section.text for section in sections)


def test_unknown_heading_keys_are_deterministic_ordinals() -> None:
    text = """FIRST CUSTOM SECTION
Alpha prose.
SECOND CUSTOM SECTION
Beta prose.
"""

    first = split_filing_sections(text, _metadata())
    second = split_filing_sections(text, _metadata())

    assert [section.section_key for section in first] == ["unknown_001", "unknown_002"]
    assert [section.document_id for section in first] == [
        section.document_id for section in second
    ]


def test_empty_sections_are_removed_without_dropping_numeric_content() -> None:
    text = """Item 1. Business

Item 2. Properties
42
"""

    sections = split_filing_sections(text, _metadata())

    assert len(sections) == 1
    assert sections[0].section_key == "item_2"
    assert sections[0].text == "Item 2. Properties\n42"


def test_8k_decimal_item_gets_canonical_key() -> None:
    sections = split_filing_sections(
        "Item 1.01 Entry into a Material Agreement\nThe agreement value is $42 million.",
        _metadata(),
    )

    assert sections[0].section_key == "item_1_01"
    assert sections[0].document_id.endswith(":item_1_01")


def test_flat_single_line_text_is_reflowed_into_sections() -> None:
    """Whitespace-collapsed (single-line) parser output still yields sections.

    Live calibration (2.2.5.3): the legacy parser emits one enormous line, so
    inline "Item N. <Title>" markers must be reflowed onto their own lines and
    TOC stubs ("Item 3. ... 19") must not become sections.
    """
    prose_a = "The Company designs and sells consumer devices worldwide. " * 12
    prose_b = "Revenue decreased due to foreign exchange headwinds this quarter. " * 12
    flat = (
        "xbrl context soup 0000320193 2025-09-28 2026-03-28 us-gaap:ProductMember "
        + "filler " * 400
        + "Item 1. Financial Statements 1 Item 2. Management's Discussion and Analysis 13 "
        + "Item 1. Business " + prose_a
        + "Item 2. Management's Discussion and Analysis of Financial Condition " + prose_b
    ).strip()
    assert "\n" not in flat

    sections = split_filing_sections(flat, _metadata())

    keys = [s.section_key for s in sections]
    assert any(k.startswith("item_1") for k in keys)
    assert any(k.startswith("item_2") for k in keys)
    # TOC stubs reflow into heading + page number only and are floored away.
    bodies = [s.text for s in sections]
    assert all(len(b) >= 80 for b in bodies)
    assert any("consumer devices" in b for b in bodies)
    assert any("foreign exchange" in b for b in bodies)


def test_normal_multiline_text_is_not_reflowed() -> None:
    """Line-structured fixtures keep byte-identical behavior (no reflow, no floor)."""
    text = "RISK FACTORS\nShort but real body line.\n"
    sections = split_filing_sections(text, _metadata())
    assert len(sections) == 1
    assert sections[0].text.splitlines()[-1] == "Short but real body line."


def test_flat_8k_decimal_item_is_reflowed() -> None:
    """Whitespace-collapsed 8-K Item 5.02 / 8.01 markers become sections."""
    prose = (
        "On September 4, 2026, the Company announced the appointment of Jane Doe "
        "as Chief Financial Officer effective immediately. " * 8
    )
    flat = (
        "xbrl soup accession header filler " * 200
        + "Item 5.02 Departure of Directors or Certain Officers; Election of Directors. "
        + prose
        + "Item 9.01 Financial Statements and Exhibits. "
        + "Exhibit 99.1 Press release dated September 4, 2026. "
        + "lorem " * 40
    ).strip()
    assert "\n" not in flat

    sections = split_filing_sections(flat, {**_metadata(), "filing_type": "8-K"})

    keys = [s.section_key for s in sections]
    assert "item_5_02" in keys
    assert any(k.startswith("item_9_01") for k in keys)
    assert any("Jane Doe" in s.text for s in sections)


def test_flat_item_colon_and_dash_headings_are_reflowed() -> None:
    """INTU/DELL-style ITEM 1: / ITEM 1 — headings survive flat-text reflow."""
    prose = "The Company designs and sells software products worldwide. " * 15
    flat = (
        "submission header filler " * 200
        + "PART I ITEM 1: Business 4 ITEM 1A: Risk Factors 15 "
        + "PART I ITEM 1 - BUSINESS BACKGROUND "
        + prose
        + "ITEM 1A — RISK FACTORS "
        + "Investing in our securities involves risk. " * 15
    ).strip()
    assert "\n" not in flat

    sections = split_filing_sections(flat, _metadata())
    keys = [s.section_key for s in sections]
    assert any(k.startswith("item_1") for k in keys)
    assert any("software products" in s.text for s in sections)


def test_item_less_flat_text_falls_back_to_full_document() -> None:
    """When no Item/PART markers exist, still emit one indexable section."""
    text = "filler " * 500 + "The board approved a special dividend of $1.00 per share."
    assert "\n" not in text.strip()

    sections = split_filing_sections(text, {**_metadata(), "filing_type": "8-K"})

    assert len(sections) == 1
    assert sections[0].section_key == "full_document"
    assert "special dividend" in sections[0].text
