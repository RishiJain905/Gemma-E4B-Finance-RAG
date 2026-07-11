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
