-- Phase 2.3 cross-store corpus metadata bridge.
CREATE TABLE IF NOT EXISTS corpus_items (
    corpus_item_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_category TEXT NOT NULL,
    provider_record_id TEXT,
    original_publisher TEXT,
    item_type TEXT NOT NULL,
    event_type TEXT,
    title TEXT NOT NULL,
    normalized_headline TEXT NOT NULL,
    syndicated_key TEXT,
    summary TEXT CHECK (summary IS NULL OR length(summary) <= 4000),
    language TEXT NOT NULL,
    published_at TEXT,
    effective_at TEXT,
    as_of_at TEXT,
    observed_at TEXT,
    accessed_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    source_url TEXT NOT NULL,
    canonical_url TEXT,
    tickers_json TEXT NOT NULL DEFAULT '[]',
    index_codes_json TEXT NOT NULL DEFAULT '[]',
    sectors_json TEXT NOT NULL DEFAULT '[]',
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    document_family TEXT NOT NULL,
    document_family_id TEXT,
    indexing_status TEXT NOT NULL CHECK (
        indexing_status IN ('pending', 'indexed', 'error', 'not_applicable')
    ),
    index_error TEXT,
    license_label TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    evidence_authority TEXT NOT NULL,
    narrative_bytes INTEGER NOT NULL DEFAULT 0,
    metadata_bytes INTEGER NOT NULL DEFAULT 0,
    is_tombstone INTEGER NOT NULL DEFAULT 0,
    retired_at TEXT,
    retention_reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS corpus_item_sources (
    corpus_item_id TEXT NOT NULL REFERENCES corpus_items(corpus_item_id) ON DELETE CASCADE,
    source_key TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_category TEXT NOT NULL,
    provider_record_id TEXT,
    original_publisher TEXT,
    source_url TEXT NOT NULL,
    canonical_url TEXT,
    published_at TEXT,
    observed_at TEXT,
    accessed_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    license_label TEXT NOT NULL,
    evidence_authority TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (corpus_item_id, source_key)
);

CREATE TABLE IF NOT EXISTS corpus_item_securities (
    corpus_item_id TEXT NOT NULL REFERENCES corpus_items(corpus_item_id) ON DELETE CASCADE,
    security_id TEXT NOT NULL REFERENCES securities(security_id),
    ticker TEXT,
    PRIMARY KEY (corpus_item_id, security_id)
);

UPDATE corpus_items SET document_family_id=corpus_item_id
WHERE document_family_id IS NULL OR document_family_id='';
