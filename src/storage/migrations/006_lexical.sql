-- Phase 2.3.7.5 persistent, content-bearing lexical chunk index.
CREATE TABLE IF NOT EXISTS lexical_index_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL DEFAULT 1,
    indexed_revision INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT,
    row_count INTEGER NOT NULL DEFAULT 0,
    rebuild_cursor INTEGER NOT NULL DEFAULT 0,
    rebuild_revision INTEGER NOT NULL DEFAULT 0
);

INSERT OR IGNORE INTO lexical_index_state (
    id, schema_version, indexed_revision, row_count
) VALUES (1, 1, 0, 0);

CREATE VIRTUAL TABLE IF NOT EXISTS corpus_fts USING fts5(
    title,
    body,
    ticker UNINDEXED,
    source_category UNINDEXED,
    item_type UNINDEXED,
    chunk_id UNINDEXED,
    family_id UNINDEXED,
    source UNINDEXED,
    event_type UNINDEXED,
    form UNINDEXED,
    item UNINDEXED,
    authority_tier UNINDEXED,
    indexing_status UNINDEXED,
    published_at UNINDEXED,
    effective_at UNINDEXED,
    as_of_at UNINDEXED
);
