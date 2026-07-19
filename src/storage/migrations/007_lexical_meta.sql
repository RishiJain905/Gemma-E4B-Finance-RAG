-- Phase 2.3.7.6 narrow lexical-chunk metadata for O(1)-ish inventory counts.
-- The persistent corpus_fts index stores the narrative body inline, so grouping
-- it for source/ticker inventory forces a full content scan (>50ms at 100k). This
-- narrow companion table mirrors only the count dimensions of every indexed chunk
-- and is maintained transactionally alongside corpus_fts, so authoritative
-- inventory counts come from an indexed GROUP BY instead of scanning Chroma or the
-- FTS content. It is body-free and therefore cheap to scan and index.
CREATE TABLE IF NOT EXISTS lexical_chunk_meta (
    chunk_id TEXT PRIMARY KEY,
    source TEXT,
    ticker TEXT
);

CREATE INDEX IF NOT EXISTS idx_lexical_chunk_meta_source
    ON lexical_chunk_meta(source);
CREATE INDEX IF NOT EXISTS idx_lexical_chunk_meta_ticker_source
    ON lexical_chunk_meta(ticker, source);
