-- Add tsvector column for full-text search (populated by application during ingestion)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS search_vector tsvector;

-- GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_documents_search_vector
    ON documents USING gin(search_vector);

-- RPC function for full-text search
CREATE OR REPLACE FUNCTION bm25_search(
    search_query TEXT,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.metadata,
        ts_rank(d.search_vector, websearch_to_tsquery('english', search_query))::FLOAT AS similarity
    FROM documents d
    WHERE d.search_vector @@ websearch_to_tsquery('english', search_query)
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Backfill search_vector for already-ingested documents.
-- Fine at small scale. For large tables (100k+ rows), batch to avoid lock
-- contention and WAL bloat:
--
--   UPDATE documents
--   SET search_vector = to_tsvector('english', content)
--   WHERE id IN (
--       SELECT id FROM documents WHERE search_vector IS NULL LIMIT 10000
--   );
--   -- Repeat until 0 rows affected, with pg_sleep(0.5) between batches.
--
-- Also consider CREATE INDEX CONCURRENTLY instead of CREATE INDEX above
-- to avoid blocking writes during index build on large tables.
UPDATE documents
SET search_vector = to_tsvector('english', content)
WHERE search_vector IS NULL;
