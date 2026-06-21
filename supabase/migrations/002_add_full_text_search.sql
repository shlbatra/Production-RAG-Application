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
