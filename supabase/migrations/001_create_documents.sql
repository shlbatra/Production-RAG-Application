-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: each row is one chunk of a source document
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate cosine similarity search
-- HNSW (Hierarchical Navigable Small World) index on the embedding column. Without this, every similarity search would scan every row (exact nearest-neighbor) — slow at scale.
-- m = 16 — each node in the HNSW graph connects to 16 neighbors. Higher = more accurate but uses more memory and slower to build. 16 is a good default.
-- ef_construction = 64 — how many candidates to consider when building the graph. Higher = better index quality but slower build time. 64 is a standard production value.
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- RPC function for similarity search (called from application code). Creates a Postgres function (callable via Supabase's .rpc('match_documents', {...}) from application code)
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5,
    match_threshold FLOAT DEFAULT 0.7
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
        1 - (d.embedding <=> query_embedding) AS similarity -- 1 - Cosine distance between embeddings for query and document chunk
    FROM documents d
    WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Row Level Security: only service role can access
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Creates a policy that allows everything — but only for the service_role. In Supabase, the service_role key bypasses RLS, so this policy effectively means:
-- service_role (our server) — full read/write access
-- anon key (browser clients) — no access (no policy grants them anything)
CREATE POLICY "Service role has full access"
ON documents FOR ALL
USING (true) WITH CHECK (true);
