# Plan: Protocol-Based Retrieval Strategies (Similarity, BM25, Hybrid)

## Context

Retrieval is currently hardcoded to pgvector cosine similarity in `DocumentStore.search_similar()`. The codebase already uses Protocol-based abstractions for parsers (`document_parser.py`) and chunkers (`chunking.py`) — same pattern here. Extract retrieval into a `RetrievalStrategy` Protocol with three implementations: similarity search (current), BM25 (keyword), and hybrid (RRF fusion of both).

## Files to Create / Modify

| File | Action |
|---|---|
| `app/retrieval.py` | **New** — Protocol, 3 strategies, factory |
| `app/document_store.py` | Modify — add `full_text_search()`, update `insert_chunks()` to populate `search_vector` |
| `app/agent.py` | Modify — use `RetrievalStrategy` instead of `document_store.search_similar()` |
| `app/config.py` | Modify — add `rag_retrieval_strategy` setting |
| `app/main.py` | Modify — build retrieval strategy in lifespan, pass to agent |
| `supabase/migrations/002_add_full_text_search.sql` | **New** — GIN index + `bm25_search()` RPC |
| `tests/test_retrieval.py` | **New** — tests for all 3 strategies |
| `.env.example` | Modify — add `RAG_RETRIEVAL_STRATEGY` |

---

## Detailed Design

### 1. `app/retrieval.py` — Protocol + 3 Strategies

Follows the exact pattern from `document_parser.py` and `chunking.py`:

```python
@runtime_checkable
class RetrievalStrategy(Protocol):
    def search(self, query: str, top_k: int, threshold: float) -> list[dict]: ...
```

Every strategy returns `list[dict]` where each dict has: `content` (str), `metadata` (dict with `source`), `similarity` (float 0-1). This is the existing contract consumed by `agent.py:89-95`.

#### a. `SimilarityRetriever`
- Wraps the current `DocumentStore.search_similar()` — no behavior change
- Constructor takes `document_store: DocumentStore`
- `search()` delegates to `document_store.search_similar(query, top_k, threshold)`

#### b. `BM25Retriever`
- Keyword-based full-text search using Postgres `tsvector`/`tsquery`
- Constructor takes `document_store: DocumentStore`
- `search()` calls a new `document_store.full_text_search(query, top_k)` method
- Returns results with `ts_rank` normalized to 0-1 as the `similarity` score

#### c. `HybridRetriever`
- Combines similarity + BM25 using Reciprocal Rank Fusion (RRF)
- Constructor takes `document_store: DocumentStore` and `k: int = 60` (RRF constant)
- `search()` calls both `search_similar()` and `full_text_search()`, then fuses scores:
  ```
  rrf_score(doc) = 1/(k + rank_similarity) + 1/(k + rank_bm25)
  ```
- Re-ranks by fused score, normalizes to 0-1, returns top_k

#### Factory

```python
_STRATEGY_MAP: dict[str, type] = {
    "similarity": SimilarityRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
}

def get_retriever(settings: Settings, document_store: DocumentStore) -> RetrievalStrategy:
    cls = _STRATEGY_MAP.get(settings.rag_retrieval_strategy)
    if cls is None:
        raise ValueError(f"Unknown retrieval strategy '{settings.rag_retrieval_strategy}'. "
                         f"Supported: {', '.join(sorted(_STRATEGY_MAP))}")
    return cls(document_store=document_store)
```

---

### 2. `supabase/migrations/002_add_full_text_search.sql` — BM25 Support

Add a GIN index and an RPC function for full-text search:

```sql
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
```

Notes:
- The `search_vector` column is **not** auto-generated — it's populated by the application during ingestion. Existing documents must be re-ingested for BM25/hybrid to find them.
- `websearch_to_tsquery` — handles natural language queries (supports `OR`, quoted phrases, `-` exclusion) without requiring manual tsquery syntax.
- `ts_rank` returns a relevance score; it's not normalized to 0-1 but we normalize in the Python layer.

---

### 3. `app/document_store.py` — Add `full_text_search()` + Update `insert_chunks()`

#### New method: `full_text_search()`

Add one new method alongside the existing `search_similar()`:

```python
def full_text_search(self, query: str, top_k: int | None = None) -> list[dict]:
    top_k = top_k or self._top_k
    with self._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM bm25_search(%s, %s)",
                (query, top_k),
            )
            results = [dict(row) for row in cur.fetchall()]

    # Normalize ts_rank scores to 0-1
    if results:
        max_score = max(r["similarity"] for r in results)
        if max_score > 0:
            for r in results:
                r["similarity"] = round(r["similarity"] / max_score, 4)

    return results
```

`search_similar()` stays unchanged — `SimilarityRetriever` wraps it directly.

#### Update `insert_chunks()` to populate `search_vector`

The existing `INSERT` statement adds `content`, `metadata`, `embedding`. After inserting each batch, compute and store the `search_vector` for newly inserted rows:

```python
# After inserting the batch with execute_values:
cur.execute(
    "UPDATE documents SET search_vector = to_tsvector('english', content) "
    "WHERE search_vector IS NULL"
)
```

The post-insert UPDATE is simpler and keeps `insert_chunks()` clean. The `WHERE search_vector IS NULL` clause ensures it only touches newly inserted rows that haven't been indexed yet.

---

### 4. `app/config.py` — Add Retrieval Strategy Setting

Add one field:

```python
rag_retrieval_strategy: str = "similarity"  # "similarity" | "bm25" | "hybrid"
```

---

### 5. `app/agent.py` — Use RetrievalStrategy

The `ProductionAgent` constructor changes to accept a `retriever` instead of `document_store`:

```python
class ProductionAgent:
    def __init__(self, retriever=None):
        ...
        self.retriever = retriever
        self.rag_enabled = retriever is not None
```

The `retrieve_context` node changes from:

```python
results = self.document_store.search_similar(
    query=user_message,
    top_k=settings.rag_top_k,
    threshold=settings.rag_similarity_threshold,
)
```

To:

```python
results = self.retriever.search(
    query=user_message,
    top_k=settings.rag_top_k,
    threshold=settings.rag_similarity_threshold,
)
```

One line changes. The rest of `retrieve_context` (source formatting, error handling) stays identical because the return shape is the same.

---

### 6. `app/main.py` — Wire It Up in Lifespan

```python
from app.retrieval import get_retriever

# In lifespan, after document_store is created:
retriever = None
if settings.rag_enabled:
    document_store = DocumentStore(settings)
    retriever = get_retriever(settings, document_store)

agent = ProductionAgent(retriever=retriever)
```

---

### 7. `.env.example` — Add Setting

```
RAG_RETRIEVAL_STRATEGY=similarity   # similarity | bm25 | hybrid
```

---

### 8. `tests/test_retrieval.py` — Tests

All tests use a mocked `DocumentStore` — no real DB needed.

| Test | What it verifies |
|---|---|
| `test_similarity_retriever_delegates` | Calls `document_store.search_similar()` with correct args |
| `test_bm25_retriever_delegates` | Calls `document_store.full_text_search()` with correct args |
| `test_bm25_normalizes_scores` | Scores are normalized to 0-1 range |
| `test_hybrid_calls_both` | Calls both `search_similar()` and `full_text_search()` |
| `test_hybrid_rrf_ranking` | Fused ranking prioritizes docs that appear in both result sets |
| `test_hybrid_deduplicates` | Same chunk from both strategies appears once, not twice |
| `test_hybrid_respects_top_k` | Returns at most `top_k` results |
| `test_get_retriever_factory` | Factory returns correct class for each strategy name |
| `test_get_retriever_unknown` | Factory raises `ValueError` for unknown strategy |
| `test_all_strategies_satisfy_protocol` | `isinstance(retriever, RetrievalStrategy)` is True |

Existing `test_agent.py` tests need the `mock_document_store` fixture updated to provide a mock `retriever` instead. The `conftest.py` fixture changes from mocking `document_store.search_similar` to mocking `retriever.search`.

---

## What Does NOT Change

- **Return contract** — all strategies return `list[dict]` with `content`, `metadata`, `similarity`
- **`/chat` endpoint** — completely untouched
- **`ingestion.py`** — untouched (it calls `document_store.insert_chunks()` which handles tsvector internally)
- **`search_similar()`** — stays on `DocumentStore`, now also wrapped by `SimilarityRetriever`

---

## Verification

1. `uv run ruff check . && uv run mypy app/` — lint + type check
2. `uv run pytest tests/test_retrieval.py -v` — new retrieval tests pass
3. `uv run pytest tests/ -v` — no regressions (agent tests updated)
4. Apply migration `002_add_full_text_search.sql` to Supabase
5. Re-ingest documents so `search_vector` is populated for existing chunks
6. Test each strategy locally:
   - `RAG_RETRIEVAL_STRATEGY=similarity uv run uvicorn app.main:app`
   - `RAG_RETRIEVAL_STRATEGY=bm25 uv run uvicorn app.main:app`
   - `RAG_RETRIEVAL_STRATEGY=hybrid uv run uvicorn app.main:app`
7. Send `/chat` requests and verify sources are returned for all 3 strategies
