# Plan: Add Supabase Document Storage for RAG

## Context

The `prod_rag` project is currently a **chat API** — it sends user messages directly to OpenAI models via LangGraph with no retrieval step. To become a true RAG system, it needs a document store, embeddings, and a retrieval pipeline.

**Choices made:**
- **Vector store**: Supabase with pgvector
- **Embedding model**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Document type**: PDF files
- **Ingestion**: Both API endpoint + CLI script for bulk loading

---

## Phase 0: Supabase Project Setup (Manual, One-Time) ✅

### 0.1 Create Supabase Project

Go to https://supabase.com/dashboard, create a new project. Note two values:
- **Project URL** (e.g., `https://abcdefgh.supabase.co`)
- **Service Role Key** (Settings > API > `service_role` key — bypasses RLS, server-side only)

### 0.2 Enable pgvector Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 0.3 Create Documents Table

```sql
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

- `content` — chunked text (each row = one chunk, not a full document)
- `metadata` — JSONB: `{"source": "report.pdf", "doc_id": "uuid", "chunk_index": 5, "total_chunks": 20}`
- `embedding` — 1536 dims matching `text-embedding-3-small`

### 0.4 Create HNSW Index for Fast Vector Search

```sql
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### 0.5 Create RPC Function for Similarity Search

```sql
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
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

- `<=>` is the cosine distance operator. `1 - distance = similarity`.
- `match_threshold` of 0.7 filters out low-relevance chunks.

### 0.6 Row Level Security

```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access"
ON documents FOR ALL
USING (true) WITH CHECK (true);
```

---

## Phase 1: New Dependencies ✅

Add to `pyproject.toml`:

```toml
"supabase>=2.15.0",
"pypdf>=5.0.0",
"python-multipart>=0.0.20",
```

- **`supabase`** — Python client for Supabase (includes PostgREST, `.rpc()`)
- **`pypdf`** — Pure-Python PDF parser, no native deps, Docker-friendly
- **`python-multipart`** — Required by FastAPI for `UploadFile` (file uploads)

We use `RecursiveCharacterTextSplitter` from `langchain_text_splitters` which is already available through the existing `langchain-openai` dependency chain. If not, add `langchain-text-splitters>=0.3.0`.

---

## Phase 2: Configuration Updates ✅

### 2a. Modify `app/config.py`

Add to `Settings`:

```python
# Supabase
supabase_url: str = ""
supabase_service_key: str = ""

# RAG Settings
embedding_model: str = "text-embedding-3-small"
rag_chunking_strategy: str = "recursive"
rag_chunk_size: int = 1000
rag_chunk_overlap: int = 200
rag_top_k: int = 5
rag_similarity_threshold: float = 0.7
max_upload_size_mb: int = 10
```

Add computed property:

```python
@property
def rag_enabled(self) -> bool:
    return bool(self.supabase_url and self.supabase_service_key)
```

Defaults to empty strings so the app starts without Supabase and degrades gracefully.

### 2b. Update `.env.example`

```
# Supabase (required for RAG features)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...your-service-role-key

# RAG Configuration (optional, these are defaults)
EMBEDDING_MODEL=text-embedding-3-small
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.7
MAX_UPLOAD_SIZE_MB=10
```

---

## Phase 3: Document Store Module ✅

### New file: `app/document_store.py`

**Class: `DocumentStore`**

Constructor creates:
- PostgreSQL connection via `psycopg2.connect(supabase_database_url)` (direct DB access, not REST API)
- OpenAI embeddings via `OpenAIEmbeddings(model=embedding_model)`

**Methods:**

| Method | Purpose |
|--------|---------|
| `generate_embedding(text)` | Embed a single text → 1536-dim vector |
| `generate_embeddings(texts)` | Batch embed multiple texts (single API call) |
| `insert_chunks(chunks)` | Insert list of `{content, metadata, embedding}` into Supabase. Batches in groups of 100. |
| `search_similar(query, top_k, threshold)` | Embed query → call `match_documents` RPC → return top-k results |
| `list_documents()` | Query distinct `doc_id` values with chunk counts |
| `delete_document(doc_id)` | Delete all chunks where `metadata->>'doc_id'` matches |
| `health_check()` | Lightweight query to verify Supabase connectivity |

---

## Phase 4: Document Parsing & Ingestion Pipeline

### 4a. New file: `app/document_parser.py` ✅

Separates **parsing** (extracting raw text from a file) from the rest of the ingestion pipeline. Uses a `Protocol` class so new file formats can be added without modifying existing code.

**Protocol: `DocumentParser`**

```python
class DocumentParser(Protocol):
    def parse(self, file: bytes, filename: str) -> str: ...
```

- `parse` takes raw file bytes + filename, returns the extracted plain text as a single string
- Protocol (structural typing) — any class with the right shape satisfies the contract, no inheritance required

**Class: `PdfParser`**

First concrete implementation. Uses `pypdf.PdfReader` to extract text page by page and joins with newlines.

**Factory: `get_parser(filename) -> DocumentParser`**

A module-level dict maps file extensions to parser instances (e.g. `{".pdf": PdfParser()}`). `get_parser` extracts the extension from the filename and looks it up. Raises `ValueError` for unsupported types. When a second format is added, implement a class with `parse` and add a mapping entry — `ingestion.py` stays unchanged.

### 4b. New file: `app/chunking.py` ✅

Same Protocol approach as the parser — separates **chunking** (splitting text into pieces) from the rest of the ingestion pipeline.

**Protocol: `ChunkingStrategy`**

```python
class ChunkingStrategy(Protocol):
    def chunk(self, text: str) -> list[str]: ...
```

- `chunk` takes a plain text string and returns a list of text chunks
- Protocol (structural typing) — any class with the right `chunk` method satisfies the contract

**Class: `RecursiveChunker`**

First concrete implementation. Wraps `RecursiveCharacterTextSplitter` from `langchain-text-splitters` with configurable `chunk_size` and `chunk_overlap`. Constructor takes these as parameters so they can be driven from `Settings`.

**Factory: `get_chunker(settings) -> ChunkingStrategy`**

Reads `settings.rag_chunking_strategy` (default `"recursive"`) and returns the matching chunker instance configured with `settings.rag_chunk_size` and `settings.rag_chunk_overlap`. Raises `ValueError` for unknown strategy names. When a second strategy is added, implement the class and add a mapping entry.

### 4c. New file: `app/ingestion.py` ✅

**Function: `ingest_document(file_bytes, filename, document_store, settings)`**

Orchestrator — the single entry point shared by the API endpoint and the CLI script:

1. Call `get_parser(filename).parse(file_bytes, filename)` to extract raw text
2. Call `get_chunker(settings).chunk(text)` to split into chunks
3. Generate UUID `doc_id`
4. Add metadata to each chunk: `{doc_id, source, chunk_index, total_chunks}`
5. Batch embed all chunks via `document_store.generate_embeddings()`
6. Call `document_store.insert_chunks()` to store
7. Return summary (doc_id, filename, chunk count)

Parsing and chunking are fully delegated to `document_parser` — `ingestion.py` only handles metadata, embedding, and storage.

---

## Phase 5: New API Models ✅

### Add to `app/models.py`

```python
class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_stored: int
    status: str
    processing_time_ms: float
    timestamp: str = Field(default_factory=...)

class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    chunk_count: int

class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int

class DocumentDeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int
    status: str
```

### Modify `ChatResponse`

Add optional field:

```python
sources: list[dict] | None = None
```

Each source: `{"source": "report.pdf", "similarity": 0.85, "chunk_preview": "first 200 chars..."}`

---

## Phase 6: New Endpoints in `app/main.py`

### 6.1 Lifespan Update

Add `document_store` as global instance. In `lifespan()`:

```python
if settings.rag_enabled:
    document_store = DocumentStore(settings)
    logger.info("Supabase document store initialized (RAG enabled)")
else:
    logger.info("Supabase not configured (RAG disabled)")
```

Pass `document_store` to `ProductionAgent(document_store=document_store)`.

### 6.2 New Endpoints

| Endpoint | Rate Limit | Description |
|----------|------------|-------------|
| `POST /documents` | 5/min | Upload PDF → ingest → return chunk count. Validates file type + size. |
| `GET /documents` | default | List all ingested documents with chunk counts |
| `DELETE /documents/{doc_id}` | default | Delete a document and all its chunks |

### 6.3 Health Check Update

Add `"document_store"` to health checks:

```python
"document_store": (
    document_store.health_check() if document_store else "not_configured"
)
```

---

## Phase 7: RAG Retrieval in the Agent

### Modify `app/agent.py`

**7.1 Expand `AgentState`:**

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
    context: list[dict]    # Retrieved chunks
    sources: list[dict]    # Source metadata for response
```

**7.2 Accept `document_store` in `ProductionAgent.__init__`:**

```python
def __init__(self, document_store=None):
    self.document_store = document_store
    self.rag_enabled = document_store is not None
```

**7.3 Add `retrieve` node:**

```python
def retrieve_context(state: AgentState) -> dict:
    if not self.rag_enabled:
        return {"context": [], "sources": []}
    try:
        user_message = state["messages"][-1].content
        results = self.document_store.search_similar(
            query=user_message, top_k=settings.rag_top_k,
            threshold=settings.rag_similarity_threshold,
        )
        sources = [{"source": r["metadata"].get("source"), "similarity": round(r["similarity"], 3),
                     "chunk_preview": r["content"][:200]} for r in results]
        return {"context": results, "sources": sources}
    except Exception:
        return {"context": [], "sources": []}  # degrade gracefully
```

**7.4 Modify `process_message` to inject context:**

If `state["context"]` has chunks, prepend a `SystemMessage`:

```
You are a helpful assistant. Use the following retrieved documents to answer
the user's question. If the documents don't contain relevant information,
say so and answer based on your general knowledge.

Retrieved Documents:
[Source: report.pdf]
<chunk text>
---
[Source: report.pdf]
<chunk text>
```

**7.5 New graph flow:**

```
START → retrieve → process → (done | fallback | error) → END
```

**7.6 Update `invoke` return:**

```python
return {
    "response": result["messages"][-1].content,
    "model_used": result.get("model_used", "unknown"),
    "error": result.get("error"),
    "sources": result.get("sources", []),
}
```

---

## Phase 8: CLI Ingestion Script

### New file: `scripts/ingest.py`

```
Usage:
  uv run python scripts/ingest.py ./data/report.pdf        # single file
  uv run python scripts/ingest.py ./data/pdfs/              # directory
```

- Validates `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` are set
- Finds all `.pdf` files (recursive if directory)
- Calls `ingest_document()` for each
- Prints progress: filename, chunks created, doc_id

---

## Phase 9: Deployment Updates

### `scripts/setup-cloud-run.sh`

Add to secrets loop:

```bash
for SECRET in OPENAI_API_KEY LANGCHAIN_API_KEY SUPABASE_URL SUPABASE_SERVICE_KEY; do
```

### `.github/workflows/deploy-cloud-run.yml`

Add to `--set-secrets`:

```
SUPABASE_SERVICE_KEY=SUPABASE_SERVICE_KEY:latest
```

Add to `--set-env-vars`:

```
SUPABASE_URL=https://your-project.supabase.co
```

(`SUPABASE_URL` is a public endpoint, not secret. `SUPABASE_SERVICE_KEY` goes through Secret Manager.)

### Dockerfile

No changes needed — `pypdf` is pure Python, no system deps. `uv sync` picks up new deps automatically.

---

## Files Summary

| File | Action | What Changes |
|------|--------|-------------|
| `app/document_store.py` | **Create** | DocumentStore class (ingest, search, list, delete) |
| `app/document_parser.py` | **Create** | DocumentParser Protocol + PdfParser + get_parser factory; ChunkingStrategy Protocol + RecursiveChunker |
| `app/ingestion.py` | **Create** | Metadata + embedding + storage pipeline (uses document_parser for text extraction and chunking) |
| `scripts/ingest.py` | **Create** | CLI bulk ingestion script |
| `app/config.py` | Modify | Add Supabase + RAG config fields |
| `app/agent.py` | Modify | Add retrieve node, context in state, system prompt |
| `app/main.py` | Modify | Init DocumentStore, add /documents endpoints, health check |
| `app/models.py` | Modify | Add document response models, sources in ChatResponse |
| `pyproject.toml` | Modify | Add supabase, pypdf, python-multipart |
| `.env.example` | Modify | Add SUPABASE_URL, SUPABASE_SERVICE_KEY, RAG settings |
| `scripts/setup-cloud-run.sh` | Modify | Add Supabase secrets |
| `.github/workflows/deploy-cloud-run.yml` | Modify | Add Supabase secrets to deploy |

---

## Graceful Degradation

The design is built around **RAG being optional**:

1. **No Supabase configured** → `rag_enabled=False`, `document_store=None`, agent skips retrieval, existing chat behavior preserved
2. **Supabase down at runtime** → retrieve node catches exceptions, returns empty context, LLM still responds
3. **Embedding API failure** → same as above, degrade to non-RAG
4. **Cache interaction** → cache keys on user message, RAG-augmented responses are cached normally (TTL=300s bounds staleness)

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parser abstraction | `Protocol` class | Structural typing — no inheritance needed; new formats added by implementing `parse` + adding a dict entry |
| PDF parser | `pypdf` via `PdfParser` | Pure Python, no native deps, Docker-friendly |
| Chunking abstraction | `Protocol` class | Same pattern as parser; swap to semantic/structure-aware chunking by implementing `chunk` |
| Text splitter | `RecursiveCharacterTextSplitter` via `RecursiveChunker` | Industry standard, respects semantic boundaries |
| Chunk size | 1000 chars / 200 overlap | Standard for general-purpose RAG; configurable via env |
| Embedding model | `text-embedding-3-small` | $0.02/1M tokens, 1536 dims, strong quality |
| Vector index | HNSW with cosine ops | Fast approximate search, pgvector standard |
| Retrieval count | Top 5, threshold 0.7 | Balances context quality vs token cost; configurable |
| Batch embedding | `embed_documents()` | Single API call for all chunks; efficient |
| Supabase client | Sync, wrapped in `asyncio.to_thread()` | Simpler than async client; non-blocking in FastAPI |

---

## Verification

1. **Supabase connection**: `GET /health` → `document_store: true`
2. **Upload**: `curl -X POST -F "file=@test.pdf" http://localhost:8000/documents` → chunks in Supabase
3. **List**: `GET /documents` → shows uploaded document
4. **RAG query**: `POST /chat` with question about PDF → response uses PDF content, `sources` field populated
5. **Fallback**: Invalid Supabase creds → `/chat` still works (direct LLM, no retrieval)
6. **CLI**: `uv run python scripts/ingest.py ./test_pdfs/` → documents appear in Supabase
7. **Delete**: `DELETE /documents/{doc_id}` → chunks removed
8. **CI**: `ruff check .`, `ruff format --check .`, `mypy app/`, `pytest tests/ -v`
