# Production RAG API

A production-ready Chat/RAG API built with **FastAPI + LangGraph + OpenAI**, featuring security, caching, observability, and deployment infrastructure.

## Architecture

```
Client Request → Rate Limiter → Security (injection + PII) → Cache → Retrieval (similarity | BM25 | hybrid) → LangGraph Agent → Output Validation → Metrics/Logging → JSON Response
```

### Request Flow

1. **Rate Limiter** — Per-IP throttling via slowapi (configurable, default 20/min)
2. **Security Middleware** — Prompt injection detection and PII masking (email, phone, SSN, credit card)
3. **Cache Layer** — SHA256-keyed in-memory cache with TTL. Returns cached response on hit, continues on miss.
4. **LangGraph Agent** — Primary model → fallback model → graceful error message. Retry logic with configurable max retries.
5. **Output Validation** — PII leak detection and harmful content filtering on LLM responses
6. **Metrics + Logging** — Structured JSON logs (ELK/Datadog-ready), request count, latency, token usage, error and cache hit rates

### Retrieval Strategies

The system supports three retrieval strategies, configurable via `RAG_RETRIEVAL_STRATEGY`:

| Strategy | How it works | Best for |
|---|---|---|
| `similarity` | Cosine similarity via pgvector embeddings | Semantic/intent-based queries |
| `bm25` | Postgres full-text search (tsvector/tsquery) | Exact keyword/term matching |
| `hybrid` (default) | RRF fusion of similarity + BM25 | Mixed queries — combines semantic understanding with exact term matching |

Strategies follow a `RetrievalStrategy` Protocol (same pattern as `DocumentParser` and `ChunkingStrategy`), so new strategies can be added without touching existing code.

### LangGraph Agent Flow

```
START → retrieve (RetrievalStrategy) → process (primary model)
                                           ├── success → END
                                           └── fail → fallback (secondary model)
                                                          ├── success → END
                                                          └── fail → error (graceful message) → END
```

## Project Structure

```
app/
├── main.py            # FastAPI app, endpoints, lifespan, rate limiting
├── config.py          # Pydantic-settings validated environment config
├── models.py          # Request/response Pydantic models
├── agent.py           # LangGraph agent with retry + fallback
├── retrieval.py       # RetrievalStrategy Protocol (similarity, BM25, hybrid)
├── document_store.py  # pgvector vector store + full-text search (ThreadedConnectionPool)
├── document_parser.py # Document parsing Protocol (PDF, text)
├── chunking.py        # Chunking strategy Protocol (recursive text splitter)
├── ingestion.py       # Document ingestion pipeline (parse → chunk → embed → store)
├── security.py        # Input sanitization, PII detection/masking, output validation
├── cache.py           # In-memory response cache with TTL
└── monitoring.py      # Structured JSON logging, metrics collector, request timer

scripts/
├── ingest.py          # CLI for batch document ingestion
└── generate_documents.py  # Generate sample insurance documents

supabase/migrations/
├── 001_create_documents.sql      # Documents table, pgvector HNSW index, match_documents RPC
└── 002_add_full_text_search.sql  # tsvector column, GIN index, bm25_search RPC
```

## Features

| Feature | Implementation | Details |
|---|---|---|
| LangSmith Tracing | `@traceable` decorators | Every request traced with metadata (EU endpoint) |
| Connection Pooling | `document_store.py` | ThreadedConnectionPool (min=2, max=10) reuses DB connections across requests |
| Graceful Degradation | `main.py` lifespan | App starts with RAG disabled if database is unreachable |
| Input Sanitization | `security.py` | Blocks prompt injection attempts |
| PII Detection/Masking | `security.py` | Redacts emails, SSNs, phone numbers, credit cards |
| Retrieval Strategies | `retrieval.py` | Similarity, BM25, and hybrid (RRF) search |
| Document Ingestion | `ingestion.py` | Parse → chunk → embed → store pipeline |
| Document Parsing | `document_parser.py` | Protocol-based PDF and text parsing |
| Error Handling + Retries | `agent.py` | Primary → fallback model with graceful degradation |
| Response Caching | `cache.py` | In-memory cache for duplicate calls |
| Rate Limiting | `main.py` + slowapi | Per-IP throttling |
| Structured Logging | `monitoring.py` | JSON logs for production aggregation |
| Metrics Collection | `monitoring.py` | Request count, latency, token usage |
| Health Checks | `main.py` `/health` | Docker/Kubernetes readiness endpoint |
| Docker Deployment | `Dockerfile` + `docker-compose.yml` | Non-root user, health check, layer caching |
| Cloud Run Deployment | GitHub Actions + Cloud Run | Artifact Registry, Secret Manager, Workload Identity Federation |

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Local Development

```bash
# Clone the repo
git clone <repo-url>
cd prod_rag

# Copy environment file and fill in your keys
cp .env.example .env

# Install dependencies
uv sync

# Run the server
uv run uvicorn app.main:app --reload --port 8000
```

### Supabase Setup

1. Create a project at [supabase.com/dashboard](https://supabase.com/dashboard)
2. Run the SQL migration against your Supabase database:

```bash
# Apply both migrations
psql -d "$SUPABASE_DATABASE_URL" -f supabase/migrations/001_create_documents.sql
psql -d "$SUPABASE_DATABASE_URL" -f supabase/migrations/002_add_full_text_search.sql
```

3. Add `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, and `SUPABASE_DATABASE_URL` to your `.env`. Use the **transaction pooler** connection string (Settings → Database → Connection string → Mode: Transaction, port 6543) rather than the direct connection (port 5432)

### Docker

```bash
docker compose up --build
```

### Google Cloud Run

Deployed manually via GitHub Actions workflow dispatch (Actions → Deploy to Cloud Run → Run workflow).

**One-time GCP setup:**

The setup script provisions all required GCP resources (Artifact Registry, Secret Manager, Workload Identity Federation, IAM). It is idempotent and safe to run multiple times.

```bash
# Required env vars
export GCP_PROJECT_ID=your-project-id
export GITHUB_REPO=owner/repo       # e.g. shlbatra/prod_rag
export GCP_REGION=us-central1       # optional, defaults to us-central1

# Run setup with default service name (prod-rag-api)
bash scripts/setup-cloud-run.sh

# Or specify a custom service name
bash scripts/setup-cloud-run.sh my-custom-api
```

After the script completes, it prints the GitHub secrets and variables to configure.

**GitHub repository configuration:**

| Name | Type | Value |
|---|---|---|
| `GCP_PROJECT_ID` | Variable | Your GCP project ID |
| `GCP_REGION` | Variable | e.g. `us-central1` |
| `GCP_WIF_PROVIDER` | Secret | Workload Identity provider resource name (printed by setup script) |
| `GCP_WIF_SERVICE_ACCOUNT` | Secret | WIF service account email (printed by setup script) |

### Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `LANGSMITH_API_KEY` | LangSmith API key | (optional) |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint | `https://eu.api.smith.langchain.com` |
| `LANGSMITH_PROJECT` | LangSmith project name | `Prod RAG Project` |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | `true` |
| `PRIMARY_MODEL` | Primary LLM model | `gpt-4.1-mini` |
| `FALLBACK_MODEL` | Fallback LLM model | `gpt-4.1-nano` |
| `SUPABASE_DATABASE_URL` | Postgres connection string (use transaction pooler) | (required for RAG) |
| `DB_POOL_MIN_CONN` | Minimum pooled DB connections | `2` |
| `DB_POOL_MAX_CONN` | Maximum pooled DB connections | `10` |
| `RAG_RETRIEVAL_STRATEGY` | Retrieval strategy | `hybrid` |
| `RAG_TOP_K` | Number of chunks to retrieve | `5` |
| `RAG_SIMILARITY_THRESHOLD` | Minimum similarity score | `0.7` |
| `RATE_LIMIT` | Rate limit per IP | `20/minute` |
| `CACHE_TTL_SECONDS` | Cache entry lifetime | `300` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Main chat endpoint |
| `POST` | `/documents` | Upload a document for RAG ingestion |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{doc_id}` | Delete a document and its chunks |
| `GET` | `/health` | Health check for Docker/K8s |
| `GET` | `/metrics` | Application metrics |
| `GET` | `/cache/stats` | Cache performance statistics |
