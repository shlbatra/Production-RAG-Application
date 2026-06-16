# Production RAG API

A production-ready Chat/RAG API built with **FastAPI + LangGraph + OpenAI**, featuring security, caching, observability, and deployment infrastructure.

## Architecture

```
Client Request → Rate Limiter → Security (injection + PII) → Cache → LangGraph Agent → Output Validation → Metrics/Logging → JSON Response
```

### Request Flow

1. **Rate Limiter** — Per-IP throttling via slowapi (configurable, default 20/min)
2. **Security Middleware** — Prompt injection detection and PII masking (email, phone, SSN, credit card)
3. **Cache Layer** — SHA256-keyed in-memory cache with TTL. Returns cached response on hit, continues on miss.
4. **LangGraph Agent** — Primary model → fallback model → graceful error message. Retry logic with configurable max retries.
5. **Output Validation** — PII leak detection and harmful content filtering on LLM responses
6. **Metrics + Logging** — Structured JSON logs (ELK/Datadog-ready), request count, latency, token usage, error and cache hit rates

### LangGraph Agent Flow

```
START → process (primary model)
           ├── success → END
           └── fail → fallback (secondary model)
                          ├── success → END
                          └── fail → error (graceful message) → END
```

## Project Structure

```
app/
├── main.py          # FastAPI app, endpoints, lifespan, rate limiting
├── config.py        # Pydantic-settings validated environment config
├── models.py        # Request/response Pydantic models
├── agent.py         # LangGraph agent with retry + fallback
├── security.py      # Input sanitization, PII detection/masking, output validation
├── cache.py         # In-memory response cache with TTL
└── monitoring.py    # Structured JSON logging, metrics collector, request timer
```

## Features

| Feature | Implementation | Details |
|---|---|---|
| LangSmith Tracing | `@traceable` decorators | Every request traced with metadata |
| Input Sanitization | `security.py` | Blocks prompt injection attempts |
| PII Detection/Masking | `security.py` | Redacts emails, SSNs, phone numbers, credit cards |
| Error Handling + Retries | `agent.py` | Primary → fallback model with graceful degradation |
| Response Caching | `cache.py` | In-memory cache for duplicate calls |
| Rate Limiting | `main.py` + slowapi | Per-IP throttling |
| Structured Logging | `monitoring.py` | JSON logs for production aggregation |
| Metrics Collection | `monitoring.py` | Request count, latency, token usage |
| Health Checks | `main.py` `/health` | Docker/Kubernetes readiness endpoint |
| Docker Deployment | `Dockerfile` + `docker-compose.yml` | Non-root user, health check, layer caching |
| Render Deployment | `render.yml` | Infrastructure as Code with secret separation |
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

### Docker

```bash
docker compose up --build
```

### Google Cloud Run

Deploys automatically on push to `main` via GitHub Actions.

**GCP prerequisites (one-time setup):**

1. Create an Artifact Registry Docker repo:
   ```bash
   gcloud artifacts repositories create prod-rag \
     --repository-format=docker \
     --location=us-central1
   ```

2. Create secrets in Secret Manager:
   ```bash
   echo -n 'sk-your-key' | gcloud secrets create OPENAI_API_KEY --data-file=-
   echo -n 'lsv2_your-key' | gcloud secrets create LANGCHAIN_API_KEY --data-file=-
   ```

3. Set up [Workload Identity Federation](https://github.com/google-github-actions/auth#workload-identity-federation-through-a-service-account) for GitHub Actions.

4. Grant the WIF service account these roles:
   - `roles/run.admin`
   - `roles/iam.serviceAccountUser`
   - `roles/artifactregistry.writer`
   - `roles/secretmanager.secretAccessor`

**GitHub repository configuration:**

| Name | Type | Value |
|---|---|---|
| `GCP_PROJECT_ID` | Variable | Your GCP project ID |
| `GCP_REGION` | Variable | e.g. `us-central1` |
| `GCP_WIF_PROVIDER` | Secret | Workload Identity provider resource name |
| `GCP_WIF_SERVICE_ACCOUNT` | Secret | WIF service account email |

### Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `LANGCHAIN_API_KEY` | LangSmith API key | (optional) |
| `PRIMARY_MODEL` | Primary LLM model | `gpt-4.1-mini` |
| `FALLBACK_MODEL` | Fallback LLM model | `gpt-4.1-nano` |
| `RATE_LIMIT` | Rate limit per IP | `20/minute` |
| `CACHE_TTL_SECONDS` | Cache entry lifetime | `300` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Main chat endpoint |
| `GET` | `/health` | Health check for Docker/K8s |
| `GET` | `/metrics` | Application metrics |
| `GET` | `/cache/stats` | Cache performance statistics |
