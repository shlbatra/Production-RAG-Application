# Issues and Gaps to Address

## 1. No Actual RAG (Retrieval) — This Is a Chat API, Not RAG Yet

The agent calls the LLM directly with no retrieval step. To make this a true RAG system, you'd need:

- A vector store using **Supabase** (pgvector via Supabase's built-in Vector support)
- A document ingestion pipeline (chunking, embedding, indexing into Supabase)
- A retrieval step in the LangGraph agent that queries Supabase for relevant context before calling the LLM

## 2. Model Config Mismatch Between `config.py` and `render.yml`

- `config.py` defaults: `gpt-4.1-mini` / `gpt-4.1-nano`
- `render.yml` specifies: `gpt-4o-mini` / `gpt-4o-mini`

These should be aligned to avoid confusion about which model is actually running in production.

## 3. Cache Limitations

The in-memory cache (`cache.py`) has documented but unaddressed limitations:

- **No size limit** — unbounded memory growth under sustained load
- **No background eviction** — stale entries linger until accessed
- **Not shared across instances** — each container has its own cache, leading to redundant LLM calls

**Recommendation:** Replace with **Redis** (or a comparable fast-lookup store like Memcached or DragonflyDB):

- **Shared cache across instances** — all containers read/write from one Redis, eliminating redundant LLM calls
- **Built-in TTL** — keys expire automatically, no lazy eviction needed
- **LRU/LFU eviction policies** — bounded memory with configurable `maxmemory-policy`
- **Persistence options** — RDB snapshots or AOF logging survive restarts
- **Sub-millisecond lookups** — negligible overhead compared to LLM call latency
- **Managed options** — Redis Cloud, AWS ElastiCache, GCP Memorystore, or Supabase Redis (keeps infra co-located with the vector store)

Use `redis-py` with async support (`aioredis`) to keep the FastAPI async flow non-blocking.

## 4. Token Counting Is a Rough Estimate

`main.py:200` uses `len(split()) * 1.3` to estimate tokens. This is fine for directional metrics but inaccurate for cost tracking or billing.

**Recommendation:** Use `tiktoken` for accurate OpenAI token counts.

## 5. No Tests

`tests/__init__.py` exists but is empty. There are no unit or integration tests covering:

- Security pipeline (injection detection, PII masking)
- Cache behavior (TTL, hit/miss)
- Agent retry/fallback logic
- API endpoint responses and error handling

## 6. Missing CORS Configuration

No `CORSMiddleware` is configured. Frontend clients making requests from a browser will be blocked by CORS policy.

**Fix:** Add `fastapi.middleware.cors.CORSMiddleware` with appropriate origin allowlist.

## 7. ~~No Cloud Run Deployment for GCP~~ RESOLVED

**Implemented:** GitHub Actions workflow (`.github/workflows/deploy-cloud-run.yml`) with Workload Identity Federation, Artifact Registry, and Secret Manager. See README for setup.

The project has deployment configs for Docker Compose (local) and Render (PaaS), but no setup for **Google Cloud Run**, which is a strong fit for production:

- Fully managed, scales to zero, pay-per-request
- Native Docker container support (the existing `Dockerfile` works as-is)
- Built-in HTTPS, load balancing, and IAM integration
- Integrates with Secret Manager for API keys, Cloud Logging for structured logs, and Artifact Registry for container images

**Recommendation:** Add a Cloud Run deployment workflow:

1. Push the Docker image to **Artifact Registry**
2. Deploy via `gcloud run deploy` or a **Cloud Build** trigger (`cloudbuild.yaml`)
3. Inject secrets (`OPENAI_API_KEY`, `LANGCHAIN_API_KEY`) via **Secret Manager** environment variables
4. Set concurrency, memory, and CPU limits appropriate for LLM call latency

## 8. Metrics Are In-Memory Only — No External Observability

The `MetricsCollector` in `monitoring.py` stores counters in process memory. Metrics are lost on restart and invisible to external dashboards. The structured JSON logs are written to stdout but not shipped anywhere.

**Recommendation:** Integrate with an observability platform such as **Datadog**, Prometheus/Grafana, or Google Cloud Monitoring:

- **Metrics:** Replace the in-memory collector with `ddtrace` (Datadog) or `prometheus_client` to emit request latency, error rate, cache hit rate, and token usage as real metrics with tags
- **Logging:** Ship structured JSON logs to Datadog Logs, Cloud Logging, or an ELK stack via a log agent or stdout forwarding
- **Tracing:** Connect LangSmith traces or OpenTelemetry spans to the observability backend for end-to-end request tracing
- **Dashboards & Alerts:** Set up alerts on error rate spikes, latency P95/P99 thresholds, and token budget burn rate

## 9. Thread/Conversation State Not Persisted

`thread_id` is passed through the request/response but the agent doesn't use it — every request is stateless. Multi-turn conversations are not supported.

**Recommendation:** Use LangGraph's built-in checkpointing (e.g., `MemorySaver` or a persistent backend) to maintain conversation state across requests.
