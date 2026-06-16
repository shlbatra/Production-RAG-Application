# Plan: Add Google Cloud Run Deployment (Issue #7)

## Context

The project has deployment configs for Docker Compose (local) and Render (PaaS), but no setup for Google Cloud Run. Issue #7 in ISSUES.md calls for adding a Cloud Run deployment workflow with Artifact Registry, Secret Manager, and CI/CD integration. The existing Dockerfile already works as-is for Cloud Run — no application code changes needed.

## Key Design Decisions

1. **Port handling**: Keep port 8000 hardcoded in the Dockerfile CMD. Tell Cloud Run to use port 8000 via `--port=8000`. This avoids touching the Dockerfile, docker-compose, or HEALTHCHECK.

2. **GitHub Actions for CI/CD**: Use a GitHub Actions workflow to build, push, and deploy on push to main. Authentication via Workload Identity Federation (keyless, no service account JSON keys).

3. **Resource limits**: 512Mi memory, 1 CPU, 80 concurrency, 120s timeout, 0-3 instances. The app is I/O-bound (waiting on LLM API calls), so these are adequate.

## Files to Create

### 1. `.github/workflows/deploy-cloud-run.yml` — GitHub Actions CI/CD workflow

Triggered on push to `main`. Steps:

1. **Checkout** code
2. **Authenticate to GCP** via `google-github-actions/auth` using Workload Identity Federation
3. **Set up Cloud SDK** via `google-github-actions/setup-gcloud`
4. **Configure Docker** for Artifact Registry (`gcloud auth configure-docker`)
5. **Build and push** Docker image to Artifact Registry, tagged with both `$GITHUB_SHA` (traceability/rollback) and `latest`
6. **Deploy to Cloud Run** with:
   - `--port=8000`
   - `--memory=512Mi`, `--cpu=1`, `--concurrency=80`, `--timeout=120`
   - `--min-instances=0`, `--max-instances=3`
   - `--set-secrets` for OPENAI_API_KEY and LANGCHAIN_API_KEY from Secret Manager
   - `--set-env-vars` for APP_ENV, LOG_LEVEL, RATE_LIMIT, CACHE_TTL_SECONDS, MAX_RETRIES, PRIMARY_MODEL, FALLBACK_MODEL, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT
   - `--allow-unauthenticated`

GitHub repository secrets/variables needed:

| Secret/Variable | Type | Purpose |
|---|---|---|
| `GCP_PROJECT_ID` | variable | GCP project ID |
| `GCP_REGION` | variable | Deployment region (e.g. us-central1) |
| `GCP_WIF_PROVIDER` | secret | Workload Identity Federation provider resource name |
| `GCP_WIF_SERVICE_ACCOUNT` | secret | Service account email for WIF |

### 2. `.gcloudignore` — Build context filter

Excludes `.git`, `.venv`, `__pycache__`, `.env`, `*.png`, `tests/`, `.claude/` from any gcloud uploads.

## Files to Modify

### 3. `README.md` — Add Cloud Run section

- Add row to the Features table: `Cloud Run Deployment | GitHub Actions + Cloud Run | Artifact Registry, Secret Manager, Workload Identity Federation`
- Add "Google Cloud Run" subsection under Setup (after Docker) with:
  - GCP prerequisites (Artifact Registry repo, Secret Manager secrets, Workload Identity Federation setup)
  - Required GitHub repository secrets/variables
  - How deployment is triggered (push to main)
  - How to set secret values in Secret Manager

### 4. `docs/ISSUES.md` — Mark issue #7 resolved

Add a resolved note to the section 7 heading.

## What Does NOT Change

- **Dockerfile** — no modifications
- **docker-compose.yml** — untouched
- **render.yml** — untouched
- **app/ source code** — zero changes

## Verification

1. Review the GitHub Actions workflow for correct step ordering and action versions
2. Confirm Workload Identity Federation auth is configured correctly (no JSON key files)
3. Confirm README instructions cover all GCP prerequisite setup
4. Confirm no existing deployment paths (docker-compose, Render) are broken
