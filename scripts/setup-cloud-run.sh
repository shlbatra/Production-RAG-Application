#!/usr/bin/env bash
set -euo pipefail

# Load .env if present
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${1:-prod-rag-api}"
REPO_NAME="prod-rag"

echo "=== Cloud Run Setup ==="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "Repo:     ${REPO_NAME}"
echo ""

# 1. Enable required APIs
echo "--- Enabling GCP APIs ---"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Create Artifact Registry repo
echo "--- Setting up Artifact Registry ---"
gcloud artifacts repositories describe "${REPO_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" 2>/dev/null \
|| gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Production RAG API images" \
  --project="${PROJECT_ID}"

# 3. Create secrets in Secret Manager
echo "--- Setting up secrets ---"
for SECRET in OPENAI_API_KEY LANGCHAIN_API_KEY; do
  if gcloud secrets describe "${SECRET}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Secret ${SECRET} already exists"
  else
    echo "Creating secret: ${SECRET}"
    printf "PLACEHOLDER" | gcloud secrets create "${SECRET}" \
      --data-file=- \
      --replication-policy=automatic \
      --project="${PROJECT_ID}"
    echo ""
    echo "  Set the real value with:"
    echo "  echo -n 'your-key' | gcloud secrets versions add ${SECRET} --data-file=- --project=${PROJECT_ID}"
    echo ""
  fi
done

# 4. Grant Cloud Run default SA access to secrets
echo "--- Configuring IAM ---"
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for SECRET in OPENAI_API_KEY LANGCHAIN_API_KEY; do
  gcloud secrets add-iam-policy-binding "${SECRET}" \
    --member="serviceAccount:${SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="${PROJECT_ID}" \
    --quiet
done

# 5. Set up Workload Identity Federation for GitHub Actions
echo "--- Setting up Workload Identity Federation ---"
WIF_POOL="github-actions-pool"
WIF_PROVIDER="github-actions-provider"

if gcloud iam workload-identity-pools describe "${WIF_POOL}" \
  --location="global" --project="${PROJECT_ID}" &>/dev/null; then
  echo "Workload Identity Pool already exists"
else
  gcloud iam workload-identity-pools create "${WIF_POOL}" \
    --location="global" \
    --display-name="GitHub Actions Pool" \
    --project="${PROJECT_ID}"
fi

if gcloud iam workload-identity-pools providers describe "${WIF_PROVIDER}" \
  --workload-identity-pool="${WIF_POOL}" \
  --location="global" --project="${PROJECT_ID}" &>/dev/null; then
  echo "Workload Identity Provider already exists"
else
  gcloud iam workload-identity-pools providers create-oidc "${WIF_PROVIDER}" \
    --workload-identity-pool="${WIF_POOL}" \
    --location="global" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --project="${PROJECT_ID}"
fi

# 6. Create a service account for GitHub Actions
GH_SA="github-actions-deployer"
GH_SA_EMAIL="${GH_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "${GH_SA_EMAIL}" --project="${PROJECT_ID}" &>/dev/null; then
  echo "Service account ${GH_SA_EMAIL} already exists"
else
  gcloud iam service-accounts create "${GH_SA}" \
    --display-name="GitHub Actions Cloud Run Deployer" \
    --project="${PROJECT_ID}"
fi

# Grant the SA required roles
for ROLE in roles/run.admin roles/iam.serviceAccountUser roles/artifactregistry.writer roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${GH_SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet
done

# Allow GitHub Actions to impersonate the SA
REPO="${GITHUB_REPO:?Set GITHUB_REPO env var (e.g. owner/repo)}"
WIF_POOL_ID=$(gcloud iam workload-identity-pools describe "${WIF_POOL}" \
  --location="global" --project="${PROJECT_ID}" --format='value(name)')

gcloud iam service-accounts add-iam-policy-binding "${GH_SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${WIF_POOL_ID}/attribute.repository/${REPO}" \
  --project="${PROJECT_ID}" \
  --quiet

# 7. Print GitHub secrets to configure
WIF_PROVIDER_ID=$(gcloud iam workload-identity-pools providers describe "${WIF_PROVIDER}" \
  --workload-identity-pool="${WIF_POOL}" \
  --location="global" --project="${PROJECT_ID}" --format='value(name)')

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Add these to your GitHub repository settings:"
echo ""
echo "  Variables:"
echo "    GCP_PROJECT_ID = ${PROJECT_ID}"
echo "    GCP_REGION     = ${REGION}"
echo ""
echo "  Secrets:"
echo "    GCP_WIF_PROVIDER        = ${WIF_PROVIDER_ID}"
echo "    GCP_WIF_SERVICE_ACCOUNT = ${GH_SA_EMAIL}"
echo ""
echo "Then set your real API keys:"
echo "  echo -n 'sk-...' | gcloud secrets versions add OPENAI_API_KEY --data-file=- --project=${PROJECT_ID}"
echo "  echo -n 'lsv2_...' | gcloud secrets versions add LANGCHAIN_API_KEY --data-file=- --project=${PROJECT_ID}"
