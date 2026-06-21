# Evaluation System for Production RAG Pipeline

## Context

This RAG system has 6 evaluable stages: parsing, chunking, retrieval, generation, security, and end-to-end. There are no evaluations today beyond unit tests with mocks. We need a golden-set-driven evaluation framework that measures each component independently and the full pipeline together, runs in CI, detects regressions, and scales to hundreds of test cases.

The design follows existing codebase patterns: Protocol-based abstractions, Pydantic models, `pydantic-settings` for config, pytest for test integration.

---

## 1. Golden Set Design

### Schema (`evals/models.py`)

Each golden case carries enough data to drive every evaluator:

```python
class GoldenCase(BaseModel):
    id: str                                    # e.g. "factual-001"
    category: Literal[
        "factual",        # answer in a single chunk
        "multi_hop",      # requires 2+ chunks
        "unanswerable",   # answer NOT in corpus — should refuse
        "adversarial",    # prompt injection attempts
        "pii",            # contains PII that should be masked
    ]
    question: str
    expected_answer: str                       # reference answer for LLM-judge
    expected_source_files: list[str]           # e.g. ["report.pdf"]
    expected_chunk_contents: list[str]         # substrings that MUST appear in retrieved chunks
    difficulty: Literal["easy", "medium", "hard"]
    tags: list[str] = []
```

### Storage: JSONL

- One case per line in `evals/golden_sets/v1.0.0.jsonl` — git-diff-friendly, streams line-by-line for large sets.
- `evals/golden_sets/manifest.json` tracks the active version.
- Companion test PDFs live in `evals/fixtures/documents/`.

### Category Distribution Target

| Category | % | Purpose |
|---|---|---|
| factual | 40% | Correct retrieval + accurate answer |
| multi_hop | 20% | Cross-chunk / cross-doc synthesis |
| unanswerable | 15% | System refuses rather than hallucinating |
| adversarial | 15% | Blocked by `InputSanitizer` |
| pii | 10% | Masked by `PIIDetector` |

---

## 2. Component-Level Evaluators

All evaluators implement a common `ComponentEvaluator` Protocol (in `evals/protocols.py`):

```python
class ComponentEvaluator(Protocol):
    def evaluate(self, cases: list[GoldenCase]) -> EvalResult: ...
```

### 2a. Chunking Evaluator (`evals/evaluators/chunking_eval.py`)

Runs `RecursiveChunker.chunk()` on fixture documents. **No LLM calls** — pure algorithmic.

| Metric | How | Threshold |
|---|---|---|
| Chunk Size Compliance | % of chunks within `chunk_size` ± 10% | >= 0.90 |
| Boundary Quality | % of chunks NOT breaking mid-sentence | >= 0.70 |
| Information Preservation | char coverage ratio of joined chunks vs original | >= 0.99 |
| Overlap Correctness | overlap between adjacent chunks matches config | >= 0.90 |

### 2b. Retrieval Evaluator (`evals/evaluators/retrieval_eval.py`)

Calls `DocumentStore.search_similar()` for each golden case. Compares results against `expected_source_files` and `expected_chunk_contents`.

| Metric | How | Threshold |
|---|---|---|
| Hit Rate | % of queries where >= 1 relevant chunk in top-k | >= 0.90 |
| MRR | Mean Reciprocal Rank of first relevant result | >= 0.70 |
| Precision@5 | relevant / retrieved at k=5 | >= 0.60 |
| Recall@5 | retrieved relevant / total relevant | >= 0.70 |
| NDCG@5 | Normalized Discounted Cumulative Gain | >= 0.65 |

**Relevance definition**: a chunk is relevant if its source file is in `expected_source_files` AND at least one `expected_chunk_contents` substring appears in the chunk content.

### 2c. Generation Evaluator — LLM-as-Judge (`evals/evaluators/generation_eval.py`)

Invokes `ProductionAgent.invoke()` for each case, then sends the response + context to an `LLMJudge` (gpt-4.1, stronger than the gpt-4.1-mini being evaluated).

**Judge** (`evals/judges/llm_judge.py`) scores each dimension 1-5 using structured prompts in `evals/judges/prompts/*.txt`:

| Dimension | What it measures | Threshold (avg) |
|---|---|---|
| Faithfulness | Is the answer grounded in retrieved context? | >= 4.0 |
| Correctness | Does it match the expected answer? | >= 3.5 |
| Relevance | Does it address the question? | >= 4.0 |
| Completeness | Does it cover all key points from expected answer? | >= 3.5 |
| Refusal Accuracy | For `unanswerable` cases, does it refuse? | >= 0.90 |

### 2d. Security Evaluator (`evals/evaluators/security_eval.py`)

Runs `SecurityPipeline.check_input()` and `PIIDetector.mask()` directly. **No LLM calls**.

| Metric | Threshold |
|---|---|
| Injection Detection Rate (adversarial cases blocked) | >= 0.95 |
| Injection False Positive Rate (legitimate cases wrongly blocked) | <= 0.05 |
| PII Masking Recall (PII instances masked) | >= 0.98 |

Extra test vectors in `evals/fixtures/security_vectors.jsonl`.

---

## 3. End-to-End Evaluator (`evals/evaluators/e2e_eval.py`)

Runs the full pipeline as `/chat` does: security check -> agent invoke -> output validation. Collects all generation quality metrics (delegated to generation evaluator) PLUS:

| Metric | How |
|---|---|
| Latency P50/P95/P99 | `RequestTimer` (existing utility in `app/monitoring.py`) wrapping each invoke |
| Cost per query | Token counts from each model x pricing table (`evals/metering.py`) |
| Error rate | % that hit the error_handler node |
| Fallback rate | % that fell back to gpt-4.1-nano |

### Cost Meter (`evals/metering.py`)

Tracks token usage per model and computes dollar cost:

```python
class CostMeter:
    PRICING = {
        "gpt-4.1-mini":           {"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000},
        "gpt-4.1-nano":           {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
        "text-embedding-3-small": {"input": 0.02 / 1_000_000},
        "gpt-4.1":                {"input": 2.00 / 1_000_000, "output": 8.00 / 1_000_000},  # judge
    }

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float: ...
    def total_cost(self) -> float: ...
    def summary(self) -> dict: ...
```

---

## 4. Infrastructure

### Directory Structure

```
evals/
    __init__.py
    models.py              # GoldenCase, EvalResult, JudgeScore, EvalReport
    protocols.py           # ComponentEvaluator Protocol
    config.py              # EvalSettings (pydantic-settings, EVAL_ prefix)
    loader.py              # load_golden_set(version, categories, max_cases)
    runner.py              # EvalRunner orchestrator
    report.py              # JSON + markdown report generation
    regression.py          # RegressionDetector — diff two runs
    metering.py            # CostMeter
    evaluators/
        __init__.py
        chunking_eval.py
        retrieval_eval.py
        generation_eval.py
        security_eval.py
        e2e_eval.py
    judges/
        __init__.py
        llm_judge.py
        prompts/
            faithfulness.txt
            correctness.txt
            relevance.txt
            completeness.txt
    fixtures/
        documents/          # test PDFs for ingestion
        security_vectors.jsonl
    golden_sets/
        manifest.json
        v1.0.0.jsonl
    results/                # .gitignored, stores run outputs
        .gitkeep
scripts/
    run_evals.py            # CLI entry point
```

### Configuration (`evals/config.py`)

`EvalSettings(BaseSettings)` with `env_prefix="EVAL_"`, following same pattern as `app/config.py`. All thresholds configurable via env vars. Key fields:

```python
class EvalSettings(BaseSettings):
    golden_set_version: str = "v1.0.0"
    golden_set_dir: str = "evals/golden_sets"

    # Thresholds (per component)
    chunking_size_compliance_min: float = 0.90
    chunking_boundary_quality_min: float = 0.70
    retrieval_hit_rate_min: float = 0.90
    retrieval_mrr_min: float = 0.70
    retrieval_precision_min: float = 0.60
    retrieval_recall_min: float = 0.70
    generation_faithfulness_min: float = 4.0
    generation_correctness_min: float = 3.5
    generation_relevance_min: float = 4.0
    security_injection_detection_min: float = 0.95
    security_false_positive_max: float = 0.05
    security_pii_recall_min: float = 0.98

    # Execution
    max_concurrency: int = 5
    judge_model: str = "gpt-4.1"
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0

    results_dir: str = "evals/results"

    model_config = {"env_prefix": "EVAL_", "env_file": ".env", "extra": "ignore"}
```

### Golden Set Loader (`evals/loader.py`)

```python
def load_golden_set(
    version: str | None = None,
    categories: list[str] | None = None,
    tags: list[str] | None = None,
    max_cases: int | None = None,
) -> list[GoldenCase]:
    """Stream JSONL line-by-line, validate with Pydantic, apply filters."""
```

### Evaluation Runner (`evals/runner.py`)

Orchestrates all evaluators:

1. Load golden cases via `loader.py` (streams JSONL, applies category/tag filters)
2. Set up test document store (ingest fixture docs into eval Supabase DB via `EVAL_SUPABASE_DATABASE_URL`)
3. Run evaluators in dependency order: chunking -> retrieval -> generation -> security -> e2e
4. Collect `EvalResult` objects into `EvalReport`
5. Tear down test data
6. Write JSON + markdown reports to `evals/results/`

### Scaling Strategy

| Component | Bottleneck | Strategy |
|---|---|---|
| Chunking | CPU only | Synchronous, fast |
| Security | CPU only | Synchronous, fast |
| Retrieval | Embedding API (3000 RPM) | `ThreadPoolExecutor(max_workers=10)` + `tenacity` retry |
| Generation | LLM API | `ThreadPoolExecutor(max_workers=5)` + `tenacity` retry with exponential backoff |
| Judge | LLM API (gpt-4.1, lower RPM) | `ThreadPoolExecutor(max_workers=3)` + `tenacity` backoff |

### CLI (`scripts/run_evals.py`)

```bash
uv run python scripts/run_evals.py                          # all evals
uv run python scripts/run_evals.py --component retrieval    # single component
uv run python scripts/run_evals.py --category factual       # filter cases
uv run python scripts/run_evals.py --max-cases 10           # smoke test
uv run python scripts/run_evals.py --compare baseline       # regression check
uv run python scripts/run_evals.py --ci                     # exit 1 on failure
```

---

## 5. Regression Detection

### Results Format (`evals/results/`)

Each run produces:

```
evals/results/
    2026-06-20T14-30-00_{git_sha}.json     # structured EvalReport
    2026-06-20T14-30-00_{git_sha}.md       # human-readable markdown
    baseline.json                           # git-tracked, updated after accepted runs
```

### EvalReport Schema (`evals/models.py`)

```python
class EvalReport(BaseModel):
    run_id: str
    timestamp: str
    git_sha: str
    git_branch: str
    golden_set_version: str
    component_results: dict[str, EvalResult]
    overall_passed: bool
    total_cost_usd: float
    total_duration_seconds: float
    environment: dict   # python version, model versions, etc.
```

### Regression Detector (`evals/regression.py`)

Compares current run against baseline. For every metric, computes `delta_pct = (current - baseline) / baseline`. Flags:

- **Warning**: delta_pct between -5% and -10%
- **Critical**: delta_pct beyond -10%

CI fails if any critical regressions exist.

```python
class RegressionDetector:
    def __init__(self, tolerance: float = 0.05): ...
    def check(self, current: EvalReport, baseline: EvalReport) -> RegressionReport: ...

class RegressionReport(BaseModel):
    regressions: list[RegressionItem]
    improvements: list[RegressionItem]
    unchanged: list[str]
    overall_regressed: bool

class RegressionItem(BaseModel):
    component: str
    metric: str
    baseline_value: float
    current_value: float
    delta: float
    delta_pct: float
    severity: Literal["warning", "critical"]
```

---

## 6. CI Integration

### PR Evaluation Job (`.github/workflows/ci.yml`)

```yaml
evals:
  runs-on: ubuntu-latest
  needs: check                      # only after lint/type/unit pass
  if: github.event_name == 'pull_request'
  steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v4
    - name: Set up Python
      run: uv python install 3.12
    - name: Install dependencies
      run: uv sync --frozen
    - name: Run evaluations
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        EVAL_SUPABASE_DATABASE_URL: ${{ secrets.EVAL_SUPABASE_DATABASE_URL }}
        EVAL_RUN: "1"
        EVAL_MAX_CONCURRENCY: "3"
      run: uv run python scripts/run_evals.py --ci --max-cases 50 --output-format json
    - name: Upload eval results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: eval-results-${{ github.sha }}
        path: evals/results/
    - name: Compare with baseline
      run: uv run python scripts/run_evals.py --compare baseline --ci
```

- PRs run a 50-case subset (~3-5 min)
- `--ci` flag exits with code 1 on threshold failure
- Uses dedicated eval Supabase DB (`EVAL_SUPABASE_DATABASE_URL`)

### Nightly Full Evaluation (`.github/workflows/evals-nightly.yml`)

```yaml
name: Nightly Evaluations
on:
  schedule:
    - cron: '0 6 * * *'        # 6 AM UTC daily
  workflow_dispatch:            # manual trigger
```

Runs the complete golden set without `--max-cases` to catch regressions the smaller CI subset might miss.

### pytest Integration (`tests/test_evals.py`)

Gated behind `EVAL_RUN=1` env var (skipped by default since evals make real API calls):

```python
pytestmark = pytest.mark.skipunless(
    os.environ.get("EVAL_RUN") == "1",
    reason="Evaluations require EVAL_RUN=1 (they make real API calls)",
)

class TestChunkingEval:
    def test_chunk_size_compliance(self, eval_runner, golden_cases): ...
    def test_boundary_quality(self, eval_runner, golden_cases): ...

class TestRetrievalEval:
    def test_hit_rate_above_threshold(self, eval_runner, golden_cases): ...
    def test_mrr_above_threshold(self, eval_runner, golden_cases): ...

class TestGenerationEval:
    def test_faithfulness_above_threshold(self, eval_runner, golden_cases): ...
    def test_correctness_above_threshold(self, eval_runner, golden_cases): ...

class TestSecurityEval:
    def test_injection_detection_rate(self, eval_runner, golden_cases): ...
    def test_pii_masking_recall(self, eval_runner, golden_cases): ...
```

---

## 7. New Dependencies

Add to `pyproject.toml` under `[dependency-groups] dev`:

```toml
"tenacity>=8.0.0",     # retry with exponential backoff for API rate limits
"tabulate>=0.9.0",     # markdown table rendering for reports
```

No production dependencies added.

---

## 8. Implementation Phases

### Phase 1: Foundation (models, config, golden set)
Create `evals/models.py`, `evals/config.py`, `evals/protocols.py`, `evals/loader.py`. Create initial golden set `v1.0.0.jsonl` with ~20 hand-curated cases and fixture PDFs.

### Phase 2: Component evaluators
Implement chunking, security, retrieval, and generation evaluators. Build the LLM judge with prompt templates.

### Phase 3: E2E + infrastructure
Implement e2e evaluator, runner, report generator, regression detector, cost meter, CLI script.

### Phase 4: CI integration
Add eval job to CI, create nightly workflow, add pytest wrapper, update `pyproject.toml` and `.gitignore`.

---

## 9. Verification

After implementation, verify by:

1. `uv run python scripts/run_evals.py --max-cases 5` — smoke test all components
2. `uv run python scripts/run_evals.py --component chunking` — each component individually
3. `uv run python scripts/run_evals.py --ci --compare baseline` — regression detection
4. `EVAL_RUN=1 uv run pytest tests/test_evals.py -v` — pytest integration
5. Inspect `evals/results/*.md` for readable reports
6. Verify CI workflow runs on a test PR
