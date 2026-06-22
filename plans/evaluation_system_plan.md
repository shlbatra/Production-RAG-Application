# Evaluation System for Production RAG Pipeline

## Context

This RAG system has 6 evaluable stages: parsing, chunking, retrieval, generation, security, and end-to-end. There are no evaluations today beyond unit tests with mocks. We need a golden-set-driven evaluation framework that measures each component independently and the full pipeline together, runs in CI, detects regressions, and scales to hundreds of test cases.

The design follows Hamel Husain's "Your AI Product Needs Evals" framework with three tiers of increasing cost and decreasing frequency:

| Level | What | When | Cost |
|---|---|---|---|
| **Level 1: Assertion Tests** | Fast, deterministic checks on structured output and component behavior | Every PR, every push | Zero LLM-judge cost |
| **Level 2: Human Annotation + Observability** | LangSmith-traced eval runs with human scoring via annotation queues | Weekly or on significant changes | Low (tracing overhead only) |
| **Level 3: LLM-as-Judge** | Automated subjective quality scoring across faithfulness, correctness, completeness | Nightly + pre-release | LLM API cost per case |

The design follows existing codebase patterns: Protocol-based abstractions, Pydantic models, `pydantic-settings` for config, pytest for test integration. The project already uses LangSmith for tracing (via LangChain); the eval system integrates with it rather than building custom observability.

---

## 1. Test Data Design

### 1a. Level 1 Test Events (`evals/fixtures/events/`)

Simple JSON files with just the input and expected deterministic outcome. One file per scenario — easy to add, easy to review in PRs:

```json
{
    "id": "billing-001",
    "category": "factual",
    "question": "What was the total revenue in Q3 2025?",
    "expected_source_files": ["q3_report.pdf"],
    "expected_category": "factual",
    "expected_refuses": false
}
```

No `expected_answer` or `expected_chunk_contents` needed — Level 1 only checks structural and categorical correctness.

### 1b. Level 2+ Golden Cases (`evals/golden_sets/`)

Full golden cases for LLM-judge evaluation. Schema in `evals/models.py`:

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
    expected_answer: str | None = None         # optional for Level 1 cases
    expected_source_files: list[str]           # e.g. ["report.pdf"]
    expected_chunk_contents: list[str] = []    # optional — substrings in retrieved chunks
    expected_refuses: bool = False             # True for unanswerable/adversarial
    difficulty: Literal["easy", "medium", "hard"]
    tags: list[str] = []
```

`expected_answer` and `expected_chunk_contents` are now optional — cases can start as Level 1 events and graduate to full golden cases as reference answers are added.

Storage: one case per line in `evals/golden_sets/v1.0.0.jsonl` — git-diff-friendly, streams line-by-line. `evals/golden_sets/manifest.json` tracks the active version. Companion test PDFs live in `evals/fixtures/documents/`.

### 1c. Eval Self-Tests (`evals/fixtures/self_tests/`)

Cases designed to intentionally fail each evaluator, verifying the eval framework itself catches problems:

```json
{
    "id": "self-test-retrieval-001",
    "question": "What is the airspeed velocity of an unladen swallow?",
    "expected_source_files": ["nonexistent_doc.pdf"],
    "should_fail": true,
    "target_evaluator": "retrieval"
}
```

During eval verification, a meta-test confirms these cases DO fail — preventing the "all green but the eval is broken" failure mode.

### Category Distribution Target (for Level 3 golden set)

| Category | % | Purpose |
|---|---|---|
| factual | 40% | Correct retrieval + accurate answer |
| multi_hop | 20% | Cross-chunk / cross-doc synthesis |
| unanswerable | 15% | System refuses rather than hallucinating |
| adversarial | 15% | Blocked by `InputSanitizer` |
| pii | 10% | Masked by `PIIDetector` |

---

## 2. Level 1: Assertion-Based Tests

Fast, deterministic checks that run on every PR. No LLM-judge calls — pure assertions on structured output and component behavior.

All evaluators implement a common `ComponentEvaluator` Protocol (in `evals/protocols.py`):

```python
class ComponentEvaluator(Protocol):
    def evaluate(self, cases: list[GoldenCase]) -> EvalResult: ...
```

### 2a. Chunking Evaluator (`evals/evaluators/chunking_eval.py`)

Runs `RecursiveChunker.chunk()` on fixture documents. Pure algorithmic — no LLM calls.

| Metric | How | Threshold |
|---|---|---|
| Chunk Size Compliance | % of chunks within `chunk_size` ± 10% | >= 0.90 |
| Boundary Quality | % of chunks NOT breaking mid-sentence | >= 0.70 |
| Information Preservation | char coverage ratio of joined chunks vs original | >= 0.99 |
| Overlap Correctness | overlap between adjacent chunks matches config | >= 0.90 |

### 2b. Retrieval Evaluator (`evals/evaluators/retrieval_eval.py`)

Calls `DocumentStore.search_similar()` for each case. Compares results against `expected_source_files` and `expected_chunk_contents`.

| Metric | How | Threshold |
|---|---|---|
| Hit Rate | % of queries where >= 1 relevant chunk in top-k | >= 0.90 |
| MRR | Mean Reciprocal Rank of first relevant result | >= 0.70 |
| Precision@5 | relevant / retrieved at k=5 | >= 0.60 |
| Recall@5 | retrieved relevant / total relevant | >= 0.70 |
| NDCG@5 | Normalized Discounted Cumulative Gain | >= 0.65 |

**Relevance definition**: a chunk is relevant if its source file is in `expected_source_files` AND (if provided) at least one `expected_chunk_contents` substring appears in the chunk content.

### 2c. Generation Structural Validator (`evals/evaluators/generation_structural_eval.py`)

Validates generation output structurally BEFORE any LLM-judge scoring. Fast, deterministic checks that catch failures without judge API cost:

| Check | What | Threshold |
|---|---|---|
| Parse Success | Response parses as expected structured output (Pydantic model) | 100% |
| Non-Empty Response | Response body is not empty or whitespace-only | 100% |
| Source Citations Present | Response includes at least one source reference | >= 0.95 |
| No Raw Errors | Response does not contain raw exception/traceback text | 100% |
| Refusal Accuracy | For `unanswerable` cases (`expected_refuses=True`), system refuses | >= 0.90 |
| Response Length | Response within configured min/max bounds | >= 0.95 |

Cases that fail structural validation are marked `FAIL` immediately — they never reach the Level 3 LLM judge.

### 2d. Security Evaluator (`evals/evaluators/security_eval.py`)

Runs `SecurityPipeline.check_input()` and `PIIDetector.mask()` directly. No LLM calls.

| Metric | Threshold |
|---|---|
| Injection Detection Rate (adversarial cases blocked) | >= 0.95 |
| Injection False Positive Rate (legitimate cases wrongly blocked) | <= 0.05 |
| PII Masking Recall (PII instances masked) | >= 0.98 |

Extra test vectors in `evals/fixtures/security_vectors.jsonl`.

---

## 3. Level 2: Human Annotation + Observability

### LangSmith Integration (`evals/tracing.py`)

The project already uses LangSmith for tracing via LangChain. The eval system integrates with it:

```python
from langsmith import Client

class EvalTracer:
    """Wraps eval runs with LangSmith tracing and manages annotation queues."""

    def __init__(self, client: Client | None = None):
        self.client = client or Client()

    def create_eval_dataset(self, cases: list[GoldenCase], name: str) -> str:
        """Upload golden cases to LangSmith Datasets for versioned tracking."""
        ...

    def trace_eval_run(self, run_name: str, cases: list[GoldenCase]) -> str:
        """Start a traced eval run — all agent invocations within are grouped."""
        ...

    def submit_for_annotation(self, run_id: str, queue_name: str = "eval-review") -> None:
        """Send eval results to LangSmith annotation queue for human review."""
        ...

    def fetch_human_scores(self, dataset_name: str) -> dict[str, float]:
        """Pull human annotation scores from LangSmith for comparison against LLM judge."""
        ...
```

### How It Fits

1. **Tracing**: Every eval run is traced in LangSmith — each `ProductionAgent.invoke()` call appears with its inputs, outputs, latency, and token usage. Replaces the need for custom latency/cost instrumentation for traced runs.
2. **Datasets**: Golden cases are synced to LangSmith Datasets alongside the local JSONL files. LangSmith Datasets provide a UI for browsing, filtering, and editing cases.
3. **Annotation Queues**: After an eval run, results are pushed to an annotation queue. Annotators review in the LangSmith UI, scoring pass/fail on each case. This provides the human baseline that validates the LLM judge's accuracy.
4. **Token/Cost Tracking**: LangSmith's built-in token tracking supplements the custom `CostMeter` — use LangSmith for per-run visibility, `CostMeter` for aggregate CI reporting.

### Human Annotation Workflow

1. Run eval suite with tracing enabled: `uv run python scripts/run_evals.py --trace`
2. Results appear in LangSmith with full traces
3. Annotators score a sample in the annotation queue (pass/fail + optional notes)
4. Periodically compare human scores vs LLM-judge scores to calibrate judge prompts
5. Cases where human and judge disagree become candidates for prompt tuning or golden set revision

---

## 4. Level 3: LLM-as-Judge (`evals/evaluators/generation_eval.py`)

Invokes `ProductionAgent.invoke()` for each case, runs structural validation first (Level 1), then sends passing responses to an `LLMJudge` (gpt-4.1, stronger than the gpt-4.1-mini being evaluated).

Only cases with `expected_answer` populated are sent to the judge — Level 1-only cases skip this.

### Judge (`evals/judges/llm_judge.py`)

Scores each dimension 1-5 using structured prompts in `evals/judges/prompts/*.txt`:

| Dimension | What it measures | Threshold (avg) |
|---|---|---|
| Faithfulness | Is the answer grounded in retrieved context? | >= 4.0 |
| Correctness | Does it match the expected answer? | >= 3.5 |
| Relevance | Does it address the question? | >= 4.0 |
| Completeness | Does it cover all key points from expected answer? | >= 3.5 |

---

## 5. End-to-End Evaluator (`evals/evaluators/e2e_eval.py`)

Runs the full pipeline as `/chat` does: security check -> agent invoke -> output validation. Collects all generation quality metrics (delegated to Level 1 structural + Level 3 judge) PLUS:

| Metric | How |
|---|---|
| Latency P50/P95/P99 | `RequestTimer` (existing utility in `app/monitoring.py`) wrapping each invoke |
| Cost per query | Token counts from each model x pricing table (`evals/metering.py`) |
| Error rate | % that hit the error_handler node |
| Fallback rate | % that fell back to gpt-4.1-nano |

### Cost Meter (`evals/metering.py`)

Tracks token usage per model and computes dollar cost. Used for aggregate CI reporting — per-run cost visibility comes from LangSmith tracing.

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

## 6. Infrastructure

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
    tracing.py             # EvalTracer — LangSmith integration
    evaluators/
        __init__.py
        chunking_eval.py
        retrieval_eval.py
        generation_structural_eval.py   # Level 1 structural checks
        generation_eval.py              # Level 3 LLM-as-judge
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
        events/             # Level 1 simple JSON test events
        self_tests/         # intentional failure cases for eval validation
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

    # Observability
    langsmith_tracing: bool = True
    langsmith_project: str = "prod-rag-evals"
    annotation_queue: str = "eval-review"

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

def load_test_events(directory: str = "evals/fixtures/events") -> list[GoldenCase]:
    """Load simple Level 1 JSON test events, converting to GoldenCase with optional fields empty."""
```

### Evaluation Runner (`evals/runner.py`)

Orchestrates all evaluators:

1. Load golden cases via `loader.py` (streams JSONL, applies category/tag filters)
2. Set up test document store (ingest fixture docs into eval Supabase DB via `EVAL_SUPABASE_DATABASE_URL`)
3. Initialize LangSmith tracing (if enabled)
4. Run evaluators in dependency order:
   - Level 1: chunking -> retrieval -> generation structural -> security
   - Level 3 (if `--level 3` or `--full`): generation LLM-judge
   - E2E (if `--e2e` or `--full`)
5. Collect `EvalResult` objects into `EvalReport`
6. Submit to LangSmith annotation queue (if `--annotate`)
7. Tear down test data
8. Write JSON + markdown reports to `evals/results/`

### Scaling Strategy

| Component | Bottleneck | Strategy |
|---|---|---|
| Chunking | CPU only | Synchronous, fast |
| Security | CPU only | Synchronous, fast |
| Generation Structural | CPU only | Synchronous, fast |
| Retrieval | Embedding API (3000 RPM) | `ThreadPoolExecutor(max_workers=10)` + `tenacity` retry |
| Generation (judge) | LLM API | `ThreadPoolExecutor(max_workers=5)` + `tenacity` retry with exponential backoff |
| Judge | LLM API (gpt-4.1, lower RPM) | `ThreadPoolExecutor(max_workers=3)` + `tenacity` backoff |

### CLI (`scripts/run_evals.py`)

```bash
uv run python scripts/run_evals.py                          # Level 1 evals (default)
uv run python scripts/run_evals.py --level 3                # include LLM-judge scoring
uv run python scripts/run_evals.py --full                   # all levels + e2e
uv run python scripts/run_evals.py --component retrieval    # single component
uv run python scripts/run_evals.py --category factual       # filter cases
uv run python scripts/run_evals.py --max-cases 10           # smoke test
uv run python scripts/run_evals.py --compare baseline       # regression check
uv run python scripts/run_evals.py --ci                     # exit 1 on failure
uv run python scripts/run_evals.py --trace                  # enable LangSmith tracing
uv run python scripts/run_evals.py --annotate               # submit to annotation queue
uv run python scripts/run_evals.py --self-test              # verify eval framework itself
```

---

## 7. Regression Detection

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
    eval_level: Literal[1, 2, 3]           # which level was run
    component_results: dict[str, EvalResult]
    overall_passed: bool
    total_cost_usd: float
    total_duration_seconds: float
    langsmith_run_url: str | None = None   # link to LangSmith trace
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

## 8. CI Integration

### PR Evaluation Job (`.github/workflows/ci.yml`)

Runs Level 1 only on PRs — fast, no LLM-judge cost:

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
    - name: Run Level 1 evaluations
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        EVAL_SUPABASE_DATABASE_URL: ${{ secrets.EVAL_SUPABASE_DATABASE_URL }}
        EVAL_RUN: "1"
        EVAL_MAX_CONCURRENCY: "3"
        EVAL_LANGSMITH_TRACING: "false"
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

- PRs run Level 1 on a 50-case subset (~1-2 min, no judge cost)
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

Runs `--full` (all levels + e2e) with `--trace` on the complete golden set. LLM-judge costs are incurred here, not on every PR.

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

class TestGenerationStructuralEval:
    def test_parse_success(self, eval_runner, golden_cases): ...
    def test_refusal_accuracy(self, eval_runner, golden_cases): ...

class TestGenerationEval:
    def test_faithfulness_above_threshold(self, eval_runner, golden_cases): ...
    def test_correctness_above_threshold(self, eval_runner, golden_cases): ...

class TestSecurityEval:
    def test_injection_detection_rate(self, eval_runner, golden_cases): ...
    def test_pii_masking_recall(self, eval_runner, golden_cases): ...

class TestEvalSelfTests:
    def test_intentional_failures_are_caught(self, eval_runner): ...
```

---

## 9. New Dependencies

Add to `pyproject.toml` under `[dependency-groups] dev`:

```toml
"tenacity>=8.0.0",     # retry with exponential backoff for API rate limits
"tabulate>=0.9.0",     # markdown table rendering for reports
"langsmith>=0.1.0",    # eval tracing, datasets, annotation queues
```

No production dependencies added.

---

## 10. Implementation Phases

### Phase 1: Level 1 — Runnable Evals in CI (delivers value immediately)
Create `evals/models.py`, `evals/config.py`, `evals/protocols.py`, `evals/loader.py`. Create ~10 simple Level 1 test events in `evals/fixtures/events/`. Implement chunking evaluator, security evaluator, and generation structural validator. Build the runner (Level 1 mode only) and CLI. Add 2-3 eval self-test cases. **Goal**: `uv run python scripts/run_evals.py` works end-to-end with assertion-based checks.

### Phase 2: LangSmith Integration + Retrieval Evaluator (observability)
Implement `evals/tracing.py` with LangSmith integration. Implement retrieval evaluator. Add `--trace` and `--annotate` CLI flags. Set up LangSmith annotation queue. **Goal**: eval runs are observable in LangSmith, human annotators can score results.

### Phase 3: LLM-as-Judge (subjective quality measurement)
Build the LLM judge with prompt templates in `evals/judges/`. Implement generation evaluator (Level 3). Expand golden set to ~50 cases with `expected_answer` filled in. Add `--level 3` and `--full` CLI flags. **Goal**: automated quality scoring on faithfulness, correctness, relevance, completeness.

### Phase 4: E2E + CI + Regression (production readiness)
Implement e2e evaluator, report generator, regression detector, cost meter. Add eval job to CI workflow (Level 1 on PRs). Create nightly workflow (full evaluation). Add pytest wrapper. Update `pyproject.toml` and `.gitignore`. **Goal**: fully automated eval pipeline in CI with regression detection.

---

## 11. Verification

After each phase, verify:

**Phase 1**:
1. `uv run python scripts/run_evals.py --max-cases 5` — smoke test Level 1 components
2. `uv run python scripts/run_evals.py --component chunking` — each component individually
3. `uv run python scripts/run_evals.py --self-test` — eval self-tests pass (intentional failures are caught)

**Phase 2**:
4. `uv run python scripts/run_evals.py --trace --max-cases 5` — verify traces appear in LangSmith
5. `uv run python scripts/run_evals.py --annotate` — verify annotation queue populated

**Phase 3**:
6. `uv run python scripts/run_evals.py --level 3 --max-cases 5` — LLM judge runs
7. Inspect `evals/results/*.md` for readable reports with judge scores

**Phase 4**:
8. `uv run python scripts/run_evals.py --full --ci --compare baseline` — full pipeline with regression detection
9. `EVAL_RUN=1 uv run pytest tests/test_evals.py -v` — pytest integration
10. Verify CI workflow runs on a test PR (Level 1 only, fast)
11. Verify nightly workflow runs full evaluation
