"""Evaluation configuration — thresholds, paths, and execution settings."""

from pydantic_settings import BaseSettings


class EvalSettings(BaseSettings):
    golden_set_version: str = "v1.0.0"
    golden_set_dir: str = "evals/golden_sets"

    # Chunking thresholds
    chunking_size_compliance_min: float = 0.90
    chunking_boundary_quality_min: float = 0.60
    chunking_info_preservation_min: float = 0.99
    chunking_overlap_correctness_min: float = 0.10
    chunking_fixtures_dir: str = "documents"

    # Security thresholds
    security_injection_detection_min: float = 0.95
    security_false_positive_max: float = 0.05
    security_vectors_path: str = "evals/fixtures/security_vectors.jsonl"

    # Retrieval thresholds
    retrieval_hit_rate_min: float = 0.90
    retrieval_mrr_min: float = 0.70
    retrieval_precision_min: float = 0.50
    retrieval_recall_min: float = 0.70
    retrieval_ndcg_min: float = 0.65
    retrieval_top_k: int = 5

    # Execution
    max_concurrency: int = 5
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0

    results_dir: str = "evals/results"

    model_config = {"env_prefix": "EVAL_", "env_file": ".env", "extra": "ignore"}
