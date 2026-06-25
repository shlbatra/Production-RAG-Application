"""Evaluation configuration — thresholds, paths, and execution settings."""

from pydantic_settings import BaseSettings


class EvalSettings(BaseSettings):
    golden_set_version: str = "v1.0.0"
    golden_set_dir: str = "evals/golden_sets"

    # Retrieval thresholds
    retrieval_hit_rate_min: float = 0.90
    retrieval_mrr_min: float = 0.70
    retrieval_precision_min: float = 0.60
    retrieval_recall_min: float = 0.70
    retrieval_ndcg_min: float = 0.65
    retrieval_top_k: int = 5

    # Execution
    max_concurrency: int = 5
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0

    results_dir: str = "evals/results"

    model_config = {"env_prefix": "EVAL_", "env_file": ".env", "extra": "ignore"}
