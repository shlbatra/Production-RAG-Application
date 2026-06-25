"""Retrieval evaluator — measures search quality against golden cases.

Uses the same retrieval strategy the app uses (similarity, BM25, or hybrid)
and compares results against expected_source_files and expected_chunk_contents.
"""

import logging
import math

from app.retrieval import RetrievalStrategy
from evals.config import EvalSettings
from evals.models import CaseResult, EvalResult, GoldenCase, MetricResult

logger = logging.getLogger(__name__)


def _is_relevant(chunk: dict, case: GoldenCase) -> bool:
    """A chunk is relevant if its source file is in expected_source_files AND
    (if provided) at least one expected_chunk_contents substring appears."""
    source = chunk.get("metadata", {}).get("source", "")
    if source not in case.expected_source_files:
        return False
    if not case.expected_chunk_contents:
        return True
    content = chunk.get("content", "")
    return any(substr in content for substr in case.expected_chunk_contents)


def _compute_dcg(relevance_flags: list[bool]) -> float:
    """Discounted Cumulative Gain."""
    dcg = 0.0
    for i, rel in enumerate(relevance_flags):
        if rel:
            dcg += 1.0 / math.log2(i + 2)
    return dcg


class RetrievalEvaluator:
    def __init__(
        self, retriever: RetrievalStrategy, settings: EvalSettings | None = None
    ) -> None:
        self._retriever = retriever
        self._settings = settings or EvalSettings()

    def evaluate(self, cases: list[GoldenCase]) -> EvalResult:
        top_k = self._settings.retrieval_top_k

        hit_count = 0
        reciprocal_ranks: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        case_results: list[CaseResult] = []

        for case in cases:
            if case.expected_refuses:
                continue

            results = self._retriever.search(
                query=case.question, top_k=top_k, threshold=0.7
            )
            relevance = [_is_relevant(r, case) for r in results]

            hit = any(relevance)
            if hit:
                hit_count += 1

            logger.debug(
                "[%s] query=%r | expected=%s | got=%s | hit=%s\n",
                case.id,
                case.question,
                case.expected_source_files,
                [
                    {
                        "source": r.get("metadata", {}).get("source", "?"),
                        "similarity": round(r.get("similarity", 0), 4),
                        "preview": r.get("content", "")[:100],
                    }
                    for r in results
                ],
                hit,
            )

            rr = 0.0
            for i, rel in enumerate(relevance):
                if rel:
                    rr = 1.0 / (i + 1)
                    break
            reciprocal_ranks.append(rr)

            relevant_retrieved = sum(relevance)
            precision = relevant_retrieved / len(results) if results else 0.0
            precisions.append(precision)

            total_relevant = len(case.expected_source_files)
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
            recalls.append(recall)

            dcg = _compute_dcg(relevance)
            ideal_relevant_count = min(total_relevant, top_k)
            ideal_relevance = [True] * ideal_relevant_count + [False] * (
                top_k - ideal_relevant_count
            )
            idcg = _compute_dcg(ideal_relevance)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

            case_metrics = {
                "hit": float(hit),
                "reciprocal_rank": rr,
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
            }
            case_passed = hit and recall >= self._settings.retrieval_recall_min
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    metrics=case_metrics,
                    passed=case_passed,
                    details=f"Retrieved {len(results)} chunks, {relevant_retrieved} relevant",
                )
            )

        evaluated_count = len(reciprocal_ranks)
        if evaluated_count == 0:
            return EvalResult(
                component="retrieval",
                metrics=[],
                case_results=[],
                passed=True,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
            )

        hit_rate = hit_count / evaluated_count
        mrr = sum(reciprocal_ranks) / evaluated_count
        avg_precision = sum(precisions) / evaluated_count
        avg_recall = sum(recalls) / evaluated_count
        avg_ndcg = sum(ndcgs) / evaluated_count

        metrics = [
            MetricResult(
                name="hit_rate",
                value=round(hit_rate, 4),
                threshold=self._settings.retrieval_hit_rate_min,
                passed=hit_rate >= self._settings.retrieval_hit_rate_min,
            ),
            MetricResult(
                name="mrr",
                value=round(mrr, 4),
                threshold=self._settings.retrieval_mrr_min,
                passed=mrr >= self._settings.retrieval_mrr_min,
            ),
            MetricResult(
                name="precision@k",
                value=round(avg_precision, 4),
                threshold=self._settings.retrieval_precision_min,
                passed=avg_precision >= self._settings.retrieval_precision_min,
            ),
            MetricResult(
                name="recall@k",
                value=round(avg_recall, 4),
                threshold=self._settings.retrieval_recall_min,
                passed=avg_recall >= self._settings.retrieval_recall_min,
            ),
            MetricResult(
                name="ndcg@k",
                value=round(avg_ndcg, 4),
                threshold=self._settings.retrieval_ndcg_min,
                passed=avg_ndcg >= self._settings.retrieval_ndcg_min,
            ),
        ]

        passed_cases = sum(1 for cr in case_results if cr.passed)
        overall_passed = all(m.passed for m in metrics)

        return EvalResult(
            component="retrieval",
            metrics=metrics,
            case_results=case_results,
            passed=overall_passed,
            total_cases=evaluated_count,
            passed_cases=passed_cases,
            failed_cases=evaluated_count - passed_cases,
        )
