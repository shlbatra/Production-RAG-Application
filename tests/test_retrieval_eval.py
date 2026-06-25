"""Tests for the retrieval evaluator — unit tests using mock retriever."""

import math
from unittest.mock import MagicMock

from evals.evaluators.retrieval_eval import (
    RetrievalEvaluator,
    _compute_dcg,
    _is_relevant,
)
from evals.models import GoldenCase


def _make_case(**overrides) -> GoldenCase:
    defaults = {
        "id": "test-001",
        "category": "factual",
        "question": "What is coverage A?",
        "expected_source_files": ["policies/PLY-FL-001.txt"],
        "expected_chunk_contents": [],
        "expected_refuses": False,
        "difficulty": "easy",
    }
    defaults.update(overrides)
    return GoldenCase(**defaults)


def _make_chunk(
    source: str, content: str = "some content", similarity: float = 0.9
) -> dict:
    return {
        "id": 1,
        "content": content,
        "metadata": {"source": source},
        "similarity": similarity,
    }


class TestIsRelevant:
    def test_relevant_by_source_only(self):
        case = _make_case(expected_source_files=["a.txt"])
        chunk = _make_chunk(source="a.txt")
        assert _is_relevant(chunk, case) is True

    def test_irrelevant_wrong_source(self):
        case = _make_case(expected_source_files=["a.txt"])
        chunk = _make_chunk(source="b.txt")
        assert _is_relevant(chunk, case) is False

    def test_relevant_with_content_match(self):
        case = _make_case(
            expected_source_files=["a.txt"],
            expected_chunk_contents=["Coverage A"],
        )
        chunk = _make_chunk(source="a.txt", content="Coverage A is $350,000")
        assert _is_relevant(chunk, case) is True

    def test_irrelevant_source_match_but_content_miss(self):
        case = _make_case(
            expected_source_files=["a.txt"],
            expected_chunk_contents=["Coverage A"],
        )
        chunk = _make_chunk(source="a.txt", content="unrelated content")
        assert _is_relevant(chunk, case) is False

    def test_missing_metadata(self):
        case = _make_case(expected_source_files=["a.txt"])
        chunk = {"id": 1, "content": "test", "metadata": {}, "similarity": 0.9}
        assert _is_relevant(chunk, case) is False


class TestComputeDcg:
    def test_all_relevant(self):
        dcg = _compute_dcg([True, True, True])
        expected = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        assert abs(dcg - expected) < 1e-9

    def test_none_relevant(self):
        assert _compute_dcg([False, False, False]) == 0.0

    def test_first_relevant(self):
        dcg = _compute_dcg([True, False, False])
        assert abs(dcg - 1 / math.log2(2)) < 1e-9

    def test_empty(self):
        assert _compute_dcg([]) == 0.0


class TestRetrievalEvaluator:
    def test_perfect_retrieval(self, eval_settings):
        retriever = MagicMock()
        retriever.search.return_value = [
            _make_chunk(
                source="policies/PLY-FL-001.txt", content="Coverage A $350,000"
            ),
        ]

        evaluator = RetrievalEvaluator(retriever, eval_settings)
        cases = [
            _make_case(
                expected_source_files=["policies/PLY-FL-001.txt"],
                expected_chunk_contents=["Coverage A"],
            )
        ]
        result = evaluator.evaluate(cases)

        assert result.passed is True
        assert result.total_cases == 1
        assert result.passed_cases == 1

        hit_rate = next(m for m in result.metrics if m.name == "hit_rate")
        assert hit_rate.value == 1.0

        mrr = next(m for m in result.metrics if m.name == "mrr")
        assert mrr.value == 1.0

    def test_no_relevant_results(self, eval_settings):
        retriever = MagicMock()
        retriever.search.return_value = [
            _make_chunk(source="wrong.txt"),
            _make_chunk(source="also_wrong.txt"),
        ]

        evaluator = RetrievalEvaluator(retriever, eval_settings)
        cases = [_make_case(expected_source_files=["policies/PLY-FL-001.txt"])]
        result = evaluator.evaluate(cases)

        hit_rate = next(m for m in result.metrics if m.name == "hit_rate")
        assert hit_rate.value == 0.0
        assert hit_rate.passed is False

    def test_relevant_at_position_two(self, eval_settings):
        retriever = MagicMock()
        retriever.search.return_value = [
            _make_chunk(source="wrong.txt"),
            _make_chunk(source="policies/PLY-FL-001.txt"),
        ]

        evaluator = RetrievalEvaluator(retriever, eval_settings)
        cases = [_make_case(expected_source_files=["policies/PLY-FL-001.txt"])]
        result = evaluator.evaluate(cases)

        mrr = next(m for m in result.metrics if m.name == "mrr")
        assert mrr.value == 0.5

    def test_multiple_cases_aggregation(self, eval_settings):
        retriever = MagicMock()
        retriever.search.side_effect = [
            [_make_chunk(source="policies/PLY-FL-001.txt")],
            [_make_chunk(source="wrong.txt")],
        ]

        evaluator = RetrievalEvaluator(retriever, eval_settings)
        cases = [
            _make_case(id="case-1", expected_source_files=["policies/PLY-FL-001.txt"]),
            _make_case(
                id="case-2", expected_source_files=["claims/CLM-FL-2024-001.txt"]
            ),
        ]
        result = evaluator.evaluate(cases)

        assert result.total_cases == 2
        assert result.passed_cases == 1
        assert result.failed_cases == 1

        hit_rate = next(m for m in result.metrics if m.name == "hit_rate")
        assert hit_rate.value == 0.5

    def test_skips_unanswerable_cases(self, eval_settings):
        retriever = MagicMock()
        evaluator = RetrievalEvaluator(retriever, eval_settings)
        cases = [_make_case(expected_refuses=True, expected_source_files=[])]
        result = evaluator.evaluate(cases)

        assert result.total_cases == 0
        retriever.search.assert_not_called()

    def test_empty_results_from_retriever(self, eval_settings):
        retriever = MagicMock()
        retriever.search.return_value = []

        evaluator = RetrievalEvaluator(retriever, eval_settings)
        cases = [_make_case(expected_source_files=["policies/PLY-FL-001.txt"])]
        result = evaluator.evaluate(cases)

        hit_rate = next(m for m in result.metrics if m.name == "hit_rate")
        assert hit_rate.value == 0.0

        precision = next(m for m in result.metrics if m.name == "precision@k")
        assert precision.value == 0.0

    def test_empty_cases_list(self, eval_settings):
        retriever = MagicMock()
        evaluator = RetrievalEvaluator(retriever, eval_settings)
        result = evaluator.evaluate([])

        assert result.passed is True
        assert result.total_cases == 0
        assert result.metrics == []

    def test_case_results_contain_details(self, eval_settings):
        retriever = MagicMock()
        retriever.search.return_value = [
            _make_chunk(source="policies/PLY-FL-001.txt"),
            _make_chunk(source="wrong.txt"),
        ]

        evaluator = RetrievalEvaluator(retriever, eval_settings)
        cases = [_make_case(expected_source_files=["policies/PLY-FL-001.txt"])]
        result = evaluator.evaluate(cases)

        assert len(result.case_results) == 1
        cr = result.case_results[0]
        assert cr.case_id == "test-001"
        assert "Retrieved 2 chunks" in cr.details
        assert "1 relevant" in cr.details

    def test_recall_with_multiple_expected_sources(self, eval_settings):
        retriever = MagicMock()
        retriever.search.return_value = [
            _make_chunk(source="claims/CLM-FL-2024-001.txt"),
        ]

        evaluator = RetrievalEvaluator(retriever, eval_settings)
        cases = [
            _make_case(
                expected_source_files=[
                    "claims/CLM-FL-2024-001.txt",
                    "adjuster_notes/NOTE-001.txt",
                ],
            )
        ]
        result = evaluator.evaluate(cases)

        recall = next(m for m in result.metrics if m.name == "recall@k")
        assert recall.value == 0.5

    def test_ndcg_perfect_ranking(self, eval_settings):
        retriever = MagicMock()
        retriever.search.return_value = [
            _make_chunk(source="policies/PLY-FL-001.txt"),
            _make_chunk(source="wrong.txt"),
            _make_chunk(source="also_wrong.txt"),
        ]

        evaluator = RetrievalEvaluator(retriever, eval_settings)
        cases = [_make_case(expected_source_files=["policies/PLY-FL-001.txt"])]
        result = evaluator.evaluate(cases)

        ndcg = next(m for m in result.metrics if m.name == "ndcg@k")
        assert ndcg.value == 1.0
