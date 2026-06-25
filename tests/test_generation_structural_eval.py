from unittest.mock import MagicMock

from evals.config import EvalSettings
from evals.evaluators.generation_structural_eval import (
    GenerationStructuralEvaluator,
    _has_raw_errors,
    _is_refusal,
)
from evals.models import GoldenCase


def _make_settings(**overrides):
    defaults = {
        "generation_parse_success_min": 1.0,
        "generation_non_empty_min": 1.0,
        "generation_source_citation_min": 0.95,
        "generation_no_raw_errors_min": 1.0,
        "generation_refusal_accuracy_min": 0.90,
        "generation_response_length_min": 0.95,
        "generation_min_response_length": 10,
        "generation_max_response_length": 5000,
    }
    defaults.update(overrides)
    return EvalSettings(**defaults)


def _make_case(
    id="test-001",
    category="factual",
    question="What is the coverage limit?",
    expected_refuses=False,
    **kwargs,
):
    defaults = {
        "id": id,
        "category": category,
        "question": question,
        "expected_answer": "Some answer",
        "expected_source_files": ["doc.txt"],
        "expected_refuses": expected_refuses,
        "difficulty": "easy",
        "tags": [],
    }
    defaults.update(kwargs)
    return GoldenCase(**defaults)


def _make_agent(responses):
    agent = MagicMock()
    agent.invoke.side_effect = list(responses)
    return agent


def _good_response(text="The coverage limit is $350,000."):
    return {
        "response": text,
        "model_used": "primary",
        "error": None,
        "sources": [
            {
                "source": "PLY-FL-001.txt",
                "similarity": 0.95,
                "chunk_preview": "Coverage A...",
            }
        ],
    }


def _refusal_response():
    return {
        "response": "I don't have sufficient context to answer this question.",
        "model_used": "primary",
        "error": None,
        "sources": [],
    }


class TestRefusalDetection:
    def test_detects_refusal_patterns(self):
        assert _is_refusal("I don't have sufficient context to answer.")
        assert _is_refusal("There is no relevant information available.")
        assert _is_refusal("I'm unable to answer that question.")

    def test_detects_access_denial_patterns(self):
        assert _is_refusal(
            "I don't have access to specific claim details or investigation outcomes."
        )
        assert _is_refusal(
            "Could you please provide more details about the Florida property?"
        )
        assert _is_refusal("You may need to contact the insurance company.")

    def test_normal_response_is_not_refusal(self):
        assert not _is_refusal("The coverage limit is $350,000.")
        assert not _is_refusal(
            "Based on the documents, the adjuster is Jennifer Walsh."
        )


class TestErrorDetection:
    def test_detects_traceback(self):
        assert _has_raw_errors("Traceback (most recent call last):\n  File ...")

    def test_detects_exception_types(self):
        assert _has_raw_errors("ValueError: invalid literal for int()")
        assert _has_raw_errors("KeyError: 'missing_key'")

    def test_detects_file_line_pattern(self):
        assert _has_raw_errors('File "/app/main.py", line 42')

    def test_clean_response_passes(self):
        assert not _has_raw_errors("The policy covers fire damage up to $500,000.")
        assert not _has_raw_errors("No errors were found in the claim.")


class TestGenerationStructuralEvaluator:
    def test_all_checks_pass(self):
        agent = _make_agent([_good_response()])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        assert result.component == "generation_structural"
        assert result.passed
        assert result.total_cases == 1
        assert result.passed_cases == 1
        assert all(m.passed for m in result.metrics)

    def test_empty_cases_passes(self):
        agent = _make_agent([])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([])

        assert result.passed
        assert result.total_cases == 0

    def test_parse_failure(self):
        agent = _make_agent(["not a dict"])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        assert not result.passed
        parse = next(m for m in result.metrics if m.name == "parse_success")
        assert parse.value == 0.0

    def test_missing_keys_fails_parse(self):
        agent = _make_agent([{"response": "hello"}])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        parse = next(m for m in result.metrics if m.name == "parse_success")
        assert parse.value == 0.0

    def test_empty_response_fails(self):
        agent = _make_agent(
            [{"response": "", "model_used": "primary", "error": None, "sources": []}]
        )
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        non_empty = next(m for m in result.metrics if m.name == "non_empty_response")
        assert non_empty.value == 0.0

    def test_whitespace_response_fails(self):
        agent = _make_agent(
            [
                {
                    "response": "   \n  ",
                    "model_used": "primary",
                    "error": None,
                    "sources": [],
                }
            ]
        )
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        non_empty = next(m for m in result.metrics if m.name == "non_empty_response")
        assert non_empty.value == 0.0

    def test_no_sources_fails_citation_check(self):
        agent = _make_agent(
            [
                {
                    "response": "The coverage limit is $350,000.",
                    "model_used": "primary",
                    "error": None,
                    "sources": [],
                }
            ]
        )
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        citation = next(
            m for m in result.metrics if m.name == "source_citations_present"
        )
        assert citation.value == 0.0

    def test_citation_not_required_for_refusal_cases(self):
        agent = _make_agent([_refusal_response()])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        case = _make_case(
            id="unans-001",
            category="unanswerable",
            expected_refuses=True,
            expected_answer=None,
            expected_source_files=[],
        )
        result = evaluator.evaluate([case])

        citation = next(
            m for m in result.metrics if m.name == "source_citations_present"
        )
        assert citation.value == 1.0

    def test_raw_errors_in_response_fails(self):
        agent = _make_agent(
            [
                {
                    "response": 'Traceback (most recent call last):\n  File "app.py", line 10',
                    "model_used": "primary",
                    "error": None,
                    "sources": [],
                }
            ]
        )
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        no_errors = next(m for m in result.metrics if m.name == "no_raw_errors")
        assert no_errors.value == 0.0

    def test_refusal_accuracy_correct_refusal(self):
        agent = _make_agent([_refusal_response()])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        case = _make_case(
            id="unans-001",
            category="unanswerable",
            expected_refuses=True,
            expected_answer=None,
            expected_source_files=[],
        )
        result = evaluator.evaluate([case])

        refusal = next(m for m in result.metrics if m.name == "refusal_accuracy")
        assert refusal.value == 1.0

    def test_refusal_accuracy_missed_refusal(self):
        agent = _make_agent([_good_response()])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        case = _make_case(
            id="unans-001",
            category="unanswerable",
            expected_refuses=True,
            expected_answer=None,
            expected_source_files=[],
        )
        result = evaluator.evaluate([case])

        refusal = next(m for m in result.metrics if m.name == "refusal_accuracy")
        assert refusal.value == 0.0

    def test_unexpected_refusal_fails_case(self):
        agent = _make_agent([_refusal_response()])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        assert result.case_results[0].passed is False
        assert "unexpected refusal" in result.case_results[0].details

    def test_response_too_short_fails(self):
        agent = _make_agent([_good_response("Short")])
        evaluator = GenerationStructuralEvaluator(
            agent, _make_settings(generation_min_response_length=10)
        )
        result = evaluator.evaluate([_make_case()])

        length = next(
            m for m in result.metrics if m.name == "response_length_compliance"
        )
        assert length.value == 0.0

    def test_response_too_long_fails(self):
        agent = _make_agent([_good_response("x" * 6000)])
        evaluator = GenerationStructuralEvaluator(
            agent, _make_settings(generation_max_response_length=5000)
        )
        result = evaluator.evaluate([_make_case()])

        length = next(
            m for m in result.metrics if m.name == "response_length_compliance"
        )
        assert length.value == 0.0

    def test_mixed_cases_computes_rates(self):
        cases = [
            _make_case(id="f-001"),
            _make_case(id="f-002"),
            _make_case(
                id="u-001",
                category="unanswerable",
                expected_refuses=True,
                expected_answer=None,
                expected_source_files=[],
            ),
        ]
        agent = _make_agent([_good_response(), _good_response(), _refusal_response()])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate(cases)

        assert result.passed
        assert result.total_cases == 3
        assert result.passed_cases == 3

        citation = next(
            m for m in result.metrics if m.name == "source_citations_present"
        )
        assert citation.value == 1.0

        refusal = next(m for m in result.metrics if m.name == "refusal_accuracy")
        assert refusal.value == 1.0

    def test_no_refusal_cases_defaults_to_pass(self):
        agent = _make_agent([_good_response()])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        refusal = next(m for m in result.metrics if m.name == "refusal_accuracy")
        assert refusal.value == 1.0
        assert refusal.passed

    def test_case_details_on_failure(self):
        agent = _make_agent(
            [
                {
                    "response": "",
                    "model_used": "primary",
                    "error": None,
                    "sources": [],
                }
            ]
        )
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        details = result.case_results[0].details
        assert "empty response" in details
        assert "no source citations" in details

    def test_case_details_on_success(self):
        agent = _make_agent([_good_response()])
        evaluator = GenerationStructuralEvaluator(agent, _make_settings())
        result = evaluator.evaluate([_make_case()])

        assert result.case_results[0].details == "All checks passed"
