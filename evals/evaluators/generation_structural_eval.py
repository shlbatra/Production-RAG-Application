"""Generation structural validator — checks response structure before LLM-judge.

Fast, deterministic checks on agent output: parse success, non-empty response,
source citations, no raw errors, refusal accuracy, and response length.
Runs ProductionAgent.invoke() for each golden case.
"""

import logging
import re
from typing import Protocol

from evals.config import EvalSettings
from evals.models import CaseResult, EvalResult, GoldenCase, MetricResult

logger = logging.getLogger(__name__)

_REFUSAL_PATTERNS = [
    "don't have sufficient context",
    "don't have enough information",
    "don't have enough context",
    "insufficient context to answer",
    "insufficient information to answer",
    "cannot answer this question",
    "unable to answer",
    "no relevant information",
    "not contain relevant information",
    "cannot find relevant information",
    "don't have the information needed",
]

_ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(
        r"(?:TypeError|ValueError|KeyError|AttributeError|ImportError"
        r"|RuntimeError|ConnectionError|TimeoutError|SyntaxError"
        r"|NameError|IndexError|FileNotFoundError|PermissionError):"
    ),
    re.compile(r'File ".*", line \d+'),
]

_REQUIRED_KEYS = {"response", "model_used", "error", "sources"}


class AgentInvoker(Protocol):
    def invoke(self, message: str) -> dict: ...


def _is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _REFUSAL_PATTERNS)


def _has_raw_errors(text: str) -> bool:
    return any(p.search(text) for p in _ERROR_PATTERNS)


class GenerationStructuralEvaluator:
    """Validates agent responses structurally against golden cases."""

    def __init__(
        self,
        agent: AgentInvoker,
        settings: EvalSettings | None = None,
    ) -> None:
        self._agent = agent
        self._settings = settings or EvalSettings()

    def evaluate(self, cases: list[GoldenCase]) -> EvalResult:
        parse_pass = 0
        non_empty_pass = 0
        citation_applicable = 0
        citation_pass = 0
        error_free_pass = 0
        refusal_applicable = 0
        refusal_pass = 0
        length_pass = 0
        case_results: list[CaseResult] = []

        for case in cases:
            logger.debug("[%s] invoking agent with: %r", case.id, case.question)
            result = self._agent.invoke(case.question)

            parsed = isinstance(result, dict) and _REQUIRED_KEYS.issubset(result.keys())
            if parsed:
                parse_pass += 1

            response_text = result.get("response", "") if parsed else ""
            sources = result.get("sources", []) if parsed else []

            non_empty = bool(response_text and response_text.strip())
            if non_empty:
                non_empty_pass += 1

            has_citations = True
            if not case.expected_refuses:
                citation_applicable += 1
                has_citations = len(sources) > 0
                if has_citations:
                    citation_pass += 1

            no_errors = not _has_raw_errors(response_text)
            if no_errors:
                error_free_pass += 1

            is_refusal = _is_refusal(response_text)
            refusal_correct = True
            if case.expected_refuses:
                refusal_applicable += 1
                refusal_correct = is_refusal
                if refusal_correct:
                    refusal_pass += 1
            else:
                refusal_correct = not is_refusal

            min_len = self._settings.generation_min_response_length
            max_len = self._settings.generation_max_response_length
            length_ok = min_len <= len(response_text) <= max_len
            if length_ok:
                length_pass += 1

            case_metrics = {
                "parsed": float(parsed),
                "non_empty": float(non_empty),
                "has_citations": float(has_citations),
                "no_errors": float(no_errors),
                "refusal_correct": float(refusal_correct),
                "length_ok": float(length_ok),
            }

            case_passed = all(
                [
                    parsed,
                    non_empty,
                    has_citations,
                    no_errors,
                    refusal_correct,
                    length_ok,
                ]
            )

            details_parts = []
            if not parsed:
                details_parts.append("parse failed")
            if not non_empty:
                details_parts.append("empty response")
            if not has_citations:
                details_parts.append("no source citations")
            if not no_errors:
                details_parts.append("raw errors detected")
            if not refusal_correct:
                if case.expected_refuses:
                    details_parts.append("expected refusal but got answer")
                else:
                    details_parts.append("unexpected refusal")
            if not length_ok:
                details_parts.append(f"length {len(response_text)} out of bounds")

            logger.debug(
                "[%s] parsed=%s non_empty=%s citations=%s errors=%s refusal=%s length=%s",
                case.id,
                parsed,
                non_empty,
                has_citations,
                no_errors,
                refusal_correct,
                length_ok,
            )

            case_results.append(
                CaseResult(
                    case_id=case.id,
                    metrics=case_metrics,
                    passed=case_passed,
                    details="; ".join(details_parts)
                    if details_parts
                    else "All checks passed",
                )
            )

        total = len(cases)
        if total == 0:
            return EvalResult(
                component="generation_structural",
                metrics=[],
                case_results=[],
                passed=True,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
            )

        parse_rate = parse_pass / total
        non_empty_rate = non_empty_pass / total
        citation_rate = (
            citation_pass / citation_applicable if citation_applicable else 1.0
        )
        error_free_rate = error_free_pass / total
        refusal_rate = refusal_pass / refusal_applicable if refusal_applicable else 1.0
        length_rate = length_pass / total

        metrics = [
            MetricResult(
                name="parse_success",
                value=round(parse_rate, 4),
                threshold=self._settings.generation_parse_success_min,
                passed=parse_rate >= self._settings.generation_parse_success_min,
            ),
            MetricResult(
                name="non_empty_response",
                value=round(non_empty_rate, 4),
                threshold=self._settings.generation_non_empty_min,
                passed=non_empty_rate >= self._settings.generation_non_empty_min,
            ),
            MetricResult(
                name="source_citations_present",
                value=round(citation_rate, 4),
                threshold=self._settings.generation_source_citation_min,
                passed=citation_rate >= self._settings.generation_source_citation_min,
            ),
            MetricResult(
                name="no_raw_errors",
                value=round(error_free_rate, 4),
                threshold=self._settings.generation_no_raw_errors_min,
                passed=error_free_rate >= self._settings.generation_no_raw_errors_min,
            ),
            MetricResult(
                name="refusal_accuracy",
                value=round(refusal_rate, 4),
                threshold=self._settings.generation_refusal_accuracy_min,
                passed=refusal_rate >= self._settings.generation_refusal_accuracy_min,
            ),
            MetricResult(
                name="response_length_compliance",
                value=round(length_rate, 4),
                threshold=self._settings.generation_response_length_min,
                passed=length_rate >= self._settings.generation_response_length_min,
            ),
        ]

        passed_cases = sum(1 for cr in case_results if cr.passed)

        return EvalResult(
            component="generation_structural",
            metrics=metrics,
            case_results=case_results,
            passed=all(m.passed for m in metrics),
            total_cases=total,
            passed_cases=passed_cases,
            failed_cases=total - passed_cases,
        )
