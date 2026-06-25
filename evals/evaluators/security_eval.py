"""Security evaluator — measures injection detection accuracy.

Pure algorithmic — no LLM calls. Runs InputSanitizer.check() against
security test vectors and computes detection rate and false positive rate.
"""

import logging
from pathlib import Path

from pydantic import BaseModel

from app.security import InputSanitizer
from evals.config import EvalSettings
from evals.models import CaseResult, EvalResult, MetricResult

logger = logging.getLogger(__name__)


class SecurityVector(BaseModel):
    id: str
    input: str
    is_injection: bool


class SecurityEvaluator:
    def __init__(
        self,
        sanitizer: InputSanitizer,
        settings: EvalSettings | None = None,
    ) -> None:
        self._sanitizer = sanitizer
        self._settings = settings or EvalSettings()

    def evaluate(self, vectors: list[SecurityVector]) -> EvalResult:
        injection_cases = [v for v in vectors if v.is_injection]
        legitimate_cases = [v for v in vectors if not v.is_injection]

        true_positives = 0
        false_negatives = 0
        false_positives = 0
        true_negatives = 0
        case_results: list[CaseResult] = []

        for v in injection_cases:
            is_safe, reason = self._sanitizer.check(v.input)
            blocked = not is_safe
            if blocked:
                true_positives += 1
            else:
                false_negatives += 1
            case_results.append(
                CaseResult(
                    case_id=v.id,
                    metrics={"blocked": float(blocked), "is_injection": 1.0},
                    passed=blocked,
                    details=reason or "Not blocked",
                )
            )

        for v in legitimate_cases:
            is_safe, reason = self._sanitizer.check(v.input)
            blocked = not is_safe
            if blocked:
                false_positives += 1
            else:
                true_negatives += 1
            case_results.append(
                CaseResult(
                    case_id=v.id,
                    metrics={"blocked": float(blocked), "is_injection": 0.0},
                    passed=not blocked,
                    details=reason or "Allowed",
                )
            )

        detection_rate = (
            true_positives / len(injection_cases) if injection_cases else 1.0
        )
        false_positive_rate = (
            false_positives / len(legitimate_cases) if legitimate_cases else 0.0
        )

        metrics = [
            MetricResult(
                name="injection_detection_rate",
                value=round(detection_rate, 4),
                threshold=self._settings.security_injection_detection_min,
                passed=detection_rate
                >= self._settings.security_injection_detection_min,
            ),
            MetricResult(
                name="injection_false_positive_rate",
                value=round(false_positive_rate, 4),
                threshold=self._settings.security_false_positive_max,
                passed=false_positive_rate
                <= self._settings.security_false_positive_max,
            ),
        ]

        passed_cases = sum(1 for cr in case_results if cr.passed)

        return EvalResult(
            component="security",
            metrics=metrics,
            case_results=case_results,
            passed=all(m.passed for m in metrics),
            total_cases=len(case_results),
            passed_cases=passed_cases,
            failed_cases=len(case_results) - passed_cases,
        )


def load_security_vectors(
    path: str = "evals/fixtures/security_vectors.jsonl",
) -> list[SecurityVector]:
    vectors: list[SecurityVector] = []
    filepath = Path(path)
    if not filepath.exists():
        return vectors
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vectors.append(SecurityVector.model_validate_json(line))
    return vectors
