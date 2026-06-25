"""Evaluation protocols — common interface for all component evaluators."""

from typing import Protocol

from evals.models import EvalResult, GoldenCase


class ComponentEvaluator(Protocol):
    def evaluate(self, cases: list[GoldenCase]) -> EvalResult: ...
