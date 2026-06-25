"""Evaluation data models — golden cases, eval results, and reports."""

from typing import Literal

from pydantic import BaseModel


class GoldenCase(BaseModel):
    id: str
    category: Literal[
        "factual",
        "multi_hop",
        "unanswerable",
        "adversarial",
        "pii",
    ]
    question: str
    expected_answer: str | None = None
    expected_source_files: list[str]
    expected_chunk_contents: list[str] = []
    expected_refuses: bool = False
    difficulty: Literal["easy", "medium", "hard"]
    tags: list[str] = []


class MetricResult(BaseModel):
    name: str
    value: float
    threshold: float
    passed: bool


class CaseResult(BaseModel):
    case_id: str
    metrics: dict[str, float]
    passed: bool
    details: str = ""


class EvalResult(BaseModel):
    component: str
    metrics: list[MetricResult]
    case_results: list[CaseResult]
    passed: bool
    total_cases: int
    passed_cases: int
    failed_cases: int
