"""Golden set loader — reads JSONL files and applies filters."""

import json
from pathlib import Path

from evals.config import EvalSettings
from evals.models import GoldenCase


def load_golden_set(
    version: str | None = None,
    categories: list[str] | None = None,
    tags: list[str] | None = None,
    max_cases: int | None = None,
) -> list[GoldenCase]:
    """Stream JSONL line-by-line, validate with Pydantic, apply filters."""
    settings = EvalSettings()
    version = version or settings.golden_set_version
    path = Path(settings.golden_set_dir) / f"{version}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")

    cases: list[GoldenCase] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = GoldenCase.model_validate_json(line)

            if categories and case.category not in categories:
                continue
            if tags and not any(t in case.tags for t in tags):
                continue

            cases.append(case)

            if max_cases and len(cases) >= max_cases:
                break

    return cases


def load_test_events(directory: str = "evals/fixtures/events") -> list[GoldenCase]:
    """Load simple Level 1 JSON test events, converting to GoldenCase."""
    path = Path(directory)
    if not path.exists():
        return []

    cases: list[GoldenCase] = []
    for file in sorted(path.glob("*.json")):
        data = json.loads(file.read_text())
        cases.append(GoldenCase.model_validate(data))

    return cases
