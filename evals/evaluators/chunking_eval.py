"""Chunking evaluator — measures structural quality of the chunking pipeline.

Pure algorithmic — no LLM calls. Reads documents from disk, runs the chunker,
and computes size compliance, boundary quality, information preservation, and
overlap correctness metrics.
"""

import logging
import re
from pathlib import Path

from app.chunking import ChunkingStrategy
from app.config import Settings
from evals.config import EvalSettings
from evals.models import CaseResult, EvalResult, MetricResult

logger = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"[.!?]\s*$")
CONTEXT_PREFIX = re.compile(r"^\[CONTEXT: [^\]]*\]\n\n")


class ChunkingEvaluator:
    def __init__(
        self,
        chunker: ChunkingStrategy,
        app_settings: Settings,
        eval_settings: EvalSettings | None = None,
    ) -> None:
        self._chunker = chunker
        self._chunk_size = app_settings.rag_chunk_size
        self._chunk_overlap = app_settings.rag_chunk_overlap
        self._settings = eval_settings or EvalSettings()

    def evaluate(self, document_paths: list[Path]) -> EvalResult:
        size_compliant_total = 0
        size_total = 0
        boundary_ok_total = 0
        boundary_total = 0
        preservation_ratios: list[float] = []
        overlap_ok_total = 0
        overlap_total = 0
        case_results: list[CaseResult] = []

        for doc_path in document_paths:
            text = doc_path.read_text()
            chunks = self._chunker.chunk(text)

            if not chunks:
                logger.debug("[%s] empty — no chunks produced", doc_path.name)
                continue

            size_ok, size_count = self._check_size_compliance(chunks)
            size_compliant_total += size_ok
            size_total += size_count

            boundary_ok, boundary_count = self._check_boundary_quality(chunks)
            boundary_ok_total += boundary_ok
            boundary_total += boundary_count

            preservation = self._check_info_preservation(text, chunks)
            preservation_ratios.append(preservation)

            overlap_ok, overlap_count = self._check_overlap_correctness(chunks)
            overlap_ok_total += overlap_ok
            overlap_total += overlap_count

            doc_passed = (
                size_ok == size_count or size_count == 0
            ) and preservation >= self._settings.chunking_info_preservation_min
            case_results.append(
                CaseResult(
                    case_id=doc_path.name,
                    metrics={
                        "chunks": len(chunks),
                        "size_compliant": size_ok / size_count if size_count else 1.0,
                        "boundary_quality": boundary_ok / boundary_count
                        if boundary_count
                        else 1.0,
                        "info_preservation": preservation,
                        "overlap_correctness": overlap_ok / overlap_count
                        if overlap_count
                        else 1.0,
                    },
                    passed=doc_passed,
                    details=f"{len(chunks)} chunks from {len(text)} chars",
                )
            )

        if not case_results:
            return EvalResult(
                component="chunking",
                metrics=[],
                case_results=[],
                passed=True,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
            )

        size_compliance = size_compliant_total / size_total if size_total else 1.0
        boundary_quality = boundary_ok_total / boundary_total if boundary_total else 1.0
        info_preservation = (
            sum(preservation_ratios) / len(preservation_ratios)
            if preservation_ratios
            else 1.0
        )
        overlap_correctness = overlap_ok_total / overlap_total if overlap_total else 1.0

        metrics = [
            MetricResult(
                name="chunk_size_compliance",
                value=round(size_compliance, 4),
                threshold=self._settings.chunking_size_compliance_min,
                passed=size_compliance >= self._settings.chunking_size_compliance_min,
            ),
            MetricResult(
                name="boundary_quality",
                value=round(boundary_quality, 4),
                threshold=self._settings.chunking_boundary_quality_min,
                passed=boundary_quality >= self._settings.chunking_boundary_quality_min,
            ),
            MetricResult(
                name="info_preservation",
                value=round(info_preservation, 4),
                threshold=self._settings.chunking_info_preservation_min,
                passed=info_preservation
                >= self._settings.chunking_info_preservation_min,
            ),
            MetricResult(
                name="overlap_correctness",
                value=round(overlap_correctness, 4),
                threshold=self._settings.chunking_overlap_correctness_min,
                passed=overlap_correctness
                >= self._settings.chunking_overlap_correctness_min,
            ),
        ]

        passed_cases = sum(1 for cr in case_results if cr.passed)

        return EvalResult(
            component="chunking",
            metrics=metrics,
            case_results=case_results,
            passed=all(m.passed for m in metrics),
            total_cases=len(case_results),
            passed_cases=passed_cases,
            failed_cases=len(case_results) - passed_cases,
        )

    @staticmethod
    def _strip_context_prefix(chunk: str) -> str:
        """Remove the [CONTEXT: ...] prefix added by ContextualChunker."""
        return CONTEXT_PREFIX.sub("", chunk)

    def _check_size_compliance(self, chunks: list[str]) -> tuple[int, int]:
        """Count chunks whose body (excluding context prefix) is within chunk_size ± 10%."""
        max_allowed = self._chunk_size * 1.10
        bodies = [self._strip_context_prefix(c) for c in chunks]
        compliant = sum(1 for b in bodies if len(b) <= max_allowed)
        return compliant, len(chunks)

    def _check_boundary_quality(self, chunks: list[str]) -> tuple[int, int]:
        """Count chunks that don't break mid-sentence (end at sentence boundary or are the last chunk)."""
        if len(chunks) <= 1:
            return len(chunks), len(chunks)
        ok = 0
        for chunk in chunks[:-1]:
            body = self._strip_context_prefix(chunk)
            if SENTENCE_END.search(body):
                ok += 1
        ok += 1
        return ok, len(chunks)

    def _check_info_preservation(self, original: str, chunks: list[str]) -> float:
        """Ratio of original non-whitespace characters preserved in joined chunk bodies."""
        original_stripped = re.sub(r"\s+", "", original)
        if not original_stripped:
            return 1.0
        bodies = [self._strip_context_prefix(c) for c in chunks]
        joined_stripped = re.sub(r"\s+", "", "".join(bodies))
        if len(joined_stripped) >= len(original_stripped):
            return 1.0
        return len(joined_stripped) / len(original_stripped)

    def _check_overlap_correctness(self, chunks: list[str]) -> tuple[int, int]:
        """For adjacent chunk pairs, check that body overlap is within ±20% of configured overlap."""
        if len(chunks) < 2 or self._chunk_overlap == 0:
            return 0, 0
        bodies = [self._strip_context_prefix(c) for c in chunks]
        ok = 0
        total = 0
        tolerance = self._chunk_overlap * 0.20
        for i in range(len(bodies) - 1):
            actual_overlap = self._measure_overlap(bodies[i], bodies[i + 1])
            if abs(actual_overlap - self._chunk_overlap) <= max(tolerance, 20):
                ok += 1
            total += 1
        return ok, total

    @staticmethod
    def _measure_overlap(chunk_a: str, chunk_b: str) -> int:
        """Find the longest suffix of chunk_a that is a prefix of chunk_b."""
        max_check = min(len(chunk_a), len(chunk_b))
        for length in range(max_check, 0, -1):
            if chunk_a.endswith(chunk_b[:length]):
                return length
        return 0
