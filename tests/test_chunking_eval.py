from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evals.config import EvalSettings
from evals.evaluators.chunking_eval import ChunkingEvaluator


def _make_app_settings(chunk_size=1000, chunk_overlap=200):
    s = MagicMock()
    s.rag_chunk_size = chunk_size
    s.rag_chunk_overlap = chunk_overlap
    return s


def _make_eval_settings(**overrides):
    defaults = {
        "chunking_size_compliance_min": 0.90,
        "chunking_boundary_quality_min": 0.70,
        "chunking_info_preservation_min": 0.99,
        "chunking_overlap_correctness_min": 0.90,
    }
    defaults.update(overrides)
    return EvalSettings(**defaults)


def _make_chunker(chunks):
    chunker = MagicMock()
    chunker.chunk.return_value = chunks
    return chunker


class TestChunkingEvaluator:
    def test_all_metrics_pass_for_well_formed_chunks(self, tmp_path):
        doc = tmp_path / "doc.txt"
        doc.write_text("This is a test document. It has multiple sentences.")

        chunker = _make_chunker(
            ["This is a test document.", "It has multiple sentences."]
        )
        evaluator = ChunkingEvaluator(
            chunker,
            _make_app_settings(chunk_size=100, chunk_overlap=0),
            _make_eval_settings(),
        )
        result = evaluator.evaluate([doc])

        assert result.component == "chunking"
        assert result.passed
        assert result.total_cases == 1

    def test_size_compliance_detects_oversized_chunks(self, tmp_path):
        doc = tmp_path / "doc.txt"
        doc.write_text("x" * 200)

        oversized = "x" * 150
        chunker = _make_chunker([oversized])
        evaluator = ChunkingEvaluator(
            chunker, _make_app_settings(chunk_size=100), _make_eval_settings()
        )
        result = evaluator.evaluate([doc])

        size_metric = next(
            m for m in result.metrics if m.name == "chunk_size_compliance"
        )
        assert size_metric.value < 1.0
        assert not size_metric.passed

    def test_size_compliance_allows_10_percent_tolerance(self, tmp_path):
        doc = tmp_path / "doc.txt"
        doc.write_text("x" * 200)

        just_over = "x" * 109
        chunker = _make_chunker([just_over])
        evaluator = ChunkingEvaluator(
            chunker, _make_app_settings(chunk_size=100), _make_eval_settings()
        )
        result = evaluator.evaluate([doc])

        size_metric = next(
            m for m in result.metrics if m.name == "chunk_size_compliance"
        )
        assert size_metric.value == 1.0

    def test_boundary_quality_detects_mid_sentence_breaks(self, tmp_path):
        doc = tmp_path / "doc.txt"
        doc.write_text("Hello world this is a test document with sentences.")

        chunker = _make_chunker(
            ["Hello world this is a", "test document with sentences."]
        )
        evaluator = ChunkingEvaluator(
            chunker,
            _make_app_settings(chunk_size=100),
            _make_eval_settings(chunking_boundary_quality_min=1.0),
        )
        result = evaluator.evaluate([doc])

        boundary_metric = next(
            m for m in result.metrics if m.name == "boundary_quality"
        )
        assert boundary_metric.value < 1.0

    def test_boundary_quality_passes_for_sentence_boundaries(self, tmp_path):
        doc = tmp_path / "doc.txt"
        doc.write_text("First sentence. Second sentence.")

        chunker = _make_chunker(["First sentence.", "Second sentence."])
        evaluator = ChunkingEvaluator(
            chunker, _make_app_settings(chunk_size=100), _make_eval_settings()
        )
        result = evaluator.evaluate([doc])

        boundary_metric = next(
            m for m in result.metrics if m.name == "boundary_quality"
        )
        assert boundary_metric.value == 1.0

    def test_info_preservation_measures_coverage(self, tmp_path):
        doc = tmp_path / "doc.txt"
        original = "ABCDEFGHIJ"
        doc.write_text(original)

        chunker = _make_chunker(["ABCDE", "FGHIJ"])
        evaluator = ChunkingEvaluator(
            chunker, _make_app_settings(chunk_size=100), _make_eval_settings()
        )
        result = evaluator.evaluate([doc])

        pres_metric = next(m for m in result.metrics if m.name == "info_preservation")
        assert pres_metric.value >= 0.99

    def test_empty_document_is_skipped(self, tmp_path):
        doc = tmp_path / "empty.txt"
        doc.write_text("")

        chunker = _make_chunker([])
        evaluator = ChunkingEvaluator(
            chunker, _make_app_settings(), _make_eval_settings()
        )
        result = evaluator.evaluate([doc])

        assert result.total_cases == 0
        assert result.passed

    def test_no_documents_returns_empty_result(self):
        chunker = _make_chunker([])
        evaluator = ChunkingEvaluator(
            chunker, _make_app_settings(), _make_eval_settings()
        )
        result = evaluator.evaluate([])

        assert result.total_cases == 0
        assert result.passed

    def test_multiple_documents_aggregated(self, tmp_path):
        doc1 = tmp_path / "a.txt"
        doc1.write_text("First document content here.")
        doc2 = tmp_path / "b.txt"
        doc2.write_text("Second document content here.")

        chunker = MagicMock()
        chunker.chunk.side_effect = [
            ["First document content here."],
            ["Second document content here."],
        ]
        evaluator = ChunkingEvaluator(
            chunker, _make_app_settings(chunk_size=100), _make_eval_settings()
        )
        result = evaluator.evaluate([doc1, doc2])

        assert result.total_cases == 2
        assert len(result.case_results) == 2

    def test_case_results_contain_document_name(self, tmp_path):
        doc = tmp_path / "report.txt"
        doc.write_text("Some content.")

        chunker = _make_chunker(["Some content."])
        evaluator = ChunkingEvaluator(
            chunker, _make_app_settings(chunk_size=100), _make_eval_settings()
        )
        result = evaluator.evaluate([doc])

        assert result.case_results[0].case_id == "report.txt"
