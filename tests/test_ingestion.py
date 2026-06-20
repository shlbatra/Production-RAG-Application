from unittest.mock import MagicMock, patch

import pytest

from app.ingestion import ingest_document


def _run_ingest(file_bytes, filename, mock_store, ingestion_settings, parse_return="First chunk.\n\nSecond chunk."):
    with (
        patch("app.ingestion.get_parser") as mock_get_parser,
        patch("app.chunking.get_settings", return_value=ingestion_settings),
    ):
        mock_parser = MagicMock()
        mock_parser.parse.return_value = parse_return
        mock_get_parser.return_value = mock_parser
        result = ingest_document(file_bytes, filename, mock_store, ingestion_settings)
    return result, mock_get_parser, mock_parser


# ingest_document
class TestIngestDocument:
    def test_returns_summary_with_doc_id_and_count(self, mock_store, ingestion_settings):
        result, _, _ = _run_ingest(b"fake-pdf", "report.pdf", mock_store, ingestion_settings)
        assert "doc_id" in result
        assert result["filename"] == "report.pdf"
        assert result["chunks_stored"] == 2

    def test_calls_parser_with_file_bytes_and_filename(self, mock_store, ingestion_settings):
        _, mock_get_parser, mock_parser = _run_ingest(
            b"raw-bytes", "doc.pdf", mock_store, ingestion_settings, parse_return="Some text."
        )
        mock_get_parser.assert_called_once_with("doc.pdf")
        mock_parser.parse.assert_called_once_with(b"raw-bytes", "doc.pdf")

    def test_embeds_all_chunks(self, mock_store, ingestion_settings):
        _run_ingest(b"fake", "test.pdf", mock_store, ingestion_settings)
        args = mock_store.generate_embeddings.call_args[0][0]
        assert len(args) == 2

    def test_inserts_chunks_with_metadata(self, mock_store, ingestion_settings):
        _run_ingest(b"fake", "report.pdf", mock_store, ingestion_settings)
        records = mock_store.insert_chunks.call_args[0][0]
        assert len(records) == 2
        for i, record in enumerate(records):
            assert "content" in record
            assert "embedding" in record
            assert record["metadata"]["source"] == "report.pdf"
            assert record["metadata"]["chunk_index"] == i
            assert record["metadata"]["total_chunks"] == 2
            assert "doc_id" in record["metadata"]

    def test_generates_unique_doc_ids(self, mock_store, ingestion_settings):
        r1, _, _ = _run_ingest(b"fake", "a.pdf", mock_store, ingestion_settings, parse_return="Some text.")
        r2, _, _ = _run_ingest(b"fake", "b.pdf", mock_store, ingestion_settings, parse_return="Some text.")
        assert r1["doc_id"] != r2["doc_id"]

    def test_raises_for_unsupported_file_type(self, mock_store, ingestion_settings):
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_document(b"data", "file.csv", mock_store, ingestion_settings)
