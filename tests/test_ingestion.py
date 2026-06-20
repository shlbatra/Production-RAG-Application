from unittest.mock import MagicMock, patch

import pytest

from app.ingestion import ingest_document


@pytest.fixture
def ingestion_settings():
    s = MagicMock()
    s.rag_chunking_strategy = "recursive"
    s.rag_chunk_size = 20
    s.rag_chunk_overlap = 0
    return s


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.generate_embeddings.side_effect = lambda texts: [[0.1] for _ in texts]
    store.insert_chunks.side_effect = lambda records: len(records)
    return store


# ingest_document
class TestIngestDocument:
    def test_returns_summary_with_doc_id_and_count(self, mock_store, ingestion_settings):
        with patch("app.ingestion.get_parser") as mock_get_parser:
            mock_parser = MagicMock()
            mock_parser.parse.return_value = "First chunk.\n\nSecond chunk."
            mock_get_parser.return_value = mock_parser

            result = ingest_document(b"fake-pdf", "report.pdf", mock_store, ingestion_settings)

        assert "doc_id" in result
        assert result["filename"] == "report.pdf"
        assert result["chunks_stored"] == 2

    def test_calls_parser_with_file_bytes_and_filename(self, mock_store, ingestion_settings):
        with patch("app.ingestion.get_parser") as mock_get_parser:
            mock_parser = MagicMock()
            mock_parser.parse.return_value = "Some text."
            mock_get_parser.return_value = mock_parser

            mock_store.generate_embeddings.return_value = [[0.1]]
            mock_store.insert_chunks.return_value = 1

            ingest_document(b"raw-bytes", "doc.pdf", mock_store, ingestion_settings)

        mock_get_parser.assert_called_once_with("doc.pdf")
        mock_parser.parse.assert_called_once_with(b"raw-bytes", "doc.pdf")

    def test_embeds_all_chunks(self, mock_store, ingestion_settings):
        with patch("app.ingestion.get_parser") as mock_get_parser:
            mock_parser = MagicMock()
            mock_parser.parse.return_value = "First chunk.\n\nSecond chunk."
            mock_get_parser.return_value = mock_parser

            ingest_document(b"fake", "test.pdf", mock_store, ingestion_settings)

        args = mock_store.generate_embeddings.call_args[0][0]
        assert len(args) == 2

    def test_inserts_chunks_with_metadata(self, mock_store, ingestion_settings):
        with patch("app.ingestion.get_parser") as mock_get_parser:
            mock_parser = MagicMock()
            mock_parser.parse.return_value = "First chunk.\n\nSecond chunk."
            mock_get_parser.return_value = mock_parser

            ingest_document(b"fake", "report.pdf", mock_store, ingestion_settings)

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
        with patch("app.ingestion.get_parser") as mock_get_parser:
            mock_parser = MagicMock()
            mock_parser.parse.return_value = "Some text."
            mock_get_parser.return_value = mock_parser

            mock_store.generate_embeddings.return_value = [[0.1]]
            mock_store.insert_chunks.return_value = 1

            r1 = ingest_document(b"fake", "a.pdf", mock_store, ingestion_settings)
            r2 = ingest_document(b"fake", "b.pdf", mock_store, ingestion_settings)

        assert r1["doc_id"] != r2["doc_id"]

    def test_raises_for_unsupported_file_type(self, mock_store, ingestion_settings):
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_document(b"data", "file.csv", mock_store, ingestion_settings)
