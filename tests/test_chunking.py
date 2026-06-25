from unittest.mock import MagicMock, patch

import pytest

from app.chunking import (
    ChunkingStrategy,
    ContextualChunker,
    RecursiveChunker,
    get_chunker,
)


def _mock_settings(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    context_header_lines: int = 5,
):
    s = MagicMock()
    s.rag_chunk_size = chunk_size
    s.rag_chunk_overlap = chunk_overlap
    s.rag_context_header_lines = context_header_lines
    return s


# RecursiveChunker.chunk
class TestRecursiveChunker:
    def test_short_text_returns_single_chunk(self):
        with patch("app.chunking.get_settings", return_value=_mock_settings()):
            chunker = RecursiveChunker()
        result = chunker.chunk("Hello world")
        assert result == ["Hello world"]

    def test_splits_long_text_into_multiple_chunks(self):
        with patch(
            "app.chunking.get_settings",
            return_value=_mock_settings(chunk_size=50, chunk_overlap=10),
        ):
            chunker = RecursiveChunker()
        text = "A" * 120
        result = chunker.chunk(text)
        assert len(result) > 1

    def test_splits_on_paragraph_boundary(self):
        with patch(
            "app.chunking.get_settings",
            return_value=_mock_settings(chunk_size=30, chunk_overlap=0),
        ):
            chunker = RecursiveChunker()
        text = "First paragraph here.\n\nSecond paragraph here."
        result = chunker.chunk(text)
        assert len(result) == 2
        assert "First" in result[0]
        assert "Second" in result[1]

    def test_chunks_overlap_when_configured(self):
        with patch(
            "app.chunking.get_settings",
            return_value=_mock_settings(chunk_size=50, chunk_overlap=10),
        ):
            chunker = RecursiveChunker()
        text = " ".join(f"word{i}" for i in range(50))
        result = chunker.chunk(text)
        assert len(result) > 1
        for i in range(1, len(result)):
            words_prev = set(result[i - 1].split())
            words_curr = set(result[i].split())
            assert words_prev & words_curr

    def test_empty_text_returns_empty_list(self):
        with patch("app.chunking.get_settings", return_value=_mock_settings()):
            chunker = RecursiveChunker()
        result = chunker.chunk("")
        assert result == []

    def test_satisfies_protocol(self):
        with patch("app.chunking.get_settings", return_value=_mock_settings()):
            chunker = RecursiveChunker()
        assert isinstance(chunker, ChunkingStrategy)


# get_chunker
class TestGetChunker:
    def _make_settings(self, strategy: str = "recursive") -> MagicMock:
        s = MagicMock()
        s.rag_chunking_strategy = strategy
        return s

    def test_returns_recursive_chunker_instance(self):
        with patch("app.chunking.get_settings", return_value=_mock_settings()):
            chunker = get_chunker(self._make_settings("recursive"))
        assert isinstance(chunker, RecursiveChunker)

    def test_returns_contextual_chunker_instance(self):
        with patch("app.chunking.get_settings", return_value=_mock_settings()):
            chunker = get_chunker(self._make_settings("contextual"))
        assert isinstance(chunker, ContextualChunker)

    def test_raises_for_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker(self._make_settings("unknown"))


class TestContextualChunker:
    def _make_chunker(
        self, header_lines: int = 5, chunk_size: int = 1000, chunk_overlap: int = 0
    ):
        with patch(
            "app.chunking.get_settings",
            return_value=_mock_settings(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                context_header_lines=header_lines,
            ),
        ):
            return ContextualChunker(RecursiveChunker())

    def test_prepends_header_to_every_chunk(self):
        chunker = self._make_chunker(header_lines=2, chunk_size=30)
        text = "POLICY NUMBER: PLY-001\nInsured: Alice\n\nCoverage A: $100k\n\nCoverage B: $50k"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.startswith("[CONTEXT: ")
            assert "PLY-001" in chunk
            assert "Alice" in chunk

    def test_skips_separator_lines(self):
        chunker = self._make_chunker(header_lines=3)
        text = "Title Line\n══════════\n----------\nReal Line Two\nReal Line Three\n\nBody text here."
        chunks = chunker.chunk(text)
        header_part = chunks[0].split("]\n\n", 1)[0]
        assert "══════════" not in header_part
        assert "----------" not in header_part
        assert "Title Line" in header_part
        assert "Real Line Two" in header_part
        assert "Real Line Three" in header_part

    def test_empty_document(self):
        chunker = self._make_chunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_header_fewer_lines_than_setting(self):
        chunker = self._make_chunker(header_lines=10)
        text = "Only One Line\n\nBody content."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert "[CONTEXT: Only One Line | Body content.]" in chunks[0]

    def test_first_chunk_has_same_prefix_format(self):
        chunker = self._make_chunker(header_lines=2, chunk_size=30)
        text = "Header One\nHeader Two\n\nChunk one text.\n\nChunk two text."
        chunks = chunker.chunk(text)
        assert chunks[0].startswith("[CONTEXT: Header One | Header Two]\n\n")

    def test_satisfies_protocol(self):
        chunker = self._make_chunker()
        assert isinstance(chunker, ChunkingStrategy)
