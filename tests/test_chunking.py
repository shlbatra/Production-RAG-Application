from unittest.mock import MagicMock

import pytest

from app.chunking import ChunkingStrategy, RecursiveChunker, get_chunker


# RecursiveChunker.chunk
class TestRecursiveChunker:
    def test_short_text_returns_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
        result = chunker.chunk("Hello world")
        assert result == ["Hello world"]

    def test_splits_long_text_into_multiple_chunks(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 120
        result = chunker.chunk(text)
        assert len(result) > 1

    def test_splits_on_paragraph_boundary(self):
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        text = "First paragraph here.\n\nSecond paragraph here."
        result = chunker.chunk(text)
        assert len(result) == 2
        assert "First" in result[0]
        assert "Second" in result[1]

    def test_chunks_overlap_when_configured(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(f"word{i}" for i in range(50))
        result = chunker.chunk(text)
        assert len(result) > 1
        for i in range(1, len(result)):
            words_prev = set(result[i - 1].split())
            words_curr = set(result[i].split())
            assert words_prev & words_curr

    def test_empty_text_returns_empty_list(self):
        chunker = RecursiveChunker()
        result = chunker.chunk("")
        assert result == []

    def test_satisfies_protocol(self):
        chunker = RecursiveChunker()
        assert isinstance(chunker, ChunkingStrategy)


# get_chunker
class TestGetChunker:
    def _make_settings(self, strategy: str = "recursive") -> MagicMock:
        s = MagicMock()
        s.rag_chunking_strategy = strategy
        s.rag_chunk_size = 500
        s.rag_chunk_overlap = 100
        return s

    def test_returns_recursive_chunker(self):
        chunker = get_chunker(self._make_settings("recursive"))
        assert isinstance(chunker, RecursiveChunker)

    def test_passes_settings_to_chunker(self):
        chunker = get_chunker(self._make_settings())
        result = chunker.chunk("Hello world")
        assert result == ["Hello world"]

    def test_raises_for_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker(self._make_settings("unknown"))
