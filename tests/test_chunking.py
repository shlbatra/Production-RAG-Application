from app.chunking import ChunkingStrategy, RecursiveChunker


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
