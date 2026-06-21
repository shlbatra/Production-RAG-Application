from unittest.mock import MagicMock, patch

import pytest

from app.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RetrievalStrategy,
    SimilarityRetriever,
    get_retriever,
)


@pytest.fixture
def mock_doc_store():
    store = MagicMock()
    store.search_similar.return_value = [
        {"id": 1, "content": "doc A", "metadata": {"source": "a.pdf"}, "similarity": 0.95},
        {"id": 2, "content": "doc B", "metadata": {"source": "b.pdf"}, "similarity": 0.80},
    ]
    store.full_text_search.return_value = [
        {"id": 2, "content": "doc B", "metadata": {"source": "b.pdf"}, "similarity": 0.9},
        {"id": 3, "content": "doc C", "metadata": {"source": "c.pdf"}, "similarity": 0.7},
    ]
    return store


class TestSimilarityRetriever:
    def test_delegates_to_search_similar(self, mock_doc_store):
        retriever = SimilarityRetriever(document_store=mock_doc_store)
        results = retriever.search(query="test", top_k=5, threshold=0.7)

        mock_doc_store.search_similar.assert_called_once_with(
            query="test", top_k=5, threshold=0.7
        )
        assert len(results) == 2

    def test_satisfies_protocol(self, mock_doc_store):
        retriever = SimilarityRetriever(document_store=mock_doc_store)
        assert isinstance(retriever, RetrievalStrategy)


class TestBM25Retriever:
    def test_delegates_to_full_text_search(self, mock_doc_store):
        retriever = BM25Retriever(document_store=mock_doc_store)
        results = retriever.search(query="test", top_k=5, threshold=0.7)

        mock_doc_store.full_text_search.assert_called_once_with(query="test", top_k=5)
        assert len(results) == 2

    def test_satisfies_protocol(self, mock_doc_store):
        retriever = BM25Retriever(document_store=mock_doc_store)
        assert isinstance(retriever, RetrievalStrategy)


class TestHybridRetriever:
    def test_calls_both_search_methods(self, mock_doc_store):
        retriever = HybridRetriever(document_store=mock_doc_store)
        retriever.search(query="test", top_k=5, threshold=0.7)

        mock_doc_store.search_similar.assert_called_once_with(
            query="test", top_k=5, threshold=0.7
        )
        mock_doc_store.full_text_search.assert_called_once_with(query="test", top_k=5)

    def test_rrf_ranking_prioritizes_overlap(self, mock_doc_store):
        """Doc B appears in both result sets and should rank highest."""
        retriever = HybridRetriever(document_store=mock_doc_store)
        results = retriever.search(query="test", top_k=5, threshold=0.7)

        assert results[0]["id"] == 2
        assert results[0]["similarity"] == 1.0

    def test_deduplicates_results(self, mock_doc_store):
        """Doc B appears in both sets but should appear only once in output."""
        retriever = HybridRetriever(document_store=mock_doc_store)
        results = retriever.search(query="test", top_k=5, threshold=0.7)

        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_respects_top_k(self, mock_doc_store):
        retriever = HybridRetriever(document_store=mock_doc_store)
        results = retriever.search(query="test", top_k=2, threshold=0.7)
        assert len(results) <= 2

    def test_empty_results(self):
        store = MagicMock()
        store.search_similar.return_value = []
        store.full_text_search.return_value = []

        retriever = HybridRetriever(document_store=store)
        results = retriever.search(query="test", top_k=5, threshold=0.7)
        assert results == []

    def test_satisfies_protocol(self, mock_doc_store):
        retriever = HybridRetriever(document_store=mock_doc_store)
        assert isinstance(retriever, RetrievalStrategy)


class TestGetRetriever:
    def test_returns_similarity_by_default(self, mock_doc_store):
        settings = MagicMock()
        settings.rag_retrieval_strategy = "similarity"
        retriever = get_retriever(settings, mock_doc_store)
        assert isinstance(retriever, SimilarityRetriever)

    def test_returns_bm25(self, mock_doc_store):
        settings = MagicMock()
        settings.rag_retrieval_strategy = "bm25"
        retriever = get_retriever(settings, mock_doc_store)
        assert isinstance(retriever, BM25Retriever)

    def test_returns_hybrid(self, mock_doc_store):
        settings = MagicMock()
        settings.rag_retrieval_strategy = "hybrid"
        retriever = get_retriever(settings, mock_doc_store)
        assert isinstance(retriever, HybridRetriever)

    def test_raises_for_unknown_strategy(self, mock_doc_store):
        settings = MagicMock()
        settings.rag_retrieval_strategy = "invalid"
        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            get_retriever(settings, mock_doc_store)
