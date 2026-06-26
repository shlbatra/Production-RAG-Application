from unittest.mock import MagicMock

import pytest

from app.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RetrievalStrategy,
    SimilarityRetriever,
    get_retriever,
)


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

    def test_filters_low_scoring_bm25_only_results(self):
        """Chunks that only appear in BM25 at low ranks should be filtered out."""
        store = MagicMock()
        store.search_similar.return_value = []
        low_rank_docs = [
            {
                "id": i,
                "content": f"doc {i}",
                "metadata": {"source": f"{i}.pdf"},
                "similarity": 0.5,
            }
            for i in range(20)
        ]
        store.full_text_search.return_value = low_rank_docs

        retriever = HybridRetriever(document_store=store, k=60)
        results = retriever.search(query="test", top_k=20, threshold=0.7)

        # 1/(60 + rank+1): rank 0 → 0.01639, rank 19 → 0.01235
        # min_rrf_score = 0.8/61 = 0.01311
        # Ranks where 1/(60+rank+1) < 0.01311 → rank+1 > 60/0.01311 - 60 ≈ 16.3
        # So ranks 0-15 pass (16 results), ranks 16-19 get filtered
        assert len(results) < 20
        for r in results:
            assert r["id"] < 16

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
