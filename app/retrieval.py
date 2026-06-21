"""
Retrieval strategies — search documents using different approaches.

Uses a Protocol so new strategies can be added without touching existing code:
implement search(query, top_k, threshold) -> list[dict] and add a mapping entry.
"""

import logging
from typing import Protocol, runtime_checkable

from app.config import Settings
from app.document_store import DocumentStore

logger = logging.getLogger(__name__)


@runtime_checkable
class RetrievalStrategy(Protocol):
    def search(self, query: str, top_k: int, threshold: float) -> list[dict]: ...


class SimilarityRetriever:
    """Vector cosine similarity search via pgvector."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        return self._store.search_similar(query=query, top_k=top_k, threshold=threshold)


class BM25Retriever:
    """Keyword-based full-text search using Postgres tsvector/tsquery."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        return self._store.full_text_search(query=query, top_k=top_k)


class HybridRetriever:
    """Combines similarity + BM25 using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, document_store: DocumentStore, k: int = 60) -> None:
        self._store = document_store
        self._k = k

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        similarity_results = self._store.search_similar(
            query=query, top_k=top_k, threshold=threshold
        )
        bm25_results = self._store.full_text_search(query=query, top_k=top_k)

        # RRF: score = sum of 1/(k + rank) across retrievers. Docs found by both
        # retrievers accumulate a higher score than docs found by only one.
        # k=60 (paper default) flattens the rank curve so presence in both lists
        # matters more than exact position in either list.
        scores: dict[int, float] = {}
        docs: dict[int, dict] = {}

        for rank, doc in enumerate(similarity_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (self._k + rank + 1)
            docs[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (self._k + rank + 1)
            docs[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        if ranked:
            max_score = ranked[0][1]
            results = []
            for doc_id, score in ranked:
                doc = docs[doc_id]
                doc["similarity"] = round(score / max_score, 4) if max_score > 0 else 0
                results.append(doc)
            return results

        return []


_STRATEGY_MAP: dict[str, type] = {
    "similarity": SimilarityRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
}


def get_retriever(
    settings: Settings, document_store: DocumentStore
) -> RetrievalStrategy:
    cls = _STRATEGY_MAP.get(settings.rag_retrieval_strategy)
    if cls is None:
        raise ValueError(
            f"Unknown retrieval strategy '{settings.rag_retrieval_strategy}'. "
            f"Supported: {', '.join(sorted(_STRATEGY_MAP))}"
        )
    return cls(document_store=document_store)
