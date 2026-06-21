"""
Document Store — pgvector-backed vector store over Supabase Postgres.
"""

import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.config import Settings

logger = logging.getLogger(__name__)


class DocumentStore:
    def __init__(self, settings: Settings) -> None:
        self._dsn = settings.supabase_database_url
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=SecretStr(settings.openai_api_key),
        )
        self._top_k = settings.rag_top_k
        self._threshold = settings.rag_similarity_threshold

    @contextmanager
    def _conn(self) -> Generator:
        conn = psycopg2.connect(self._dsn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def generate_embedding(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def insert_chunks(self, chunks: list[dict]) -> int:
        inserted = 0
        with self._conn() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(chunks), 100):
                    batch = chunks[i : i + 100]
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO documents (content, metadata, embedding) VALUES %s",
                        [
                            (
                                c["content"],
                                psycopg2.extras.Json(c["metadata"]),
                                str(c["embedding"]),
                            )
                            for c in batch
                        ],
                    )
                    inserted += len(batch)
                cur.execute(
                    "UPDATE documents SET search_vector = to_tsvector('english', content) "
                    "WHERE search_vector IS NULL"
                )
        logger.info("Inserted %d chunks", inserted)
        return inserted

    def full_text_search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Keyword-based search using Postgres tsvector/tsquery via bm25_search RPC."""
        top_k = top_k or self._top_k
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM bm25_search(%s, %s)",
                    (query, top_k),
                )
                results = [dict(row) for row in cur.fetchall()]

        if results:
            max_score = max(r["similarity"] for r in results)
            if max_score > 0:
                for r in results:
                    r["similarity"] = round(r["similarity"] / max_score, 4)

        return results

    def search_similar(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        top_k = top_k or self._top_k
        threshold = threshold or self._threshold
        embedding = self.generate_embedding(query)

        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM match_documents(%s::vector, %s, %s)",
                    (str(embedding), top_k, threshold),
                )
                return [dict(row) for row in cur.fetchall()]

    def list_documents(self) -> list[dict]:
        """Query distinct documents by grouping chunks on doc_id and source, returning each document's chunk count ordered by most recently ingested first."""
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        metadata->>'doc_id' AS doc_id,
                        metadata->>'source' AS source,
                        COUNT(*) AS chunk_count
                    FROM documents
                    WHERE metadata->>'doc_id' IS NOT NULL
                    GROUP BY metadata->>'doc_id', metadata->>'source'
                    ORDER BY MIN(created_at) DESC
                """)
                return [dict(row) for row in cur.fetchall()]

    def delete_document(self, doc_id: str) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE metadata->>'doc_id' = %s",
                    (doc_id,),
                )
                return cur.rowcount

    def health_check(self) -> bool:
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            logger.exception("Document store health check failed")
            return False
