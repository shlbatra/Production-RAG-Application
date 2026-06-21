from unittest.mock import ANY, MagicMock, patch

import psycopg2
import pytest

from app.document_store import DocumentStore


# DocumentStore.__init__
class TestInit:
    def test_stores_dsn_and_defaults(self, settings):
        with (
            patch("app.document_store.OpenAIEmbeddings") as mock_emb,
            patch("app.document_store.ThreadedConnectionPool") as mock_pool,
        ):
            store = DocumentStore(settings)

        assert store._dsn == settings.supabase_database_url
        assert store._top_k == 5
        assert store._threshold == 0.7
        mock_emb.assert_called_once_with(
            model=settings.embedding_model,
            api_key=ANY,
        )
        mock_pool.assert_called_once_with(
            minconn=settings.db_pool_min_conn,
            maxconn=settings.db_pool_max_conn,
            dsn=settings.supabase_database_url,
        )


# DocumentStore.generate_embedding / generate_embeddings
class TestGenerateEmbedding:
    def test_delegates_to_embed_query(self, store):
        store._embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        result = store.generate_embedding("hello")
        store._embeddings.embed_query.assert_called_once_with("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_delegates_to_embed_documents(self, store):
        store._embeddings.embed_documents.return_value = [[0.1], [0.2]]
        result = store.generate_embeddings(["a", "b"])
        store._embeddings.embed_documents.assert_called_once_with(["a", "b"])
        assert result == [[0.1], [0.2]]


# DocumentStore.insert_chunks
class TestInsertChunks:
    def test_inserts_single_batch(self, store, mock_conn):
        conn, cursor = mock_conn
        store._pool.getconn.return_value = conn
        chunks = [
            {
                "content": "text1",
                "metadata": {"doc_id": "abc"},
                "embedding": [0.1, 0.2],
            },
            {
                "content": "text2",
                "metadata": {"doc_id": "abc"},
                "embedding": [0.3, 0.4],
            },
        ]

        with patch("app.document_store.psycopg2.extras.execute_values") as mock_exec:
            result = store.insert_chunks(chunks)

        assert result == 2
        mock_exec.assert_called_once()
        args = mock_exec.call_args
        assert "INSERT INTO documents" in args[0][1]

    def test_batches_in_groups_of_100(self, store, mock_conn):
        conn, cursor = mock_conn
        store._pool.getconn.return_value = conn
        chunks = [
            {"content": f"text{i}", "metadata": {"doc_id": "abc"}, "embedding": [0.1]}
            for i in range(150)
        ]

        with patch("app.document_store.psycopg2.extras.execute_values") as mock_exec:
            result = store.insert_chunks(chunks)

        assert result == 150
        assert mock_exec.call_count == 2

    def test_empty_chunks(self, store, mock_conn):
        conn, _ = mock_conn
        store._pool.getconn.return_value = conn

        with patch("app.document_store.psycopg2.extras.execute_values") as mock_exec:
            result = store.insert_chunks([])

        assert result == 0
        mock_exec.assert_not_called()


# DocumentStore.search_similar
class TestSearchSimilar:
    def test_embeds_query_and_calls_rpc(self, store, mock_conn):
        conn, cursor = mock_conn
        store._pool.getconn.return_value = conn
        store._embeddings.embed_query.return_value = [0.1, 0.2]
        cursor.fetchall.return_value = [
            {
                "id": 1,
                "content": "chunk",
                "metadata": {"source": "doc.pdf"},
                "similarity": 0.9,
            },
        ]

        results = store.search_similar("test query", top_k=3, threshold=0.8)

        store._embeddings.embed_query.assert_called_once_with("test query")
        cursor.execute.assert_called_once()
        sql, params = cursor.execute.call_args[0]
        assert "match_documents" in sql
        assert params[1] == 3
        assert params[2] == 0.8
        assert len(results) == 1
        assert results[0]["similarity"] == 0.9

    def test_uses_defaults_when_not_specified(self, store, mock_conn):
        conn, cursor = mock_conn
        store._pool.getconn.return_value = conn
        store._embeddings.embed_query.return_value = [0.1]
        cursor.fetchall.return_value = []

        store.search_similar("query")

        _, params = cursor.execute.call_args[0]
        assert params[1] == 5
        assert params[2] == 0.7


# DocumentStore.list_documents
class TestListDocuments:
    def test_returns_grouped_documents(self, store, mock_conn):
        conn, cursor = mock_conn
        store._pool.getconn.return_value = conn
        cursor.fetchall.return_value = [
            {"doc_id": "abc", "source": "report.pdf", "chunk_count": 10},
            {"doc_id": "def", "source": "notes.pdf", "chunk_count": 5},
        ]

        results = store.list_documents()

        assert len(results) == 2
        assert results[0]["doc_id"] == "abc"
        assert results[1]["chunk_count"] == 5
        sql = cursor.execute.call_args[0][0]
        assert "GROUP BY" in sql
        assert "doc_id" in sql


# DocumentStore.delete_document
class TestDeleteDocument:
    def test_deletes_by_doc_id(self, store, mock_conn):
        conn, cursor = mock_conn
        store._pool.getconn.return_value = conn
        cursor.rowcount = 10

        result = store.delete_document("abc-123")

        assert result == 10
        sql, params = cursor.execute.call_args[0]
        assert "DELETE FROM documents" in sql
        assert params == ("abc-123",)


# DocumentStore.health_check
class TestHealthCheck:
    def test_returns_true_on_success(self, store, mock_conn):
        conn, _ = mock_conn
        store._pool.getconn.return_value = conn

        assert store.health_check() is True

    def test_returns_false_on_failure(self, store):
        store._pool.getconn.side_effect = psycopg2.OperationalError(
            "connection refused"
        )

        assert store.health_check() is False


# DocumentStore._conn
class TestConnContextManager:
    def test_commits_on_success_and_returns_to_pool(self, store):
        conn = MagicMock()
        store._pool.getconn.return_value = conn

        with store._conn():
            pass

        conn.commit.assert_called_once()
        store._pool.putconn.assert_called_once_with(conn)

    def test_rolls_back_on_error_and_returns_to_pool(self, store):
        conn = MagicMock()
        store._pool.getconn.return_value = conn

        with pytest.raises(ValueError):
            with store._conn():
                raise ValueError("boom")

        conn.rollback.assert_called_once()
        store._pool.putconn.assert_called_once_with(conn)


# DocumentStore.close
class TestClose:
    def test_closes_all_pool_connections(self, store):
        store.close()
        store._pool.closeall.assert_called_once()
