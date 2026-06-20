from unittest.mock import MagicMock, patch

import pytest

from app.document_store import DocumentStore


@pytest.fixture
def settings():
    s = MagicMock()
    s.supabase_database_url = "postgresql://user:pass@localhost:5432/testdb"
    s.embedding_model = "text-embedding-3-small"
    s.openai_api_key = "test-key"
    s.rag_top_k = 5
    s.rag_similarity_threshold = 0.7
    return s


@pytest.fixture
def store(settings):
    with patch("app.document_store.OpenAIEmbeddings"):
        return DocumentStore(settings)


@pytest.fixture
def mock_conn():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor
