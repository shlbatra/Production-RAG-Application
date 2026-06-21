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
    s.db_pool_min_conn = 2
    s.db_pool_max_conn = 10
    return s


@pytest.fixture
def store(settings):
    with patch("app.document_store.OpenAIEmbeddings"), \
         patch("app.document_store.ThreadedConnectionPool"):
        return DocumentStore(settings)


@pytest.fixture
def mock_conn():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


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


@pytest.fixture
def mock_settings():
    with patch("app.agent.get_settings") as mock:
        s = MagicMock()
        s.primary_model = "gpt-4.1-mini"
        s.fallback_model = "gpt-4.1-nano"
        s.openai_api_key = "test-key"
        s.max_retries = 3
        s.rag_top_k = 5
        s.rag_similarity_threshold = 0.7
        mock.return_value = s
        yield s


@pytest.fixture
def mock_doc_store():
    store = MagicMock()
    store.search_similar.return_value = [
        {
            "id": 1,
            "content": "doc A",
            "metadata": {"source": "a.pdf"},
            "similarity": 0.95,
        },
        {
            "id": 2,
            "content": "doc B",
            "metadata": {"source": "b.pdf"},
            "similarity": 0.80,
        },
    ]
    store.full_text_search.return_value = [
        {
            "id": 2,
            "content": "doc B",
            "metadata": {"source": "b.pdf"},
            "similarity": 0.9,
        },
        {
            "id": 3,
            "content": "doc C",
            "metadata": {"source": "c.pdf"},
            "similarity": 0.7,
        },
    ]
    return store


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.search.return_value = [
        {
            "id": 1,
            "content": "Python is a programming language.",
            "metadata": {"source": "intro.pdf", "doc_id": "abc"},
            "similarity": 0.92,
        },
        {
            "id": 2,
            "content": "Python was created by Guido van Rossum.",
            "metadata": {"source": "history.pdf", "doc_id": "def"},
            "similarity": 0.85,
        },
    ]
    return retriever


@pytest.fixture
def mock_document_store():
    store = MagicMock()
    store.search_similar.return_value = [
        {
            "id": 1,
            "content": "Python is a programming language.",
            "metadata": {"source": "intro.pdf", "doc_id": "abc"},
            "similarity": 0.92,
        },
        {
            "id": 2,
            "content": "Python was created by Guido van Rossum.",
            "metadata": {"source": "history.pdf", "doc_id": "def"},
            "similarity": 0.85,
        },
    ]
    return store
