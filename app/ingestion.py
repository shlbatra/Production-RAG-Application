"""
Document ingestion pipeline — parse, chunk, embed, and store documents.

Single entry point shared by the API endpoint and the CLI script.
"""

import logging
import uuid

from app.chunking import get_chunker
from app.config import Settings
from app.document_parser import get_parser
from app.document_store import DocumentStore

logger = logging.getLogger(__name__)


def ingest_document(
    file_bytes: bytes,
    filename: str,
    document_store: DocumentStore,
    settings: Settings,
) -> dict:
    parser = get_parser(filename)
    text = parser.parse(file_bytes, filename)

    chunker_cls = get_chunker(settings)
    chunker = chunker_cls()
    chunks = chunker.chunk(text)

    doc_id = uuid.uuid4().hex

    embeddings = document_store.generate_embeddings(
        [c for c in chunks],
    )

    records = [
        {
            "content": chunks[i],
            "metadata": {
                "doc_id": doc_id,
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
            "embedding": embeddings[i],
        }
        for i in range(len(chunks))
    ]

    inserted = document_store.insert_chunks(records)

    logger.info(
        "Document ingested",
        extra={
            "extra_data": {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": inserted,
            }
        },
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_stored": inserted,
    }
