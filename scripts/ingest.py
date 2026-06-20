"""
CLI script for bulk document ingestion into the RAG vector store.

Usage:
    uv run python scripts/ingest.py ./data/report.pdf        # single file
    uv run python scripts/ingest.py ./data/pdfs/              # directory (recursive)
"""

import sys
from pathlib import Path

from app.config import get_settings
from app.document_store import DocumentStore
from app.ingestion import ingest_document

SUPPORTED_EXTENSIONS = {".pdf"}


def find_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"Error: unsupported file type '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            sys.exit(1)
        return [path]

    if path.is_dir():
        files = sorted(
            f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"No supported files found in {path}")
            sys.exit(1)
        return files

    print(f"Error: {path} does not exist")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/ingest.py <file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])

    settings = get_settings()
    if not settings.rag_enabled:
        print("Error: SUPABASE_DATABASE_URL must be set for ingestion")
        sys.exit(1)

    document_store = DocumentStore(settings)
    files = find_files(target)

    print(f"Found {len(files)} file(s) to ingest\n")

    succeeded = 0
    failed = 0

    for file_path in files:
        file_bytes = file_path.read_bytes()
        try:
            result = ingest_document(
                file_bytes, file_path.name, document_store, settings
            )
            print(
                f"  {result['filename']}: {result['chunks_stored']} chunks (doc_id: {result['doc_id']})"
            )
            succeeded += 1
        except Exception as e:
            print(f"  {file_path.name}: FAILED — {e}")
            failed += 1

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
