"""
CLI script for bulk document ingestion into the RAG vector store.

Usage:
    uv run python scripts/ingest.py ./data/report.pdf        # single file
    uv run python scripts/ingest.py ./data/pdfs/              # directory (recursive)
    uv run python scripts/ingest.py ./documents/ --metadata ./documents/metadata.json
    uv run python scripts/ingest.py ./documents/ --clear  # delete all existing chunks first
"""

import json
import sys
from pathlib import Path

from app.config import get_settings
from app.document_store import DocumentStore
from app.ingestion import ingest_document

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


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


def load_metadata(metadata_path: Path, base_dir: Path) -> dict:
    """Load metadata.json keyed by relative path from the documents root."""
    with open(metadata_path) as f:
        raw = json.load(f)
    result = {}
    for rel_path, meta in raw.items():
        abs_path = (base_dir / rel_path).resolve()
        result[str(abs_path)] = meta
    return result


def main() -> None:
    args = sys.argv[1:]
    metadata_path = None
    clear = False

    if "--clear" in args:
        args.remove("--clear")
        clear = True

    if "--metadata" in args:
        idx = args.index("--metadata")
        if idx + 1 >= len(args):
            print("Error: --metadata requires a path argument")
            sys.exit(1)
        metadata_path = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2 :]

    if len(args) != 1:
        print(
            "Usage: uv run python scripts/ingest.py <file_or_directory> [--metadata metadata.json] [--clear]"
        )
        sys.exit(1)

    target = Path(args[0])

    settings = get_settings()
    if not settings.rag_enabled:
        print("Error: SUPABASE_DATABASE_URL must be set for ingestion")
        sys.exit(1)

    document_store = DocumentStore(settings)

    if clear:
        deleted = document_store.clear_all()
        print(f"Cleared {deleted} existing chunks\n")

    files = find_files(target)

    file_metadata = {}
    if metadata_path:
        base_dir = metadata_path.parent
        file_metadata = load_metadata(metadata_path, base_dir)

    print(f"Found {len(files)} file(s) to ingest\n")

    succeeded = 0
    failed = 0

    for file_path in files:
        file_bytes = file_path.read_bytes()
        extra_metadata = file_metadata.get(str(file_path.resolve()), {})
        try:
            result = ingest_document(
                file_bytes,
                file_path.name,
                document_store,
                settings,
                extra_metadata=extra_metadata,
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
