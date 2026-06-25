"""
Chunking strategies — split raw text into chunks for embedding.

Uses a Protocol so new strategies can be added without touching existing code:
implement chunk(text) -> list[str] and add a mapping entry in _CHUNKER_MAP.
"""

from typing import Protocol, runtime_checkable, Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings


@runtime_checkable
class ChunkingStrategy(Protocol):
    def chunk(self, text: str) -> list[str]: ...


class RecursiveChunker:
    def __init__(self) -> None:
        settings = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> list[str]:
        return self._splitter.split_text(text)


class ContextualChunker:
    """Decorator that prepends a document header to every chunk from a base chunker."""

    def __init__(self, base: ChunkingStrategy) -> None:
        settings = get_settings()
        self._base = base
        self._header_lines = settings.rag_context_header_lines

    def _extract_header(self, text: str) -> str:
        """Extract first N non-blank, non-decorative lines as a context prefix."""
        lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if all(c in "=═─-" for c in stripped):
                continue
            lines.append(stripped)
            if len(lines) >= self._header_lines:
                break
        return " | ".join(lines)

    def chunk(self, text: str) -> list[str]:
        header = self._extract_header(text)
        chunks = self._base.chunk(text)
        if not header:
            return chunks
        prefix = f"[CONTEXT: {header}]\n\n"
        return [prefix + chunk for chunk in chunks]


_CHUNKER_MAP: dict[str, Callable[[], ChunkingStrategy]] = {
    "recursive": lambda: RecursiveChunker(),
    "contextual": lambda: ContextualChunker(RecursiveChunker()),
}


def get_chunker(settings: Settings) -> ChunkingStrategy:
    factory = _CHUNKER_MAP.get(settings.rag_chunking_strategy)
    if factory is None:
        supported = sorted(_CHUNKER_MAP.keys())
        raise ValueError(
            f"Unknown chunking strategy '{settings.rag_chunking_strategy}'. "
            f"Supported: {', '.join(supported)}"
        )
    return factory()
