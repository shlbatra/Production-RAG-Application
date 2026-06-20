"""
Chunking strategies — split raw text into chunks for embedding.

Uses a Protocol so new strategies can be added without touching existing code:
implement chunk(text) -> list[str] and add a mapping entry in _STRATEGY_MAP.
"""

from typing import Protocol, runtime_checkable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings


@runtime_checkable
class ChunkingStrategy(Protocol):
    def chunk(self, text: str) -> list[str]: ...


class RecursiveChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> list[str]:
        return self._splitter.split_text(text)


_STRATEGY_MAP: dict[str, type[RecursiveChunker]] = {
    "recursive": RecursiveChunker,
}


def get_chunker(settings: Settings) -> type:
    cls = _STRATEGY_MAP.get(settings.rag_chunking_strategy)
    if cls is None:
        supported = sorted(_STRATEGY_MAP.keys())
        raise ValueError(
            f"Unknown chunking strategy '{settings.rag_chunking_strategy}'. "
            f"Supported: {', '.join(supported)}"
        )
    return cls
