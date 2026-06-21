"""
Document parsing — extract raw text from uploaded files.

Uses a Protocol so new formats can be added without touching existing code:
implement parse(file, filename) -> str and add a mapping entry.
"""

import io
import logging
from typing import Protocol, runtime_checkable

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, file: bytes, filename: str) -> str: ...


class PdfParser:
    def parse(self, file: bytes, filename: str) -> str:
        reader = PdfReader(io.BytesIO(file))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)


class TextParser:
    def parse(self, file: bytes, filename: str) -> str:
        return file.decode("utf-8")


_EXTENSION_MAP: dict[str, DocumentParser] = {
    ".pdf": PdfParser(),
    ".txt": TextParser(),
}


def get_parser(filename: str) -> DocumentParser:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    parser = _EXTENSION_MAP.get(ext)
    if parser is None:
        supported = sorted(_EXTENSION_MAP.keys())
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(supported)}"
        )
    return parser
