# Plan: Contextual Chunking — Prepend Document Header to Every Chunk

## Problem

Retrieval returns the wrong policy's coverage summary for name-specific queries like
"What is the Coverage A dwelling limit for **Maria Gonzalez's** Florida policy?"

All 5 policies have nearly identical COVERAGE SUMMARY chunks — same structure, same
field names — so their embeddings are almost indistinguishable. The policyholder name
("Maria Gonzalez") only appears in chunk 0 (declarations page), not in the coverage
summary chunk. Both vector search and BM25 lack the signal to rank the correct document
higher.

## Solution

**Contextual chunking**: extract the first N non-blank lines from the raw document text
(the "document header") and prepend them to every chunk before embedding. This gives
each chunk the identifying context (policy number, insured name, claim ID, state) that
distinguishes it from structurally similar chunks in other documents.

### Why first-N-lines instead of regex parsing

Every document type in the corpus has its identifiers in the opening lines:

| Document Type    | First ~5 lines contain                                    |
|------------------|-----------------------------------------------------------|
| Policies         | `POLICY NUMBER: PLY-FL-001`, `Named Insured: Maria Gonzalez`, `State: FL` |
| Claims           | `Claim ID: CLM-FL-2024-001`, `Policy: PLY-FL-001`, peril type |
| Adjuster Notes   | `ADJUSTER FIELD NOTES — CLM-FL-2024-001`, `Adjuster: Thomas Rivera` |
| Regulations      | `STATE OF FLORIDA — INSURANCE CLAIMS REGULATIONS SUMMARY` |

A generic first-N-lines approach works across all types without format-specific parsers
and won't break when new document types are added.

### What a chunk looks like after the change

Before:
```
COVERAGE SUMMARY
──────────────────────────────
Coverage A — Dwelling: $350,000
Coverage B — Other Structures: $35,000
...
```

After:
```
[CONTEXT: HOMEOWNERS POLICY — HO-3 SPECIAL FORM | POLICY NUMBER: PLY-FL-001 | Named Insured: Maria Gonzalez | Property Address: 4521 Palm Beach Blvd, Fort Lauderdale, FL 33301 | Policy Period: 01/01/2024 to 01/01/2025]

COVERAGE SUMMARY
──────────────────────────────
Coverage A — Dwelling: $350,000
Coverage B — Other Structures: $35,000
...
```

Both the embedding model and BM25 tsvector now see "Maria Gonzalez" and "PLY-FL-001" in
this chunk, allowing correct ranking.

## Files to Create / Modify

| # | File | Action | Purpose |
|---|---|---|---|
| 1 | `app/chunking.py` | Modify | Add `ContextualChunker` that wraps `RecursiveChunker`, prepending header to each chunk |
| 2 | `app/config.py` | Modify | Add `rag_context_header_lines: int = 5` setting |
| 3 | `app/ingestion.py` | Modify | Pass full document text to chunker so it can extract the header |
| 4 | `tests/test_chunking.py` | Modify | Add tests for `ContextualChunker` |
| 5 | `tests/test_ingestion.py` | Modify | Update ingestion tests to verify context prefix flows through |

No migration needed — the context is part of the chunk `content` stored in the existing
`documents` table. Existing documents require re-ingestion to pick up the change.

---

## Detailed Design

### 1. `app/config.py` — New setting

```python
rag_context_header_lines: int = 5   # number of non-blank lines from doc start to use as context
```

### 2. `app/chunking.py` — ContextualChunker

New chunker that wraps any base chunker. Extracts the document header, then prepends it
to each chunk produced by the base chunker.

```python
class ContextualChunker:
    def __init__(self) -> None:
        settings = get_settings()
        self._base = RecursiveChunker()
        self._header_lines = settings.rag_context_header_lines

    def _extract_header(self, text: str) -> str:
        """Extract first N non-blank lines as a single-line context prefix."""
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip decorative separator lines (=====, ─────)
            if all(c in "=─-─" for c in stripped):
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
```

Update `_STRATEGY_MAP`:
```python
_STRATEGY_MAP = {
    "recursive": RecursiveChunker,
    "contextual": ContextualChunker,
}
```

### 3. `app/config.py` — Default strategy change

```python
rag_chunking_strategy: str = "contextual"   # was "recursive"
```

### 4. `app/ingestion.py` — No interface change needed

The chunker already receives the full document text (`chunker.chunk(text)`), so the
contextual chunker can extract the header from it directly. No changes to the ingestion
pipeline interface.

### 5. Tests

**`tests/test_chunking.py`** — Add:
- `test_contextual_chunker_prepends_header` — verify prefix appears on every chunk
- `test_contextual_chunker_skips_separator_lines` — decorative lines excluded from header
- `test_contextual_chunker_empty_document` — graceful handling
- `test_contextual_chunker_header_fewer_lines_than_setting` — short documents use available lines
- `test_contextual_chunker_does_not_duplicate_on_first_chunk` — chunk 0 has the same prefix format

**`tests/test_ingestion.py`** — Update existing tests to use "contextual" strategy if
the default changes.

---

## Re-ingestion

After deploying, existing documents need re-ingestion to get the context prefix:

```bash
# Clear existing chunks and re-ingest
uv run python scripts/ingest.py ./documents/ --metadata ./documents/metadata.json
```

The `insert_chunks` method inserts new rows — existing duplicates should be cleared
first (delete by doc source, or truncate the table before re-ingestion).

## Expected Impact

- Queries mentioning policyholder names, claim IDs, or states will rank the correct
  document's chunks higher in both vector and BM25 search
- The `[CONTEXT: ...]` prefix adds ~100-200 chars per chunk — within the 1000-char
  chunk size budget (may slightly increase total chunk count)
- No impact on retrieval strategy code — the change is fully in the chunking layer
