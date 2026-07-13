# Plan: Chunking Evaluator

## Context

The evaluation system plan (section 2a) defines a chunking evaluator that measures
algorithmic quality of the chunking pipeline. The retrieval evaluator is already
implemented and integrated into `scripts/run_evals.py`. This plan adds the chunking
evaluator following the same patterns.

Unlike the retrieval evaluator, the chunking evaluator is **pure algorithmic** — no
LLM calls, no database, no embeddings. It reads fixture documents from disk, runs the
chunker, and measures structural quality metrics.

## Metrics

| Metric | How | Threshold |
|---|---|---|
| Chunk Size Compliance | % of chunks within `chunk_size` ± 10% | >= 0.90 |
| Boundary Quality | % of chunks NOT breaking mid-sentence | >= 0.70 |
| Information Preservation | char coverage ratio of joined chunks vs original | >= 0.99 |
| Overlap Correctness | overlap between adjacent chunks matches config ± 20% | >= 0.90 |

## Files to Create / Modify

| # | File | Action | Purpose |
|---|---|---|---|
| 1 | `evals/evaluators/chunking_eval.py` | Create | ChunkingEvaluator class |
| 2 | `evals/config.py` | Modify | Add chunking threshold settings |
| 3 | `scripts/run_evals.py` | Modify | Wire up chunking evaluator + add to --component choices |
| 4 | `tests/test_chunking_eval.py` | Create | Unit tests for the evaluator |
| 5 | `evals/fixtures/documents/` | Create | Copy sample docs for eval fixtures |

## Design

### ChunkingEvaluator

The evaluator takes a `ChunkingStrategy` instance and a list of document paths.
It does NOT use `GoldenCase` — chunking quality is measured per-document, not
per-question. Instead it accepts document paths directly and produces an `EvalResult`.

For each document:
1. Read the raw text
2. Run `chunker.chunk(text)` to get chunks
3. Compute per-document metrics
4. Aggregate across all documents

### Integration with run_evals.py

The chunking evaluator runs before retrieval (no external dependencies needed).
It reads documents from `evals/fixtures/documents/` (symlinked or copied from
`documents/`). The `--component chunking` flag runs it in isolation.
