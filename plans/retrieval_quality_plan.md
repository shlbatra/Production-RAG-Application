# Fix: Improve Retrieval Quality

## Problem

The retrieval eval passes only 4/13 answerable cases (hit_rate 0.46, recall 0.44). All 14 expected source documents exist in the corpus — the failures are retrieval quality, not missing data. Additionally, 1 of 2 unanswerable cases gets tangentially-related documents returned (flood insurance query matches Florida policy docs), causing the generation structural evaluator's refusal_accuracy to be 0.5.

### Failure Breakdown

| Pattern | Cases | Root Cause |
|---|---|---|
| 0 chunks above threshold | factual-001, 004, 006, 009, 010 | Correct doc exists but no chunk exceeds 0.7 cosine similarity |
| Wrong chunk retrieved | factual-003, multi-003 | Tangentially-related chunks from wrong docs score higher |
| Multi-hop recall gap | multi-001, multi-002 | Only 1 of 3 expected sources retrieved per query |
| False retrieval (unanswerable) | unanswerable-001 | Florida policy chunks surface for "flood insurance" query |

### Root Causes

1. **BM25 bypasses the similarity threshold.** `HybridRetriever` calls `full_text_search` which returns top_k results with no relevance floor. Keyword matches on common terms ("Florida", "policy", "claim") surface irrelevant documents. After RRF normalization, the top result always gets similarity=1.0 regardless of actual relevance.

2. **Similarity threshold 0.7 is too aggressive for `text-embedding-3-small`.** Five cases with correct documents in the corpus return 0 chunks because no chunk exceeds 0.7 cosine similarity. The embedding model produces lower absolute similarity scores for domain-specific insurance content.

3. **No post-retrieval relevance filtering.** Whatever the DB returns goes straight to the agent — there's no cross-encoder re-ranker or minimum-quality gate on the merged RRF results.

4. **Single-query retrieval misses multi-hop cases.** Multi-hop questions need chunks from 2-3 documents, but a single embedding can only be semantically close to one document's content.

## Implementation

### Step 1: Add minimum-score filtering to HybridRetriever

The HybridRetriever's RRF normalization creates artificial similarity scores (top result always = 1.0). Add a threshold filter on the raw RRF score to remove results that only matched via one weak retriever leg.

A document found by both retrievers gets an RRF score of ~2/(k+1). A document found by only one retriever gets ~1/(k+1). Filter at a fraction of the two-retriever baseline to keep only results with strong evidence from at least one retriever.

**File: `app/retrieval.py`, class `HybridRetriever.search`**

```python
def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
    similarity_results = self._store.search_similar(
        query=query, top_k=top_k, threshold=threshold
    )
    bm25_results = self._store.full_text_search(query=query, top_k=top_k)

    scores: dict[int, float] = {}
    docs: dict[int, dict] = {}

    for rank, doc in enumerate(similarity_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (self._k + rank + 1)
        docs[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (self._k + rank + 1)
        docs[doc_id] = doc

    # Filter: require at least 80% of a single-retriever top-1 score.
    # This removes results that only weakly matched one retriever.
    min_rrf_score = 0.8 / (self._k + 1)
    ranked = [
        (doc_id, score)
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if score >= min_rrf_score
    ][:top_k]

    if ranked:
        max_score = ranked[0][1]
        results = []
        for doc_id, score in ranked:
            doc = docs[doc_id]
            doc["similarity"] = round(score / max_score, 4) if max_score > 0 else 0
            results.append(doc)
        return results

    return []
```

This prevents BM25-only keyword matches (low RRF score) from surfacing when there's no vector similarity support. The unanswerable-001 case ("flood insurance") likely matches Florida docs only via BM25 keywords — this filter should drop those.

### Step 2: Lower similarity threshold from 0.7 to 0.55

Five cases return 0 chunks because no chunk reaches 0.7 cosine similarity with `text-embedding-3-small`. This model produces lower absolute similarity scores than `text-embedding-ada-002`. A 0.55 threshold retains relevant chunks while the RRF min-score filter (Step 1) prevents low-quality results from leaking through.

**File: `app/config.py`**

```python
rag_similarity_threshold: float = 0.55  # was 0.7
```

This is the vector-search leg's threshold only. The RRF min-score filter provides the quality floor for the merged results.

### Step 3: Increase top_k from 5 to 8 for multi-hop recall

Multi-hop questions need chunks from 2-3 documents. With top_k=5 and a 0.7 threshold, only 1 of 3 expected sources gets retrieved. Increasing to 8 gives the retriever more room to surface secondary and tertiary sources.

**File: `app/config.py`**

```python
rag_top_k: int = 8  # was 5
```

### Step 4: Update tests

**File: `tests/test_retrieval.py`**

- Update `TestHybridRetriever` to verify the min-score filter: add a test case where a doc only appears in BM25 with a low score and verify it gets filtered out.
- Update existing tests that assert on the number of returned results (the threshold change may affect mock expectations).

**File: `tests/test_agent.py`**

- No changes needed — agent tests mock the retriever, so they're unaffected by threshold changes.

### Step 5: Run evals and verify improvement

```bash
# Retrieval eval — expect hit_rate and recall improvement
uv run python scripts/run_evals.py --component retrieval -v --report

# Generation structural eval — expect refusal_accuracy improvement  
uv run python scripts/run_evals.py --component generation_structural -v --report
```

**Expected outcomes:**
- Retrieval: hit_rate should improve from 0.46 toward 0.90 (5 zero-chunk cases should now return results)
- Retrieval: recall should improve from 0.44 toward 0.70 (multi-hop cases get more sources)
- Generation structural: refusal_accuracy should improve from 0.50 toward 1.00 (unanswerable-001 no longer gets irrelevant Florida docs via BM25)

### Step 6: Tune if needed

If the threshold/filter changes over-correct (too many irrelevant results or legitimate results filtered):
- Adjust `min_rrf_score` multiplier (0.8 is conservative — lower to 0.6 if too aggressive, raise to 1.0 if too permissive)
- Adjust similarity threshold between 0.50-0.65 based on eval results
- Consider raising top_k further if multi-hop recall is still low

## Out of Scope

- **Embedding model upgrade** (e.g., `text-embedding-3-large`) — would improve absolute similarity scores but adds cost and requires re-ingestion. Consider as a follow-up if threshold tuning is insufficient.
- **Cross-encoder re-ranking** — a dedicated re-ranker model would be the best long-term solution for precision but adds latency and model dependency. Worth planning separately.
- **Query decomposition for multi-hop** — splitting multi-hop questions into sub-queries would improve recall but adds complexity and LLM calls. Separate initiative.
- **Chunk size tuning** — current 1000/200 is reasonable for the document sizes (28-121 lines). Not the primary failure mode.

## Files Changed

| File | Change |
|---|---|
| `app/retrieval.py` | Add min-score filter to `HybridRetriever.search` |
| `app/config.py` | Lower `rag_similarity_threshold` to 0.55, raise `rag_top_k` to 8 |
| `tests/test_retrieval.py` | Add hybrid min-score filter test, update threshold-sensitive assertions |
