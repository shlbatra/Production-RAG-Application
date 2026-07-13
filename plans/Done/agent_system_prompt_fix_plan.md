# Fix: Always Send System Prompt to Agent (DONE)

## Problem

The RAG system prompt is only injected into the LLM call when retrieval returns results (`if state.get("context")`). When retrieval returns nothing — either because the question is unanswerable or because similarity scores fall below the threshold — the LLM receives no system prompt at all and responds as a generic assistant.

This causes two failures surfaced by the generation structural evaluator:

1. **No refusal for unanswerable questions** — The system prompt tells the LLM to "say you don't have sufficient context to answer the question" when documents aren't relevant. Without the prompt, the LLM improvises: asking clarifying questions, answering from training data, or saying "I don't have access" instead of the expected refusal.

2. **False confidence on retrieval misses** — When retrieval misses a factual question (e.g., the document exists but similarity is below 0.7), the LLM answers from general knowledge with no indication that it's operating without source documents. The user has no way to tell a grounded answer from a hallucinated one.

### Current Behavior

```
User asks question
  → Retriever returns results?
    YES → System prompt + documents injected → LLM answers with grounding
    NO  → No system prompt → LLM answers as generic ChatGPT
```

### Desired Behavior

```
User asks question
  → Retriever returns results?
    YES → System prompt + documents injected → LLM answers with grounding
    NO  → System prompt (no documents) injected → LLM refuses clearly
```

## Affected Code

Both `process_message` and `try_fallback` in `app/agent.py` have the same pattern:

```python
# Lines 102-121 (process_message) and 123-145 (try_fallback)
if state.get("context"):
    chunks_text = "\n---\n".join(...)
    messages.insert(0, SystemMessage(content=RAG_SYSTEM_PROMPT + chunks_text))
```

The system prompt `RAG_SYSTEM_PROMPT` is a single string that combines the persona ("You are a helpful assistant"), the grounding instruction ("Use the following retrieved documents"), the refusal instruction ("If the documents don't contain relevant information, say you don't have sufficient context"), and the document header ("Retrieved Documents:").

This single string doesn't work when there are no documents — "Use the following retrieved documents to answer" makes no sense when there are none.

## Implementation

### Step 1: Split the system prompt into base + document sections

Replace the single `RAG_SYSTEM_PROMPT` with two parts:

```python
RAG_SYSTEM_PROMPT_BASE = (
    "You are a helpful assistant that answers questions based on retrieved documents. "
    "If no documents were retrieved or the documents don't contain relevant information, "
    "say you don't have sufficient context to answer the question. "
    "Do not answer from general knowledge."
)

RAG_SYSTEM_PROMPT_DOCS_HEADER = "\n\nRetrieved Documents:\n"
```

Key changes from the current prompt:
- Explicitly handles the "no documents retrieved" case
- Adds "Do not answer from general knowledge" to prevent hallucinated answers on retrieval misses
- Separates the document header so it's only appended when documents exist

### Step 2: Always inject the system prompt

In both `process_message` and `try_fallback`, always prepend the system prompt. Only append documents when they exist:

```python
def process_message(state: AgentState) -> dict:
    try:
        messages = list(state["messages"])
        system_content = RAG_SYSTEM_PROMPT_BASE
        if state.get("context"):
            chunks_text = "\n---\n".join(
                f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['content']}"
                for c in state["context"]
            )
            system_content += RAG_SYSTEM_PROMPT_DOCS_HEADER + chunks_text
        messages.insert(0, SystemMessage(content=system_content))
        response = self.primary_llm.invoke(messages)
        return {"messages": [response], "error": None, "model_used": "primary"}
    except Exception as e:
        ...
```

Apply the same change to `try_fallback`.

### Step 3: Extract shared prompt-building logic

Both `process_message` and `try_fallback` build messages identically. Extract to a helper to avoid the duplication:

```python
def _build_messages(state: AgentState) -> list[BaseMessage]:
    messages = list(state["messages"])
    system_content = RAG_SYSTEM_PROMPT_BASE
    if state.get("context"):
        chunks_text = "\n---\n".join(
            f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['content']}"
            for c in state["context"]
        )
        system_content += RAG_SYSTEM_PROMPT_DOCS_HEADER + chunks_text
    messages.insert(0, SystemMessage(content=system_content))
    return messages
```

### Step 4: Update existing tests

Update tests in `tests/test_agent.py` that assert on system message behavior:

- `TestContextInjection::test_system_message_prepended_with_context` — still passes, system message now includes base + documents
- `TestContextInjection::test_no_system_message_without_context` — **must change**: now expects a system message with just the base prompt (no documents section)
- Add new test: `test_system_message_without_context_instructs_refusal` — verify the base-only system message contains the refusal instruction

### Step 5: Verify with generation structural eval

Run `uv run python scripts/run_evals.py --component generation_structural -v` and confirm:
- Refusal accuracy improves (unanswerable cases now get the refusal instruction)
- Source citation rate may stay the same (retrieval misses are a separate issue) but false-confidence answers should become proper refusals

## Out of Scope

- **Retrieval quality** — 5 factual cases get 0 retrieval hits due to the similarity threshold. That's a retrieval/ingestion issue, not an agent prompt issue. The system prompt fix will cause these to correctly refuse rather than hallucinate, which is the right behavior until retrieval is improved.
- **Refusal pattern narrowing in the evaluator** — After this fix, the agent should consistently use "I don't have sufficient context" (the exact system prompt phrasing). The broader patterns added to the evaluator can stay as a safety net but should rarely trigger.

## Files Changed

| File | Change |
|---|---|
| `app/agent.py` | Split prompt, always inject system message, extract `_build_messages` |
| `tests/test_agent.py` | Update `test_no_system_message_without_context`, add refusal instruction test |
