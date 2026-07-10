# Tool Calling Plan

*Updated 2026-06-26 — reflects system prompt split (PR #55) and retrieval quality filtering (PR #57).*

## Goal

Add tool calling to the RAG agent so the LLM can **decide when to search** rather than always retrieving context upfront. Today the graph is linear: `retrieve → process → (fallback) → (error)`. With tool calling, the LLM receives the user question first, decides if it needs documents, calls a `search_documents` tool, reads the results, and then answers. If the knowledge base has no relevant results, the LLM can fall back to a `web_search` tool before giving up — an "agentic RAG" pattern with web fallback.

### Expected Flow

```
User question
  → LLM decides: needs context? → search_documents tool
    → Results found → LLM answers from documents
    → NO_RESULTS / LOW_RELEVANCE → LLM tries web_search tool
      → Web results found → LLM answers from web (cites URLs)
      → Nothing found → LLM tells user it could not find the information
  → LLM decides: general question → responds directly (no tools)
```

## Why Tool Calling Over Always-Retrieve

| Current (always-retrieve) | With tools |
|---|---|
| Every query hits the vector DB, even "hi" or "what's 2+2" | LLM only searches when it actually needs context |
| Single retrieval pass — can't refine the search | LLM can search multiple times with different queries |
| Retrieval prompt is the raw user message | LLM can reformulate the query before searching |
| No fallback when docs don't have the answer | LLM can fall back to web search if docs return nothing |
| Lower latency per query (one round-trip) | Slightly higher latency when tools are used, but saves unnecessary retrievals |

## Current Architecture (What We're Changing)

```
app/agent.py — ProductionAgent
├── AgentState (messages, error, retry_count, model_used, context, sources)
├── Graph: START → retrieve → process → [fallback | error | END]
├── retrieve_context() always calls self.retriever.search()
├── _build_messages() always injects system prompt (with or without docs)
└── RAG_SYSTEM_PROMPT_BASE + RAG_SYSTEM_PROMPT_DOCS_HEADER (split prompt)
```

Recent changes to be aware of:
- **System prompt is always injected** — `_build_messages()` prepends `RAG_SYSTEM_PROMPT_BASE` to every request. When retrieval returns context, it appends `RAG_SYSTEM_PROMPT_DOCS_HEADER` + chunks. This helper and both prompt constants will be removed when moving to tool calling.
- **Retrieval quality thresholds** — vector similarity threshold is 0.55 (not 0.7), BM25 has a 0.3 normalized score floor, and HybridRetriever has an RRF min-score filter (0.8/(k+1)). The `search_documents` tool wraps the retriever as-is, so these thresholds carry over automatically.

The LLM (`ChatOpenAI`) is already from langchain-openai which supports OpenAI function/tool calling natively. LangGraph supports tool nodes out of the box.

## Design

### 1. Define Tools — `app/tools.py` (new file)

Two tools: `search_documents` (knowledge base) and `web_search` (internet fallback).

**a) `search_documents`** — wraps the existing retriever:

```python
from langchain_core.tools import tool

def create_search_tool(retriever, top_k: int, threshold: float):
    @tool
    def search_documents(query: str) -> str:
        """Search the knowledge base for documents relevant to the query.
        Use this when the user asks about topics that might be covered in uploaded documents."""
        results = retriever.search(query=query, top_k=top_k, threshold=threshold)
        if not results:
            return "No relevant documents found."
        formatted = []
        for r in results:
            source = r["metadata"].get("source", "unknown")
            formatted.append(f"[Source: {source}]\n{r['content']}")
        return "\n---\n".join(formatted)
    return search_documents
```

- Wraps the existing `RetrievalStrategy` — no changes to retrieval.py or document_store.py
- Returns formatted text the LLM can reason over
- `top_k` and `threshold` come from settings (currently top_k=5, threshold=0.55)
- All retrieval quality filters (BM25 min_score=0.3, RRF min-score) apply automatically since the tool calls the same retriever

**b) `web_search`** — Tavily-based internet fallback:

```python
from langchain_community.tools.tavily_search import TavilySearchResults

def create_web_search_tool(max_results: int = 3):
    return TavilySearchResults(
        max_results=max_results,
        name="web_search",
        description=(
            "Search the web for current information. "
            "Use this ONLY when search_documents returns no relevant results "
            "and the user's question requires factual information you don't have."
        ),
    )
```

- Tavily returns structured results (title, url, snippet) — no HTML parsing needed
- `max_results=3` keeps token usage and latency low
- The tool description guides the LLM to prefer docs first, web second
- Disabled by default — opt-in via `WEB_SEARCH_ENABLED=true` + `TAVILY_API_KEY`

### 2. Modify the Agent Graph — `app/agent.py`

Replace the linear `retrieve → process` flow with a tool-calling loop:

```
START → agent_node → should_continue?
                      ├── "tools" → tool_node → agent_node (loop)
                      ├── "end"   → END
                      └── "error" → fallback → END / error → END
```

Key changes to `ProductionAgent`:

**a) Bind tools to the LLM**
```python
from langgraph.prebuilt import ToolNode

tools = []
if self.rag_enabled:
    tools.append(create_search_tool(retriever, settings.rag_top_k, settings.rag_similarity_threshold))  # threshold=0.55
if settings.web_search_enabled:
    tools.append(create_web_search_tool(max_results=settings.web_search_max_results))

self.primary_llm = ChatOpenAI(...).bind_tools(tools)
self.fallback_llm = ChatOpenAI(...).bind_tools(tools)
```

No graph structure changes needed for web search — the same `agent → should_continue? → tools → agent` loop handles both tools.

**b) New `agent_node`** — replaces `retrieve_context`, `_build_messages`, and `process_message`:
```python
def agent_node(state: AgentState) -> dict:
    messages = list(state["messages"])
    messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    response = self.primary_llm.invoke(messages)
    return {"messages": [response], "model_used": "primary"}
```

This eliminates `_build_messages()`, `RAG_SYSTEM_PROMPT_BASE`, and `RAG_SYSTEM_PROMPT_DOCS_HEADER` — the tool-aware `SYSTEM_PROMPT` replaces all three.

**c) Routing function** — checks if the LLM wants to call a tool:
```python
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return "end"
```

**d) ToolNode** — LangGraph's prebuilt node that executes the tool calls and returns ToolMessages:
```python
tool_node = ToolNode(tools)
```

**e) Updated graph**:
```python
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
```

**f) Error handling / fallback** — wrap `agent_node` in try/except. On failure, route to `fallback_node` (same logic but uses `self.fallback_llm`). The fallback LLM also has tools bound, so it can search if needed.

### 3. Update AgentState

Remove `context` and `sources` fields (no longer pre-fetched). Sources can be extracted post-hoc from ToolMessages in the final state:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
```

### 4. Extract Sources from Tool Calls — `app/agent.py`

After the graph runs, parse the message history to find tool calls and extract source info for the API response. Distinguish document vs web sources so the client knows the provenance:

```python
def _extract_sources(self, messages: list[BaseMessage]) -> list[dict]:
    sources = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.name == "search_documents":
                # parse the formatted text to extract source names
                sources.append({"type": "document", ...})
            elif msg.name == "web_search":
                # parse web results (Tavily returns title + url)
                sources.append({"type": "web", ...})
    return sources
```

### 5. Add a Max Tool Calls Guard

Prevent infinite loops by capping the number of tool-calling rounds (e.g., 3):

```python
def should_continue(state: AgentState) -> str:
    tool_call_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    if tool_call_count >= MAX_TOOL_ROUNDS:
        return "end"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"
```

### 5a. Tool Ordering Guardrails

The LLM's decision to escalate (docs → web → give up) is probabilistic. Three layers make it reliable:

**Layer 1 — Explicit tool output signals.** Return unambiguous strings so the LLM doesn't have to interpret vague snippets:

```python
# In search_documents tool
if not results:
    return "NO_RESULTS: No relevant documents found in the knowledge base."

if all(r["similarity"] < 0.75 for r in results):
    return "LOW_RELEVANCE: Documents were found but none are highly relevant.\n\n" + formatted
```

Note: the vector threshold (0.55) and BM25 floor (0.3) already filter out weak results before they reach the tool. The `LOW_RELEVANCE` check here is a higher bar — it flags results that passed the retrieval filters but are still borderline. The LLM sees `NO_RESULTS` or `LOW_RELEVANCE` as a clear signal rather than guessing from content.

**Layer 2 — Enforce tool ordering in `should_continue`.** Don't rely solely on the prompt — programmatically block `web_search` if `search_documents` hasn't been called yet:

```python
def should_continue(state: AgentState) -> str:
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if len(tool_messages) >= MAX_TOOL_ROUNDS:
        return "end"

    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return "end"

    # Enforce ordering: block web_search unless search_documents has already run
    has_searched_docs = any(m.name == "search_documents" for m in tool_messages)
    requested_tools = [tc["name"] for tc in last.tool_calls]

    if "web_search" in requested_tools and not has_searched_docs:
        # Strip web_search from tool_calls, force docs first
        last.tool_calls = [tc for tc in last.tool_calls if tc["name"] != "web_search"]
        if not last.tool_calls:
            return "end"

    return "tools"
```

This means even if the LLM skips straight to `web_search`, the guard strips it out. The LLM can't bypass the ordering.

**Layer 3 — Stronger system prompt language:**

```python
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base and the web.\n\n"
    "STRICT RULES:\n"
    "1. You MUST call search_documents BEFORE web_search. Never skip this step.\n"
    "2. Only call web_search if search_documents returned NO_RESULTS or LOW_RELEVANCE.\n"
    "3. If both tools return nothing useful, tell the user you could not find the information.\n"
    "4. For general conversation (greetings, math, opinions), respond directly — no tools.\n"
    "5. When using web results, always cite the source URLs.\n"
)
```

### 6. Update System Prompt

The current split prompt (`RAG_SYSTEM_PROMPT_BASE` + `RAG_SYSTEM_PROMPT_DOCS_HEADER`) always injects context via `_build_messages()`. With tool calling, the LLM decides when to search, so the prompt changes from "here are your documents" to "here are your tools":

```python
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base and the web.\n\n"
    "STRICT RULES:\n"
    "1. You MUST call search_documents BEFORE web_search. Never skip this step.\n"
    "2. Only call web_search if search_documents returned NO_RESULTS or LOW_RELEVANCE.\n"
    "3. If both tools return nothing useful, tell the user you could not find the information.\n"
    "4. For general conversation (greetings, math, opinions), respond directly — no tools.\n"
    "5. When using web results, always cite the source URLs.\n"
)
```

This replaces `RAG_SYSTEM_PROMPT_BASE`, `RAG_SYSTEM_PROMPT_DOCS_HEADER`, and `_build_messages()` — all three are deleted.

### 7. Config — `app/config.py`

```python
max_tool_calls: int = 3  # max tool-calling rounds per request

# Web search
web_search_enabled: bool = False
tavily_api_key: str = ""
web_search_max_results: int = 3
```

Web search is disabled by default — opt-in via env var `WEB_SEARCH_ENABLED=true` + `TAVILY_API_KEY`.

### 8. Unchanged / Minor Changes

- **`app/retrieval.py`** — untouched, tool wraps the existing retriever (RRF min-score filter and threshold carry over)
- **`app/document_store.py`** — untouched (BM25 min_score=0.3 floor carries over)
- **`app/main.py`** — untouched, `agent.invoke()` signature stays the same
- **`app/models.py`** — add `type` field to `SourceReference` so client can distinguish `"document"` vs `"web"` sources:
  ```python
  class SourceReference(BaseModel):
      source: str
      similarity: float
      chunk_preview: str
      type: str = "document"  # "document" or "web"
  ```
- **Cache / security / monitoring** — untouched

### 9. Tests — `tests/test_agent.py`

Update existing agent tests:
- Test that a knowledge question triggers the `search_documents` tool call
- Test that a general question ("hi") does NOT trigger a tool call
- Test max tool call guard (mock LLM to always request tools, verify it caps out)
- Test fallback still works when primary errors during a tool-calling loop
- Test source extraction from tool messages distinguishes `"document"` vs `"web"` types
- Test that LLM calls `web_search` after `search_documents` returns no results
- Test that LLM does NOT call `web_search` when docs provide a good answer
- Test `web_search_enabled=False` means the web tool is not available
- Test tool ordering guard blocks `web_search` before `search_documents`
- Test max tool calls guard works with two tools

### 10. Dependency

Add `tavily-py` to `pyproject.toml` (integrates with langchain via `langchain-community`):

```toml
"tavily-py>=0.5.0",
```

### 11. Non-RAG Mode

When no retriever is configured (`retriever=None`), don't bind any tools — the graph becomes a straight `agent → END` with no tool node. This preserves the current behavior when RAG is disabled.

## File Changes Summary

| File | Change |
|---|---|
| `app/tools.py` | **New** — `create_search_tool()`, `create_web_search_tool()` |
| `app/agent.py` | Rewrite graph to tool-calling loop, update state, extract sources (doc + web) |
| `app/config.py` | Add `max_tool_calls`, `web_search_enabled`, `tavily_api_key`, `web_search_max_results` |
| `app/models.py` | Add `type` field to `SourceReference` |
| `pyproject.toml` | Add `tavily-py` dependency |
| `tests/test_agent.py` | Update/add tests for tool calling and web search fallback |

## Implementation Order

1. Create `app/tools.py` with `search_documents` and `web_search` tools
2. Add config settings (`max_tool_calls`, web search)
3. Add `tavily-py` dependency to `pyproject.toml`
4. Rewrite `app/agent.py` — new graph, state, source extraction, guardrails
5. Update `app/models.py` — add source type field
6. Update tests
7. Manual test via `/chat` endpoint (docs-only, web fallback, general conversation)
