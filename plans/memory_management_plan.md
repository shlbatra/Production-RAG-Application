# Memory Management Plan

## Problem

The agent is stateless — every `/chat` request starts from scratch with a single `HumanMessage`. There is no conversation history, so:

- Users can't ask follow-up questions ("What about section 3?" after "Summarize the document")
- The LLM can't reference its own prior answers
- Each request re-triggers tool calls even when the context was just retrieved
- The `thread_id` generated in `main.py` is thrown away — it's not used for threading

The Redis cache helps with identical queries, but doesn't help with multi-turn conversations.

## Goal

Add conversation memory so users can have multi-turn interactions, while keeping context within LLM token limits and storing conversations persistently.

## Current Architecture (What Changes)

```
POST /chat  →  new HumanMessage  →  agent.invoke()  →  response
              (no history)          (no prior context)
```

Key files:
- `app/agent.py` — `ProductionAgent.invoke()` creates fresh state every call
- `app/main.py` — generates `thread_id` but never reuses it
- `app/models.py` — `ChatRequest` has no `thread_id` field
- `app/cache.py` — caches by query hash, not by conversation

## Design

### 1. Thread-Based Conversation Store — `app/conversation_store.py` (new file)

Store conversation history in Supabase (same Postgres instance as documents). Each message belongs to a `thread_id`.

**Schema** (new Supabase migration):

```sql
CREATE TABLE conversations (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('human', 'ai', 'tool', 'system')),
    content     TEXT NOT NULL,
    tool_name   TEXT,           -- for tool messages
    tool_call_id TEXT,          -- for tool messages
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_conversations_thread ON conversations (thread_id, created_at);
```

**ConversationStore class:**

```python
class ConversationStore:
    def __init__(self, pool: ThreadedConnectionPool) -> None:
        self._pool = pool

    def save_message(self, thread_id: str, role: str, content: str, ...) -> None:
        """Persist a single message."""

    def get_history(self, thread_id: str, limit: int = 50) -> list[BaseMessage]:
        """Load conversation history as LangChain messages, ordered by created_at."""

    def delete_thread(self, thread_id: str) -> int:
        """Delete all messages in a thread."""
```

Why Postgres over Redis for conversations:
- Conversations need durability — Redis TTL would silently drop mid-conversation history
- Already have a Postgres connection pool via `DocumentStore`
- Can share the same `ThreadedConnectionPool` instance (pass it in, don't create a new one)
- Redis stays for response caching (short-lived, TTL-appropriate)

### 2. Update the API Contract — `app/models.py`

**a) Accept `thread_id` on requests:**

```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    thread_id: str | None = None  # omit for new conversation
```

When `thread_id` is None, generate a new one (current behavior). When provided, load and continue that conversation.

**b) The response already returns `thread_id`** — no change needed there. Clients use the `thread_id` from the first response to continue the conversation.

### 3. Context Window Management — `app/memory.py` (new file)

The LLM has a finite context window. As conversations grow, we need to keep the message list within token limits without losing important context.

**Strategy: Sliding window with summary**

```python
class ConversationMemory:
    def __init__(self, max_tokens: int = 8000, summary_threshold: int = 6000):
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold

    def prepare_messages(self, history: list[BaseMessage]) -> list[BaseMessage]:
        """Trim or summarize history to fit within token limits."""
        token_count = self._count_tokens(history)

        if token_count <= self.max_tokens:
            return history

        # Keep most recent messages, summarize older ones
        return self._summarize_and_trim(history)

    def _summarize_and_trim(self, history: list[BaseMessage]) -> list[BaseMessage]:
        """Split history into old (summarized) and recent (kept verbatim)."""
        # 1. Find split point — keep the last N messages that fit in max_tokens/2
        # 2. Summarize everything before the split into a single SystemMessage
        # 3. Return [summary] + recent_messages

    def _count_tokens(self, messages: list[BaseMessage]) -> int:
        """Estimate token count. Use tiktoken for accuracy."""
```

Three approaches, from simplest to most sophisticated:

| Approach | How it works | Trade-off |
|---|---|---|
| **Sliding window** | Keep last N messages, drop older ones | Simple, but loses early context |
| **Summary + window** | Summarize old messages into a single message, keep recent verbatim | Extra LLM call for summarization, but preserves key context |
| **Token-budget trim** | Count tokens, drop oldest messages until under budget | No extra LLM calls, but might drop mid-topic |

**Recommendation: Start with token-budget trim, add summary later.**

Token-budget trim is simplest to implement and doesn't add latency (no LLM summarization call). It works well for most conversations. Summary can be added as a follow-up when we see users hitting context limits on long conversations.

```python
def trim_to_budget(self, history: list[BaseMessage], budget: int) -> list[BaseMessage]:
    """Drop oldest messages (after the first human message) until under budget."""
    if not history:
        return history

    total = sum(self._msg_tokens(m) for m in history)
    if total <= budget:
        return history

    # Always keep the first message (original question gives context)
    # and all recent messages. Drop from the middle.
    trimmed = list(history)
    while len(trimmed) > 2 and sum(self._msg_tokens(m) for m in trimmed) > budget:
        trimmed.pop(1)  # remove second-oldest

    return trimmed
```

### 4. Wire Memory into the Agent — `app/agent.py`

Update `ProductionAgent.invoke()` to accept and use conversation history:

```python
def invoke(self, message: str, history: list[BaseMessage] | None = None) -> dict:
    messages = []
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=message))

    result = self.graph.invoke({
        "messages": messages,
        "error": None,
        "retry_count": 0,
        "model_used": "",
    })
    # ... rest unchanged
```

The agent node already prepends the system prompt to `state["messages"]`, so history messages are naturally included.

### 5. Wire Memory into the Endpoint — `app/main.py`

Update the `/chat` endpoint to load/save conversation history:

```python
@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    # Determine thread_id
    thread_id = body.thread_id or uuid.uuid4().hex

    # Load conversation history (if continuing a thread)
    history = []
    if body.thread_id and conversation_store:
        history = conversation_store.get_history(body.thread_id)
        history = memory.prepare_messages(history)  # trim to budget

    # ... security check, cache lookup ...

    # Invoke agent with history
    result = agent.invoke(cleaned_message, history=history)

    # Save messages to conversation store
    if conversation_store:
        conversation_store.save_message(thread_id, "human", cleaned_message)
        conversation_store.save_message(thread_id, "ai", result["response"])

    # ... rest unchanged, thread_id already returned in response
```

### 6. Cache Behavior with Memory

The current cache keys on `sha256(query)`. With conversation memory, the same query can have different answers depending on conversation context.

**Options:**

| Option | Key | Pros | Cons |
|---|---|---|---|
| A. Disable cache for threaded requests | query hash | Simple, correct | Loses caching for returning conversations |
| B. Include thread_id in cache key | `sha256(thread_id + query)` | Thread-specific caching | Cache almost never hits (unique per turn) |
| C. Cache only first-turn requests | query hash (only when no thread_id) | Preserves current behavior for new queries | Clear, minimal change |

**Recommendation: Option C** — cache only when `thread_id` is None (new conversation). Continuing a conversation should always go to the LLM since context differs.

```python
# In /chat endpoint
is_new_conversation = body.thread_id is None

if is_new_conversation:
    cached_response = cache.get(cleaned_message)
    if cached_response is not None:
        # ... return cached
```

### 7. Conversation Management Endpoints

Add endpoints to manage conversations:

```python
@app.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    """Get conversation history."""

@app.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    """Delete a conversation thread."""
```

### 8. Config — `app/config.py`

```python
# Memory / Conversation
memory_enabled: bool = True
memory_max_tokens: int = 8000         # max tokens for conversation context
memory_max_messages: int = 50         # max messages to load from store
```

### 9. Non-Memory Mode

When `memory_enabled=False` or no Postgres is configured, the agent behaves exactly as it does today — stateless, per-request. The `thread_id` is still generated and returned but not used for threading.

This means:
- `conversation_store` is None → history loading/saving is skipped
- `memory.prepare_messages()` is never called
- Cache works as today (keyed on query hash)
- No new database table needed

## File Changes Summary

| File | Change |
|---|---|
| `app/conversation_store.py` | **New** — `ConversationStore` class (Postgres-backed) |
| `app/memory.py` | **New** — `ConversationMemory` class (token-budget trimming) |
| `app/models.py` | Add `thread_id` field to `ChatRequest` |
| `app/agent.py` | Accept `history` param in `invoke()` |
| `app/main.py` | Load/save conversation history, conditional caching |
| `app/config.py` | Add memory settings |
| `supabase/migrations/` | New migration for `conversations` table |
| `tests/test_memory.py` | **New** — tests for trimming logic |
| `tests/test_conversation_store.py` | **New** — tests for persistence |
| `tests/test_agent.py` | Add tests for multi-turn invocation |

## Implementation Order

1. Add config settings (`memory_enabled`, `memory_max_tokens`, `memory_max_messages`)
2. Create `conversations` table migration
3. Build `ConversationStore` (save/load/delete)
4. Build `ConversationMemory` (token-budget trimming)
5. Update `ChatRequest` model to accept `thread_id`
6. Update `ProductionAgent.invoke()` to accept history
7. Wire everything in `main.py` — load history, invoke with context, save messages, conditional cache
8. Add conversation management endpoints
9. Tests
10. Manual test: multi-turn conversation via `/chat`

## Future Improvements (Not in This Plan)

- **Summary-based compression** — LLM-generated summaries of old messages instead of dropping them
- **Per-user threads** — associate threads with authenticated users (requires auth)
- **Conversation TTL** — auto-expire old threads (cron job or Postgres `pg_cron`)
- **Semantic memory** — extract and store key facts from conversations for long-term recall
- **Tool message persistence** — store intermediate tool calls/results (currently only human + AI messages are saved, which is sufficient for the LLM to maintain context)
