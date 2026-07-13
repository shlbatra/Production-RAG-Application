# Plan: Remove thread_id from /chat Request

## Context

`thread_id` is declared on `ChatRequest` with `default_factory=lambda: uuid.uuid4().hex`. Callers never need to supply it — it's auto-generated server-side. Exposing it in the request body implies the API supports multi-turn conversations tied to a thread, but it doesn't — the agent is stateless and `thread_id` is only used for log correlation and echoed back in the response.

## What to change

### 1. `app/models.py` — move generation out of the request

- Remove `thread_id` from `ChatRequest` (only `message` remains)
- Keep `thread_id` on `ChatResponse` (still useful as a request correlation ID)

### 2. `app/main.py` — generate thread_id at the start of the handler

Generate `thread_id` once at the top of the `chat()` function and use it throughout:

```python
async def chat(request: Request, body: ChatRequest):
    thread_id = uuid.uuid4().hex
    ...
```

Then replace all `body.thread_id` references with `thread_id` (9 occurrences across logging and response construction).

### 3. `docs/test_questions.md` — no change needed

The curl examples already omit `thread_id`.

## Files touched

| File | Change |
|------|--------|
| `app/models.py` | Remove `thread_id` field from `ChatRequest`, remove `uuid` import if unused |
| `app/main.py` | Add `import uuid`, generate `thread_id` at top of handler, replace 9 `body.thread_id` → `thread_id` |

## What stays the same

- `ChatResponse` still returns `thread_id` — it's a useful correlation ID for clients to reference in support requests or log searches
- All log entries still include `thread_id` for tracing a request through the pipeline
