# ReAct Agent Plan

## Goal

Evolve the RAG agent from a fixed linear pipeline into an explicit **ReAct loop** — *Reason → Act → Observe → Adjust/Decide* — so the LLM reasons about what it knows, takes an action (search docs, search web, or answer), observes the result, and then **decides** whether the observation is good enough or whether it must adjust (reformulate, search again, escalate, or stop). Today the graph is linear (`retrieve → process → fallback → error`) with no reasoning, no self-correction, and exactly one retrieval pass.

ReAct makes the agent's control flow driven by its own reasoning about intermediate results, not by a hard-coded edge order.

## Relationship to the Tool Calling Plan

This plan **builds on** [`tool_calling_plan.md`](tool_calling_plan.md), which introduces the `search_documents` / `web_search` tools and the `agent → tools → agent` loop. Tool calling gives the agent the *ability to act*; ReAct adds the missing three parts:

| Tool Calling Plan gives us | ReAct adds on top |
|---|---|
| **Act** — LLM can call tools | **Reason** — explicit "thought" before each action |
| Raw tool output returned to the LLM | **Observe** — a structured reflection on whether the observation answered the question |
| Loop until the LLM stops calling tools | **Adjust/Decide** — a dedicated decision step: refine the query, escalate the tool, or finalize |

If the tool-calling plan is not yet implemented, implement its **Version 1** (`search_documents` + loop) first — this plan assumes that scaffolding (`app/tools.py`, `ToolNode`, `bind_tools`, `should_continue`) exists.

## The ReAct Cycle Mapped to LangGraph

```
                ┌─────────────────────────────────────────────┐
                │                                             │
START → reason → act ──(tool_calls)──→ observe → decide ──────┤ (continue: adjusted query)
          │       │                                  │
          │       └──(no tool, direct answer)──→ END │
          │                                          │
          └──(error)──→ fallback → END               └──(finalize)──→ END
                                                     └──(give_up)──→ END
```

- **reason** — LLM produces a short thought: what does the user want, what do I already know, what should I do next. This is captured as a `Thought` in state (and in the message trace for LangSmith).
- **act** — LLM emits a tool call (or answers directly). Reuses the tool-calling `agent_node` + `ToolNode`.
- **observe** — after the tool runs, a lightweight node summarizes the observation and grades it (`sufficient` / `insufficient` / `empty`). This is the *reflection* the current agent completely lacks.
- **decide/adjust** — routing based on the observation grade + a step budget:
  - observation sufficient → **finalize** (LLM writes the answer, END)
  - observation insufficient but budget remains → **adjust**: reformulate the query or escalate `search_documents → web_search`, loop back to `reason`
  - budget exhausted or nothing found → **give_up** gracefully

## Design

### 1. Extend `AgentState` — `app/agent.py`

Add ReAct bookkeeping fields on top of the tool-calling state:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
    # ReAct additions
    step: int                     # current reason→act→observe cycle count
    scratchpad: list[dict]        # [{thought, action, observation, grade}, ...]
    last_grade: str               # "sufficient" | "insufficient" | "empty"
    escalation: str               # "docs" | "web" | "exhausted"
```

`scratchpad` is the ReAct trace — it makes the agent's reasoning inspectable, testable, and loggable, and it is fed back into the prompt so the LLM does not repeat a query that already failed.

### 2. `reason` node — the "Reason" step

A node that asks the LLM for its next thought **before** it acts. Use a structured prompt that forces a plan:

```python
def reason(state: AgentState) -> dict:
    scratchpad_text = _render_scratchpad(state["scratchpad"])
    messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        *state["messages"],
        SystemMessage(content=f"Reasoning so far:\n{scratchpad_text}\n\n"
                              "Think step by step: what do you still need, and what is "
                              "the single best next action? Then act."),
    ]
    response = self.primary_llm.invoke(messages)  # tools bound
    return {"messages": [response], "model_used": "primary", "step": state["step"] + 1}
```

The same LLM call both reasons and (via bound tools) acts — modern tool-calling models interleave a natural-language thought with the tool call in one turn, so `reason` and `act` collapse into one LLM invocation. We keep them conceptually separate for the graph and the scratchpad.

### 3. `act` — reuse the tool node

No new code beyond the tool-calling plan: `ToolNode(tools)` executes whatever tool the `reason` step requested and appends a `ToolMessage`. If the `reason` step produced **no** tool call (the model chose to answer directly), route straight to `finalize`.

### 4. `observe` node — the "Observe" step

After the tool runs, grade the observation. Keep this cheap — a rule-based grade first, LLM grade only when ambiguous:

```python
def observe(state: AgentState) -> dict:
    last_tool_msg = _last_tool_message(state["messages"])
    content = last_tool_msg.content

    # Rule-based fast path using the tool output signals from the tool-calling plan
    if content.startswith("NO_RESULTS"):
        grade = "empty"
    elif content.startswith("LOW_RELEVANCE"):
        grade = "insufficient"
    else:
        grade = self._grade_observation(state, content)  # optional LLM grader

    entry = {
        "thought": _last_ai_thought(state["messages"]),
        "action": last_tool_msg.name,
        "observation": content[:500],
        "grade": grade,
    }
    return {"scratchpad": state["scratchpad"] + [entry], "last_grade": grade}
```

`_grade_observation` (optional, behind a config flag) asks a cheap model: *"Does this observation contain enough information to answer the user's question? Answer sufficient/insufficient."* Rule-based signals (`NO_RESULTS`, `LOW_RELEVANCE`) from the tool-calling plan cover most cases without an extra LLM call.

### 5. `decide` router — the "Adjust/Decide" step

The heart of ReAct: choose the next move from the observation grade, the step budget, and the escalation ladder.

```python
def decide(state: AgentState) -> str:
    if state["step"] >= settings.react_max_steps:
        return "give_up"
    grade = state["last_grade"]
    if grade == "sufficient":
        return "finalize"
    # insufficient or empty → adjust and loop
    if state["escalation"] == "docs":
        return "adjust"        # reformulate / escalate to web, loop to reason
    if state["escalation"] == "web":
        return "adjust"
    return "give_up"           # exhausted the ladder
```

Escalation ladder, enforced programmatically (not left to the prompt), reusing the tool-ordering guard from the tool-calling plan §5a:

```
docs (reformulated query) → web → give_up
```

The `adjust` transition updates `state["escalation"]` and injects a hint into the scratchpad so the next `reason` step knows to **reformulate** ("the query X returned nothing, try a broader/different phrasing") or **escalate** ("docs exhausted, use web_search"). This is the self-correction the current agent lacks entirely.

### 6. `finalize` and `give_up` nodes

```python
def finalize(state: AgentState) -> dict:
    # LLM writes the final answer grounded ONLY in observed tool results
    messages = [SystemMessage(content=REACT_ANSWER_PROMPT), *state["messages"]]
    response = self.primary_llm_no_tools.invoke(messages)  # tools NOT bound → forces an answer
    return {"messages": [response], "model_used": state["model_used"]}

def give_up(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=(
        "I looked through the knowledge base"
        + (" and the web" if settings.web_search_enabled else "")
        + " but couldn't find enough information to answer that confidently."))]}
```

Binding tools off in `finalize` prevents the model from looping again when we've decided to answer.

### 7. Graph wiring — `app/agent.py`

```python
graph.add_node("reason", reason)
graph.add_node("act", tool_node)          # ToolNode from tool-calling plan
graph.add_node("observe", observe)
graph.add_node("finalize", finalize)
graph.add_node("give_up", give_up)
graph.add_node("fallback", try_fallback)  # kept from current agent

graph.add_edge(START, "reason")
graph.add_conditional_edges("reason", route_after_reason,
    {"act": "act", "finalize": "finalize", "error": "fallback"})
graph.add_edge("act", "observe")
graph.add_conditional_edges("observe", decide,
    {"adjust": "reason", "finalize": "finalize", "give_up": "give_up"})
graph.add_edge("finalize", END)
graph.add_edge("give_up", END)
graph.add_edge("fallback", END)
```

`route_after_reason` = "act" if the last message has `tool_calls`, "finalize" if it produced a direct answer, "error" on exception (→ fallback, preserving today's resilience).

### 8. System prompts — `app/agent.py`

```python
REACT_SYSTEM_PROMPT = (
    "You are a research assistant that answers strictly from retrieved evidence.\n"
    "Work in a Reason → Act → Observe loop:\n"
    "1. REASON about what the user needs and what you still lack.\n"
    "2. ACT by calling exactly one tool (search_documents first, web_search only "
    "   after documents come back empty or irrelevant).\n"
    "3. You will then OBSERVE the result and decide your next step.\n"
    "Do not answer from general knowledge. If evidence is insufficient after "
    "searching, say so. Never repeat a query that already returned nothing — "
    "reformulate it instead."
)

REACT_ANSWER_PROMPT = (
    "Write the final answer using ONLY the observations gathered above. "
    "Cite sources. If the observations do not fully answer the question, say "
    "what is missing rather than guessing."
)
```

### 9. Config — `app/config.py`

```python
# ReAct loop
react_enabled: bool = True          # false → fall back to tool-calling / linear agent
react_max_steps: int = 4            # max reason→act→observe cycles (guards infinite loops)
react_llm_grader: bool = False      # use an LLM to grade observations (else rule-based only)
react_grader_model: str = "gpt-4.1-nano"  # cheap model for observation grading
```

`react_max_steps` is the hard budget backstop — combined with the tool-ordering guard it guarantees termination.

### 10. Source & trace extraction — `app/agent.py`

- Sources: reuse `_extract_sources` from the tool-calling plan (parse `ToolMessage`s).
- Expose the ReAct trace: return `scratchpad` in the `invoke()` result so it can be surfaced for debugging / included in LangSmith. Optionally add it to the API response behind a `?debug=true` flag.

```python
return {
    "response": result["messages"][-1].content,
    "model_used": result.get("model_used", "unknown"),
    "error": result.get("error"),
    "sources": self._extract_sources(result["messages"]),
    "reasoning_trace": result.get("scratchpad", []),  # ReAct steps
}
```

### 11. Guardrails (termination & safety)

| Risk | Guard |
|---|---|
| Infinite reason/act loop | `react_max_steps` budget checked in `decide` |
| Re-issuing a failed query | Scratchpad fed back into `reason` prompt + prompt rule |
| Skipping docs → web ordering | Programmatic tool-ordering guard (tool-calling plan §5a) |
| Model answering from general knowledge | `finalize` grounded-only prompt + output validation (existing) |
| Extra latency/cost from grading | Rule-based grade first; LLM grader opt-in via `react_llm_grader` |

### 12. Non-RAG / disabled mode

- `react_enabled=False` → use the existing tool-calling agent (or the current linear agent if tools aren't built yet). Keep both graphs behind a factory so behavior is switchable via config, matching the `RetrievalStrategy` protocol pattern used elsewhere.
- `retriever=None` → no `search_documents` tool; the loop degenerates to `reason → finalize` (direct answer), same graceful degradation as today.

## Observability

- Each node is LangSmith-traced (`@traceable`), so the full Reason→Act→Observe→Decide chain shows up as nested spans.
- `monitoring.py`: add counters for `react_steps_per_request`, `react_escalations`, and `react_give_ups` to watch how often the loop adjusts vs. answers on the first pass.

## Evaluation

- Reuse `evals/` harness. Add cases that require **multi-step reasoning**: questions whose answer needs a reformulated query, and questions answerable only after doc→web escalation.
- Compare linear vs. tool-calling vs. ReAct on the existing eval set: track answer quality, retrieval recall, avg steps, and latency. Store results under `evals/results/` as usual.

## File Changes Summary

| File | Change |
|---|---|
| `app/agent.py` | Add `reason`/`observe`/`decide`/`finalize`/`give_up` nodes, rewire graph, extend state, scratchpad rendering, trace in `invoke()` |
| `app/tools.py` | Reused from tool-calling plan (no change if already built) |
| `app/config.py` | Add `react_enabled`, `react_max_steps`, `react_llm_grader`, `react_grader_model` |
| `app/models.py` | Optional: add `reasoning_trace` field to `ChatResponse` (debug) |
| `app/monitoring.py` | Add ReAct step/escalation/give-up metrics |
| `tests/test_agent.py` | ReAct-specific tests (below) |
| `evals/` | Add multi-step reasoning eval cases |

## Tests — `tests/test_agent.py`

- Single-pass: a question answered by the first doc search → 1 step, `finalize`.
- Reformulation: first query graded `empty`, agent adjusts and succeeds on the second → scratchpad has 2 entries.
- Escalation: docs empty → agent escalates to `web_search` (when enabled) before answering.
- Budget guard: mock the grader to always return `insufficient`; assert the loop stops at `react_max_steps` and hits `give_up`.
- Ordering guard: assert `web_search` is never called before `search_documents`.
- Direct answer: a greeting ("hi") → no tool call, straight to `finalize`.
- Fallback: primary LLM raises during `reason` → routes to `fallback`, still returns a response.
- Trace: `invoke()` returns a non-empty `reasoning_trace` with `{thought, action, observation, grade}` entries.
- `react_enabled=False` → falls back to the prior agent behavior.

## Implementation Order

1. Ensure the tool-calling plan Version 1 scaffolding exists (`app/tools.py`, `ToolNode`, `bind_tools`).
2. Add ReAct config settings.
3. Extend `AgentState` with `step` / `scratchpad` / `last_grade` / `escalation`.
4. Implement `reason`, `observe`, `decide`, `finalize`, `give_up` nodes + scratchpad rendering.
5. Rewire the graph behind a `react_enabled` factory.
6. Add source + reasoning-trace extraction to `invoke()`.
7. Add monitoring counters.
8. Tests, then manual `/chat` verification.
9. Run evals (linear vs. tool-calling vs. ReAct) and record results.
```