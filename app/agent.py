"""
LangGraph Agent with Production Error Handling

Agentic RAG: instead of always retrieving up front, the LLM is given a
`search_documents` tool and decides *when* to search. The graph is a
tool-calling loop — `agent → (tools → agent)* → END` — with model fallback
and graceful error handling folded into the agent node.
"""

import logging
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from typing_extensions import Annotated, TypedDict

from app.config import get_settings
from app.tools import create_search_tool

logger = logging.getLogger(__name__)

# Tool-aware system prompt. Unlike the old always-retrieve prompt, context is
# no longer injected — the LLM pulls it in on demand via search_documents.
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base of documents. "
    "When the user asks a question that might be answered by the knowledge base, "
    "use the search_documents tool to find relevant information before answering. "
    "If the tool returns no relevant results, say you don't have sufficient "
    "context to answer rather than guessing. "
    "For general conversation or questions unrelated to the knowledge base, "
    "respond directly without searching."
)

# Sentinel returned by the search tool when nothing is found (see app/tools.py).
# Kept in sync here so _extract_sources can skip it.
_NO_RESULTS = "No relevant documents found."


class AgentState(TypedDict):
    """
    State for the production agent.

    Uses Annotated with add_messages so tool calls, tool results, and model
    responses accumulate across loop iterations. Context/sources are no longer
    pre-fetched — sources are derived post-hoc from tool messages.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


class ProductionAgent:
    """
    Production LangGraph agent with:
    - Tool-calling loop (LLM decides when to search)
    - Model fallback on primary failure
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self, retriever=None):
        settings = get_settings()

        self.retriever = retriever
        self.rag_enabled = retriever is not None
        self.max_retries = settings.max_retries
        self.max_tool_calls = settings.max_tool_calls

        # Only expose tools when RAG is configured. With no retriever the graph
        # degrades to a straight agent → END with no tool node (non-RAG mode).
        self.tools = []
        if self.rag_enabled:
            self.tools.append(
                create_search_tool(
                    retriever,
                    settings.rag_top_k,
                    settings.rag_similarity_threshold,
                )
            )

        primary = ChatOpenAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.openai_api_key,
        )
        fallback = ChatOpenAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.openai_api_key,
        )

        # Bind tools so the models can emit tool calls. When there are no tools,
        # use the raw models so they always answer directly.
        if self.tools:
            self.primary_llm = primary.bind_tools(self.tools)
            self.fallback_llm = fallback.bind_tools(self.tools)
        else:
            self.primary_llm = primary
            self.fallback_llm = fallback

        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the tool-calling state machine."""

        def _build_messages(state: AgentState) -> list[BaseMessage]:
            """Prepend the system prompt to the running message history."""
            messages = list(state["messages"])
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
            return messages

        def _invoke_llm(messages: list[BaseMessage]) -> tuple[BaseMessage, str]:
            """Call primary; on failure fall back to the secondary model.

            Raises if both fail — the caller records that as an agent error.
            """
            try:
                return self.primary_llm.invoke(messages), "primary"
            except Exception:
                logger.exception("Primary LLM failed, trying fallback")
                return self.fallback_llm.invoke(messages), "fallback"

        def agent_node(state: AgentState) -> dict:
            """Run the LLM. It either answers directly or requests a tool call."""
            try:
                messages = _build_messages(state)
                response, model_used = _invoke_llm(messages)
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": model_used,
                }
            except Exception as e:
                logger.exception("Both primary and fallback LLMs failed")
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """Return a graceful error message."""
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I'm sorry, I'm having trouble processing your request "
                            "right now. Please try again in a moment."
                        )
                    )
                ],
                "model_used": "error_handler",
            }

        def route_after_agent(state: AgentState) -> str:
            """Decide the next step after the agent runs.

            error → error handler; a pending tool call → tools (unless the
            per-request tool-call budget is exhausted); otherwise finish.
            """
            if state.get("error") is not None:
                return "error"

            # Cap the number of tool rounds to prevent infinite search loops.
            tool_call_count = sum(
                1 for m in state["messages"] if isinstance(m, ToolMessage)
            )
            if tool_call_count >= self.max_tool_calls:
                return "end"

            last = state["messages"][-1]
            if self.tools and getattr(last, "tool_calls", None):
                return "tools"
            return "end"

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("error", handle_error)
        graph.add_edge(START, "agent")

        if self.tools:
            graph.add_node("tools", ToolNode(self.tools))
            graph.add_conditional_edges(
                "agent",
                route_after_agent,
                {"tools": "tools", "end": END, "error": "error"},
            )
            graph.add_edge("tools", "agent")
        else:
            graph.add_conditional_edges(
                "agent",
                route_after_agent,
                {"end": END, "error": "error"},
            )

        graph.add_edge("error", END)
        return graph.compile()

    def _extract_sources(self, messages: list[BaseMessage]) -> list[dict]:
        """Recover source references from search_documents tool results.

        The tool returns `[Source: <name>]\\n<content>` blocks joined by
        `\\n---\\n` (see app/tools.py). This parses that format back into the
        structured sources the API response expects. Kept in sync with the tool.
        """
        sources: list[dict] = []
        for msg in messages:
            if not isinstance(msg, ToolMessage) or msg.name != "search_documents":
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if not content or content == _NO_RESULTS:
                continue
            for block in content.split("\n---\n"):
                header, _, body = block.partition("\n")
                if header.startswith("[Source: ") and header.endswith("]"):
                    source = header[len("[Source: ") : -1]
                else:
                    source = "unknown"
                sources.append(
                    {
                        "source": source,
                        "similarity": None,
                        "chunk_preview": body[:200],
                    }
                )
        return sources

    @traceable(name="production_rag_agent_invoke")
    def invoke(self, message: str) -> dict:
        """
        Invoke agent with user message.
        Returns: {"response": str, "model_used": str, "error": str | None, "sources": list}
        """
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "error": None,
                "retry_count": 0,
                "model_used": "",
            }
        )

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
            "sources": self._extract_sources(result["messages"]),
        }
