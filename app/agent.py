"""
LangGraph Agent with Tool Calling
The LLM decides when to search the knowledge base or the web,
with retry logic, model fallback, and structured state management.
"""

import logging
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from typing_extensions import Annotated, TypedDict

from app.config import get_settings
from app.tools import create_search_tool, create_web_search_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base and the web.\n\n"
    "STRICT RULES:\n"
    "1. You MUST call search_documents BEFORE web_search. Never skip this step.\n"
    "2. Only call web_search if search_documents returned NO_RESULTS or LOW_RELEVANCE.\n"
    "3. If both tools return nothing useful, tell the user you could not find the information.\n"
    "4. For general conversation (greetings, math, opinions), respond directly — no tools.\n"
    "5. When using web results, always cite the source URLs.\n"
)


class AgentState(TypedDict):
    """State for production agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


class ProductionAgent:
    """
    Production LangGraph agent with:
    - Tool calling (search_documents + web_search)
    - Retry on failure (model fallback)
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self, retriever=None):
        settings = get_settings()

        self.retriever = retriever
        self.rag_enabled = retriever is not None
        self.max_retries = settings.max_retries
        self.max_tool_calls = settings.max_tool_calls

        tools = []
        if self.rag_enabled:
            tools.append(
                create_search_tool(
                    retriever,
                    settings.rag_top_k,
                    settings.rag_similarity_threshold,
                )
            )
        if settings.web_search_enabled:
            tools.append(
                create_web_search_tool(max_results=settings.web_search_max_results)
            )

        self.tools = tools

        self.primary_llm = ChatOpenAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.openai_api_key,
        )

        self.fallback_llm = ChatOpenAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.openai_api_key,
        )

        if tools:
            self.primary_llm = self.primary_llm.bind_tools(tools)
            self.fallback_llm = self.fallback_llm.bind_tools(tools)

        self.graph = self._build_graph()

    def _build_graph(self):
        """Build langgraph state machine with tool-calling loop."""

        def agent_node(state: AgentState) -> dict:
            """Invoke the primary LLM (with tool bindings)."""
            try:
                messages = list(state["messages"])
                messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
                response = self.primary_llm.invoke(messages)
                return {"messages": [response], "error": None, "model_used": "primary"}
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def fallback_node(state: AgentState) -> dict:
            """Fallback to secondary model."""
            try:
                messages = list(state["messages"])
                messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
                response = self.fallback_llm.invoke(messages)
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """Return graceful error message."""
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

        def should_continue(state: AgentState) -> str:
            """Route after agent: call tools, fallback on error, or finish."""
            if state.get("error"):
                if state["retry_count"] < self.max_retries:
                    return "fallback"
                return "error"

            last = state["messages"][-1]

            if not hasattr(last, "tool_calls") or not last.tool_calls:
                return "end"

            tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
            if len(tool_messages) >= self.max_tool_calls:
                return "end"

            has_searched_docs = any(m.name == "search_documents" for m in tool_messages)
            requested_tools = [tc["name"] for tc in last.tool_calls]

            if "web_search" in requested_tools and not has_searched_docs:
                last.tool_calls = [tc for tc in last.tool_calls if tc["name"] != "web_search"]
                if not last.tool_calls:
                    return "end"

            return "tools"

        def route_after_fallback(state: AgentState) -> str:
            """Route after fallback attempt."""
            if state.get("error") is None:
                last = state["messages"][-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    return "tools"
                return "end"
            return "error"

        graph = StateGraph(AgentState)

        graph.add_node("agent", agent_node)
        graph.add_node("fallback", fallback_node)
        graph.add_node("error", handle_error)

        if self.tools:
            tool_node = ToolNode(self.tools)
            graph.add_node("tools", tool_node)
            graph.add_edge("tools", "agent")

            graph.add_edge(START, "agent")
            graph.add_conditional_edges(
                "agent",
                should_continue,
                {"tools": "tools", "end": END, "fallback": "fallback", "error": "error"},
            )
            graph.add_conditional_edges(
                "fallback",
                route_after_fallback,
                {"tools": "tools", "end": END, "error": "error"},
            )
        else:
            graph.add_edge(START, "agent")
            graph.add_conditional_edges(
                "agent",
                should_continue,
                {"end": END, "fallback": "fallback", "error": "error"},
            )
            graph.add_conditional_edges(
                "fallback",
                route_after_fallback,
                {"end": END, "error": "error"},
            )

        graph.add_edge("error", END)
        return graph.compile()

    def _extract_sources(self, messages: list[BaseMessage]) -> list[dict]:
        """Parse tool messages to extract source references for the API response."""
        sources = []
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            if msg.name == "search_documents":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content.startswith("NO_RESULTS:"):
                    continue
                text = content
                if text.startswith("LOW_RELEVANCE:"):
                    text = text.split("\n\n", 1)[-1]
                for block in text.split("\n---\n"):
                    source_name = "unknown"
                    if block.startswith("[Source: "):
                        end = block.index("]")
                        source_name = block[9:end]
                    preview = block.split("\n", 1)[-1][:200] if "\n" in block else block[:200]
                    sources.append({
                        "source": source_name,
                        "similarity": 0.0,
                        "chunk_preview": preview,
                        "type": "document",
                    })
            elif msg.name == "web_search":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                try:
                    import json
                    results = json.loads(content) if content.startswith("[") else []
                except (json.JSONDecodeError, ValueError):
                    results = []
                for r in results:
                    if isinstance(r, dict):
                        sources.append({
                            "source": r.get("url", "unknown"),
                            "similarity": 0.0,
                            "chunk_preview": r.get("content", r.get("snippet", ""))[:200],
                            "type": "web",
                        })
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

        sources = self._extract_sources(result["messages"])

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
            "sources": sources,
        }
