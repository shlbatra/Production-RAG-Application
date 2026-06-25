"""
LangGraph Agent with Production Error Handling
Retry logic, model fallback, RAG retrieval, and structured state management.
"""

import logging
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable
from typing_extensions import Annotated, TypedDict

from app.config import get_settings

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT_BASE = (
    "You are a helpful assistant that answers questions based on retrieved documents. "
    "If no documents were retrieved or the documents don't contain relevant information, "
    "say you don't have sufficient context to answer the question. "
    "Do not answer from general knowledge."
)

RAG_SYSTEM_PROMPT_DOCS_HEADER = "\n\nRetrieved Documents:\n"


class AgentState(TypedDict):
    """
    State for production agent
    Uses Annotated with add_messages reducer for message accumulation
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
    context: list[dict]
    sources: list[dict]


class ProductionAgent:
    """
    Production LangGraph agent with:
    - Retry on failure (model fallback)
    - Graceful error handling
    - RAG retrieval
    - LangSmith tracing
    """

    def __init__(self, retriever=None):
        settings = get_settings()

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

        self.retriever = retriever
        self.rag_enabled = retriever is not None
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build langgraph state machine"""
        settings = get_settings()

        def retrieve_context(state: AgentState) -> dict:
            if not self.rag_enabled:
                return {"context": [], "sources": []}
            try:
                user_message = state["messages"][-1].content
                results = self.retriever.search(
                    query=user_message,
                    top_k=settings.rag_top_k,
                    threshold=settings.rag_similarity_threshold,
                )
                sources = [
                    {
                        "source": r["metadata"].get("source", "unknown"),
                        "similarity": round(r["similarity"], 3),
                        "chunk_preview": r["content"][:200],
                    }
                    for r in results
                ]
                return {"context": results, "sources": sources}
            except Exception:
                logger.exception("RAG retrieval failed, degrading gracefully")
                return {"context": [], "sources": []}

        def _build_messages(state: AgentState) -> list[BaseMessage]:
            """Build message list with system prompt always included."""
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

        def process_message(state: AgentState) -> dict:
            """Try to process message with primary model"""
            try:
                messages = _build_messages(state)
                response = self.primary_llm.invoke(messages)
                return {"messages": [response], "error": None, "model_used": "primary"}
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """Fallback to secondary model."""
            try:
                messages = _build_messages(state)
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
            """Return graceful error message"""
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

        def route_after_process(state: AgentState) -> str:
            """Decide what to do after primary model attempt"""
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """Decide what to do after fallback attempt."""
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        graph = StateGraph(AgentState)

        graph.add_node("retrieve", retrieve_context)
        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "process")
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        graph.add_conditional_edges(
            "fallback", route_after_fallback, {"done": END, "error": "error"}
        )
        graph.add_edge("error", END)
        return graph.compile()

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
                "context": [],
                "sources": [],
            }
        )

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
            "sources": result.get("sources", []),
        }
