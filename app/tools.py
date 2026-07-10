"""
Tool definitions for the agentic RAG pipeline.

Two tools: search_documents (knowledge base) and web_search (internet fallback).
The LLM decides when to call each tool based on the user's question.
"""

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


def create_search_tool(retriever, top_k: int, threshold: float):
    """Wrap the existing retriever as a LangChain tool the LLM can call."""

    @tool
    def search_documents(query: str) -> str:
        """Search the knowledge base for documents relevant to the query.
        Use this when the user asks about topics that might be covered in uploaded documents."""
        results = retriever.search(query=query, top_k=top_k, threshold=threshold)

        if not results:
            return "NO_RESULTS: No relevant documents found in the knowledge base."

        if all(r["similarity"] < 0.75 for r in results):
            formatted = _format_results(results)
            return (
                "LOW_RELEVANCE: Documents were found but none are highly relevant.\n\n"
                + formatted
            )

        return _format_results(results)

    return search_documents


def _format_results(results: list[dict]) -> str:
    formatted = []
    for r in results:
        source = r["metadata"].get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{r['content']}")
    return "\n---\n".join(formatted)


def create_web_search_tool(max_results: int = 3):
    """Create a Tavily web search tool for internet fallback."""
    return TavilySearchResults(
        max_results=max_results,
        name="web_search",
        description=(
            "Search the web for current information. "
            "Use this ONLY when search_documents returned NO_RESULTS or LOW_RELEVANCE "
            "and the user's question requires factual information you don't have."
        ),
    )
