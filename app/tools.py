"""
LangChain tools for the RAG agent.

Tools let the LLM decide *when* to search rather than always retrieving
context upfront. Each factory closes over its dependencies (retriever,
settings) and returns a `@tool`-decorated callable ready to bind to the LLM.
"""

import logging

from langchain_core.tools import tool

from app.retrieval import RetrievalStrategy

logger = logging.getLogger(__name__)


# create_search_tool is an adapter. On one side is your retriever (a normal
# Python object with config). On the other side is the LLM (which can only pass
# a query string and can only read text back). The factory + @tool + string
# formatting together translate between those two worlds.
def create_search_tool(retriever: RetrievalStrategy, top_k: int, threshold: float):
    """Build a `search_documents` tool that wraps the given retriever.

    The tool exposes the existing `RetrievalStrategy` to the LLM as a callable
    it can invoke with a reformulated query. Results are returned as formatted
    text the model can reason over — retrieval.py is not modified.
    """

    @tool
    def search_documents(query: str) -> str:
        """Search the knowledge base for documents relevant to the query.

        Use this when the user asks about topics that might be covered in
        uploaded documents. Returns the most relevant document chunks, each
        prefixed with its source, or a message when nothing relevant is found.
        """
        results = retriever.search(query=query, top_k=top_k, threshold=threshold)
        logger.info("search_documents(query=%r) -> %d result(s)", query, len(results))
        if not results:
            return "No relevant documents found."

        formatted = []
        for r in results:
            source = r["metadata"].get("source", "unknown")
            formatted.append(f"[Source: {source}]\n{r['content']}")
        return "\n---\n".join(formatted)

    return search_documents
