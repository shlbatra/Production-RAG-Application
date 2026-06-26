from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from app.agent import (
    RAG_SYSTEM_PROMPT_BASE,
    RAG_SYSTEM_PROMPT_DOCS_HEADER,
    ProductionAgent,
)


# ProductionAgent.__init__
class TestInit:
    def test_rag_enabled_when_retriever_provided(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI"):
            agent = ProductionAgent(retriever=mock_retriever)
        assert agent.rag_enabled is True
        assert agent.retriever is mock_retriever

    def test_rag_disabled_when_no_retriever(self, mock_settings):
        with patch("app.agent.ChatOpenAI"):
            agent = ProductionAgent()
        assert agent.rag_enabled is False
        assert agent.retriever is None

    def test_rag_disabled_with_explicit_none(self, mock_settings):
        with patch("app.agent.ChatOpenAI"):
            agent = ProductionAgent(retriever=None)
        assert agent.rag_enabled is False


# retrieve node
class TestRetrieveContext:
    def test_retrieves_and_formats_sources(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="answer")
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("What is Python?")

        mock_retriever.search.assert_called_once_with(
            query="What is Python?",
            top_k=5,
            threshold=0.7,
        )
        assert result["sources"] == [
            {
                "source": "intro.pdf",
                "similarity": 0.92,
                "chunk_preview": "Python is a programming language.",
            },
            {
                "source": "history.pdf",
                "similarity": 0.85,
                "chunk_preview": "Python was created by Guido van Rossum.",
            },
        ]

    def test_skips_retrieval_when_rag_disabled(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="answer")
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        result = agent.invoke("hello")
        assert result["sources"] == []

    def test_degrades_gracefully_on_search_error(self, mock_settings):
        retriever = MagicMock()
        retriever.search.side_effect = RuntimeError("connection lost")

        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="fallback answer")
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent(retriever=retriever)

        result = agent.invoke("hello")
        assert result["response"] == "fallback answer"
        assert result["sources"] == []


# context injection into LLM prompt
class TestContextInjection:
    def test_system_message_prepended_with_context(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="rag answer")
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent(retriever=mock_retriever)

        agent.invoke("What is Python?")

        call_args = mock_llm.invoke.call_args[0][0]
        assert call_args[0].content.startswith(RAG_SYSTEM_PROMPT_BASE)
        assert RAG_SYSTEM_PROMPT_DOCS_HEADER in call_args[0].content
        assert "[Source: intro.pdf]" in call_args[0].content
        assert "Python is a programming language." in call_args[0].content
        assert call_args[1].content == "What is Python?"

    def test_system_message_without_context_has_base_prompt(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="plain answer")
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        agent.invoke("hello")

        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].content == RAG_SYSTEM_PROMPT_BASE
        assert RAG_SYSTEM_PROMPT_DOCS_HEADER not in call_args[0].content
        assert call_args[1].content == "hello"

    def test_system_message_without_context_instructs_refusal(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="plain answer")
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        agent.invoke("hello")

        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        assert "don't have sufficient context" in system_content
        assert "Do not answer from general knowledge" in system_content


# invoke return value
class TestInvokeReturn:
    def test_includes_sources_in_result(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="answer")
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("test")
        assert "sources" in result
        assert len(result["sources"]) == 2
        assert result["response"] == "answer"
        assert result["model_used"] == "primary"
        assert result["error"] is None

    def test_empty_sources_when_no_rag(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="answer")
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        result = agent.invoke("test")
        assert result["sources"] == []


# fallback with RAG context
class TestFallbackWithRAG:
    def test_fallback_receives_context(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            primary = MagicMock()
            primary.invoke.side_effect = RuntimeError("primary down")
            fallback = MagicMock()
            fallback.invoke.return_value = AIMessage(content="fallback answer")
            mock_llm_cls.side_effect = [primary, fallback]
            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("What is Python?")

        assert result["model_used"] == "fallback"
        assert result["response"] == "fallback answer"
        call_args = fallback.invoke.call_args[0][0]
        assert call_args[0].content.startswith(RAG_SYSTEM_PROMPT_BASE)
        assert RAG_SYSTEM_PROMPT_DOCS_HEADER in call_args[0].content
        assert "[Source: intro.pdf]" in call_args[0].content
