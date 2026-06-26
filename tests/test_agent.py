from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from app.agent import SYSTEM_PROMPT, ProductionAgent


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

    def test_tools_bound_when_rag_enabled(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent(retriever=mock_retriever)

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "search_documents"
        mock_llm.bind_tools.assert_called()

    def test_no_tools_when_no_retriever(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        assert len(agent.tools) == 0
        mock_llm.bind_tools.assert_not_called()


class TestToolCalling:
    def test_knowledge_question_triggers_search_tool(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()

            tool_call_response = AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "search_documents",
                    "args": {"query": "What is Python?"},
                }],
            )
            final_response = AIMessage(content="Python is a programming language.")

            mock_llm.invoke.side_effect = [tool_call_response, final_response]
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("What is Python?")
        assert result["response"] == "Python is a programming language."
        assert result["model_used"] == "primary"
        mock_retriever.search.assert_called_once()

    def test_general_question_no_tool_call(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="Hello! How can I help?")
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("hi")
        assert result["response"] == "Hello! How can I help?"
        mock_retriever.search.assert_not_called()
        assert result["sources"] == []


class TestMaxToolCallsGuard:
    def test_caps_tool_calls(self, mock_settings, mock_retriever):
        mock_settings.max_tool_calls = 2

        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()

            tool_call = AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "search_documents",
                    "args": {"query": "test"},
                }],
            )
            final_response = AIMessage(content="Done.")

            mock_llm.invoke.side_effect = [tool_call, tool_call, final_response]
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("search everything")
        assert mock_retriever.search.call_count <= 2


class TestToolOrderingGuard:
    def test_blocks_web_search_before_doc_search(self, mock_settings, mock_retriever):
        mock_settings.web_search_enabled = True
        mock_settings.web_search_max_results = 3

        with (
            patch("app.agent.ChatOpenAI") as mock_llm_cls,
            patch("app.agent.create_web_search_tool") as mock_web_tool_fn,
            patch("app.agent.ToolNode") as mock_tool_node_cls,
        ):
            mock_web_tool = MagicMock()
            mock_web_tool.name = "web_search"
            mock_web_tool_fn.return_value = mock_web_tool

            mock_tool_node_cls.return_value = MagicMock()

            mock_llm = MagicMock()

            web_only_call = AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "web_search",
                    "args": {"query": "test"},
                }],
            )
            final_response = AIMessage(content="Could not find info.")

            mock_llm.invoke.side_effect = [web_only_call, final_response]
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("test query")
        assert result["response"] == ""
        mock_retriever.search.assert_not_called()
        assert mock_llm.invoke.call_count == 1


class TestFallback:
    def test_fallback_on_primary_error(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            primary = MagicMock()
            primary.invoke.side_effect = RuntimeError("primary down")
            primary.bind_tools.return_value = primary

            fallback = MagicMock()
            fallback.invoke.return_value = AIMessage(content="fallback answer")
            fallback.bind_tools.return_value = fallback

            mock_llm_cls.side_effect = [primary, fallback]
            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("What is Python?")
        assert result["model_used"] == "fallback"
        assert result["response"] == "fallback answer"

    def test_error_handler_when_both_fail(self, mock_settings):
        mock_settings.max_retries = 1

        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            primary = MagicMock()
            primary.invoke.side_effect = RuntimeError("primary down")

            fallback = MagicMock()
            fallback.invoke.side_effect = RuntimeError("fallback down")

            mock_llm_cls.side_effect = [primary, fallback]
            agent = ProductionAgent()

        result = agent.invoke("test")
        assert result["model_used"] == "error_handler"
        assert "trouble processing" in result["response"]


class TestSourceExtraction:
    def test_extracts_document_sources(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        tool_msg = ToolMessage(
            content="[Source: intro.pdf]\nPython is great.\n---\n[Source: guide.pdf]\nLearn Python.",
            name="search_documents",
            tool_call_id="call_1",
        )
        sources = agent._extract_sources([tool_msg])
        assert len(sources) == 2
        assert sources[0]["source"] == "intro.pdf"
        assert sources[0]["type"] == "document"
        assert sources[1]["source"] == "guide.pdf"

    def test_extracts_web_sources(self, mock_settings):
        import json

        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        web_results = json.dumps([
            {"url": "https://example.com", "content": "Some web content"},
            {"url": "https://other.com", "content": "Other content"},
        ])
        tool_msg = ToolMessage(
            content=web_results,
            name="web_search",
            tool_call_id="call_2",
        )
        sources = agent._extract_sources([tool_msg])
        assert len(sources) == 2
        assert sources[0]["source"] == "https://example.com"
        assert sources[0]["type"] == "web"
        assert sources[1]["source"] == "https://other.com"

    def test_skips_no_results(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        tool_msg = ToolMessage(
            content="NO_RESULTS: No relevant documents found in the knowledge base.",
            name="search_documents",
            tool_call_id="call_1",
        )
        sources = agent._extract_sources([tool_msg])
        assert sources == []

    def test_handles_low_relevance(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        tool_msg = ToolMessage(
            content="LOW_RELEVANCE: Documents were found but none are highly relevant.\n\n[Source: file.pdf]\nSome content.",
            name="search_documents",
            tool_call_id="call_1",
        )
        sources = agent._extract_sources([tool_msg])
        assert len(sources) == 1
        assert sources[0]["source"] == "file.pdf"
        assert sources[0]["type"] == "document"


class TestSystemPrompt:
    def test_system_prompt_includes_tool_instructions(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="hi")
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            agent = ProductionAgent(retriever=mock_retriever)

        agent.invoke("hello")

        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        assert "search_documents" in system_content
        assert "web_search" in system_content
        assert "STRICT RULES" in system_content

    def test_system_prompt_always_prepended(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="hi")
            mock_llm_cls.return_value = mock_llm

            agent = ProductionAgent()

        agent.invoke("hello")

        call_args = mock_llm.invoke.call_args[0][0]
        assert call_args[0].content == SYSTEM_PROMPT
        assert call_args[1].content == "hello"


class TestWebSearchConfig:
    def test_web_search_disabled_by_default(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            agent = ProductionAgent(retriever=mock_retriever)

        tool_names = [t.name for t in agent.tools]
        assert "search_documents" in tool_names
        assert "web_search" not in tool_names

    def test_web_search_enabled_adds_tool(self, mock_settings, mock_retriever):
        mock_settings.web_search_enabled = True
        mock_settings.web_search_max_results = 3

        with (
            patch("app.agent.ChatOpenAI") as mock_llm_cls,
            patch("app.agent.create_web_search_tool") as mock_web_tool_fn,
            patch("app.agent.ToolNode") as mock_tool_node_cls,
        ):
            mock_web_tool = MagicMock()
            mock_web_tool.name = "web_search"
            mock_web_tool_fn.return_value = mock_web_tool

            mock_tool_node_cls.return_value = MagicMock()

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            agent = ProductionAgent(retriever=mock_retriever)

        tool_names = [t.name for t in agent.tools]
        assert "web_search" in tool_names


class TestInvokeReturn:
    def test_includes_sources_from_tool_calls(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()

            tool_call_response = AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "search_documents",
                    "args": {"query": "test"},
                }],
            )
            final_response = AIMessage(content="answer")

            mock_llm.invoke.side_effect = [tool_call_response, final_response]
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("test")
        assert "sources" in result
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


class TestNonRagMode:
    def test_straight_through_without_tools(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="plain answer")
            mock_llm_cls.return_value = mock_llm
            agent = ProductionAgent()

        result = agent.invoke("hello")
        assert result["response"] == "plain answer"
        assert result["model_used"] == "primary"
        assert agent.tools == []
