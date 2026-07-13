from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from app.agent import SYSTEM_PROMPT, ProductionAgent


def _mock_llm(*responses):
    """Build a mock LLM whose bind_tools() returns itself.

    ProductionAgent calls `.bind_tools(tools)` on the model; returning self
    keeps the configured .invoke responses in effect after binding.
    Pass one response to reuse it for every call, or several to script a
    sequence across tool-calling rounds.
    """
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    if len(responses) == 1:
        llm.invoke.return_value = responses[0]
    else:
        llm.invoke.side_effect = list(responses)
    return llm


def _tool_call(query, call_id="1"):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search_documents", "args": {"query": query}, "id": call_id}
        ],
    )


# ProductionAgent.__init__
class TestInit:
    def test_rag_enabled_binds_tools(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as cls:
            cls.return_value = _mock_llm(AIMessage(content="hi"))
            agent = ProductionAgent(retriever=mock_retriever)
        assert agent.rag_enabled is True
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "search_documents"

    def test_rag_disabled_has_no_tools(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as cls:
            cls.return_value = _mock_llm(AIMessage(content="hi"))
            agent = ProductionAgent()
        assert agent.rag_enabled is False
        assert agent.tools == []


# Tool-calling behavior
class TestToolCalling:
    def test_knowledge_question_triggers_search(self, mock_settings, mock_retriever):
        # First the LLM asks to search; after seeing results it answers.
        llm = _mock_llm(
            _tool_call("What is Python?"),
            AIMessage(content="Python is a programming language."),
        )
        with patch("app.agent.ChatOpenAI", return_value=llm):
            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("What is Python?")

        mock_retriever.search.assert_called_once_with(
            query="What is Python?", top_k=5, threshold=0.55
        )
        assert result["response"] == "Python is a programming language."
        assert result["model_used"] == "primary"
        assert result["error"] is None

    def test_general_question_does_not_search(self, mock_settings, mock_retriever):
        # No tool_calls -> the agent answers directly, retriever untouched.
        llm = _mock_llm(AIMessage(content="Hello!"))
        with patch("app.agent.ChatOpenAI", return_value=llm):
            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("hi")

        mock_retriever.search.assert_not_called()
        assert result["response"] == "Hello!"
        assert result["sources"] == []

    def test_max_tool_calls_guard_caps_loop(self, mock_settings, mock_retriever):
        # LLM always wants to search; the guard must stop it at max_tool_calls.
        # Each round returns a fresh message with a unique tool-call id so the
        # add_messages reducer appends (rather than deduping a reused instance).
        mock_settings.max_tool_calls = 3
        rounds = {"n": 0}

        def always_search(_messages):
            rounds["n"] += 1
            return _tool_call("loop", call_id=str(rounds["n"]))

        llm = _mock_llm()
        llm.invoke.side_effect = always_search
        with patch("app.agent.ChatOpenAI", return_value=llm):
            agent = ProductionAgent(retriever=mock_retriever)

        agent.invoke("keep searching")

        assert mock_retriever.search.call_count == 3

    def test_system_prompt_is_tool_aware(self, mock_settings, mock_retriever):
        llm = _mock_llm(AIMessage(content="answer"))
        with patch("app.agent.ChatOpenAI", return_value=llm):
            agent = ProductionAgent(retriever=mock_retriever)

        agent.invoke("hi")

        sent = llm.invoke.call_args[0][0]
        assert sent[0].content == SYSTEM_PROMPT
        assert "search_documents" in sent[0].content


# Source extraction from tool messages
class TestSourceExtraction:
    def test_extracts_sources_from_tool_results(self, mock_settings, mock_retriever):
        llm = _mock_llm(
            _tool_call("Python"),
            AIMessage(content="Python is a language."),
        )
        with patch("app.agent.ChatOpenAI", return_value=llm):
            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("What is Python?")

        srcs = result["sources"]
        assert [s["source"] for s in srcs] == ["intro.pdf", "history.pdf"]
        assert srcs[0]["chunk_preview"].startswith("Python is a programming language.")
        assert srcs[0]["similarity"] is None

    def test_no_sources_when_tool_finds_nothing(self, mock_settings):
        retriever = MagicMock()
        retriever.search.return_value = []
        llm = _mock_llm(
            _tool_call("obscure"),
            AIMessage(content="I don't have sufficient context."),
        )
        with patch("app.agent.ChatOpenAI", return_value=llm):
            agent = ProductionAgent(retriever=retriever)

        result = agent.invoke("something obscure")
        assert result["sources"] == []

    def test_extract_sources_ignores_non_tool_messages(self, mock_settings):
        with patch("app.agent.ChatOpenAI", return_value=_mock_llm(AIMessage("x"))):
            agent = ProductionAgent()
        messages = [
            AIMessage(content="just a chat reply"),
            ToolMessage(
                content="[Source: a.pdf]\nbody", name="other_tool", tool_call_id="1"
            ),
        ]
        assert agent._extract_sources(messages) == []


# Model fallback
class TestFallback:
    def test_fallback_used_when_primary_fails(self, mock_settings, mock_retriever):
        with patch("app.agent.ChatOpenAI") as cls:
            primary = _mock_llm()
            primary.invoke.side_effect = RuntimeError("primary down")
            fallback = _mock_llm(AIMessage(content="fallback answer"))
            cls.side_effect = [primary, fallback]
            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("What is Python?")
        assert result["model_used"] == "fallback"
        assert result["response"] == "fallback answer"

    def test_error_handler_when_both_fail(self, mock_settings):
        with patch("app.agent.ChatOpenAI") as cls:
            primary = _mock_llm()
            primary.invoke.side_effect = RuntimeError("primary down")
            fallback = _mock_llm()
            fallback.invoke.side_effect = RuntimeError("fallback down")
            cls.side_effect = [primary, fallback]
            agent = ProductionAgent()

        result = agent.invoke("hello")
        assert result["model_used"] == "error_handler"
        assert "trouble processing" in result["response"]


# invoke return contract
class TestInvokeReturn:
    def test_result_has_required_keys(self, mock_settings, mock_retriever):
        llm = _mock_llm(AIMessage(content="answer"))
        with patch("app.agent.ChatOpenAI", return_value=llm):
            agent = ProductionAgent(retriever=mock_retriever)

        result = agent.invoke("test")
        assert set(result) >= {"response", "model_used", "error", "sources"}
        assert result["response"] == "answer"
        assert result["model_used"] == "primary"
        assert result["error"] is None
