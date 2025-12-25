"""Tests for QueryAgent utility."""
import pytest
from promptscreen.utils import QueryAgent


class TestQueryAgent:
    """Test suite for QueryAgent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        # Note: This won't actually call Ollama in tests
        return QueryAgent(model="test-model", system_prompt="You are helpful")

    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.system_prompt == "You are helpful"
        assert agent.history_lst == []

    def test_clear_history(self, agent):
        """Test clearing conversation history."""
        agent.history_lst = [("q1", "a1"), ("q2", "a2")]
        agent.clear_history()
        assert agent.history_lst == []

    def test_generate_query_without_history(self, agent):
        """Test query generation without history."""
        prompt = agent._generate_query("test query", history=False)
        assert "test query" in prompt
        assert "System prompt" in prompt
        assert "Conversation History" not in prompt

    def test_generate_query_with_history(self, agent):
        """Test query generation with history."""
        agent.history_lst = [("previous question", "previous answer")]
        prompt = agent._generate_query("new query", history=True)

        assert "new query" in prompt
        assert "Conversation History" in prompt
        assert "previous question" in prompt
        assert "previous answer" in prompt
