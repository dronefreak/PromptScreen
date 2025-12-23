"""Tests for AttackResult dataclass."""
import pytest
from datetime import datetime
from promptscreen.evaluation import AttackResult


class TestAttackResult:
    """Test suite for AttackResult."""

    def test_creation_with_all_fields(self):
        """Test creating result with all fields."""
        result = AttackResult(
            prompt="test prompt",
            response="test response",
            classification_time=1.5,
            success=True,
            confidence=0.9,
            timestamp=datetime.now(),
            attack_type="attempted",
            model="test-model",
            reasoning="Test reasoning",
            harmful_content=True,
            guardrail_bypass=True,
        )

        assert result.prompt == "test prompt"
        assert result.response == "test response"
        assert result.classification_time == 1.5
        assert result.success is True
        assert result.confidence == 0.9
        assert result.attack_type == "attempted"

    def test_creation_minimal_fields(self):
        """Test creating result with minimal fields."""
        result = AttackResult(
            prompt="test",
            response="test",
            classification_time=0.0,
            success=False,
            confidence=0.0,
            timestamp=datetime.now(),
            attack_type="blocked",
            model="test",
        )

        assert result.reasoning is None
        assert result.harmful_content is None
        assert result.guardrail_bypass is None

    def test_blocked_attack_result(self):
        """Test result for blocked attack."""
        result = AttackResult(
            prompt="malicious prompt",
            response="",
            classification_time=0.0,
            success=False,
            confidence=0.0,
            timestamp=datetime.now(),
            attack_type="blocked",
            model="test",
            reasoning="Blocked by guard",
        )

        assert result.attack_type == "blocked"
        assert result.response == ""
        assert result.success is False