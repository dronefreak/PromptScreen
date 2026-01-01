"""Tests for YARA Scanner guard."""

import pytest

from promptscreen.defence import Scanner


class TestScanner:
    """Test suite for YARA Scanner."""

    @pytest.fixture
    def guard(self):
        """Create scanner instance with bundled rules."""
        return Scanner()  # Uses bundled YARA rules

    def test_initialization_with_bundled_rules(self, guard):
        """Test scanner initializes with bundled rules."""
        assert guard.compiled_rules is not None
        assert guard.rules_dir is not None

    def test_allows_benign_prompt(self, guard, sample_benign_prompts):
        """Test that benign prompts are allowed."""
        for prompt in sample_benign_prompts:
            result = guard.analyse(prompt)
            assert result.get_verdict() is True, f"Should allow: {prompt}"

    def test_empty_prompt(self, guard):
        """Test handling of empty prompt."""
        result = guard.analyse("")
        assert result.get_verdict() is True

    def test_whitespace_only_prompt(self, guard):
        """Test handling of whitespace-only prompt."""
        result = guard.analyse("   \n\t  ")
        assert result.get_verdict() is True

    def test_result_has_correct_type(self, guard):
        """Test that result type indicates YARA scanner."""
        result = guard.analyse("test prompt")
        assert "YARA" in result.get_type() or "yara" in result.get_type().lower()

    def test_custom_rules_directory(self, temp_model_dir):
        """Test scanner with custom rules directory (should fail gracefully)."""
        with pytest.raises(FileNotFoundError):
            Scanner(rules_dir=str(temp_model_dir / "nonexistent"))
