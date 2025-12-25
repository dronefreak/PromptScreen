"""Tests for AnalysisResult dataclass."""

from promptscreen.defence.ds.analysis_result import AnalysisResult


class TestAnalysisResult:
    """Test suite for AnalysisResult."""

    def test_safe_result_creation(self):
        """Test creating a safe result."""
        result = AnalysisResult(type="Clean", is_safe=True)

        assert result.get_verdict() is True
        assert result.get_type() == "Clean"
        assert result.is_safe is True

    def test_unsafe_result_creation(self):
        """Test creating an unsafe result."""
        result = AnalysisResult(type="Injection detected", is_safe=False)

        assert result.get_verdict() is False
        assert result.get_type() == "Injection detected"
        assert result.is_safe is False

    def test_result_with_detailed_type(self):
        """Test result with detailed reasoning."""
        result = AnalysisResult(
            type="Attack pattern detected: keyword 'ignore' found", is_safe=False
        )

        assert not result.get_verdict()
        assert "ignore" in result.get_type().lower()
