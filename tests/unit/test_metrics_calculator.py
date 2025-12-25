"""Tests for MetricsCalculator."""
from datetime import datetime

import pytest
from promptscreen.evaluation import AttackResult, MetricsCalculator


class TestMetricsCalculator:
    """Test suite for MetricsCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return MetricsCalculator()

    @pytest.fixture
    def successful_attack(self):
        """Create successful attack result."""
        return AttackResult(
            prompt="attack prompt",
            response="harmful response",
            classification_time=1.5,
            success=True,
            confidence=0.9,
            timestamp=datetime.now(),
            attack_type="attempted",
            model="test-model",
            reasoning="Attack succeeded",
            harmful_content=True,
            guardrail_bypass=True,
        )

    @pytest.fixture
    def failed_attack(self):
        """Create failed attack result."""
        return AttackResult(
            prompt="attack prompt",
            response="safe response",
            classification_time=2.0,
            success=False,
            confidence=0.3,
            timestamp=datetime.now(),
            attack_type="attempted",
            model="test-model",
            reasoning="Attack failed",
            harmful_content=False,
            guardrail_bypass=False,
        )

    @pytest.fixture
    def blocked_attack(self):
        """Create blocked attack result."""
        return AttackResult(
            prompt="attack prompt",
            response="",
            classification_time=0.0,
            success=False,
            confidence=0.0,
            timestamp=datetime.now(),
            attack_type="blocked",
            model="test-model",
            reasoning="Blocked by guard",
            harmful_content=False,
            guardrail_bypass=False,
        )

    def test_initialization(self, calculator):
        """Test calculator initializes with empty results."""
        assert calculator.attack_results == []

    def test_add_result(self, calculator, successful_attack):
        """Test adding a result."""
        calculator.add_result(successful_attack)
        assert len(calculator.attack_results) == 1
        assert calculator.attack_results[0] == successful_attack

    def test_calculate_asr_no_results(self, calculator):
        """Test ASR calculation with no results."""
        asr = calculator.calculate_asr()
        assert "error" in asr

    def test_calculate_asr_all_blocked(self, calculator, blocked_attack):
        """Test ASR when all attacks are blocked."""
        calculator.add_result(blocked_attack)
        calculator.add_result(blocked_attack)

        asr = calculator.calculate_asr()
        assert asr["overall_asr"] == 0.0
        assert asr["successful_attacks"] == 0
        assert asr["blocked_attacks"] == 2
        assert asr["attempted_attacks"] == 0

    def test_calculate_asr_mixed_results(
        self, calculator, successful_attack, failed_attack, blocked_attack
    ):
        """Test ASR with mixed results."""
        # 1 successful, 1 failed (both attempted), 1 blocked
        calculator.add_result(successful_attack)
        calculator.add_result(failed_attack)
        calculator.add_result(blocked_attack)

        asr = calculator.calculate_asr()

        # ASR = successful / attempted (not including blocked)
        assert asr["successful_attacks"] == 1
        assert asr["attempted_attacks"] == 2  # Only non-blocked
        assert asr["blocked_attacks"] == 1
        assert asr["overall_asr"] == 0.5  # 1 success / 2 attempts

    def test_calculate_asr_all_successful(self, calculator, successful_attack):
        """Test ASR when all attacks succeed."""
        calculator.add_result(successful_attack)
        calculator.add_result(successful_attack)

        asr = calculator.calculate_asr()
        assert asr["overall_asr"] == 1.0
        assert asr["success_percentage"] == 100.0

    def test_calculate_time_metrics_no_successful(self, calculator, failed_attack):
        """Test time metrics with no successful attacks."""
        calculator.add_result(failed_attack)

        time_metrics = calculator.calculate_time_to_classify()
        assert "error" in time_metrics

    def test_calculate_time_metrics_single_success(self, calculator, successful_attack):
        """Test time metrics with single successful attack."""
        calculator.add_result(successful_attack)

        time_metrics = calculator.calculate_time_to_classify()
        assert time_metrics["successful_attempts"] == 1
        assert time_metrics["average_time_to_classify"] == 1.5
        assert time_metrics["median_time_to_classify"] == 1.5
        assert time_metrics["std_deviation"] == 0.0

    def test_calculate_time_metrics_multiple_success(self, calculator):
        """Test time metrics with multiple successful attacks."""
        # Create 3 successful attacks with different times
        for time in [1.0, 2.0, 3.0]:
            result = AttackResult(
                prompt="test",
                response="test",
                classification_time=time,
                success=True,
                confidence=0.8,
                timestamp=datetime.now(),
                attack_type="attempted",
                model="test",
            )
            calculator.add_result(result)

        time_metrics = calculator.calculate_time_to_classify()
        assert time_metrics["successful_attempts"] == 3
        assert time_metrics["average_time_to_classify"] == 2.0
        assert time_metrics["median_time_to_classify"] == 2.0
        assert time_metrics["min_time"] == 1.0
        assert time_metrics["max_time"] == 3.0
        assert time_metrics["total_classification_time"] == 6.0

    def test_generate_report(self, calculator, successful_attack, failed_attack):
        """Test report generation."""
        calculator.add_result(successful_attack)
        calculator.add_result(failed_attack)

        asr = calculator.calculate_asr()
        time = calculator.calculate_time_to_classify()
        report = calculator.generate_report(asr, time)

        assert "LLM Attack Simulation Report" in report
        assert "ATTACK SUCCESS RATE METRICS" in report
        assert "TIME-TO-CLASSIFY METRICS" in report
