"""PromptScreen - Production-ready prompt injection defense for LLMs.

A comprehensive library for defending Large Language Models against 
prompt injection and jailbreak attacks.
"""

__version__ = "0.1.0"

# Core defence classes
from .defence import (
    AbstractDefence,
    AnalysisResult,
    HeuristicVectorAnalyzer,
    JailbreakInferenceAPI,
    ClassifierCluster,
    ShieldGemma2BClassifier,
    VectorDBScanner,
    Scanner,
    InjectionScanner,
    PolymorphicPromptAssembler,
)

# Evaluation framework (optional - only if user installs [eval] extras)
try:
    from .evaluation import AttackEvaluator, MetricsCalculator
    _has_evaluation = True
except ImportError:
    _has_evaluation = False
    AttackEvaluator = None  # type: ignore
    MetricsCalculator = None  # type: ignore

# API server (optional - only if user installs [api] extras)
try:
    from .api import create_app
    _has_api = True
except ImportError:
    _has_api = False
    create_app = None  # type: ignore

__all__ = [
    "__version__",
    "AbstractDefence",
    "AnalysisResult",
    "HeuristicVectorAnalyzer",
    "JailbreakInferenceAPI",
    "ClassifierCluster",
    "ShieldGemma2BClassifier",
    "VectorDBScanner",
    "Scanner",
    "InjectionScanner",
    "PolymorphicPromptAssembler",
    "AttackEvaluator",
    "MetricsCalculator",
    "create_app",
]