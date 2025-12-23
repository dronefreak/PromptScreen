"""PromptScreen - Production-ready prompt injection defense for LLMs.

A comprehensive library for defending Large Language Models against 
prompt injection and jailbreak attacks.
"""

__version__ = "0.1.0"

# Import commonly used classes for convenience
from .defence import (
    AbstractDefence,
    AnalysisResult,
    HeuristicVectorAnalyzer,
    JailbreakInferenceAPI,
)

__all__ = [
    "__version__",
    "AbstractDefence",
    "AnalysisResult",
    "HeuristicVectorAnalyzer",
    "JailbreakInferenceAPI",
]