"""Defence implementations for prompt injection detection."""

from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult
from .heuristic_channel import HeuristicVectorAnalyzer
from .injection_regex import InjectionScanner
from .linear_svm import JailbreakInferenceAPI
from .ppa_defence import PolymorphicPromptAssembler
from .scanner import Scanner
from .vectordb import VectorDB, VectorDBScanner

# Optional ML guards (require torch/transformers)
try:
    from .classifier_cluster import ClassifierCluster

    _has_ml = True
except ImportError:
    ClassifierCluster = None  # type: ignore
    _has_ml = False

try:
    from .shieldgemma import ShieldGemma2BClassifier

    _has_shieldgemma = True
except ImportError:
    ShieldGemma2BClassifier = None  # type: ignore
    _has_shieldgemma = False

__all__ = [
    "AbstractDefence",
    "AnalysisResult",
    "HeuristicVectorAnalyzer",
    "JailbreakInferenceAPI",
    "Scanner",
    "InjectionScanner",
    "PolymorphicPromptAssembler",
    "VectorDB",
    "VectorDBScanner",
    # Optional - may be None if ML deps not installed
    "ClassifierCluster",
    "ShieldGemma2BClassifier",
]
