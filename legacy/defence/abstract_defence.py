from abc import ABC, abstractmethod
from defence.ds.analysis_result import AnalysisResult

class AbstractDefence(ABC):
    @abstractmethod
    def analyse(self, query: str) -> AnalysisResult:
        pass
