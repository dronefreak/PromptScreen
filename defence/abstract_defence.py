from abc import ABC, abstractmethod

class AbstractDefence(ABC):
    @abstractmethod
    def is_safe(self, query: str) -> bool:
        pass
