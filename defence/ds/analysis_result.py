class AnalysisResult():
    def __init__(self, type: str, is_safe: bool):
        self.type: str = type
        self.is_safe: bool = is_safe

    def get_verdict(self) -> bool:
        return self.is_safe

    def get_type(self) -> str:
        return self.type
