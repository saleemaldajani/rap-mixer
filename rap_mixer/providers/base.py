from typing import Protocol


class SemanticAnalyzer(Protocol):
    def analyze_bars(self, text: str) -> dict[str, float]: ...

