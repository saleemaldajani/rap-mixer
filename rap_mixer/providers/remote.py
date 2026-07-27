from pydantic import SecretStr


class OpenAICompatibleSemanticAnalyzer:
    def __init__(self, provider: str, key: SecretStr):
        self.provider = provider
        self._key = key

    def analyze_bars(self, text: str) -> dict[str, float]:
        raise RuntimeError("Provider unavailable")

