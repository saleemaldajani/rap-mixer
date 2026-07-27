from rap_mixer.analysis.lyrics import lyric_features


class DeterministicSemanticAnalyzer:
    def analyze_bars(self, text: str) -> dict[str, float]:
        return lyric_features(text)

