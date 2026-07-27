def voice_summary(audio: dict[str, float]) -> float:
    return max(0, min(100, 40 + audio.get("dynamic_range", 0) * 2 + audio.get("onset_rate", 0) * 4))

