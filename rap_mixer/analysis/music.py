def music_summary(audio: dict[str, float]) -> float:
    return max(0, min(100, 50 + audio.get("beat_strength", 0) * 35 - audio.get("clipping", 0) * 30))

