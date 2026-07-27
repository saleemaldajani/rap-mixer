def bar_duration(bpm: float, beats_per_bar: int = 4) -> float:
    return beats_per_bar * 60 / max(1, bpm)

